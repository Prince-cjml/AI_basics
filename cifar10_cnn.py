import os
import argparse
import datetime
import json
import logging
from contextlib import nullcontext
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.distributed as dist

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP


def get_model_state_dict(model):
    # For wrapped models, persist the underlying module weights.
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state_dict(model, state_dict, strict=True):
    """
    Load checkpoints saved from either single-GPU or DataParallel runs.
    Handles key prefix conversion for "module." automatically.
    """
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if not model_keys or not ckpt_keys:
        model.load_state_dict(state_dict, strict=strict)
        return

    model_has_module_prefix = model_keys[0].startswith('module.')
    ckpt_has_module_prefix = ckpt_keys[0].startswith('module.')

    if model_has_module_prefix == ckpt_has_module_prefix:
        model.load_state_dict(state_dict, strict=strict)
        return

    if ckpt_has_module_prefix and not model_has_module_prefix:
        converted = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(converted, strict=strict)
        return

    if model_has_module_prefix and not ckpt_has_module_prefix:
        converted = {f'module.{k}': v for k, v in state_dict.items()}
        model.load_state_dict(converted, strict=strict)
        return


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    if not is_dist_avail_and_initialized():
        return True
    return dist.get_rank() == 0


def reduce_sum(value, device, dtype=torch.float64):
    tensor = torch.tensor(value, device=device, dtype=dtype)
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN training')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of DataLoader worker processes')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'],
                        help='optimizer to use (sgd, adam, adamw)')
    parser.add_argument('--multi-gpu', type=str, default='auto', choices=['auto', 'ddp', 'off'],
                        help='multi-GPU mode: auto (use DDP when launched with torchrun), ddp (require DDP), off (single GPU)')
    parser.add_argument('--min-per-gpu-batch', type=int, default=64,
                        help='reserved for future auto policy tuning')
    # Compat with torch.distributed.launch style arguments.
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--amp', dest='amp', action='store_true',
                        help='enable mixed precision training on CUDA (default: enabled)')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                        help='disable mixed precision training')

    # Model structure
    parser.add_argument('--conv-layers', type=int, default=3, help='number of conv blocks')
    parser.add_argument('--conv-channels', type=int, nargs='+', default=[64, 128, 256],
                        help='output channels for each conv block')
    parser.add_argument('--pool-every', type=int, default=2,
                        help='insert pooling after every N conv blocks (0 = none)')
    parser.add_argument('--pool-type', type=str, default='max', choices=['max', 'avg', 'none'])
    parser.add_argument('--fc-layers', type=int, nargs='*', default=[512],
                        help='sizes for fully-connected layers')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--arch', type=str, default='cnn', choices=['cnn', 'cnn-vit'],
                        help='model architecture to use')

    # ViT-specific params (used when --arch cnn-vit)
    parser.add_argument('--vit-layers', type=int, default=6, help='number of transformer encoder layers for ViT')
    parser.add_argument('--vit-dim', type=int, default=256, help='latent dimension / embedding size for ViT')
    parser.add_argument('--vit-heads', type=int, default=8, help='number of attention heads for ViT')
    parser.add_argument('--vit-mlp-dim', type=int, default=512, help='feedforward hidden size in transformer')
    parser.add_argument('--vit-dropout', type=float, default=0.1, help='dropout inside transformer layers')

    # Data & logging
    parser.add_argument('--disable-augment', action='store_true')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'none'])
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--model-name', type=str, default='cnn', help='model name for checkpoint and figure folders')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many training steps between progress logs')

    parser.set_defaults(amp=True)
    return parser.parse_args()


class Net(nn.Module):
    """
    Build a configurable CNN from lists of conv channels and FC sizes.
    Each conv block: Conv2d -> BatchNorm2d -> ReLU -> optional Pool
    After convs we apply AdaptiveAvgPool2d((1,1)) then FC stack.
    """
    def __init__(self, in_channels=3, conv_channels=(64, 128, 256), pool_every=2, pool_type='max', fc_layers=(512,), dropout=0.5, num_classes=10):
        super().__init__()
        assert len(conv_channels) >= 1
        self.conv_blocks = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(conv_channels):
            layers = [nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True)]
            # decide pooling insertion by position and pool_every
            if pool_every > 0 and ((i + 1) % pool_every == 0):
                if pool_type == 'max':
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                elif pool_type == 'avg':
                    layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            self.conv_blocks.append(nn.Sequential(*layers))
            prev = ch

        # global pooling to fixed-size feature
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # build fc stack
        fc_modules = []
        in_feat = prev * 1 * 1
        for idx, fch in enumerate(fc_layers):
            fc_modules.append(nn.Linear(in_feat, fch, bias=False))
            fc_modules.append(nn.BatchNorm1d(fch))
            fc_modules.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                fc_modules.append(nn.Dropout(dropout))
            in_feat = fch
        # final classifier
        fc_modules.append(nn.Linear(in_feat, num_classes))
        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_experiment_key(args):
    conv_spec = '-'.join(str(c) for c in args.conv_channels[:args.conv_layers])
    fc_spec = '-'.join(str(f) for f in args.fc_layers) if args.fc_layers else 'none'
    # include scheduler description in key for reproducibility
    if args.scheduler == 'step':
        sched_spec = f'step_s{args.step_size}_g{args.gamma}'
    elif args.scheduler == 'cosine':
        sched_spec = 'cosine'
    else:
        sched_spec = 'nosched'
    # include optimizer in key
    opt_spec = args.optimizer
    key = f'conv{conv_spec}_fc{fc_spec}_bs{args.batch_size}_opt{opt_spec}_lr{args.lr}_sched{sched_spec}_drop{args.dropout}_seed{args.seed}'
    # include architecture and ViT params if requested
    if getattr(args, 'arch', 'cnn') == 'cnn-vit':
        key = f'arch{args.arch}_vitl{args.vit_layers}_vitd{args.vit_dim}_vith{args.vit_heads}_vitm{args.vit_mlp_dim}_' + key
    else:
        key = f'arch{getattr(args, "arch", "cnn")}_' + key
    return key


class CNN_ViT(nn.Module):
    """
    CNN feature extractor whose spatial feature-map tokens are fed to a ViT encoder.
    The conv blocks mirror the original Net conv construction (no final FCs).
    """
    def __init__(self, in_channels=3, conv_channels=(64, 128, 256), pool_every=2, pool_type='max',
                 vit_layers=6, vit_dim=256, vit_heads=8, vit_mlp_dim=512, vit_dropout=0.1, num_classes=10, input_size=32):
        super().__init__()
        assert len(conv_channels) >= 1
        self.conv_blocks = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(conv_channels):
            layers = [nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                      nn.BatchNorm2d(ch),
                      nn.ReLU(inplace=True)]
            if pool_every > 0 and ((i + 1) % pool_every == 0):
                if pool_type == 'max':
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                elif pool_type == 'avg':
                    layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            self.conv_blocks.append(nn.Sequential(*layers))
            prev = ch

        # compute spatial size after convs (assumes input spatial size known and integer division by powers of two)
        pools = 0 if pool_every == 0 else (len(conv_channels) // pool_every)
        down = 2 ** pools
        assert input_size % down == 0, 'input size not divisible by pooling factor'
        self.spatial = input_size // down
        self.num_patches = self.spatial * self.spatial
        self.patch_dim = prev

        # projection from CNN feature dim to ViT latent dim
        self.proj = nn.Linear(self.patch_dim, vit_dim)

        # class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, vit_dim))

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=vit_dim, nhead=vit_heads,
                                                   dim_feedforward=vit_mlp_dim, dropout=vit_dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=vit_layers)

        # final classifier from cls token
        self.classifier = nn.Linear(vit_dim, num_classes)

        # initialize params
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C) where N=H*W
        x = self.proj(x)  # (B, N, vit_dim)

        cls = self.cls_token.expand(B, -1, -1)
        # if pos_embed size mismatches (shouldn't for fixed args), interpolate
        if self.pos_embed.size(1) != 1 + x.size(1):
            # recreate pos_embed to match current sequence length
            pos = torch.zeros(1, 1 + x.size(1), self.pos_embed.size(2), device=x.device)
            nn.init.trunc_normal_(pos, std=0.02)
            pos_embed = pos
        else:
            pos_embed = self.pos_embed

        x = torch.cat([cls, x], dim=1)
        x = x + pos_embed
        x = self.encoder(x)
        cls_out = x[:, 0]
        logits = self.classifier(cls_out)
        return logits


def get_ckpt_dir(args):
    key = build_experiment_key(args)
    d = os.path.join('model_ckpts', f"{args.model_name}_{key}")
    os.makedirs(d, exist_ok=True)
    return d


def get_fig_dir(args):
    key = build_experiment_key(args)
    d = os.path.join('figures', f"{args.model_name}_{key}")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    args = parse_args()

    # Use spawn workers to avoid CUDA+fork deadlocks on Linux.
    try:
        torch.multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass

    # Distributed setup from torchrun environment.
    env_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    env_rank = int(os.environ.get('RANK', '0'))
    env_local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    use_ddp = False
    if args.multi_gpu in ('auto', 'ddp') and env_world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError('DDP requires CUDA; launch without torchrun or use --multi-gpu off on CPU-only hosts.')
        dist.init_process_group(backend='nccl', init_method='env://')
        use_ddp = True
        env_rank = dist.get_rank()
        env_world_size = dist.get_world_size()
        env_local_rank = int(os.environ.get('LOCAL_RANK', str(env_rank)))
        torch.cuda.set_device(env_local_rank)
    elif args.multi_gpu == 'ddp' and env_world_size <= 1:
        raise RuntimeError('`--multi-gpu ddp` requires torchrun, e.g. `torchrun --nproc_per_node=8 cifar10_cnn.py ...`')

    # When torchrun is used with --multi-gpu off, only rank0 should continue.
    if args.multi_gpu == 'off' and env_world_size > 1 and env_rank != 0:
        return

    # Setup logging: ensure `model_ckpts/` and `logs/` exist and log into `logs/`
    os.makedirs('model_ckpts', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    run_key = build_experiment_key(args)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_fname = os.path.join('logs', f"{args.model_name}_{run_key}_{ts}.log")
    logger = logging.getLogger(f'cifar10_train_rank{env_rank}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s %(message)s')
    if is_main_process():
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        fh = logging.FileHandler(log_fname)
        fh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    if is_main_process():
        logger.info('Logging to file: %s', log_fname)
        logger.info('Run config: %s', json.dumps(vars(args), sort_keys=True))

    # Reproducibility
    seed = args.seed + (env_rank if use_ddp else 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # Data transforms (preserve exact normalization requested)
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)

    if args.disable_augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std) # Maps [0, 1] to [-1, 1]
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # Datasets and loaders
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    local_batch_size = args.batch_size
    if use_ddp:
        if args.batch_size < env_world_size:
            raise ValueError(f'batch-size ({args.batch_size}) must be >= WORLD_SIZE ({env_world_size}) for DDP')
        if args.batch_size % env_world_size != 0 and is_main_process():
            logger.warning('Global batch size %d is not divisible by world size %d; using floor division per-rank batch %d.',
                           args.batch_size, env_world_size, args.batch_size // env_world_size)
        local_batch_size = max(1, args.batch_size // env_world_size)

    loader_workers = args.num_workers if not use_ddp else max(0, args.num_workers // env_world_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=env_world_size, rank=env_rank, shuffle=True) if use_ddp else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=env_world_size, rank=env_rank, shuffle=False) if use_ddp else None

    train_loader_kwargs = {
        'batch_size': local_batch_size,
        'shuffle': train_sampler is None,
        'sampler': train_sampler,
        'num_workers': loader_workers,
        'pin_memory': True,
    }
    test_loader_kwargs = {
        'batch_size': local_batch_size,
        'shuffle': False,
        'sampler': test_sampler,
        'num_workers': loader_workers,
        'pin_memory': True,
    }
    if loader_workers > 0:
        train_loader_kwargs['multiprocessing_context'] = 'spawn'
        train_loader_kwargs['persistent_workers'] = True
        test_loader_kwargs['multiprocessing_context'] = 'spawn'
        test_loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    test_loader = DataLoader(test_dataset, **test_loader_kwargs)

    # Model
    conv_chans = args.conv_channels
    # ensure conv_chans length matches conv_layers if provided fewer values
    if len(conv_chans) < args.conv_layers:
        # extend with last value
        conv_chans = conv_chans + [conv_chans[-1]] * (args.conv_layers - len(conv_chans))
    else:
        conv_chans = conv_chans[:args.conv_layers]

    if args.arch == 'cnn':
        model = Net(conv_channels=conv_chans, pool_every=args.pool_every, pool_type=args.pool_type,
                    fc_layers=tuple(args.fc_layers), dropout=args.dropout)
    elif args.arch == 'cnn-vit':
        model = CNN_ViT(conv_channels=conv_chans, pool_every=args.pool_every, pool_type=args.pool_type,
                        vit_layers=args.vit_layers, vit_dim=args.vit_dim, vit_heads=args.vit_heads,
                        vit_mlp_dim=args.vit_mlp_dim, vit_dropout=args.vit_dropout, num_classes=10,
                        input_size=32)
    else:
        model = Net(conv_channels=conv_chans, pool_every=args.pool_every, pool_type=args.pool_type,
                    fc_layers=tuple(args.fc_layers), dropout=args.dropout)

    # Device selection (MLU support retained if available)
    use_mlu = False
    try:
        use_mlu = torch.mlu.is_available()
    except Exception:
        use_mlu = False

    if use_mlu:
        device = torch.device('mlu:0')
    else:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{env_local_rank}' if use_ddp else 'cuda:0')
        else:
            device = torch.device('cpu')

    model = model.to(device)
    if device.type == 'cuda':
        # Ampere+ benefits from TF32 for conv/matmul-heavy training workloads.
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if use_ddp:
        model = DDP(model, device_ids=[env_local_rank], output_device=env_local_rank)
        logger.info('Enabled DistributedDataParallel on %d GPUs (rank %d, local_rank %d)', env_world_size, env_rank, env_local_rank)
    elif device.type == 'cuda':
        visible = torch.cuda.device_count()
        if visible > 1 and args.multi_gpu != 'off':
            logger.info('Multiple GPUs visible (%d) but DDP is off. Use torchrun to enable high-performance multi-GPU.', visible)
        logger.info('Single-GPU training on %s', device)
    logger.info(f'Using device: {device}')
    if is_main_process():
        logger.info('Batch sizes: global=%d local=%d workers_per_rank=%d', args.batch_size,
                    local_batch_size, loader_workers)

    # Loss / optimizer / scheduler
    criterion = nn.CrossEntropyLoss()
    # create optimizer based on user choice
    opt_name = args.optimizer.lower()
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        # fallback to SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    logger.info('Using optimizer: %s', opt_name)
    if opt_name in ('adam', 'adamw') and args.lr > 0.01:
        logger.warning('Learning rate %.4g is unusually high for %s (typical is around 1e-3).', args.lr, opt_name)
    scheduler = None
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Optional wandb
    if args.use_wandb and is_main_process():
        try:
            import wandb
            wandb.init(project='cifar10_cnn', config=vars(args))
            use_wandb = True
        except Exception as e:
            logger.warning('wandb requested but not available or failed to init: %s', e)
            use_wandb = False
    else:
        use_wandb = False

    start_epoch = 0
    best_acc = 0.0

    # Resume checkpoint if requested
    if args.resume:
        if os.path.isfile(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            load_model_state_dict(model, ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optim_state'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_acc = ckpt.get('best_acc', 0.0)
            if is_main_process():
                logger.info(f"Resumed from {args.resume}, starting at epoch {start_epoch}")
        else:
            if is_main_process():
                logger.warning('Resume path not found: %s', args.resume)

    # training history for plotting
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    # Training loop
    use_amp = bool(args.amp and device.type == 'cuda' and not use_mlu)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if is_main_process():
        logger.info('Mixed precision (AMP): %s', 'enabled' if use_amp else 'disabled')
        logger.info('Starting training loop from epoch %d to %d', start_epoch + 1, args.epochs)
    for epoch in range(start_epoch, args.epochs):
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        steps_per_epoch = len(train_loader)
        if is_main_process():
            logger.info('Epoch %d has %d train steps per rank (log interval: %d)', epoch + 1, steps_per_epoch, args.log_interval)

        amp_ctx = (lambda: torch.amp.autocast('cuda', enabled=True)) if use_amp else nullcontext
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            with amp_ctx():
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(1)
            batch_acc = (preds == labels).float().sum().item()
            running_loss += loss.item() * images.size(0)
            running_acc += batch_acc
            total += images.size(0)

            # Always show first and last step so long intervals never look frozen.
            if is_main_process() and ((i + 1) == 1 or (i + 1) % args.log_interval == 0 or (i + 1) == steps_per_epoch):
                logger.info('Epoch [%d/%d] Step [%d/%d] Loss: %.4f', epoch + 1, args.epochs, i + 1, len(train_loader), loss.item())

        global_loss_sum = reduce_sum(running_loss, device)
        global_correct_sum = reduce_sum(running_acc, device)
        global_count = reduce_sum(total, device)
        epoch_loss = global_loss_sum / max(1.0, global_count)
        epoch_acc = global_correct_sum / max(1.0, global_count)
        if is_main_process():
            logger.info('Epoch %d Train Loss: %.4f Acc: %.4f', epoch + 1, epoch_loss, epoch_acc)
        if use_wandb:
            wandb.log({'train/loss': epoch_loss, 'train/acc': epoch_acc, 'epoch': epoch + 1})

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                with amp_ctx():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        global_test_loss_sum = reduce_sum(test_loss, device)
        global_test_correct = reduce_sum(correct, device)
        global_test_count = reduce_sum(total, device)
        test_loss = global_test_loss_sum / max(1.0, global_test_count)
        test_acc = global_test_correct / max(1.0, global_test_count)
        if is_main_process():
            logger.info('Epoch %d Test Loss: %.4f Acc: %.4f', epoch + 1, test_loss, test_acc)
        if use_wandb:
            wandb.log({'test/loss': test_loss, 'test/acc': test_acc, 'epoch': epoch + 1})

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        # record history
        if is_main_process():
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

        # Checkpoint: save in experiment-specific folder with epoch and test-acc in filename
        if is_main_process():
            ckpt_dir = get_ckpt_dir(args)
            save_dict = {
                'epoch': epoch,
                'model_state': get_model_state_dict(model),
                'optim_state': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': vars(args)
            }
            # save last
            torch.save(save_dict, os.path.join(ckpt_dir, 'last.pth'))

            # save epoch file with test accuracy
            epoch_fname = f'epoch_{epoch+1}_acc{test_acc:.4f}.pth'
            torch.save(save_dict, os.path.join(ckpt_dir, epoch_fname))

            # save best
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({**save_dict, 'best_acc': best_acc}, os.path.join(ckpt_dir, 'best.pth'))
                logger.info('Saved new best checkpoint: %s', os.path.join(ckpt_dir, 'best.pth'))

            # Save figures after each epoch
            fig_dir = get_fig_dir(args)
            # accuracy plot
            try:
                epochs = list(range(1, len(history['train_acc']) + 1))
                plt.figure()
                plt.plot(epochs, history['train_acc'], label='train_acc')
                plt.plot(epochs, history['test_acc'], label='test_acc')
                plt.xlabel('epoch')
                plt.ylabel('accuracy')
                plt.legend()
                plt.grid(True)
                acc_path = os.path.join(fig_dir, 'accuracy.png')
                plt.savefig(acc_path)
                plt.close()

                # loss plot
                plt.figure()
                plt.plot(epochs, history['train_loss'], label='train_loss')
                plt.plot(epochs, history['test_loss'], label='test_loss')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.legend()
                plt.grid(True)
                loss_path = os.path.join(fig_dir, 'loss.png')
                plt.savefig(loss_path)
                plt.close()
            except Exception as e:
                logger.warning('Failed to save plots: %s', e)

            # persist machine-readable epoch metrics under logs/
            metrics_path = os.path.join('logs', f"{args.model_name}_{run_key}_{ts}_metrics.json")
            try:
                with open(metrics_path, 'w') as f:
                    json.dump(history, f, indent=2)
            except Exception as e:
                logger.warning('Failed to write metrics json: %s', e)

    if is_main_process():
        logger.info('Training complete. Best test acc: %.4f', best_acc)

    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
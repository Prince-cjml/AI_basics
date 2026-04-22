#!/usr/bin/env python3
"""
简单的 Markdown -> PDF 转换器脚本。
优先尝试使用 pypandoc（需系统安装 pandoc），否则调用系统 pandoc。
用法:
  python scripts/md_to_pdf.py REPORT.md -o REPORT.pdf
"""
import argparse
import os
import shutil
import subprocess
import sys


def convert_with_pypandoc(md_path, out_path):
    try:
        import pypandoc
    except Exception:
        return False
    try:
        pypandoc.convert_file(md_path, 'pdf', outputfile=out_path)
        return True
    except Exception:
        return False


def convert_with_pandoc(md_path, out_path):
    pandoc = shutil.which('pandoc')
    if not pandoc:
        return False
    # Try with xelatex engine first for better font handling
    cmd = [pandoc, md_path, '-o', out_path, '--pdf-engine=xelatex']
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        # fallback without specifying engine
        try:
            subprocess.check_call([pandoc, md_path, '-o', out_path])
            return True
        except subprocess.CalledProcessError:
            return False


def main():
    parser = argparse.ArgumentParser(description='Convert Markdown to PDF')
    parser.add_argument('md', help='Markdown file to convert')
    parser.add_argument('-o', '--output', help='Output PDF path', default=None)
    args = parser.parse_args()

    md_path = args.md
    if not os.path.exists(md_path):
        print(f'Markdown file not found: {md_path}', file=sys.stderr)
        sys.exit(2)

    out_path = args.output or os.path.splitext(md_path)[0] + '.pdf'

    # Try pypandoc first
    ok = convert_with_pypandoc(md_path, out_path)
    if ok:
        print(f'Converted {md_path} -> {out_path} via pypandoc')
        return

    # Try system pandoc
    ok = convert_with_pandoc(md_path, out_path)
    if ok:
        print(f'Converted {md_path} -> {out_path} via pandoc')
        return

    print('Failed to convert Markdown to PDF. Please install pandoc (and a TeX engine like xelatex) or pypandoc.', file=sys.stderr)
    print('Ubuntu example: sudo apt install pandoc texlive-xetex', file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    main()

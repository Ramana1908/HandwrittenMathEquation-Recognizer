"""Build the training folder `data/` with the 24 symbol classes the models use.

Usage:
    python scripts/download_data.py                  # download from Kaggle via kagglehub
    python scripts/download_data.py --source PATH    # use an already-extracted copy

Kaggle downloads may require a Kaggle account and API token; see
https://github.com/Kaggle/kagglehub#authenticate
"""
import argparse
import os
import shutil
import subprocess
import sys

KAGGLE_DATASET = "xainano/handwrittenmathsymbols"

CLASSES = ['forward_slash', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           '+', '-', '(', ')', 'd', 'X', 'int', 'lt', 'leq', 'gt', 'geq', '=', 'neq']

# Alternative folder names to accept for a class, in order of preference.
ALIASES = {'forward_slash': ['forward_slash', 'forward_slash_new']}


def find_class_root(root):
    """Return the directory under `root` that contains the class folders."""
    for dirpath, dirnames, _ in os.walk(root):
        if {'int', 'neq', 'X'} <= set(dirnames):
            return dirpath
    return None


def extract_archives(root):
    """Kaggle ships the images as data.rar; unpack it next to the archive.

    Python's standard library can't read RAR, so this uses the system `tar`
    (bsdtar, built into macOS and Windows 10+), which can.
    """
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(('.rar', '.zip')):
                continue
            path = os.path.join(dirpath, name)
            print(f"Extracting {path} (this takes about a minute) ...")
            try:
                if name.endswith('.zip'):
                    shutil.unpack_archive(path, dirpath)
                else:
                    subprocess.run(['tar', '-xf', path, '-C', dirpath], check=True)
            except (OSError, subprocess.CalledProcessError, shutil.ReadError):
                sys.exit(f"Could not extract {path}. Extract it manually "
                         f"(e.g. `unar`, `7z x` or `unrar x`) and rerun with --source {dirpath}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', help='folder containing the extracted Kaggle class folders')
    parser.add_argument('--out', default='data', help='output folder (default: data)')
    args = parser.parse_args()

    if args.source:
        source_root = args.source
    else:
        try:
            import kagglehub
        except ImportError:
            sys.exit("kagglehub is not installed. Run `pip install kagglehub`, or pass --source.")
        source_root = kagglehub.dataset_download(KAGGLE_DATASET)
        print("Downloaded to", source_root)

    class_root = find_class_root(source_root)
    if class_root is None and not args.source:
        extract_archives(source_root)
        class_root = find_class_root(source_root)
    if class_root is None:
        sys.exit(f"Could not find the class folders (e.g. 'int', 'neq', 'X') under {source_root}")

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for cls in CLASSES:
        src = next((os.path.join(class_root, name) for name in ALIASES.get(cls, [cls])
                    if os.path.isdir(os.path.join(class_root, name))), None)
        if src is None:
            sys.exit(f"Class folder for '{cls}' not found in {class_root}")
        dst = os.path.join(args.out, cls)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        count = len(os.listdir(dst))
        total += count
        print(f"{cls:>14}: {count:6d} images")
    print(f"Done: {total} images in {len(CLASSES)} classes -> {args.out}/")


if __name__ == '__main__':
    main()

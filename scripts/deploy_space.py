"""Publish the web demo (app.py) as a Hugging Face Space.

    pip install huggingface_hub
    hf auth login                                   # once, with a token that has write access
    python scripts/deploy_space.py YOUR_USERNAME/handwritten-math

Creates the Space if needed and uploads only what the demo needs: the app, the
recognizer module, the trained weights, the example images and requirements.txt.
Run it again to push updates.
"""
import argparse
import os

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ['app.py', 'recognizer.py', 'requirements.txt', 'weights/*.pth', 'examples/*']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('space', help='Space id, e.g. your-username/handwritten-math')
    parser.add_argument('--private', action='store_true', help='create the Space as private')
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.space, repo_type='space', space_sdk='gradio',
                    private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.space, repo_type='space', folder_path=ROOT,
                      allow_patterns=FILES, commit_message='Update demo')
    api.upload_file(repo_id=args.space, repo_type='space',
                    path_or_fileobj=os.path.join(ROOT, 'deploy', 'space_README.md'),
                    path_in_repo='README.md', commit_message='Update Space card')
    print(f'Done. The Space builds in a few minutes: https://huggingface.co/spaces/{args.space}')


if __name__ == '__main__':
    main()

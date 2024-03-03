import os
import warnings
from pathlib import Path
from argparse import ArgumentParser

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
if HF_ACCESS_TOKEN is None:
    warnings.warn("`HF_ACCESS_TOKEN` env variable not set! Certain gated models requires authentication.")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repository ID.")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Cache directory for downloading and saving the HuggingFace models.")

    args = parser.parse_args()

    cache_dir = Path.cwd() / args.cache_dir
    cache_dir.mkdir(exist_ok=True)
    snapshot_download(repo_id=args.repo_id, cache_dir=cache_dir, token=HF_ACCESS_TOKEN)
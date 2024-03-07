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
    parser.add_argument(
        "--ignore_patterns", 
        type=str, 
        default="", 
        help=(
            "If provided, files matching any of the patterns are not downloaded. "
            "Multiple patterns should be comma separated (i.e., stablelm-*,*.gguf)"
        )
    )

    args = parser.parse_args()
    cache_dir = Path.cwd() / args.cache_dir
    cache_dir.mkdir(exist_ok=True)
    if ignore_patterns := args.ignore_patterns:
        ignore_patterns = args.ignore_patterns.split(",")
    snapshot_download(
        repo_id=args.repo_id, 
        cache_dir=cache_dir, 
        token=HF_ACCESS_TOKEN,
        ignore_patterns=ignore_patterns
    )
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a Lumora checkpoint to a Hugging Face model repo.")
    parser.add_argument("--repo-id", required=True, help="Target model repo, e.g. nur9211/ct-rate-vlm-model.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to the local .pt checkpoint.")
    parser.add_argument("--filename", default=None, help="Filename to use in the Hugging Face repo.")
    parser.add_argument("--private", action="store_true", help="Create the model repo as private if it does not exist.")
    parser.add_argument("--token", default=None, help="HF token. Omit to use cached login or HF_TOKEN.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint path is not a file: {checkpoint}")

    path_in_repo = args.filename or checkpoint.name
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=args.token)
    api = HfApi(token=args.token)
    api.upload_file(
        path_or_fileobj=checkpoint,
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Upload {path_in_repo}",
    )
    print(f"Uploaded {checkpoint} to https://huggingface.co/{args.repo_id}/blob/main/{path_in_repo}")


if __name__ == "__main__":
    main()

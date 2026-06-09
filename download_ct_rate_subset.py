import argparse
import os
import shutil
from pathlib import Path

from ct_rate_utils import (
    DEFAULT_DATA_DIR,
    REPORT_CSV_FILES,
    build_entries,
    download_hub_file,
    download_one_volume,
)


GIB = 1024**3


def gib(value):
    return value / GIB


def dir_size(path):
    path = Path(path)
    if not path.exists():
        return 0

    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except FileNotFoundError:
                pass
    return total


def free_bytes(path):
    return shutil.disk_usage(path).free


def resolve_token(token):
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def ensure_metadata(data_dir, token):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    metadata_files = ["README.md", *REPORT_CSV_FILES.values()]
    print(f"Ensuring CT-RATE metadata/report files in {data_dir.resolve()}")
    for filename in metadata_files:
        local_path = data_dir / filename
        if local_path.exists():
            print(f"  exists: {filename}")
            continue
        print(f"  downloading: {filename}")
        download_hub_file(filename, data_dir, token)


def local_volume_count(data_dir):
    data_dir = Path(data_dir)
    train_count = len(list((data_dir / "dataset" / "train_fixed").rglob("*.nii.gz")))
    valid_count = len(list((data_dir / "dataset" / "valid_fixed").rglob("*.nii.gz")))
    return train_count, valid_count


def volume_plan(data_dir, min_valid_volumes):
    train_entries, _ = build_entries(data_dir, "train")
    valid_entries, _ = build_entries(data_dir, "valid")
    _, existing_valid_count = local_volume_count(data_dir)

    valid_needed = []
    if existing_valid_count < min_valid_volumes:
        for entry in valid_entries:
            if existing_valid_count + len(valid_needed) >= min_valid_volumes:
                break
            if not Path(entry["volume_path"]).exists():
                valid_needed.append(("valid", entry))

    train_needed = [("train", entry) for entry in train_entries if not Path(entry["volume_path"]).exists()]
    return valid_needed + train_needed


def download_to_target(args):
    data_dir = Path(args.data_dir)
    token = resolve_token(args.token)
    target_bytes = int(args.target_gb * GIB)
    min_free_bytes = int(args.min_free_gb * GIB)

    ensure_metadata(data_dir, token)

    current_size = dir_size(data_dir)
    free = free_bytes(".")
    train_count, valid_count = local_volume_count(data_dir)
    print(f"Current CT-RATE size : {gib(current_size):.2f} GiB")
    print(f"Current free space   : {gib(free):.2f} GiB")
    print(f"Local volumes        : train={train_count:,}, valid={valid_count:,}")
    print(f"Target CT-RATE size  : {args.target_gb:.2f} GiB")
    print(f"Minimum free buffer  : {args.min_free_gb:.2f} GiB")

    if current_size >= target_bytes:
        print("Target already reached. Nothing to download.")
        return

    if free <= min_free_bytes:
        raise SystemExit(
            f"Not enough free space to start. Free={gib(free):.2f} GiB, "
            f"required buffer={args.min_free_gb:.2f} GiB."
        )

    plan = volume_plan(data_dir, args.min_valid_volumes)
    if not plan:
        print("No additional CT-RATE volumes were found in the report CSVs.")
        return

    downloaded = 0
    skipped = 0
    for split, entry in plan:
        current_size = dir_size(data_dir)
        free = free_bytes(".")

        if current_size >= target_bytes:
            print(f"Stop: target reached at {gib(current_size):.2f} GiB.")
            break
        if free <= min_free_bytes:
            print(f"Stop: free-space buffer reached. Free={gib(free):.2f} GiB.")
            break
        if downloaded >= args.max_new_volumes:
            print(f"Stop: max new volume limit reached ({args.max_new_volumes}).")
            break

        volume_name = entry["volume_name"]
        volume_path = Path(entry["volume_path"])
        if volume_path.exists():
            skipped += 1
            continue

        print(
            f"Downloading {split} volume {volume_name} "
            f"(data={gib(current_size):.2f} GiB, free={gib(free):.2f} GiB)"
        )
        if args.dry_run:
            downloaded += 1
            continue

        before = dir_size(data_dir)
        path = download_one_volume(split, volume_name, data_dir, token)
        after = dir_size(data_dir)
        downloaded += 1
        print(f"  saved: {path}")
        print(f"  added: {gib(after - before):.2f} GiB; total={gib(after):.2f} GiB")

    final_size = dir_size(data_dir)
    final_free = free_bytes(".")
    train_count, valid_count = local_volume_count(data_dir)
    print("\nDownload summary")
    print("-" * 32)
    print(f"Final CT-RATE size : {gib(final_size):.2f} GiB")
    print(f"Final free space   : {gib(final_free):.2f} GiB")
    print(f"Local volumes      : train={train_count:,}, valid={valid_count:,}")
    print(f"New volumes        : {downloaded:,}")
    print(f"Skipped existing   : {skipped:,}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a resumable CT-RATE subset up to a target local size."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--target-gb", type=float, default=10.0, help="Target total size for data/ct_rate.")
    parser.add_argument("--min-free-gb", type=float, default=4.0, help="Stop before free space drops below this.")
    parser.add_argument("--min-valid-volumes", type=int, default=1, help="Ensure at least this many validation volumes.")
    parser.add_argument("--max-new-volumes", type=int, default=200, help="Safety cap for one run.")
    parser.add_argument("--token", default=None, help="HF token. Omit to use cached login or HF_TOKEN env var.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without downloading files.")
    return parser.parse_args()


if __name__ == "__main__":
    download_to_target(parse_args())

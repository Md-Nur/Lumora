"""
convert_and_clean.py
──────────────────────────────────────────────────────────────────────────────
Pipeline: 3D NIfTI → 2D PNG → CSV update → delete .nii.gz → (optional) download next batch.

Workflow
────────
1. Find all local .nii.gz volumes in data/ct_rate.
2. Convert each to a single RGB PNG (224×224) using max-intensity projections
   along the axial, coronal, and sagittal axes merged as R, G, B channels.
3. Save PNGs to  data/ct_rate/images/{split}/  (same stem, .png extension).
4. Add / update an `image_path` column in both train and validation CSVs so
   the training code can load PNGs instead of NIfTI volumes.
5. Delete the original .nii.gz file to reclaim disk space.
6. (Optional) Run download_ct_rate_subset.py to pull the next batch of 3D files.

Usage
─────
# Convert all local .nii.gz, then download 5 GiB more:
    python convert_and_clean.py --then-download

# Convert only 10 volumes (good for testing):
    python convert_and_clean.py --limit 10

# Dry-run: show what would happen, no file changes:
    python convert_and_clean.py --dry-run

# Custom target after download:
    python convert_and_clean.py --then-download --target-gb 18

Options
───────
--data-dir        Path to CT-RATE data root  [default: data/ct_rate]
--image-dir       Where to store PNGs        [default: data/ct_rate/images]
--limit           Max volumes to convert in one run
--dry-run         Print plan, no changes
--then-download   After converting, run download_ct_rate_subset.py
--target-gb       Target total GB after download  [default: 18]
--min-free-gb     Minimum free-space buffer for download  [default: 5]
--min-valid       Minimum validation volumes to keep  [default: 3]
--max-new         Max new volumes to download in one run  [default: 200]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm

from ct_rate_utils import (
    DEFAULT_DATA_DIR,
    REPORT_CSV_FILES,
    nifti_to_rgb_image,
)


# ── helpers ──────────────────────────────────────────────────────────────────

GIB = 1024 ** 3


def gib(b):
    return b / GIB


def free_bytes(path="."):
    import shutil
    return shutil.disk_usage(path).free


def split_of(nii_path: Path) -> str:
    """Infer 'train' or 'valid' from the volume file path."""
    parts = nii_path.parts
    for p in parts:
        if "train" in p:
            return "train"
        if "valid" in p:
            return "valid"
    return "train"  # fallback


def png_path(nii_path: Path, image_dir: Path) -> Path:
    """Return the target PNG path for a given NIfTI path."""
    split = split_of(nii_path)
    return image_dir / split / (nii_path.stem.removesuffix(".nii") + ".png")


def find_local_volumes(data_dir: Path):
    """Return all .nii.gz files under data_dir, sorted."""
    return sorted(data_dir.rglob("*.nii.gz"))


def update_csvs(data_dir: Path, image_dir: Path, converted: dict[str, str], dry_run: bool):
    """
    Add / update the `image_path` column in both CSVs.

    converted: {volume_stem_without_ext -> absolute png path}
    """
    import pandas as pd

    for split, csv_rel in REPORT_CSV_FILES.items():
        csv_path = data_dir / csv_rel
        if not csv_path.exists():
            print(f"  [skip] CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Find the volume-name column
        vol_col = None
        for candidate in ("VolumeName", "volume_name", "volume", "Volume",
                          "FileName", "filename", "Image", "image"):
            if candidate in df.columns:
                vol_col = candidate
                break
        if vol_col is None:
            print(f"  [skip] Could not find volume column in {csv_path.name}")
            continue

        # Ensure image_path column exists
        if "image_path" not in df.columns:
            df["image_path"] = ""

        updated = 0
        for idx, row in df.iterrows():
            raw_name = str(row[vol_col]).strip()
            stem = raw_name.removesuffix(".nii.gz").removesuffix(".nii")
            if stem in converted:
                df.at[idx, "image_path"] = converted[stem]
                df.at[idx, vol_col] = stem + ".png"
                updated += 1

        print(f"  CSV [{split}]: {updated:,} rows updated → {csv_path.name}")
        if not dry_run:
            df.to_csv(csv_path, index=False)
        else:
            print("    (dry-run: not saved)")


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(args):
    data_dir = Path(args.data_dir).resolve()
    image_dir = Path(args.image_dir).resolve()

    volumes = find_local_volumes(data_dir)
    if not volumes:
        print("No .nii.gz volumes found. Nothing to convert.")
        return

    if args.limit:
        volumes = volumes[: args.limit]

    print(f"Found {len(volumes):,} local .nii.gz volume(s) to convert.")
    print(f"PNG output dir : {image_dir}")
    print(f"Free disk space: {gib(free_bytes()):.2f} GiB")
    if args.dry_run:
        print("DRY-RUN mode — no files will be written or deleted.\n")

    converted: dict[str, str] = {}   # stem → absolute png path
    errors: list[str] = []

    for nii in tqdm(volumes, desc="Converting"):
        out = png_path(nii, image_dir)
        stem = nii.name.removesuffix(".nii.gz").removesuffix(".nii")

        # Skip if PNG already exists (idempotent)
        if out.exists():
            converted[stem] = str(out)
            if not args.dry_run:
                nii.unlink()
                # Remove empty parent dirs
                for parent in [nii.parent, nii.parent.parent, nii.parent.parent.parent]:
                    try:
                        parent.rmdir()
                    except OSError:
                        break
            continue

        try:
            img = nifti_to_rgb_image(nii)
        except Exception as exc:
            errors.append(f"{nii.name}: {exc}")
            print(f"\n  [error] {nii.name}: {exc}")
            continue

        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out, format="PNG", optimize=True)
            nii.unlink()
            # Remove empty parent dirs
            for parent in [nii.parent, nii.parent.parent, nii.parent.parent.parent]:
                try:
                    parent.rmdir()
                except OSError:
                    break

        converted[stem] = str(out)

    print(f"\nConversion summary")
    print("-" * 40)
    print(f"Converted : {len(converted):,}")
    print(f"Errors    : {len(errors):,}")
    print(f"Free space: {gib(free_bytes()):.2f} GiB (after deletes)")

    if converted:
        print("\nUpdating CSV files...")
        update_csvs(data_dir, image_dir, converted, dry_run=args.dry_run)

    if errors:
        print("\nFailed volumes:")
        for e in errors:
            print(f"  {e}")

    if args.then_download:
        print("\nStarting next download batch...")
        cmd = [
            sys.executable, "download_ct_rate_subset.py",
            "--data-dir", str(args.data_dir),
            "--target-gb", str(args.target_gb),
            "--min-free-gb", str(args.min_free_gb),
            "--min-valid-volumes", str(args.min_valid),
            "--max-new-volumes", str(args.max_new),
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print("Download finished with warnings.")
        else:
            print("Download complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CT-RATE 3D .nii.gz volumes to 2D PNGs, update CSVs, free space."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path,
                        help="CT-RATE data root (default: data/ct_rate)")
    parser.add_argument("--image-dir", default=Path("data/ct_rate/images"), type=Path,
                        help="Output directory for PNG images (default: data/ct_rate/images)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max volumes to convert in one run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without making changes")
    parser.add_argument("--then-download", action="store_true",
                        help="Download next batch of 3D files after converting")
    parser.add_argument("--target-gb", type=float, default=18.0,
                        help="Target total GB for download (default: 18)")
    parser.add_argument("--min-free-gb", type=float, default=5.0,
                        help="Minimum free-space buffer during download (default: 5)")
    parser.add_argument("--min-valid", type=int, default=3,
                        help="Minimum validation volumes to ensure (default: 3)")
    parser.add_argument("--max-new", type=int, default=200,
                        help="Max new volumes to download in one run (default: 200)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

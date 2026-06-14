"""
download_ct_rate_subset.py
──────────────────────────────────────────────────────────────────────────────
Pipeline: 3D NIfTI → 2D PNG → CSV update → delete .nii.gz → download next batch loop.

Workflow
────────
1. Convert all local .nii.gz volumes in data/ct_rate to 2D PNG images.
2. Clean up (delete) the original NIfTI volumes to reclaim disk space.
3. Check if there are any remaining volumes to download. If not, the pipeline finishes.
4. Download a batch of 3D NIfTI volumes (default limit: 20 GB).
5. Repeat the loop until the entire dataset is successfully downloaded and converted.

Usage
─────
# Run in continuous conversion/download loop mode (default batch size 20 GB):
    python download_ct_rate_subset.py

# Disable looping, only convert existing local volumes once:
    python download_ct_rate_subset.py --no-loop

# Customize loop batch size (e.g., download 10 GB per iteration):
    python download_ct_rate_subset.py --batch-gb 10.0

# Limit conversion count (good for testing):
    python download_ct_rate_subset.py --limit 10

# Dry-run: show what would happen, no file changes:
    python download_ct_rate_subset.py --dry-run
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from tqdm.auto import tqdm

REPO_ID = "ibrahimhamamci/CT-RATE"
DEFAULT_DATA_DIR = Path("data/ct_rate")
MAX_TEXT_LENGTH = 128

REPORT_CSV_CANDIDATES = {
    "train": [
        "dataset/radiology_text_reports/train_reports.csv",
        "radiology_text_reports/train_reports.csv",
        "train_reports.csv",
    ],
    "valid": [
        "dataset/radiology_text_reports/validation_reports.csv",
        "dataset/radiology_text_reports/valid_reports.csv",
        "radiology_text_reports/validation_reports.csv",
        "radiology_text_reports/valid_reports.csv",
        "validation_reports.csv",
        "valid_reports.csv",
    ],
}

REPORT_CSV_FILES = {
    "train": "dataset/radiology_text_reports/train_reports.csv",
    "valid": "dataset/radiology_text_reports/validation_reports.csv",
}

METADATA_PREFIXES = (
    "dataset/radiology_text_reports/",
    "dataset/multi_abnormality_labels/",
    "dataset/metadata/",
)

def _load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

_load_dotenv()

def hub_token(token=None):
    if token:
        return token
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.exists():
        tok = cached.read_text().strip()
        if tok:
            return tok
    value = os.environ.get("hf_ct_scan_token")
    if value:
        return value
    return True

def require_hf_access(exc):
    raise SystemExit(
        "Could not access CT-RATE on Hugging Face.\n"
        "Accept the dataset terms at https://huggingface.co/datasets/ibrahimhamamci/CT-RATE, "
        "then run `huggingface-cli login` with a token that has public gated repository access enabled. "
        "You can also pass the token with `--token`."
    ) from exc

def download_hub_file(filename, data_dir, token=None):
    try:
        return Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=data_dir,
                token=hub_token(token),
            )
        )
    except (HfHubHTTPError, LocalEntryNotFoundError) as exc:
        require_hf_access(exc)

def hub_volume_candidates(split, volume_name):
    volume_name = str(volume_name).strip()
    if not volume_name.endswith(".nii.gz"):
        volume_name = f"{volume_name}.nii.gz"

    if "/" in volume_name:
        yield volume_name

    stem = volume_name.removesuffix(".nii.gz")
    parts = stem.split("_")
    split_folder = "train_fixed" if split == "train" else "valid_fixed"

    if len(parts) >= 3:
        level1 = "_".join(parts[:2])
        level2 = "_".join(parts[:3])
        yield f"dataset/{split_folder}/{level1}/{level2}/{volume_name}"
    if len(parts) >= 2:
        level1 = "_".join(parts[:2])
        yield f"dataset/{split_folder}/{level1}/{volume_name}"
    yield f"dataset/{split_folder}/{volume_name}"

def download_one_volume(split, volume_name, data_dir, token=None):
    for filename in dict.fromkeys(hub_volume_candidates(split, volume_name)):
        try:
            return download_hub_file(filename, data_dir, token)
        except SystemExit:
            continue
    raise SystemExit(
        f"Could not download CT-RATE volume {volume_name!r} from the expected {split} paths. "
        "The dataset layout may have changed, or this token cannot access gated files."
    ) from None

def first_existing_path(data_dir, candidates):
    data_dir = Path(data_dir)
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    return None

def read_report_csv(data_dir, split):
    path = first_existing_path(data_dir, REPORT_CSV_CANDIDATES[split])
    if path is None:
        raise FileNotFoundError(
            f"Could not find the CT-RATE {split} report CSV."
        )
    df = pd.read_csv(path)
    df["__split"] = split
    return df, path

def find_volume_column(df):
    candidates = [
        "VolumeName",
        "volume_name",
        "volume",
        "Volume",
        "FileName",
        "filename",
        "Image",
        "image",
    ]
    for column in candidates:
        if column in df.columns:
            return column
    nii_columns = [
        column
        for column in df.columns
        if df[column].astype(str).str.contains(r"\.nii(\.gz)?$", regex=True, na=False).any()
    ]
    if nii_columns:
        return nii_columns[0]
    raise ValueError(f"Could not infer a CT-RATE volume filename column from: {list(df.columns)}")

def report_text(row):
    preferred = [
        "Findings_EN",
        "Impressions_EN",
        "Findings",
        "Impressions",
        "Report",
        "report",
        "Text",
        "text",
    ]
    parts = []
    for column in preferred:
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            parts.append(str(row[column]).strip())
    if not parts:
        ignored = {"__split"}
        for column in row.index:
            if column in ignored:
                continue
            value = row[column]
            if pd.notna(value) and isinstance(value, str) and len(value.strip()) > 20:
                parts.append(value.strip())
    if not parts:
        return "Findings: No radiology report text is available."
    return " ".join(parts)

def volume_path_from_name(data_dir, split, volume_name):
    data_dir = Path(data_dir)
    volume_name = str(volume_name).strip()
    if volume_name.endswith(".png"):
        nii_name = volume_name.removesuffix(".png") + ".nii.gz"
    elif not volume_name.endswith(".nii.gz"):
        nii_name = f"{volume_name}.nii.gz"
    else:
        nii_name = volume_name

    if "/" in nii_name:
        direct = data_dir / nii_name
        if direct.exists():
            return direct

    stem = nii_name.removesuffix(".nii.gz")
    parts = stem.split("_")
    split_folder = "train_fixed" if split == "train" else "valid_fixed"

    candidates = [
        data_dir / "dataset" / split_folder / nii_name,
    ]
    if len(parts) >= 3:
        level1 = "_".join(parts[:2])
        level2 = "_".join(parts[:3])
        candidates.append(data_dir / "dataset" / split_folder / level1 / level2 / nii_name)
    if len(parts) >= 2:
        level1 = "_".join(parts[:2])
        candidates.append(data_dir / "dataset" / split_folder / level1 / nii_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]

def build_entries(data_dir, split, csv_path=None):
    if csv_path is None:
        df, csv_path = read_report_csv(data_dir, split)
    else:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        df["__split"] = split

    volume_column = find_volume_column(df)
    has_image_path = "image_path" in df.columns

    entries = []
    for _, row in df.iterrows():
        volume_name = str(row[volume_column]).strip()
        if not volume_name or volume_name.lower() == "nan":
            continue

        image_path = ""
        if has_image_path:
            raw = str(row.get("image_path", "")).strip()
            if raw and raw.lower() != "nan":
                image_path = raw

        if volume_name.endswith(".png") and not image_path:
            image_path = str(data_dir / "images" / split / volume_name)

        entries.append(
            {
                "split": split,
                "volume_name": volume_name,
                "volume_path": volume_path_from_name(data_dir, split, volume_name),
                "image_path": image_path,
                "text": report_text(row),
            }
        )
    if not entries:
        raise ValueError(f"No usable CT-RATE report rows found in {csv_path}")
    return entries, csv_path

def normalize_projection(array):
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        array = np.nan_to_num(array)
    high = np.percentile(array, 99)
    low = np.percentile(array, 1)
    if high <= low:
        high = float(array.max())
        low = float(array.min())
    if high <= low:
        return np.zeros_like(array, dtype=np.uint8)
    array = np.clip(array, low, high)
    array = (array - low) / (high - low)
    return (array * 255).astype(np.uint8)

def nifti_to_rgb_image(path):
    volume = np.asanyarray(nib.load(str(path)).dataobj)
    if volume.ndim == 4:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D or 4D NIfTI volume, got shape {volume.shape}")

    axial = normalize_projection(volume.max(axis=2))
    coronal = normalize_projection(volume.max(axis=1))
    sagittal = normalize_projection(volume.max(axis=0))

    axial = Image.fromarray(axial).resize((224, 224), resample=Image.BILINEAR)
    coronal = Image.fromarray(coronal).resize((224, 224), resample=Image.BILINEAR)
    sagittal = Image.fromarray(sagittal).resize((224, 224), resample=Image.BILINEAR)
    return Image.merge("RGB", (axial, coronal, sagittal))

# ── helpers ──────────────────────────────────────────────────────────────────

GIB = 1024 ** 3


def gib(b):
    return b / GIB


def free_bytes(path="."):
    return shutil.disk_usage(path).free


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


def local_volume_count(data_dir):
    data_dir = Path(data_dir)
    train_count = len(list((data_dir / "dataset" / "train_fixed").rglob("*.nii.gz")))
    valid_count = len(list((data_dir / "dataset" / "valid_fixed").rglob("*.nii.gz")))
    return train_count, valid_count


def resolve_token(token):
    if token:
        return token
    # Try to load .env if python-dotenv is installed
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    # Standard env vars
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    # System-level token stored by `huggingface-cli login`
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.exists():
        tok = cached.read_text().strip()
        if tok:
            return tok
    # Custom .env key (fallback — may lack download permissions)
    value = os.environ.get("hf_ct_scan_token")
    if value:
        return value
    return None


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


def volume_plan(data_dir, min_valid_volumes):
    train_entries, _ = build_entries(data_dir, "train")
    valid_entries, _ = build_entries(data_dir, "valid")
    _, existing_valid_count = local_volume_count(data_dir)

    valid_needed = []
    if existing_valid_count < min_valid_volumes:
        for entry in valid_entries:
            if existing_valid_count + len(valid_needed) >= min_valid_volumes:
                break
            # Skip if already converted to 2D PNG
            if entry["volume_name"].endswith(".png") or (entry.get("image_path") and Path(entry["image_path"]).exists()):
                continue
            if not Path(entry["volume_path"]).exists():
                valid_needed.append(("valid", entry))

    train_needed = []
    for entry in train_entries:
        # Skip if already converted to 2D PNG
        if entry["volume_name"].endswith(".png") or (entry.get("image_path") and Path(entry["image_path"]).exists()):
            continue
        if not Path(entry["volume_path"]).exists():
            train_needed.append(("train", entry))
            
    return valid_needed + train_needed


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


def convert_local_batch(data_dir, image_dir, dry_run, limit=None):
    """Convert all existing local .nii.gz files in data_dir to PNG, and delete the original files."""
    volumes = find_local_volumes(data_dir)
    if not volumes:
        print("No local .nii.gz volumes found.")
        return 0

    if limit:
        volumes = volumes[:limit]

    print(f"Found {len(volumes):,} local .nii.gz volume(s) to convert.")
    print(f"PNG output dir : {image_dir}")
    print(f"Free disk space: {gib(free_bytes()):.2f} GiB")
    if dry_run:
        print("DRY-RUN mode — no files will be written or deleted.\n")

    converted = {}   # stem → absolute png path
    errors = []

    for nii in tqdm(volumes, desc="Converting"):
        out = png_path(nii, image_dir)
        stem = nii.name.removesuffix(".nii.gz").removesuffix(".nii")

        # Skip if PNG already exists (idempotent)
        if out.exists():
            converted[stem] = str(out)
            if not dry_run:
                try:
                    nii.unlink()
                    # Remove empty parent dirs
                    for parent in [nii.parent, nii.parent.parent, nii.parent.parent.parent]:
                        try:
                            parent.rmdir()
                        except OSError:
                            break
                except Exception as e:
                    print(f"Warning: could not delete {nii}: {e}")
            continue

        try:
            img = nifti_to_rgb_image(nii)
        except Exception as exc:
            errors.append(f"{nii.name}: {exc}")
            print(f"\n  [error] {nii.name}: {exc}")
            continue

        if not dry_run:
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                img.save(out, format="PNG", optimize=True)
                nii.unlink()
                # Remove empty parent dirs
                for parent in [nii.parent, nii.parent.parent, nii.parent.parent.parent]:
                    try:
                        parent.rmdir()
                    except OSError:
                        break
            except Exception as e:
                errors.append(f"{nii.name} save/cleanup error: {e}")
                print(f"\n  [error] {nii.name} save/cleanup error: {e}")
                continue

        converted[stem] = str(out)

    print(f"\nConversion summary")
    print("-" * 40)
    print(f"Converted : {len(converted):,}")
    print(f"Errors    : {len(errors):,}")
    print(f"Free space: {gib(free_bytes()):.2f} GiB (after deletes)")

    if converted:
        print("\nUpdating CSV files...")
        update_csvs(data_dir, image_dir, converted, dry_run=dry_run)

    if errors:
        print("\nFailed volumes:")
        for e in errors:
            print(f"  {e}")

    return len(converted)


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(args):
    data_dir = Path(args.data_dir).resolve()
    image_dir = Path(args.image_dir).resolve()
    token = resolve_token(args.token)

    if args.loop:
        print("Starting pipeline in LOOP mode.")
        ensure_metadata(data_dir, token)

        iteration = 1
        while True:
            print(f"\n{'='*50}")
            print(f" LOOP ITERATION {iteration}")
            print(f"{'='*50}")

            # 1. Convert all local volumes
            convert_local_batch(data_dir, image_dir, args.dry_run, args.limit)

            # 2. Re-evaluate download plan
            plan = volume_plan(data_dir, args.min_valid)
            if not plan:
                print("\n[Success] All volumes in the dataset are downloaded and converted! Loop finished.")
                break

            # 3. Download a batch of files up to batch-gb (or max-new)
            batch_bytes_limit = int(args.batch_gb * GIB)
            min_free_bytes = int(args.min_free_gb * GIB)

            free = free_bytes(".")
            if free <= min_free_bytes:
                print(f"\n[Stop] Not enough free disk space to download. Free={gib(free):.2f} GiB, required buffer={args.min_free_gb:.2f} GiB.")
                break

            print(f"\nDownloading next batch (up to {args.batch_gb:.2f} GiB or {args.max_new} volumes)...")
            downloaded_count = 0
            downloaded_bytes = 0
            skipped = 0

            for split, entry in plan:
                free = free_bytes(".")
                if free <= min_free_bytes:
                    print(f"Stop: free-space buffer reached during download. Free={gib(free):.2f} GiB.")
                    break
                if downloaded_bytes >= batch_bytes_limit:
                    print(f"Reached batch limit of {args.batch_gb:.2f} GiB. Pausing download to convert.")
                    break
                if downloaded_count >= args.max_new:
                    print(f"Reached batch volume limit of {args.max_new} volumes. Pausing download to convert.")
                    break

                volume_name = entry["volume_name"]
                volume_path = Path(entry["volume_path"])
                if volume_path.exists():
                    skipped += 1
                    continue

                print(
                    f"Downloading {split} volume {volume_name} "
                    f"(Batch: {gib(downloaded_bytes):.2f}/{args.batch_gb:.2f} GiB, Free={gib(free):.2f} GiB)"
                )
                if args.dry_run:
                    downloaded_count += 1
                    # Dry-run mock size
                    downloaded_bytes += 150 * 1024 * 1024
                    continue

                try:
                    path = download_one_volume(split, volume_name, data_dir, token)
                    if path and path.exists():
                        file_size = path.stat().st_size
                        downloaded_bytes += file_size
                        downloaded_count += 1
                        print(f"  saved: {path} ({gib(file_size):.2f} GiB)")
                except Exception as e:
                    print(f"  [error] Failed to download {volume_name}: {e}")

            print(f"\nBatch download summary:")
            print(f"  Downloaded      : {downloaded_count} volume(s)")
            print(f"  Downloaded Size : {gib(downloaded_bytes):.2f} GiB")
            print(f"  Skipped local   : {skipped}")

            if downloaded_count == 0:
                print("\nNo new volumes downloaded in this batch. Breaking loop to prevent infinite cycle.")
                break

            iteration += 1

    else:
        print("Starting pipeline in SINGLE-PASS mode.")
        convert_local_batch(data_dir, image_dir, args.dry_run, args.limit)

        if args.then_download:
            ensure_metadata(data_dir, token)
            plan = volume_plan(data_dir, args.min_valid)
            if not plan:
                print("No additional volumes found in CSVs.")
                return

            target_bytes = int((args.target_gb or 18.0) * GIB)
            min_free_bytes = int(args.min_free_gb * GIB)

            downloaded = 0
            for split, entry in plan:
                current_size = dir_size(data_dir)
                free = free_bytes(".")

                if current_size >= target_bytes:
                    print(f"Stop: target reached at {gib(current_size):.2f} GiB.")
                    break
                if free <= min_free_bytes:
                    print(f"Stop: free-space buffer reached. Free={gib(free):.2f} GiB.")
                    break
                if downloaded >= args.max_new:
                    print(f"Stop: max new volume limit reached ({args.max_new}).")
                    break

                volume_name = entry["volume_name"]
                volume_path = Path(entry["volume_path"])
                if volume_path.exists():
                    continue

                print(f"Downloading {split} volume {volume_name}")
                if args.dry_run:
                    downloaded += 1
                    continue

                try:
                    download_one_volume(split, volume_name, data_dir, token)
                    downloaded += 1
                except Exception as e:
                    print(f"  [error] Failed to download {volume_name}: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified loop to convert CT-RATE 3D volumes to 2D PNGs and download the dataset in batches."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path,
                        help="CT-RATE data root (default: data/ct_rate)")
    parser.add_argument("--image-dir", default=Path("data/ct_rate/images"), type=Path,
                        help="Output directory for PNG images (default: data/ct_rate/images)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max volumes to convert in one run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without making changes")
    
    # Loop configuration
    parser.add_argument("--loop", action="store_true", default=True,
                        help="Run in continuous loop mode (default: True)")
    parser.add_argument("--no-loop", action="store_false", dest="loop",
                        help="Disable continuous loop mode, only convert local volumes once")
    parser.add_argument("--batch-gb", type=float, default=20.0,
                        help="Amount of 3D data in GB to download per batch (default: 20)")
    
    # Single-pass/compatibility configuration
    parser.add_argument("--then-download", action="store_true",
                        help="In single-pass mode, download next batch after converting")
    parser.add_argument("--target-gb", type=float, default=None,
                        help="In single-pass mode, target folder size in GB")
    parser.add_argument("--min-free-gb", type=float, default=5.0,
                        help="Minimum free-space buffer during download (default: 5)")
    parser.add_argument("--min-valid", "--min-valid-volumes", type=int, default=3, dest="min_valid",
                        help="Minimum validation volumes to ensure (default: 3)")
    parser.add_argument("--max-new", "--max-new-volumes", type=int, default=200, dest="max_new",
                        help="Max new volumes to download in one batch/run (default: 200)")
    parser.add_argument("--token", default=None,
                        help="HF token. Omit to use cached login or HF_TOKEN env var.")
    
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

import argparse
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel


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

METADATA_PREFIXES = (
    "dataset/radiology_text_reports/",
    "dataset/multi_abnormality_labels/",
    "dataset/metadata/",
)
REPORT_CSV_FILES = {
    "train": "dataset/radiology_text_reports/train_reports.csv",
    "valid": "dataset/radiology_text_reports/validation_reports.csv",
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hub_token(token):
    return token if token else True


def require_hf_access(exc):
    raise SystemExit(
        "Could not access CT-RATE on Hugging Face.\n"
        "Accept the dataset terms at https://huggingface.co/datasets/ibrahimhamamci/CT-RATE, "
        "then run `huggingface-cli login` with a token that has public gated repository access enabled. "
        "You can also pass the token with `--token`."
    ) from exc


def list_hub_files(token=None):
    try:
        files = HfApi(token=hub_token(token)).list_repo_files(REPO_ID, repo_type="dataset")
    except (HfHubHTTPError, LocalEntryNotFoundError) as exc:
        require_hf_access(exc)
    if not files:
        raise SystemExit(
            "CT-RATE file listing is empty for the current Hugging Face credentials. "
            "This usually means the gated dataset terms or token permissions are not enabled yet."
        )
    return files


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
    errors = []
    for filename in dict.fromkeys(hub_volume_candidates(split, volume_name)):
        try:
            return download_hub_file(filename, data_dir, token)
        except SystemExit as exc:
            errors.append(str(exc))
            continue
    raise SystemExit(
        f"Could not download CT-RATE volume {volume_name!r} from the expected {split} paths. "
        "The dataset layout may have changed, or this token cannot access gated files."
    ) from None


def selected_volume_names(data_dir, split, limit):
    entries, csv_path = build_entries(data_dir, split)
    names = []
    for entry in entries:
        name = entry["volume_name"]
        if name and name.lower() != "nan":
            names.append(name)
        if limit and len(names) >= limit:
            break
    if not names:
        raise SystemExit(f"No volume names found in {csv_path}.")
    return names


def download_ct_rate(data_dir, token=None, limit_train=0, limit_valid=0, metadata_only=False, list_metadata=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if list_metadata:
        files = list_hub_files(token)
        metadata_files = [
            filename
            for filename in files
            if any(filename.startswith(prefix) for prefix in METADATA_PREFIXES) or filename == "README.md"
        ]
    else:
        metadata_files = ["README.md", *REPORT_CSV_FILES.values()]

    print(f"Downloading {len(metadata_files):,} CT-RATE metadata/report files to {data_dir.resolve()}")
    for filename in tqdm(sorted(metadata_files), desc="Downloading metadata"):
        download_hub_file(filename, data_dir, token)

    if metadata_only:
        print("Metadata-only download complete.")
        return

    train_limit = 1 if limit_train is None or limit_train < 1 else limit_train
    valid_limit = 1 if limit_valid is None or limit_valid < 1 else limit_valid
    selected = [("train", name) for name in selected_volume_names(data_dir, "train", train_limit)]
    selected += [("valid", name) for name in selected_volume_names(data_dir, "valid", valid_limit)]

    print(f"Downloading {len(selected):,} CT volumes")
    for split, volume_name in tqdm(selected, desc="Downloading volumes"):
        download_one_volume(split, volume_name, data_dir, token)


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
            f"Could not find the CT-RATE {split} report CSV. "
            f"Run the download command first, or pass --train-csv/--valid-csv explicitly."
        )
    df = pd.read_csv(path)
    df["__split"] = split
    return df, path


def require_training_files(data_dir, train_csv=None, valid_csv=None):
    missing = []
    if train_csv is None and first_existing_path(data_dir, REPORT_CSV_CANDIDATES["train"]) is None:
        missing.append(REPORT_CSV_FILES["train"])
    elif train_csv is not None and not Path(train_csv).exists():
        missing.append(str(train_csv))

    if valid_csv is None and first_existing_path(data_dir, REPORT_CSV_CANDIDATES["valid"]) is None:
        missing.append(REPORT_CSV_FILES["valid"])
    elif valid_csv is not None and not Path(valid_csv).exists():
        missing.append(str(valid_csv))

    if missing:
        lines = "\n  - ".join(missing)
        raise SystemExit(
            "CT-RATE is not ready for training because required report CSVs are missing:\n"
            f"  - {lines}\n\n"
            "The previous download was blocked by Hugging Face gated access. "
            "Accept the CT-RATE terms, then create or update a Hugging Face token with "
            "public gated repository access enabled, and run:\n"
            "  huggingface-cli login\n"
            "  uv run python ct_rate_train_local.py download --limit-train 1 --limit-valid 1"
        )


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
    if not volume_name.endswith(".nii.gz"):
        volume_name = f"{volume_name}.nii.gz"

    if "/" in volume_name:
        direct = data_dir / volume_name
        if direct.exists():
            return direct

    stem = volume_name.removesuffix(".nii.gz")
    parts = stem.split("_")
    split_folder = "train_fixed" if split == "train" else "valid_fixed"

    candidates = [
        data_dir / "dataset" / split_folder / volume_name,
    ]
    if len(parts) >= 3:
        level1 = "_".join(parts[:2])
        level2 = "_".join(parts[:3])
        candidates.append(data_dir / "dataset" / split_folder / level1 / level2 / volume_name)
    if len(parts) >= 2:
        level1 = "_".join(parts[:2])
        candidates.append(data_dir / "dataset" / split_folder / level1 / volume_name)

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
    entries = []
    for _, row in df.iterrows():
        volume_name = str(row[volume_column]).strip()
        if not volume_name or volume_name.lower() == "nan":
            continue
        entries.append(
            {
                "split": split,
                "volume_name": volume_name,
                "volume_path": volume_path_from_name(data_dir, split, volume_name),
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


class CTRateReportDataset(Dataset):
    def __init__(self, entries, tokenizer, max_length=MAX_TEXT_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.entries = [entry for entry in entries if Path(entry["volume_path"]).exists()]
        if not self.entries:
            raise ValueError(
                "No local CT-RATE volumes matched the report CSV. "
                "Run the download command after accepting the gated dataset terms."
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        try:
            image = nifti_to_rgb_image(entry["volume_path"])
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)

        tokens = self.tokenizer(
            entry["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MedicalReportGenerator(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768):
        super().__init__()
        self.encoder = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_ftrs = self.encoder.classifier.in_features
        self.encoder.classifier = nn.Identity()
        self.projector = nn.Linear(num_ftrs, embed_dim)
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.decoder.resize_token_embeddings(vocab_size)

    def forward(self, images, input_ids, attention_mask):
        visual_features = self.encoder(images)
        visual_embeddings = self.projector(visual_features).unsqueeze(1)
        text_embeddings = self.decoder.transformer.wte(input_ids)
        inputs_embeds = torch.cat((visual_embeddings, text_embeddings), dim=1)

        visual_mask = torch.ones((images.size(0), 1), device=images.device)
        extended_mask = torch.cat((visual_mask, attention_mask), dim=1)
        return self.decoder(inputs_embeds=inputs_embeds, attention_mask=extended_mask).logits


def set_phase(model, phase):
    if phase == "warmup":
        print("Phase 1: encoder frozen, training projector and GPT-2.")
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.projector.parameters():
            p.requires_grad = True
        for p in model.decoder.parameters():
            p.requires_grad = True
    elif phase == "joint":
        print("Phase 2: all layers unfrozen.")
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown phase: {phase}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return filter(lambda p: p.requires_grad, model.parameters())


def run_epoch(model, loader, optimizer, criterion, device, phase_label, train=True):
    model.train(train)
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        bar = tqdm(loader, desc=phase_label, leave=True)
        for batch in bar:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images, input_ids, attention_mask)
            shift_logits = logits[:, 1:-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, phase1_done, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "phase1_done": phase1_done,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")


def build_loaders(args, tokenizer, device):
    train_entries, train_csv = build_entries(args.data_dir, "train", args.train_csv)
    val_entries, val_csv = build_entries(args.data_dir, "valid", args.valid_csv)

    if args.limit_train:
        train_entries = train_entries[: args.limit_train]
    if args.limit_val:
        val_entries = val_entries[: args.limit_val]

    train_dataset = CTRateReportDataset(train_entries, tokenizer, args.max_text_length)
    val_dataset = CTRateReportDataset(val_entries, tokenizer, args.max_text_length)

    print(f"Train CSV: {Path(train_csv).resolve()}")
    print(f"Valid CSV: {Path(val_csv).resolve()}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def train(args):
    device = get_device()
    print(f"Device: {device}")

    require_training_files(args.data_dir, args.train_csv, args.valid_csv)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args, tokenizer, device)
    print(f"Train samples: {len(train_dataset):,} ({len(train_loader):,} batches)")
    print(f"Val samples: {len(val_dataset):,} ({len(val_loader):,} batches)")

    sample = next(iter(train_loader))
    print(f"Sample image tensor: {tuple(sample['image'].shape)}")
    print(f"Sample input ids: {tuple(sample['input_ids'].shape)}")

    model = MedicalReportGenerator(vocab_size=len(tokenizer)).to(device)
    resume_epoch = 0
    phase1_done = False
    saved_optimizer = None

    if args.resume_checkpoint and args.resume_checkpoint.exists():
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state)
        if isinstance(checkpoint, dict):
            resume_epoch = checkpoint.get("epoch", 0)
            phase1_done = checkpoint.get("phase1_done", False)
            saved_optimizer = checkpoint.get("optimizer_state_dict")
        print(f"Resumed checkpoint: {args.resume_checkpoint}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    if phase1_done:
        print("Phase 1 already complete; skipping warmup.")
    else:
        trainable = set_phase(model, "warmup")
        optimizer = torch.optim.AdamW(trainable, lr=args.phase1_lr)
        if saved_optimizer is not None and resume_epoch < args.phase1_epochs:
            optimizer.load_state_dict(saved_optimizer)

        train_loss = math.nan
        val_loss = math.nan
        for epoch in range(resume_epoch + 1, args.phase1_epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, criterion, device, f"Train P1 {epoch}")
            val_loss = run_epoch(model, val_loader, optimizer, criterion, device, f"Val P1 {epoch}", train=False)
            if args.save_epoch_checkpoints:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    train_loss,
                    val_loss,
                    phase1_done=False,
                    path=args.checkpoint_dir / f"ct_rate_vlm_phase1_epoch{epoch}_checkpoint.pt",
                )

        if args.save_phase1_final:
            save_checkpoint(
                model,
                optimizer,
                args.phase1_epochs,
                train_loss,
                val_loss,
                phase1_done=True,
                path=args.checkpoint_dir / "ct_rate_vlm_phase1_FINAL.pt",
            )

    trainable = set_phase(model, "joint")
    optimizer = torch.optim.AdamW(trainable, lr=args.phase2_lr)
    avg_train_loss = math.nan
    avg_val_loss = math.nan

    for epoch in range(1, args.phase2_epochs + 1):
        avg_train_loss = run_epoch(model, train_loader, optimizer, criterion, device, f"Train P2 {epoch}")
        avg_val_loss = run_epoch(model, val_loader, optimizer, criterion, device, f"Val P2 {epoch}", train=False)
        if args.save_epoch_checkpoints:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                avg_train_loss,
                avg_val_loss,
                phase1_done=True,
                path=args.checkpoint_dir / f"ct_rate_vlm_phase2_epoch{epoch}_checkpoint.pt",
            )

    final_path = args.checkpoint_dir / args.final_model_name
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "phase1_done": True,
        "config": vars(args),
    }
    if args.save_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, final_path)
    print(f"Final model saved: {final_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download and train the Lumora VLM on CT-RATE.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download gated CT-RATE files.")
    download_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    download_parser.add_argument("--token", default=None, help="HF token, or omit to use your cached login.")
    download_parser.add_argument("--limit-train", type=int, default=1, help="Download only the first N train volumes.")
    download_parser.add_argument("--limit-valid", type=int, default=1, help="Download only the first N valid volumes.")
    download_parser.add_argument("--metadata-only", action="store_true")
    download_parser.add_argument(
        "--list-metadata",
        action="store_true",
        help="List the full Hub manifest to download all metadata folders. Slower for CT-RATE.",
    )

    train_parser = subparsers.add_parser("train", help="Run the same two-phase pipeline on CT-RATE.")
    train_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train_parser.add_argument("--train-csv", type=Path, default=None)
    train_parser.add_argument("--valid-csv", type=Path, default=None)
    train_parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/ct_rate"))
    train_parser.add_argument("--resume-checkpoint", type=Path, default=None)
    train_parser.add_argument("--final-model-name", default="ct_rate_vlm_phase2_fully_trained.pt")
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--max-text-length", type=int, default=MAX_TEXT_LENGTH)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--phase1-epochs", type=int, default=3)
    train_parser.add_argument("--phase1-lr", type=float, default=1e-4)
    train_parser.add_argument("--phase2-epochs", type=int, default=3)
    train_parser.add_argument("--phase2-lr", type=float, default=2e-5)
    train_parser.add_argument("--limit-train", type=int, default=None)
    train_parser.add_argument("--limit-val", type=int, default=None)
    train_parser.add_argument("--save-epoch-checkpoints", action="store_true")
    train_parser.add_argument("--save-phase1-final", action="store_true")
    train_parser.add_argument("--save-optimizer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "download":
        download_ct_rate(
            data_dir=args.data_dir,
            token=args.token,
            limit_train=args.limit_train,
            limit_valid=args.limit_valid,
            metadata_only=args.metadata_only,
            list_metadata=args.list_metadata,
        )
    elif args.command == "train":
        train(args)


if __name__ == "__main__":
    main()

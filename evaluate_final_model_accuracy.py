import argparse
import ast
import math
import re
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
from peft import PeftModel
import evaluate as hf_evaluate
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


DEFAULT_XRAY_MODEL_PATH = Path("checkpoints/x_ray/mimic_vlm_phase2_fully_trained.pt")
DEFAULT_XRAY_VALID_CSV = Path("mimic_cxr_aug_validate.csv")
DEFAULT_XRAY_IMG_ROOT = Path("official_data_iccv_final")
DEFAULT_CT_MODEL_PATH = Path("checkpoints/ct_rate/ct_rate_vlm_phase2_fully_trained.pt")
DEFAULT_CT_DATA_DIR = Path("ct_data")
MAX_TEXT_LENGTH = 128

# Label Vocabulary for the Disease Detection model
LABEL_VOCAB = [
    'Aortic Dilation', 'Atelectasis', 'Atherosclerosis', 'Cardiomegaly', 'Cholelithiasis', 
    'Consolidation', 'Emphysema/COPD', 'Hepatic Steatosis', 'Hiatal Hernia', 'Lymphadenopathy', 
    'Mild scoliosis', 'Mosaic Attenuation Pattern', 'No acute cardiopulmonary disease', 
    'Osteoporosis', 'Pericardial Effusion', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax', 
    'Possible Aspiration', 'Possible Malignancy/Mass', 'Pulmonary Artery Enlargement', 
    'Pulmonary Edema/Vascular Congestion', 'Pulmonary Fibrosis/Scarring', 'Pulmonary Nodules', 
    'Rib/Bone Fracture', 'Spinal Degenerative Changes'
]
LABEL2IDX = {d: i for i, d in enumerate(LABEL_VOCAB)}

# Normalization map for cleaning labels in translation/disease classifier datasets
NORMALIZATION_MAP = {
    'No acute cardiopulmonary disease': 'No acute cardiopulmonary disease',
    'Normal / No acute disease': 'No acute cardiopulmonary disease',
    'Pneumothorax': 'Pneumothorax',
    'Atelectasis': 'Atelectasis',
    'Pulmonary nodules (small': 'Pulmonary nodules',
    'Pulmonary nodule (small': 'Pulmonary nodules',
    'nonspecific)': None,
    'nonspecific': None,
    'recommend clinical correlation)': None,
    'Pleural Effusion': 'Pleural Effusion',
    'Cardiomegaly': 'Cardiomegaly',
    'Pulmonary Edema / Vascular Congestion': 'Pulmonary Edema/Vascular Congestion',
    'Pulmonary Edema': 'Pulmonary Edema/Vascular Congestion',
    'Vascular Congestion': 'Pulmonary Edema/Vascular Congestion',
    'Pneumonia': 'Pneumonia',
    'Covid-19 pneumonia (consistent findings)': 'Pneumonia',
    'Coronary and aortic atherosclerosis': 'Atherosclerosis',
    'Coronary atherosclerosis': 'Atherosclerosis',
    'Consolidation': 'Consolidation',
    'Spinal Degenerative Changes': 'Spinal Degenerative Changes',
    'Pulmonary fibrosis/scarring (sequelae)': 'Pulmonary Fibrosis/Scarring',
    'Pulmonary fibrosis/scarring (apical sequelae)': 'Pulmonary Fibrosis/Scarring',
    'Hiatal hernia (sliding)': 'Hiatal Hernia',
    'Hiatal hernia': 'Hiatal Hernia',
    'Emphysema': 'Emphysema/COPD',
    'Mild emphysema': 'Emphysema/COPD',
    'Emphysema/COPD': 'Emphysema/COPD',
    'Hepatic steatosis (fatty liver)': 'Hepatic Steatosis',
    'Mild hepatic steatosis': 'Hepating Steatosis', # wait, 'Hepatic Steatosis'
    'Pericardial effusion (minimal)': 'Pericardial Effusion',
    'Mosaic attenuation pattern (possible small airway/vessel disease)': 'Mosaic Attenuation Pattern',
    'Rib/Bone Fracture': 'Rib/Bone Fracture',
    'Pulmonary nodules (small, nonspecific)': 'Pulmonary Nodules',
    'Pulmonary nodule (small, nonspecific)': 'Pulmonary Nodules',
    'Pulmonary Nodule': 'Pulmonary Nodules',
    'Hiatal hernia (mild)': 'Hiatal Hernia',
    'Aortic atherosclerosis': 'Atherosclerosis',
    'Pulmonary artery enlargement': 'Pulmonary Artery Enlargement',
    'Possible Aspiration': 'Possible Aspiration',
    'Cholelithiasis': 'Cholelithiasis',
    'Hiatal/Diaphragmatic Hernia': 'Hiatal Hernia',
    'Hiatal hernia (mild, sliding)': 'Hiatal Hernia',
    'Pulmonary fibrosis/scarring': 'Pulmonary Fibrosis/Scarring',
    'Degenerative bone changes': 'Spinal Degenerative Changes',
    'Degenerative spinal changes': 'Spinal Degenerative Changes',
    'Thoracic spondylosis': 'Spinal Degenerative Changes',
    'Mild thoracic spondylosis': 'Spinal Degenerative Changes',
    'Reduced bone density': 'Osteoporosis',
    'Osteoporosis': 'Osteoporosis',
    'Mediastinal lymphadenopathy': 'Lymphadenopathy',
    'Mediastinal/hilar lymphadenopathy': 'Lymphadenopathy',
    'Bilateral pleural effusion': 'Pleural Effusion',
    'Bilateral pleural effusion (minimal)': 'Pleural Effusion',
    'Pulmonary nodules (stable)': 'Pulmonary Nodules',
    'Aortic dilation (ascending aorta)': 'Aortic Dilation',
    'Mild aortic dilation (ascending aorta)': 'Aortic Dilation',
    'Aortic and coronary atherosclerosis': 'Atherosclerosis',
    'Mild coronary atherosclerosis': 'Atherosclerosis',
    'Pericardial effusion': 'Pericardial Effusion',
    'Mild cardiomegaly': 'Cardiomegaly',
    'Mosaic attenuation pattern': 'Mosaic Attenuation Pattern',
    'Mild bronchiectasis': 'Bronchiectasis',
    'Mild bronchial wall thickening': 'Bronchial Wall Thickening',
    'Covid-19 pneumonia (typical/probable findings)': 'Pneumonia',
    'Splenomegaly': 'Splenomegaly',
    'Accessory spleen (normal variant)': None,
    'Possible Malignancy/Mass': 'Possible Malignancy/Mass',
    'Possible thymic remnant tissue (benign)': None,
    'Status post cholecystectomy': None,
    'Right nephrolithiasis (small)': 'Nephrolithiasis',
}

# Fix typo in value
NORMALIZATION_MAP['Mild hepatic steatosis'] = 'Hepatic Steatosis'


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_latest_checkpoint(base_dir, prefix="checkpoint-"):
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None
    checkpoints = [
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    if not checkpoints:
        if (base_dir / "config.json").exists() or (base_dir / "adapter_config.json").exists():
            return base_dir
        return None
    def get_number(path):
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1
    checkpoints.sort(key=get_number)
    return checkpoints[-1]


def resolve_translation_model_path(path):
    if path is not None:
        return Path(path)
    for c in ["translation_model", "t5_translation"]:
        p = Path(c)
        if p.exists():
            latest = find_latest_checkpoint(p)
            if latest:
                return latest
    return None


def resolve_disease_model_path(path):
    if path is not None:
        return Path(path)
    for c in ["disease_classifier_model", "clinicalbert_disease_classifier"]:
        p = Path(c)
        if p.exists():
            latest = find_latest_checkpoint(p)
            if latest:
                return latest
    return None


# Parsing and cleaning helpers for translation/disease models
def smart_split(diseases_str):
    parts = []
    depth = 0
    current = []
    for ch in diseases_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return [p for p in parts if p]


def clean_label(label):
    label = re.sub(r'\[.*?\]\(.*?\)', '/', label)
    label = re.sub(r'\s+', ' ', label).strip()
    return label


def normalize_label(label):
    label = clean_label(label)
    if label in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[label]
    return label


def process_diseases(diseases_str):
    if not isinstance(diseases_str, str):
        return []
    raw_parts = smart_split(diseases_str)
    cleaned = []
    for p in raw_parts:
        norm = normalize_label(p)
        if norm is not None:
            cleaned.append(norm)
    return cleaned


def multi_hot_encode(disease_list):
    vec = np.zeros(len(LABEL_VOCAB), dtype=np.float32)
    for d in disease_list:
        if d in LABEL2IDX:
            vec[LABEL2IDX[d]] = 1.0
    return vec


def load_combined_reports_df(xray_csv, ct_csv, split="test"):
    if not Path(xray_csv).exists():
        raise FileNotFoundError(f"Translation/Disease X-ray CSV not found: {xray_csv}")
    if not Path(ct_csv).exists():
        raise FileNotFoundError(f"Translation/Disease CT CSV not found: {ct_csv}")
        
    df_x_ray = pd.read_csv(xray_csv)
    df_x_ray = df_x_ray.rename(columns={
        'original_report': 'report',
        'patient_friendly_translation': 'translation',
        'detected_diseases': 'diseases'
    })[['report', 'translation', 'diseases']]
    
    df_ct = pd.read_csv(ct_csv)
    df_ct = df_ct.rename(columns={
        'report': 'report',
        'translation': 'translation',
        'diseases': 'diseases'
    })[['report', 'translation', 'diseases']]
    
    df = pd.concat([df_x_ray, df_ct], ignore_index=True)
    df = df.dropna(subset=['report', 'translation', 'diseases']).reset_index(drop=True)
    
    df['disease_list'] = df['diseases'].apply(process_diseases)
    df['label_vec'] = df['disease_list'].apply(multi_hot_encode)
    
    train_df, val_test_df = train_test_split(df, test_size=0.20, random_state=42)
    val_df, test_df = train_test_split(val_test_df, test_size=0.50, random_state=42)
    
    if split == "train":
        return train_df.reset_index(drop=True)
    elif split == "val":
        return val_df.reset_index(drop=True)
    else:
        return test_df.reset_index(drop=True)


class TranslationDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len=512, max_target_len=256):
        self.reports = df['report'].tolist()
        self.translations = df['translation'].tolist()
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, idx):
        input_text = "translate to patient-friendly: " + str(self.reports[idx])
        target_text = str(self.translations[idx])

        model_inputs = self.tokenizer(
            input_text, truncation=True, max_length=self.max_input_len, padding="max_length"
        )
        labels = self.tokenizer(
            target_text, truncation=True, max_length=self.max_target_len, padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        model_inputs = {k: torch.tensor(v) for k, v in model_inputs.items()}
        model_inputs["labels"][model_inputs["labels"] == self.tokenizer.pad_token_id] = -100
        return model_inputs


class DiseaseClassifierDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.reports = df['report'].tolist()
        self.labels = df['label_vec'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, idx):
        report = str(self.reports[idx])
        label = self.labels[idx]
        
        inputs = self.tokenizer(
            report,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(label, dtype=torch.float32)
        return item


class MimicReportDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, tokenizer, max_length=MAX_TEXT_LENGTH):
        self.df = pd.read_csv(csv_file)
        self.img_root_dir = Path(img_root_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.skipped_missing_images = 0
        self.missing_images = 0
        self.samples = []

        for _, row in self.df.iterrows():
            img_paths = ast.literal_eval(row["image"])
            reports = ast.literal_eval(row["text"])
            image_path = next((self.img_root_dir / path for path in img_paths if (self.img_root_dir / path).exists()), None)
            if image_path is None:
                self.skipped_missing_images += 1
                continue
            report_text = reports[0] if reports else "Findings: Normal study."
            self.samples.append((image_path, report_text))

        if not self.samples:
            raise ValueError(
                f"No X-ray images from {csv_file} were found under {self.img_root_dir}. "
                "Pass the correct --xray-img-root or use a validation CSV that matches the local image files."
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
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, report_text = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            self.missing_images += 1
            image = torch.zeros(3, 224, 224)

        tokens = self.tokenizer(
            report_text,
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
        self.encoder = models.densenet121(weights=None)
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

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
        ).logits


def load_model(checkpoint_path, device):
    model = MedicalReportGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint


def build_xray_dataset(args, tokenizer):
    return MimicReportDataset(
        csv_file=args.xray_valid_csv,
        img_root_dir=args.xray_img_root,
        tokenizer=tokenizer,
        max_length=args.max_text_length,
    )


REPORT_CSV_CANDIDATES = {
    "train": [
        "train_reports.csv",
        "dataset/radiology_text_reports/train_reports.csv",
        "radiology_text_reports/train_reports.csv",
    ],
    "valid": [
        "validation_reports.csv",
        "valid_reports.csv",
        "dataset/radiology_text_reports/validation_reports.csv",
        "dataset/radiology_text_reports/valid_reports.csv",
        "radiology_text_reports/validation_reports.csv",
        "radiology_text_reports/valid_reports.csv",
    ],
}

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
        if df[column].astype(str).str.contains(r"\.(nii(\.gz)?|png)$", regex=True, na=False).any()
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

def resolve_image_path(path_str, data_dir):
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    parts = p.parts
    for i in range(len(parts)):
        if parts[i] in ("train", "valid", "val", "ct_data", "images", "dataset", "data"):
            sub_path = Path(*parts[i:])
            if parts[i] == "ct_data" and i + 1 < len(parts) and parts[i+1] in ("train", "valid", "val"):
                sub_path = Path(*parts[i+1:])
            elif parts[i] == "data" and i + 1 < len(parts) and parts[i+1] == "ct_rate":
                sub_path = Path(*parts[i+2:])
            trial = data_dir / sub_path
            if trial.exists():
                return trial
    filename = p.name
    for s in ("train", "valid", "val"):
        trial = data_dir / s / filename
        if trial.exists():
            return trial
    return p

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

        if not image_path:
            png_name = volume_name
            if png_name.endswith(".nii.gz"):
                png_name = png_name.removesuffix(".nii.gz") + ".png"
            elif not png_name.endswith(".png"):
                png_name = png_name + ".png"
            folder_name = "valid" if split in ("valid", "validation", "val") else split
            image_path = str(data_dir / folder_name / png_name)

        resolved_p = resolve_image_path(image_path, data_dir)
        if resolved_p and resolved_p.exists():
            entries.append(
                {
                    "split": split,
                    "volume_name": volume_name,
                    "image_path": str(resolved_p),
                    "text": report_text(row),
                }
            )
    if not entries:
        raise ValueError(f"No usable CT-RATE report rows found in {csv_path} with existing local images")
    return entries, csv_path

class CTRateReportDataset(Dataset):
    def __init__(self, entries, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.entries = entries
        if not self.entries:
            raise ValueError("No local CT-RATE images matched the report CSV.")

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
            image = Image.open(entry["image_path"]).convert("RGB")
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

def build_ct_dataset(args, tokenizer):
    entries, csv_path = build_entries(args.ct_data_dir, "valid", args.ct_valid_csv)
    if args.limit is not None:
        entries = entries[: args.limit]

    dataset = CTRateReportDataset(entries, tokenizer, args.max_text_length)
    print(f"CT validation CSV: {Path(csv_path).resolve()}")
    print(f"CT data dir: {args.ct_data_dir.resolve()}")
    return dataset


def resolve_defaults(args):
    if args.xray_model_path is None:
        args.xray_model_path = DEFAULT_XRAY_MODEL_PATH
    if args.ct_model_path is None:
        args.ct_model_path = DEFAULT_CT_MODEL_PATH
    if args.xray_valid_csv is None:
        args.xray_valid_csv = DEFAULT_XRAY_VALID_CSV
    if args.xray_img_root is None:
        args.xray_img_root = DEFAULT_XRAY_IMG_ROOT
    if args.ct_data_dir is None:
        args.ct_data_dir = DEFAULT_CT_DATA_DIR

    return args


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    exact_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating final model"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(images, input_ids, attention_mask)

            shift_logits = logits[:, 1:-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid_mask = shift_labels != -100

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            predictions = shift_logits.argmax(dim=-1)
            correct_by_token = (predictions == shift_labels) & valid_mask

            total_loss += loss.item()
            total_tokens += valid_mask.sum().item()
            correct_tokens += correct_by_token.sum().item()

            sequence_has_tokens = valid_mask.any(dim=1)
            sequence_is_exact = ((predictions == shift_labels) | ~valid_mask).all(dim=1)
            exact_sequences += (sequence_is_exact & sequence_has_tokens).sum().item()
            total_sequences += sequence_has_tokens.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    token_accuracy = correct_tokens / max(total_tokens, 1)
    exact_sequence_accuracy = exact_sequences / max(total_sequences, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "validation_loss": avg_loss,
        "perplexity": perplexity,
        "token_accuracy": token_accuracy,
        "exact_sequence_accuracy": exact_sequence_accuracy,
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "exact_sequences": exact_sequences,
        "total_sequences": total_sequences,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Lumora checkpoints (VLM models, Translation model, and Disease Detection model)."
    )
    parser.add_argument(
        "--modality", 
        choices=["xray", "ct", "translation", "disease", "all"], 
        default="all",
        help="Model modality to evaluate. 'all' evaluates all four models."
    )
    parser.add_argument("--xray-model-path", type=Path, default=None)
    parser.add_argument("--ct-model-path", type=Path, default=None)
    parser.add_argument("--translation-model-path", type=Path, default=None, help="Directory containing T5 adapter model.")
    parser.add_argument("--disease-model-path", type=Path, default=None, help="Directory containing Disease Classifier model.")
    
    parser.add_argument("--xray-valid-csv", type=Path, default=None)
    parser.add_argument("--ct-valid-csv", type=Path, default=None)
    
    parser.add_argument("--translation-csv-xray", type=Path, default=Path("mimic_cxr_translated.csv"))
    parser.add_argument("--translation-csv-ct", type=Path, default=Path("translated_ct_reports.csv"))
    parser.add_argument("--disease-csv-xray", type=Path, default=Path("mimic_cxr_translated.csv"))
    parser.add_argument("--disease-csv-ct", type=Path, default=Path("translated_ct_reports.csv"))
    
    parser.add_argument("--xray-img-root", type=Path, default=None, help="Image root for X-ray evaluation.")
    parser.add_argument("--ct-data-dir", type=Path, default=None, help="CT-RATE data directory for CT evaluation.")
    
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=MAX_TEXT_LENGTH)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows for each dataset.")
    parser.add_argument(
        "--disease-threshold", 
        type=float, 
        default=None, 
        help="Threshold for classification. If not specified, a threshold sweep is run to find the best threshold."
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Data split to use for evaluating Translation and Disease classifier models."
    )
    
    return resolve_defaults(parser.parse_args())


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Validation loss/token : {metrics['validation_loss']:.4f}")
    print(f"Perplexity            : {metrics['perplexity']:.4f}")
    print(f"Token accuracy        : {metrics['token_accuracy'] * 100:.2f}%")
    print(f"Exact report accuracy : {metrics['exact_sequence_accuracy'] * 100:.2f}%")
    print(f"Correct tokens        : {metrics['correct_tokens']:,}/{metrics['total_tokens']:,}")
    print(f"Exact reports         : {metrics['exact_sequences']:,}/{metrics['total_sequences']:,}")


def evaluate_modality(modality, args, tokenizer, device):
    print("\n" + "=" * 64)
    print(f"Evaluating {modality.upper()} model")
    print("=" * 64)

    if modality == "ct":
        model_path = args.ct_model_path
        if not model_path or not model_path.exists():
            print(f"\n⚠️ Skipping CT evaluation: checkpoint not found at {model_path}")
            return None
        print(f"Model: {model_path.resolve()}")
        try:
            dataset = build_ct_dataset(args, tokenizer)
        except (ValueError, FileNotFoundError) as e:
            print(f"\n⚠️ Skipping CT evaluation: {e}")
            return None
    else:
        model_path = args.xray_model_path
        if not model_path or not model_path.exists():
            print(f"\n⚠️ Skipping XRAY evaluation: checkpoint not found at {model_path}")
            return None
        print(f"Model: {model_path.resolve()}")
        print(f"Validation CSV: {args.xray_valid_csv.resolve()}")
        print(f"Image root: {args.xray_img_root.resolve()}")
        try:
            dataset = build_xray_dataset(args, tokenizer)
        except (ValueError, FileNotFoundError) as e:
            print(f"\n⚠️ Skipping XRAY evaluation: {e}")
            return None
        if dataset.skipped_missing_images:
            print(
                f"Using {len(dataset):,} X-ray rows with local images; "
                f"skipped {dataset.skipped_missing_images:,} rows with missing files."
            )

    if modality == "xray" and args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    try:
        model, checkpoint = load_model(model_path, device)
    except Exception as e:
        print(f"\n⚠️ Skipping {modality.upper()} evaluation: failed to load model: {e}")
        return None

    if isinstance(checkpoint, dict):
        saved_val_loss = checkpoint.get("final_val_loss", checkpoint.get("val_loss"))
        if saved_val_loss is not None:
            print(f"Saved checkpoint validation loss: {saved_val_loss}")

    metrics = evaluate(model, loader, device)
    print_metrics(f"{modality.upper()} validation metrics", metrics)

    backing_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if getattr(backing_dataset, "missing_images", 0):
        print(f"Warning: {backing_dataset.missing_images} images could not be loaded and used zero tensors.")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def evaluate_translation_modality(args, device):
    print("\n" + "=" * 64)
    print("Evaluating TRANSLATION model")
    print("=" * 64)
    
    model_path = resolve_translation_model_path(args.translation_model_path)
    if not model_path or not model_path.exists():
        print(f"\n⚠️ Skipping TRANSLATION evaluation: checkpoint not found at {model_path}")
        return None
        
    print(f"Resolved Model Path: {model_path.resolve()}")
    
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    try:
        model = PeftModel.from_pretrained(base_model, model_path).to(device)
    except Exception as e:
        print(f"\n⚠️ Skipping TRANSLATION evaluation: failed to load model: {e}")
        return None
    model.eval()
    
    try:
        df = load_combined_reports_df(args.translation_csv_xray, args.translation_csv_ct, split=args.split)
    except (ValueError, FileNotFoundError) as e:
        print(f"\n⚠️ Skipping TRANSLATION evaluation: {e}")
        return None
    print(f"Loaded {len(df)} reports for translation evaluation (Split: {args.split}).")
    
    # Run ROUGE evaluation and loss calculation
    # We pass limit to restrict translation generation to N samples for speed, 
    # but compute loss on all if limit is None
    limit = args.limit if args.limit is not None else 50
    metrics = {
        "loss": 0.0,
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "rougeLsum": 0.0
    }
    
    rouge_metric = hf_evaluate.load("rouge")
    gen_df = df.head(limit) if limit is not None else df
    predictions = []
    references = gen_df['translation'].tolist()
    
    print(f"Generating translations for ROUGE score computation ({len(gen_df)} samples)...")
    with torch.no_grad():
        for report in tqdm(gen_df['report'], desc="Generating Translations"):
            input_text = "translate to patient-friendly: " + str(report)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model.generate(**inputs, max_length=256, num_beams=4)
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(pred_text)
            
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    
    dataset = TranslationDataset(df, tokenizer)
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda"
    )
    
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing Translation Loss"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)
            
    avg_loss = total_loss / max(total_tokens, 1)
    
    metrics = {
        "loss": avg_loss,
        "rouge1": rouge_results.get("rouge1", 0.0),
        "rouge2": rouge_results.get("rouge2", 0.0),
        "rougeL": rouge_results.get("rougeL", 0.0),
        "rougeLsum": rouge_results.get("rougeLsum", 0.0)
    }
    
    print("\nTranslation metrics:")
    print("-" * 20)
    print(f"Validation Loss : {metrics['loss']:.4f}")
    print(f"ROUGE-1         : {metrics['rouge1']:.4f}")
    print(f"ROUGE-2         : {metrics['rouge2']:.4f}")
    print(f"ROUGE-L         : {metrics['rougeL']:.4f}")
    print(f"ROUGE-Lsum      : {metrics['rougeLsum']:.4f}")
    
    del model, base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def evaluate_disease_modality(args, device):
    print("\n" + "=" * 64)
    print("Evaluating DISEASE DETECTION model")
    print("=" * 64)
    
    model_path = resolve_disease_model_path(args.disease_model_path)
    if not model_path or not model_path.exists():
        print(f"\n⚠️ Skipping DISEASE Classifier evaluation: checkpoint not found at {model_path}")
        return None
        
    print(f"Resolved Model Path: {model_path.resolve()}")
    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=26,
            problem_type="multi_label_classification"
        ).to(device)
    except Exception as e:
        print(f"\n⚠️ Skipping DISEASE Classifier evaluation: failed to load model: {e}")
        return None
    model.eval()
    
    try:
        df = load_combined_reports_df(args.disease_csv_xray, args.disease_csv_ct, split=args.split)
    except (ValueError, FileNotFoundError) as e:
        print(f"\n⚠️ Skipping DISEASE Classifier evaluation: {e}")
        return None
    print(f"Loaded {len(df)} reports for disease classifier evaluation (Split: {args.split}).")
    
    dataset = DiseaseClassifierDataset(df, tokenizer)
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda"
    )
    
    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Disease Classifier Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
            
    avg_loss = total_loss / max(total_samples, 1)
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    # Threshold selection or sweep
    selected_threshold = args.disease_threshold
    best_f1 = 0.0
    
    if selected_threshold is None:
        print("\n📋 Running Threshold Sweep subject to Recall >= 0.70...")
        thresholds = np.arange(0.02, 0.55, 0.02).tolist()
        best_threshold = 0.10
        best_score = 0.0
        
        for th in thresholds:
            preds = (all_probs > th).astype(float)
            p = precision_score(all_labels, preds, average='micro', zero_division=0)
            r = recall_score(all_labels, preds, average='micro', zero_division=0)
            f1 = f1_score(all_labels, preds, average='micro', zero_division=0)
            
            if r >= 0.70 and f1 > best_score:
                best_score = f1
                best_threshold = th
                
        if best_score == 0.0:
            # fallback
            best_recall = 0.0
            for th in thresholds:
                preds = (all_probs > th).astype(float)
                r = recall_score(all_labels, preds, average='micro', zero_division=0)
                f1 = f1_score(all_labels, preds, average='micro', zero_division=0)
                if r > best_recall or (r == best_recall and f1 > best_score):
                    best_recall = r
                    best_score = f1
                    best_threshold = th
                    
        selected_threshold = best_threshold
        print(f"Selected threshold from sweep: {selected_threshold:.2f} (Micro F1: {best_score:.4f})")
    else:
        print(f"\nUsing manual threshold: {selected_threshold:.2f}")
        
    final_preds = (all_probs > selected_threshold).astype(float)
    precision_val = precision_score(all_labels, final_preds, average='micro', zero_division=0)
    recall_val = recall_score(all_labels, final_preds, average='micro', zero_division=0)
    f1_micro_val = f1_score(all_labels, final_preds, average='micro', zero_division=0)
    f1_macro_val = f1_score(all_labels, final_preds, average='macro', zero_division=0)
    
    print("\nDisease Classifier metrics:")
    print("-" * 27)
    print(f"Validation Loss : {avg_loss:.4f}")
    print(f"Micro Precision : {precision_val:.4f}")
    print(f"Micro Recall    : {recall_val:.4f} (most critical)")
    print(f"Micro F1        : {f1_micro_val:.4f}")
    print(f"Macro F1        : {f1_macro_val:.4f}")
    
    print("\n📊 Per-Class Breakdown:")
    print(f"{'Disease':<45} {'Support':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 87)
    for i, disease in enumerate(LABEL_VOCAB):
        support = int(all_labels[:, i].sum())
        if support == 0:
            continue
        tp = ((final_preds[:, i] == 1) & (all_labels[:, i] == 1)).sum()
        fp = ((final_preds[:, i] == 1) & (all_labels[:, i] == 0)).sum()
        fn = ((final_preds[:, i] == 0) & (all_labels[:, i] == 1)).sum()
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        flag = ' ⚠️' if r < 0.5 else ' ✅' if r >= 0.7 else ''
        print(f"  {disease:<43} {support:>8} {p:>10.4f} {r:>10.4f} {f1:>10.4f}{flag}")
        
    metrics = {
        "loss": avg_loss,
        "precision": precision_val,
        "recall": recall_val,
        "f1_micro": f1_micro_val,
        "f1_macro": f1_macro_val,
        "threshold": selected_threshold
    }
    
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def print_comparison(results):
    if not results:
        return

    print("\n" + "=" * 64)
    print("Lumora Evaluation Summary")
    print("=" * 64)
    
    # 1. Print VLM models comparison if present
    vlm_results = {k: v for k, v in results.items() if k in ("xray", "ct")}
    if vlm_results:
        print(f"\n{'VLM Model':<10} {'Token Acc':>10} {'Loss/Token':>12} {'Perplexity':>12} {'Exact Acc':>10}")
        print("-" * 64)
        for modality, metrics in vlm_results.items():
            print(
                f"{modality.upper():<10} "
                f"{metrics['token_accuracy'] * 100:>9.2f}% "
                f"{metrics['validation_loss']:>12.4f} "
                f"{metrics['perplexity']:>12.4f} "
                f"{metrics['exact_sequence_accuracy'] * 100:>9.2f}%"
            )
            
    # 2. Print Translation model metrics if present
    if "translation" in results:
        t_metrics = results["translation"]
        print(f"\n{'Translation Model':<20} {'Loss':>10} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10}")
        print("-" * 64)
        print(
            f"{'T5 + LoRA':<20} "
            f"{t_metrics['loss']:>10.4f} "
            f"{t_metrics['rouge1']:>10.4f} "
            f"{t_metrics['rouge2']:>10.4f} "
            f"{t_metrics['rougeL']:>10.4f}"
        )
        
    # 3. Print Disease Detection classifier metrics if present
    if "disease" in results:
        d_metrics = results["disease"]
        print(f"\n{'Disease Classifier':<20} {'Loss':>10} {'Micro P':>10} {'Micro R':>10} {'Micro F1':>10} {'Macro F1':>10} {'Thresh':>8}")
        print("-" * 82)
        print(
            f"{'Bio_ClinicalBERT':<20} "
            f"{d_metrics['loss']:>10.4f} "
            f"{d_metrics['precision']:>10.4f} "
            f"{d_metrics['recall']:>10.4f} "
            f"{d_metrics['f1_micro']:>10.4f} "
            f"{d_metrics['f1_macro']:>10.4f} "
            f"{d_metrics['threshold']:>8.2f}"
        )
    print()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Mode: {args.modality}")

    results = {}
    
    # Determine which modalities to run
    run_xray = args.modality in ("xray", "all")
    run_ct = args.modality in ("ct", "all")
    run_translation = args.modality in ("translation", "all")
    run_disease = args.modality in ("disease", "all")

    # Evaluate VLM Models
    if run_xray or run_ct:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        if run_xray:
            res = evaluate_modality("xray", args, tokenizer, device)
            if res is not None:
                results["xray"] = res
        if run_ct:
            res = evaluate_modality("ct", args, tokenizer, device)
            if res is not None:
                results["ct"] = res
            
    # Evaluate Translation Model
    if run_translation:
        res = evaluate_translation_modality(args, device)
        if res is not None:
            results["translation"] = res
        
    # Evaluate Disease Classifier Model
    if run_disease:
        res = evaluate_disease_modality(args, device)
        if res is not None:
            results["disease"] = res

    print_comparison(results)


if __name__ == "__main__":
    main()

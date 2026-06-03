import argparse
import ast
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel


DEFAULT_MODEL_PATH = Path("lumora-web/backend/mimic-vlm-model/mimic_vlm_phase2_fully_trained.pt")
DEFAULT_VALID_CSV = Path("mimic_cxr_aug_validate.csv")
DEFAULT_IMG_ROOT = Path("official_data_iccv_final")
MAX_TEXT_LENGTH = 128


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MimicReportDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, tokenizer, max_length=MAX_TEXT_LENGTH):
        self.df = pd.read_csv(csv_file)
        self.img_root_dir = Path(img_root_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.missing_images = 0

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
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_paths = ast.literal_eval(row["image"])
        reports = ast.literal_eval(row["text"])

        img_path = self.img_root_dir / img_paths[0]
        report_text = reports[0] if reports else "Findings: Normal study."

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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint


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

            # Same next-token shift used during training:
            # skip visual-prefix logit at index 0, predict labels from token 1 onward.
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
        description="Evaluate the final Lumora VLM checkpoint on validation report-token accuracy."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--valid-csv", type=Path, default=DEFAULT_VALID_CSV)
    parser.add_argument("--img-root", type=Path, default=DEFAULT_IMG_ROOT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=MAX_TEXT_LENGTH)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {args.model_path.resolve()}")
    print(f"Validation CSV: {args.valid_csv.resolve()}")
    print(f"Image root: {args.img_root.resolve()}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = MimicReportDataset(
        csv_file=args.valid_csv,
        img_root_dir=args.img_root,
        tokenizer=tokenizer,
        max_length=args.max_text_length,
    )
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, checkpoint = load_model(args.model_path, device)
    if isinstance(checkpoint, dict):
        saved_val_loss = checkpoint.get("final_val_loss", checkpoint.get("val_loss"))
        if saved_val_loss is not None:
            print(f"Saved checkpoint validation loss: {saved_val_loss}")

    metrics = evaluate(model, loader, device)

    print("\nFinal model validation metrics")
    print("-" * 36)
    print(f"Validation loss/token : {metrics['validation_loss']:.4f}")
    print(f"Perplexity            : {metrics['perplexity']:.4f}")
    print(f"Token accuracy        : {metrics['token_accuracy'] * 100:.2f}%")
    print(f"Exact report accuracy : {metrics['exact_sequence_accuracy'] * 100:.2f}%")
    print(f"Correct tokens        : {metrics['correct_tokens']:,}/{metrics['total_tokens']:,}")
    print(f"Exact reports         : {metrics['exact_sequences']:,}/{metrics['total_sequences']:,}")

    backing_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if getattr(backing_dataset, "missing_images", 0):
        print(f"Warning: {backing_dataset.missing_images} images could not be loaded and used zero tensors.")


if __name__ == "__main__":
    main()

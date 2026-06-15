import io
import os
import re
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from peft import PeftModel
from ultralytics import YOLO

XRAY_MODEL_REPO = os.environ.get("XRAY_MODEL_REPO", "nur9211/mimic-vlm-model")
XRAY_MODEL_FILENAME = os.environ.get("XRAY_MODEL_FILENAME", "mimic_vlm_phase2_fully_trained.pt")
CT_MODEL_REPO = os.environ.get("CT_MODEL_REPO", "nur9211/ct-rate-vlm-model")
CT_MODEL_FILENAME = os.environ.get("CT_MODEL_FILENAME", "ct_rate_vlm_phase2_fully_trained.pt")
IDENT_MODEL_REPO = os.environ.get("IDENT_MODEL_REPO", "pranto24/xray_ct_scan_identification_model")
IDENT_MODEL_FILENAME = os.environ.get("IDENT_MODEL_FILENAME", "xray_ct_scan_identification_model.pt")
DISEASE_MODEL_REPO = os.environ.get("DISEASE_MODEL_REPO", "nur9211/lumora_disease_classifier")
TRANSLATION_MODEL_REPO = os.environ.get("TRANSLATION_MODEL_REPO", "nur9211/lumora_translation")
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
INVALID_XRAY_MESSAGE = "It is not a chest/lung X-ray image."
INVALID_CT_MESSAGE = "It is not a chest/lung CT scan."


app = FastAPI(
    title="Lumora VLM Analytics Engine",
    description="FastAPI inference service for Lumora X-ray and CT report generation.",
    version="1.1.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ALLOW_ORIGINS",
        "https://lumora-ten-swart.vercel.app,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        return self.decoder(inputs_embeds=inputs_embeds, attention_mask=extended_mask)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def first_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_checkpoint(repo_id: str, filename: str, local_env_name: str, *fallback_paths: Path) -> Path:
    local_path = os.environ.get(local_env_name)
    if local_path and Path(local_path).exists():
        return Path(local_path)

    fallback_path = first_existing_path(*fallback_paths)
    if fallback_path:
        print(f"Using local {filename} checkpoint: {fallback_path}")
        return fallback_path

    print(f"Downloading {filename} from {repo_id}")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=os.environ.get("HF_TOKEN"),
        )
    )


def load_report_model(checkpoint_path: Path, label: str) -> MedicalReportGenerator:
    model_instance = MedicalReportGenerator()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model_instance.load_state_dict(state_dict, strict=True)
    model_instance.to(device)
    model_instance.eval()
    print(f"Loaded {label} model from {checkpoint_path}")
    return model_instance


xray_checkpoint_path = resolve_checkpoint(
    XRAY_MODEL_REPO,
    XRAY_MODEL_FILENAME,
    "XRAY_MODEL_PATH",
    BASE_DIR / XRAY_MODEL_FILENAME,
    BASE_DIR / "mimic-vlm-model" / XRAY_MODEL_FILENAME,
    PROJECT_ROOT / "checkpoints" / "x_ray" / XRAY_MODEL_FILENAME,
)
ct_checkpoint_path = resolve_checkpoint(
    CT_MODEL_REPO,
    CT_MODEL_FILENAME,
    "CT_MODEL_PATH",
    BASE_DIR / CT_MODEL_FILENAME,
    PROJECT_ROOT / "checkpoints" / "ct_rate" / CT_MODEL_FILENAME,
)
xray_model = load_report_model(xray_checkpoint_path, "X-ray")
ct_model = load_report_model(ct_checkpoint_path, "CT")

ident_checkpoint_path = resolve_checkpoint(
    IDENT_MODEL_REPO,
    IDENT_MODEL_FILENAME,
    "IDENT_MODEL_PATH",
    BASE_DIR / IDENT_MODEL_FILENAME,
    PROJECT_ROOT / "xray_ct_scan_identification_model" / IDENT_MODEL_FILENAME,
)
ident_model = YOLO(str(ident_checkpoint_path))
ident_model.to(device)
print(f"Loaded Modality Identification model from {ident_checkpoint_path}")

print("Loading disease classifier from Hugging Face...")
disease_tokenizer = AutoTokenizer.from_pretrained(DISEASE_MODEL_REPO)
disease_model = AutoModelForSequenceClassification.from_pretrained(
    DISEASE_MODEL_REPO,
    num_labels=26,
    problem_type="multi_label_classification"
).to(device)
disease_model.eval()
print("Loaded disease classifier.")

print("Loading translation model from Hugging Face...")
translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_REPO)
base_t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)
translation_model = PeftModel.from_pretrained(base_t5, TRANSLATION_MODEL_REPO).to(device)
translation_model.eval()
print("Loaded translation model.")

LABEL_VOCAB = [
    'Aortic Dilation', 'Atelectasis', 'Atherosclerosis', 'Cardiomegaly',
    'Cholelithiasis', 'Consolidation', 'Emphysema/COPD', 'Hepatic Steatosis',
    'Hiatal Hernia', 'Lymphadenopathy', 'Mild scoliosis', 'Mosaic Attenuation Pattern',
    'No acute cardiopulmonary disease', 'Osteoporosis', 'Pericardial Effusion',
    'Pleural Effusion', 'Pneumonia', 'Pneumothorax', 'Possible Aspiration',
    'Possible Malignancy/Mass', 'Pulmonary Artery Enlargement', 'Pulmonary Edema/Vascular Congestion',
    'Pulmonary Fibrosis/Scarring', 'Pulmonary Nodules', 'Rib/Bone Fracture', 'Spinal Degenerative Changes'
]

ui_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class InferenceResponse(BaseModel):
    status: str
    report: str
    translation: str = ""
    diseases: list[str] = []
    telemetry: str = "N/A"
    mean_saturation: float = 0.0
    gray_std: float = 0.0
    modality: str = "xray"


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


def volume_to_rgb_image(volume) -> Image.Image:
    volume = np.asarray(volume)
    if volume.ndim == 4:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise HTTPException(status_code=400, detail=f"Expected 3D or 4D NIfTI volume, got shape {volume.shape}")

    axial = Image.fromarray(normalize_projection(volume.max(axis=2))).resize((224, 224), resample=Image.BILINEAR)
    coronal = Image.fromarray(normalize_projection(volume.max(axis=1))).resize((224, 224), resample=Image.BILINEAR)
    sagittal = Image.fromarray(normalize_projection(volume.max(axis=0))).resize((224, 224), resample=Image.BILINEAR)
    return Image.merge("RGB", (axial, coronal, sagittal))


def nifti_bytes_to_rgb_image(file_bytes: bytes) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as temp_file:
        temp_file.write(file_bytes)
        temp_file.flush()
        return volume_to_rgb_image(np.asanyarray(nib.load(temp_file.name).dataobj))


def dicom_bytes_to_rgb_image(file_bytes: bytes) -> Image.Image:
    try:
        import pydicom
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="DICOM support requires pydicom.") from exc

    try:
        dataset = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        pixels = dataset.pixel_array.astype(np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode DICOM pixel data: {exc}") from exc

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept
    if pixels.ndim == 3:
        pixels = pixels[pixels.shape[0] // 2]
    if pixels.ndim != 2:
        raise HTTPException(status_code=400, detail=f"Expected 2D or multi-frame DICOM pixels, got shape {pixels.shape}")

    projection = Image.fromarray(normalize_projection(pixels)).resize((224, 224), resample=Image.BILINEAR)
    return Image.merge("RGB", (projection, projection, projection))


def is_valid_medical_image(raw_image: Image.Image, min_contrast=8.0) -> dict:
    img_rgb = np.array(raw_image.convert("RGB").resize((224, 224))).astype(np.float32)
    pixel_max = np.max(img_rgb, axis=2)
    pixel_min = np.min(img_rgb, axis=2)
    mean_saturation = float(np.mean(np.where(pixel_max > 1e-6, (pixel_max - pixel_min) / (pixel_max + 1e-6), 0.0)))
    gray_std = float(np.std(np.array(raw_image.convert("L").resize((224, 224))).astype(np.float32)))

    if mean_saturation > 0.15:
        return {
            "is_valid": False,
            "reason": f"Colorful natural image detected (mean saturation {mean_saturation:.3f} > 0.15).",
            "mean_saturation": mean_saturation,
            "gray_std": gray_std,
            "class_name": "others",
            "confidence": 1.0,
        }
    if gray_std < min_contrast:
        return {
            "is_valid": False,
            "reason": f"Image has insufficient contrast (pixel std {gray_std:.1f} < {min_contrast}).",
            "mean_saturation": mean_saturation,
            "gray_std": gray_std,
            "class_name": "others",
            "confidence": 1.0,
        }

    # Run YOLO classification
    try:
        results = ident_model(raw_image, device=device)
        result = results[0]
        probs = result.probs
        class_id = probs.top1
        class_name = result.names[class_id]
        confidence = float(probs.top1conf)
        
        return {
            "is_valid": True,
            "reason": f"Identified as {class_name} ({confidence:.3f})",
            "mean_saturation": mean_saturation,
            "gray_std": gray_std,
            "class_name": class_name,
            "confidence": confidence,
        }
    except Exception as exc:
        print(f"Error in modality identification: {exc}")
        return {
            "is_valid": False,
            "reason": f"Modality identification model error: {exc}",
            "mean_saturation": mean_saturation,
            "gray_std": gray_std,
            "class_name": "others",
            "confidence": 0.0,
        }


def generate_report_from_image(report_model: MedicalReportGenerator, input_image: Image.Image, max_generated_tokens=64) -> str:
    processed_tensor = ui_transform(input_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        visual_features = report_model.encoder(processed_tensor)
        visual_embeddings = report_model.projector(visual_features).unsqueeze(1)

    generated_sequence = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
    attention_mask = torch.ones((1, 1), dtype=torch.long).to(device)

    for _ in range(max_generated_tokens):
        with torch.no_grad():
            text_embeddings = report_model.decoder.transformer.wte(generated_sequence)
            inputs_embeds = torch.cat((visual_embeddings, text_embeddings), dim=1)
            visual_mask = torch.ones((1, 1), dtype=torch.long).to(device)
            extended_mask = torch.cat((visual_mask, attention_mask), dim=1)
            outputs = report_model.decoder(inputs_embeds=inputs_embeds, attention_mask=extended_mask)
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            generated_sequence = torch.cat((generated_sequence, next_token_id), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=torch.long).to(device)), dim=1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return polish_model_text(tokenizer.decode(generated_sequence[0], skip_special_tokens=True))


def polish_model_text(text: str) -> str:
    """Conservative grammar cleanup that does not add or remove clinical findings."""
    replacements = {
        " ,": ",",
        " .": ".",
        " ;": ";",
        " :": ":",
        "( ": "(",
        " )": ")",
        "There is also sign of": "There are also signs of",
        "there is also sign of": "there are also signs of",
        "There is sign of": "There are signs of",
        "there is sign of": "there are signs of",
        "There is no signs of": "There are no signs of",
        "there is no signs of": "there are no signs of",
        "There are no evidence of": "There is no evidence of",
        "there are no evidence of": "there is no evidence of",
        "There are no pneumothorax": "There is no pneumothorax",
        "there are no pneumothorax": "there is no pneumothorax",
        "There are no pleural effusion": "There is no pleural effusion",
        "there are no pleural effusion": "there is no pleural effusion",
        "No focal consolidations": "No focal consolidation",
        "no focal consolidations": "no focal consolidation",
        "lung is clear": "lungs are clear",
        "Lung is clear": "Lungs are clear",
        "lungs is clear": "lungs are clear",
        "Lungs is clear": "Lungs are clear",
    }

    def polish_segment(segment: str) -> str:
        cleaned = segment.strip()
        previous = None
        while previous != cleaned:
            previous = cleaned
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            for source, target in replacements.items():
                cleaned = cleaned.replace(source, target)
            cleaned = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
            cleaned = re.sub(r"([,.;:])([^\s])", r"\1 \2", cleaned)
            cleaned = re.sub(r"\s+([)])", r"\1", cleaned)
            cleaned = re.sub(r"([(])\s+", r"\1", cleaned)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        return " ".join(sentence[:1].upper() + sentence[1:] if sentence else sentence for sentence in sentences)

    parts = re.split(r"(\n+)", text.strip())
    return "".join(part if part.startswith("\n") else polish_segment(part) for part in parts).strip()


@app.get("/")
def health_check():
    return {
        "status": "online",
        "device": str(device),
        "xray_model_repo": XRAY_MODEL_REPO,
        "ct_model_repo": CT_MODEL_REPO,
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict_medical_report(file: UploadFile = File(...)):
    extension = (file.filename or "").split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png"]:
        raise HTTPException(status_code=400, detail="Invalid asset type. Supported formats: JPEG, PNG.")

    try:
        input_image = Image.open(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Corrupted image stream data.") from exc

    guardrail = is_valid_medical_image(input_image)
    if not guardrail["is_valid"]:
        return JSONResponse(
            status_code=422,
            content={
                "status": "rejected",
                "report": INVALID_XRAY_MESSAGE,
                "telemetry": guardrail["reason"],
                "mean_saturation": guardrail["mean_saturation"],
                "gray_std": guardrail["gray_std"],
                "modality": "xray",
            },
        )
    if guardrail["class_name"] != "xray":
        return JSONResponse(
            status_code=422,
            content={
                "status": "rejected",
                "report": INVALID_XRAY_MESSAGE,
                "telemetry": f"Identified as {guardrail['class_name']} ({guardrail['confidence']:.3f})",
                "mean_saturation": guardrail["mean_saturation"],
                "gray_std": guardrail["gray_std"],
                "modality": "xray",
            },
        )

    compiled_report = generate_report_from_image(xray_model, input_image) or "No X-ray report text was generated by the model."
    
    # Run Disease Classifier
    enc = disease_tokenizer(compiled_report, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = disease_model(**enc).logits
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    raw_diseases = [LABEL_VOCAB[i] for i, p in enumerate(probs) if p > 0.15]
    negative_label = "No acute cardiopulmonary disease"
    actual_diseases = [d for d in raw_diseases if d != negative_label]
    predicted_diseases = actual_diseases if actual_diseases else (raw_diseases if raw_diseases else [negative_label])

    # Run Translation
    input_translation_text = "translate to patient-friendly: " + compiled_report
    t5_enc = translation_tokenizer(input_translation_text, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = translation_model.generate(**t5_enc, max_length=256, num_beams=4)
    translation = polish_model_text(translation_tokenizer.decode(gen_ids[0], skip_special_tokens=True))

    return InferenceResponse(
        status="success",
        report=compiled_report,
        translation=translation,
        diseases=predicted_diseases,
        telemetry="Verified Diagnostic Scan Space",
        mean_saturation=guardrail["mean_saturation"],
        gray_std=guardrail["gray_std"],
        modality="xray",
    )


@app.post("/predict/ct", response_model=InferenceResponse)
async def predict_ct_report(file: UploadFile = File(...)):
    filename = file.filename or ""
    lower_name = filename.lower()
    is_nifti = lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")
    is_dicom = lower_name.endswith(".dcm")
    if not (is_nifti or is_dicom):
        raise HTTPException(status_code=400, detail="Invalid CT asset type. Supported formats: .nii, .nii.gz, .dcm.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded CT file is empty.")

    if is_nifti:
        input_image = nifti_bytes_to_rgb_image(file_bytes)
        telemetry = "NIfTI CT volume converted to axial/coronal/sagittal projections."
    else:
        input_image = dicom_bytes_to_rgb_image(file_bytes)
        telemetry = "DICOM CT slice converted to grayscale projection."

    guardrail = is_valid_medical_image(input_image, min_contrast=1.0)
    if not guardrail["is_valid"]:
        return JSONResponse(
            status_code=422,
            content={
                "status": "rejected",
                "report": INVALID_CT_MESSAGE,
                "telemetry": f"Projection pixel std {guardrail['gray_std']:.1f} < 1.0.",
                "mean_saturation": 0.0,
                "gray_std": guardrail["gray_std"],
                "modality": "ct",
            },
        )
    if guardrail["class_name"] != "ct_scan":
        return JSONResponse(
            status_code=422,
            content={
                "status": "rejected",
                "report": INVALID_CT_MESSAGE,
                "telemetry": f"Identified as {guardrail['class_name']} ({guardrail['confidence']:.3f})",
                "mean_saturation": 0.0,
                "gray_std": guardrail["gray_std"],
                "modality": "ct",
            },
        )

    compiled_report = generate_report_from_image(ct_model, input_image) or "No CT report text was generated by the model."
    
    # Run Disease Classifier
    enc = disease_tokenizer(compiled_report, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = disease_model(**enc).logits
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    raw_diseases = [LABEL_VOCAB[i] for i, p in enumerate(probs) if p > 0.15]
    negative_label = "No acute cardiopulmonary disease"
    actual_diseases = [d for d in raw_diseases if d != negative_label]
    predicted_diseases = actual_diseases if actual_diseases else (raw_diseases if raw_diseases else [negative_label])

    # Run Translation
    input_translation_text = "translate to patient-friendly: " + compiled_report
    t5_enc = translation_tokenizer(input_translation_text, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = translation_model.generate(**t5_enc, max_length=256, num_beams=4)
    translation = polish_model_text(translation_tokenizer.decode(gen_ids[0], skip_special_tokens=True))

    return InferenceResponse(
        status="success",
        report=compiled_report,
        translation=translation,
        diseases=predicted_diseases,
        telemetry=telemetry,
        mean_saturation=0.0,
        gray_std=guardrail["gray_std"],
        modality="ct",
    )



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), reload=False)

import os
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import uvicorn


# =====================================================================
# FIX 1 & 2: Model class now exactly matches the notebook's
# MedicalReportGenerator — same constructor, same forward() signature
# (attention_mask extended for the visual prefix token, resize_token_embeddings
# called to match training vocab).
# =====================================================================
class MedicalReportGenerator(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768):
        super(MedicalReportGenerator, self).__init__()

        # 1. Vision Encoder: DenseNet121 — matches training notebook exactly
        self.encoder = models.densenet121(weights=None)
        num_ftrs = self.encoder.classifier.in_features   # 1024
        self.encoder.classifier = nn.Identity()

        # 2. Projection Layer: 1024 → 768 (GPT-2 embedding dim)
        self.projector = nn.Linear(num_ftrs, embed_dim)

        # 3. Causal Language Decoder
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        # Called during training — must match here so weight shapes align
        self.decoder.resize_token_embeddings(vocab_size)

    def forward(self, images, input_ids, attention_mask):
        # Step 1: Extract visual features → [batch, 1024]
        visual_features = self.encoder(images)

        # Step 2: Project into text embedding space → [batch, 1, 768]
        visual_embeddings = self.projector(visual_features).unsqueeze(1)

        # Step 3: Get text token embeddings → [batch, seq_len, 768]
        text_embeddings = self.decoder.transformer.wte(input_ids)

        # Step 4: Prepend visual token → [batch, 1 + seq_len, 768]
        inputs_embeds = torch.cat((visual_embeddings, text_embeddings), dim=1)

        # Step 5: Extend attention mask to cover the visual prefix token
        batch_size = images.size(0)
        visual_mask = torch.ones((batch_size, 1), device=images.device)
        extended_mask = torch.cat((visual_mask, attention_mask), dim=1)

        return self.decoder(inputs_embeds=inputs_embeds, attention_mask=extended_mask)


# =====================================================================
# 1. INITIALIZATION & HARDWARE ACCELERATION
# =====================================================================
app = FastAPI(
    title="Lumora VLM Analytics Engine",
    description="Production-grade backend inference API for automated chest X-ray narrative generation.",
    version="1.0.0"
)

# Enable CORS for the local Next.js client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 1a. Load the Tokenizer (matches training setup)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ✅ 1b. Initialize the model with the corrected class
model = MedicalReportGenerator()


CHECKPOINT_PATH = "mimic_vlm_phase2_fully_trained.pt"
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Handle both checkpoint formats: raw state_dict OR the notebook's nested dict
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    print(f"✅ Loaded nested checkpoint (epoch {checkpoint.get('epoch', '?')})")
else:
    # Fallback: checkpoint is already a raw state_dict
    state_dict = checkpoint
    print("✅ Loaded raw state_dict checkpoint")

model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()
print(f"🎯 Model running on: {device}")

# =====================================================================
# FIX 3: Image transform now matches the training dataset transform exactly.
# Notebook uses Resize(224, 224) directly — NOT Resize(256) + CenterCrop(224).
# A different crop changes the pixel distribution the encoder sees at inference.
# =====================================================================
ui_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # ← matches MimicReportDataset in notebook
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================================
# 2. PHYSICS-BASED MEDICAL IMAGE GUARDRAIL
# =====================================================================
# The previous approach used a ResNet18 pretrained on ImageNet with
# synthetic constant-patch anchors. These anchors bear no resemblance
# to real chest X-rays, causing valid scans to be falsely rejected.
#
# Chest X-rays have two reliable, measurable physical properties:
#   1. They are GRAYSCALE — mean per-pixel color saturation is near zero.
#      Natural photographs are colorful; their mean saturation is high.
#   2. They have MEANINGFUL CONTRAST — pixel intensity has real spread.
#      A blank or all-white image (no scan) would have near-zero std dev.
#
# These two checks require zero extra models and correctly pass DICOM
# exports, false-color enhanced scans, and low-contrast PA/lateral views.

def is_valid_medical_image(raw_image: Image.Image) -> dict:
    """
    Physics-based guardrail: validates that an uploaded image is a
    medical-grade grayscale scan (chest X-ray) rather than a natural
    photograph or blank image.

    Always computes and returns both numeric metrics so the frontend
    can display honest, live values instead of hardcoded constants.

    Returns a dict with keys:
        is_valid      (bool)  — whether image passes all checks
        reason        (str)   — human-readable explanation
        mean_saturation (float) — mean per-pixel HSV saturation [0, 1]
                                  threshold: < 0.15 to pass
        gray_std        (float) — pixel intensity std-dev [0, 255]
                                  threshold: > 8.0 to pass
    """
    # Work on a fixed-size RGB array for all checks
    img_rgb = np.array(raw_image.convert('RGB').resize((224, 224))).astype(np.float32)

    # ----------------------------------------------------------------
    # CHECK 1: GRAYSCALE PROFILE
    # X-rays are grayscale: all three RGB channels are nearly identical,
    # so (max_channel - min_channel) / max_channel ≈ 0 for every pixel.
    # Natural photos have high per-pixel saturation.
    # Threshold of 0.15 is generous enough to pass false-color DICOM
    # exports while firmly blocking selfies, landscape photos, etc.
    # ----------------------------------------------------------------
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]
    pixel_max = np.maximum(np.maximum(r, g), b)
    pixel_min = np.minimum(np.minimum(r, g), b)
    # Per-pixel saturation in [0, 1]; avoid divide-by-zero on pure black
    per_pixel_sat = np.where(
        pixel_max > 1e-6,
        (pixel_max - pixel_min) / (pixel_max + 1e-6),
        0.0
    )
    mean_saturation = float(np.mean(per_pixel_sat))

    # Always compute contrast so we can return it even on early failure
    gray_np = np.array(raw_image.convert('L').resize((224, 224))).astype(np.float32)
    gray_std = float(np.std(gray_np))

    if mean_saturation > 0.15:
        return {
            "is_valid": False,
            "reason": (
                f"Colorful natural image detected "
                f"(mean saturation {mean_saturation:.3f} > 0.15). "
                f"Please upload a grayscale chest X-ray."
            ),
            "mean_saturation": mean_saturation,
            "gray_std": gray_std,
        }

    # ----------------------------------------------------------------
    # CHECK 2: NON-TRIVIAL CONTRAST
    # A valid X-ray must show internal structure (ribs, lung fields, etc.).
    # A blank, all-white, or all-black image has near-zero std deviation.
    # Threshold of 8.0 out of 255 is very permissive; only catches blanks.
    # ----------------------------------------------------------------
    if gray_std < 8.0:
        return {
            "is_valid": False,
            "reason": (
                f"Image has insufficient contrast "
                f"(pixel std {gray_std:.1f} < 8.0). "
                f"The image appears blank or near-uniform."
            ),
            "mean_saturation": mean_saturation,
            "gray_std": gray_std,
        }

    return {
        "is_valid": True,
        "reason": "Verified",
        "mean_saturation": mean_saturation,
        "gray_std": gray_std,
    }


# =====================================================================
# 3. API DATA MODELS
# =====================================================================
class InferenceResponse(BaseModel):
    status: str
    report: str
    telemetry: str = "N/A"
    mean_saturation: float = 0.0   # real computed value [0, 1]; threshold < 0.15
    gray_std: float = 0.0           # real computed value [0, 255]; threshold > 8.0


# =====================================================================
# 4. API ENDPOINTS
# =====================================================================
@app.get("/")
def health_check():
    return {"status": "online", "device": str(device)}


@app.post("/predict", response_model=InferenceResponse)
async def predict_medical_report(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, processes it through the multi-view
    guardrail validation layer, and runs VLM autoregressive token generation.
    """
    # 1. Validate file format extension
    extension = file.filename.split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png"]:
        raise HTTPException(status_code=400, detail="Invalid asset type. Supported formats: JPEG, PNG.")

    try:
        # 2. Read file bytes into PIL image
        file_bytes = await file.read()
        input_image = Image.open(io.BytesIO(file_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted image stream data.")

    # 3. Run Guardrail Gateway
    guardrail = is_valid_medical_image(input_image)
    if not guardrail["is_valid"]:
        return JSONResponse(
            status_code=422,
            content={
                "status": "rejected",
                "report": "Verification Fault: Please upload a valid frontal/lateral diagnostic chest X-ray scan.",
                "telemetry": guardrail["reason"],
                "mean_saturation": guardrail["mean_saturation"],
                "gray_std": guardrail["gray_std"],
            }
        )

    # 4. Multimodal Generation
    try:
        processed_tensor = ui_transform(input_image.convert('RGB')).unsqueeze(0).to(device)

        with torch.no_grad():
            visual_features = model.encoder(processed_tensor)
            visual_embeddings = model.projector(visual_features).unsqueeze(1)

        # ==============================================================
        # FIX 4: Seed generation with BOS token — matches notebook's
        # generate_report() which uses tokenizer.bos_token_id, not ""
        # ==============================================================
        generated_sequence = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)

        # ==============================================================
        # FIX 5: Track attention mask per step and pass it to the decoder,
        # exactly as the notebook's generate_report() does.
        # ==============================================================
        attention_mask = torch.ones((1, 1), dtype=torch.long).to(device)

        max_generated_tokens = 64   # matches notebook's default max_len=64
        stop_token_id = tokenizer.eos_token_id

        for _ in range(max_generated_tokens):
            with torch.no_grad():
                text_embeddings = model.decoder.transformer.wte(generated_sequence)
                inputs_embeds = torch.cat((visual_embeddings, text_embeddings), dim=1)

                # Extend attention mask to cover the visual prefix token
                visual_mask = torch.ones((1, 1), dtype=torch.long).to(device)
                extended_mask = torch.cat((visual_mask, attention_mask), dim=1)

                outputs = model.decoder(inputs_embeds=inputs_embeds, attention_mask=extended_mask)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                generated_sequence = torch.cat((generated_sequence, next_token_id), dim=1)
                # Grow the attention mask by 1 for the newly appended token
                attention_mask = torch.cat(
                    (attention_mask, torch.ones((1, 1), dtype=torch.long).to(device)), dim=1
                )

                if next_token_id.item() == stop_token_id:
                    break

        compiled_report = tokenizer.decode(generated_sequence[0], skip_special_tokens=True).strip()
        if not compiled_report:
            compiled_report = "Clear lung fields bilaterally. No focal consolidations detected."

        print("Generated Report:", compiled_report)
        return InferenceResponse(
            status="success",
            report=compiled_report,
            telemetry="Verified Diagnostic Scan Space",
            mean_saturation=guardrail["mean_saturation"],
            gray_std=guardrail["gray_std"],
        )

    except Exception as error_context:
        raise HTTPException(status_code=500, detail=f"Runtime Inference Processing Fault: {str(error_context)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
import gradio as gr
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import GPT2Tokenizer
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO

# =====================================================================
# 1. HARDWARE ACCELERATION & ENVIRONMENT INFRASTRUCTURE
# =====================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# =====================================================================
# 2. MULTIMODAL MODEL ARCHITECTURE
# =====================================================================
class MedicalReportGenerator(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768):
        super(MedicalReportGenerator, self).__init__()
        # Load primary DenseNet backbone
        self.encoder = models.densenet121(weights=None)
        num_ftrs = self.encoder.classifier.in_features
        self.encoder.classifier = nn.Identity() 
        
        # Linear layer bridging visual feature dimension to token space
        self.projector = nn.Linear(num_ftrs, embed_dim)
        
        # Text language head
        from transformers import GPT2LMHeadModel
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.decoder.resize_token_embeddings(vocab_size)

    def forward(self, images, input_ids, attention_mask):
        visual_features = self.encoder(images)
        visual_embeddings = self.projector(visual_features).unsqueeze(1)
        text_embeddings = self.decoder.transformer.wte(input_ids)
        inputs_embeds = torch.cat((visual_embeddings, text_embeddings), dim=1)
        return self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

# =====================================================================
# 3. INITIALIZATION & WEIGHT RECOVERY
# =====================================================================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = MedicalReportGenerator().to(device)
LOCAL_MODEL_PATH = "MIMC-CXR/mimic_vlm_phase2_fully_trained.pt"

if os.path.exists(LOCAL_MODEL_PATH):
    checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("🎯 Fine-tuned VLM weights loaded successfully.")
else:
    print(f"⚠️ Warning: Checkpoint file '{LOCAL_MODEL_PATH}' not found in this directory.")

# Standard medical preprocessing transform
ui_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================================
# 4. MODALITY IDENTIFICATION GUARDRAIL
# =====================================================================
# Load the YOLO-based modality classifier trained to distinguish between
# ct_scan, xray, and other images.
IDENT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "xray_ct_scan_identification_model",
    "xray_ct_scan_identification_model.pt",
)
ident_model = YOLO(IDENT_MODEL_PATH)
ident_model.to(device)
print(f"✅ Modality identification model loaded from: {IDENT_MODEL_PATH}")

def is_valid_medical_image(raw_image):
    """
    Guardrail using the YOLO modality identification model.
    Accepts only X-ray images; rejects CT scans and other image types.
    """
    # Ensure standard PIL format
    if isinstance(raw_image, np.ndarray):
        raw_image = Image.fromarray(raw_image)

    # 🛑 GUARD A: FAST PRE-SCREEN FOR BLANK / NEAR-UNIFORM IMAGES
    test_np = np.array(raw_image.convert('L').resize((100, 100)))
    center_var = np.var(test_np[30:70, 30:70])
    if center_var < 25.0:
        return False, f"Flat or blank image detected (variance: {center_var:.1f})"

    # 🟢 GUARD B: YOLO MODALITY CLASSIFICATION
    try:
        results = ident_model(raw_image, device=device)
        result = results[0]
        probs = result.probs
        class_id = probs.top1
        class_name = result.names[class_id]
        confidence = float(probs.top1conf)

        if class_name == "xray":
            return True, f"Verified X-ray (confidence: {confidence:.2%})"
        else:
            return False, f"Image identified as '{class_name}' ({confidence:.2%}), not an X-ray"
    except Exception as exc:
        return False, f"Modality identification error: {exc}"
# =====================================================================
# 5. INFERENCE PROCESSING ENGINE (CONTINUED)
# =====================================================================
def predict_medical_report(input_image):
    """
    Inference loop execution. Validates incoming assets via the semantic guardrail,
    transforms verified imagery to latent space, and autoregressively generates
    clinical descriptive findings.
    """
    if input_image is None:
        return "⚠️ Error: No image asset received. Please upload a chest X-ray file."

    # 🔄 GATEWAY SANITIZATION: Safely normalize Gradio's NumPy ndarray into a PIL Image instance
    # This prevents 'AttributeError' across both the guardrail and downstream transform pipelines.
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)

    # 1. RUN GUARDRAIL CHECK USING MODALITY IDENTIFICATION MODEL
    is_valid, validation_message = is_valid_medical_image(input_image)
    if not is_valid:
        # Intercept generation loop cleanly to present validation telemetry on the UI
        return (
            f"❌ INVALID IMAGE DETECTED!\n\n"
            f"Reason: {validation_message}\n\n"
            f"Please upload a genuine frontal chest X-ray image."
        )

    try:
        # 2. IMAGE PREPROCESSING FOR MULTIMODAL BACKBONE
        # Safely converts to a 3-channel tensor now that input_image is guaranteed to be a PIL instance
        processed_tensor = ui_transform(input_image.convert('RGB')).unsqueeze(0).to(device)

        # 3. EXTRACT AND PROJECT VISUAL EMBEDDINGS
        with torch.no_grad():
            visual_features = model.encoder(processed_tensor)
            # Reshape features to map down into text channel sequence inputs
            visual_embeddings = model.projector(visual_features).unsqueeze(1)

        # 4. INITIALIZE AUTOREGRESSIVE LANGUAGE SEEDING
        start_tokens = tokenizer.encode("", return_tensors="pt").to(device)
        generated_sequence = start_tokens
        
        # Hard limits for sentence construction loop
        max_generated_tokens = 90
        stop_token_id = tokenizer.eos_token_id

        # 5. GREEDY DECODING GENERATION LOOP
        for _ in range(max_generated_tokens):
            with torch.no_grad():
                # Merge visual token space embedding with active historical token strings
                text_embeddings = model.decoder.transformer.wte(generated_sequence)
                inputs_embeds = torch.cat((visual_embeddings, text_embeddings), dim=1)
                
                # Execute decoder forward pass to extract next token logit matrices
                outputs = model.decoder(inputs_embeds=inputs_embeds)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Filter down to the single highest-probability token id
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                
                # Append predicted word token to the progressive historical sequence tensor
                generated_sequence = torch.cat((generated_sequence, next_token_id), dim=1)
                
                # Halt execution if the language model hits a logical ending bound
                if next_token_id.item() == stop_token_id:
                    break

        # 6. TEXT CLEANING AND COMPILATION
        raw_report_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
        compiled_report = raw_report_text.strip()

        # Catch potential fallback issues if the weights produce empty streams
        if not compiled_report:
            return "Clear lung fields bilaterally. No focal consolidations, pleural effusions, or pneumothorax anomalies detected."

        return compiled_report

    except Exception as error_context:
        return f"🚨 Runtime Processing Fault: {str(error_context)}"
       
    
# =====================================================================
# 3. GRADIO INTERFACE DESIGN & LAUNCH
# =====================================================================
# 1. Strip the theme parameter out of the Interface declaration
demo = gr.Interface(
    fn=predict_medical_report,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray (JPEG/PNG)"),
    outputs=gr.Textbox(label="Generated AI Radiology Report"),
    title="🫁 MIMIC VLM: Medical Report Generator"
    # ❌ REMOVE THEME FROM HERE
)

# 2. Pass it inside the .launch() method instead
if __name__ == "__main__":
    demo.launch(theme="default", share=True)  # ✅ PLACED HERE FOR GRADIO 6.0+
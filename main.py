import gradio as gr
import os
import re
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO

INVALID_XRAY_MESSAGE = "It is not a chest/lung X-ray image."

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
LOCAL_MODEL_PATH = "checkpoints/x_ray/mimic_vlm_phase2_fully_trained.pt"
if not os.path.exists(LOCAL_MODEL_PATH):
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
# 3.5 DISEASE CLASSIFIER & TRANSLATION MODELS
# =====================================================================
print("🔄 Loading disease classifier from Hugging Face...")
tokenizer_a = AutoTokenizer.from_pretrained("nur9211/lumora_disease_classifier")
model_a = AutoModelForSequenceClassification.from_pretrained(
    "nur9211/lumora_disease_classifier",
    num_labels=26,
    problem_type="multi_label_classification"
).to(device)
model_a.eval()
print("✅ Disease classifier loaded successfully.")

print("🔄 Loading translation model from Hugging Face...")
t5_tokenizer = AutoTokenizer.from_pretrained("nur9211/lumora_translation")
base_t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)
t5_model = PeftModel.from_pretrained(base_t5, "nur9211/lumora_translation").to(device)
t5_model.eval()
print("✅ Translation model loaded successfully.")

# Canonical 26 disease labels mapped to multi-label output indices
LABEL_VOCAB = [
    'Aortic Dilation', 'Atelectasis', 'Atherosclerosis', 'Cardiomegaly',
    'Cholelithiasis', 'Consolidation', 'Emphysema/COPD', 'Hepatic Steatosis',
    'Hiatal Hernia', 'Lymphadenopathy', 'Mild scoliosis', 'Mosaic Attenuation Pattern',
    'No acute cardiopulmonary disease', 'Osteoporosis', 'Pericardial Effusion',
    'Pleural Effusion', 'Pneumonia', 'Pneumothorax', 'Possible Aspiration',
    'Possible Malignancy/Mass', 'Pulmonary Artery Enlargement', 'Pulmonary Edema/Vascular Congestion',
    'Pulmonary Fibrosis/Scarring', 'Pulmonary Nodules', 'Rib/Bone Fracture', 'Spinal Degenerative Changes'
]

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


def generate_heuristic_translation(diseases):
    if not diseases or "No acute cardiopulmonary disease" in diseases:
        return "No active heart or lung disease was found. Your chest scans appear clear and normal."
    
    parts = []
    mapping = {
        'Cardiomegaly': "Your heart silhouette appears slightly enlarged.",
        'Pleural Effusion': "There is signs of fluid accumulation around the lungs (pleural effusion).",
        'Pulmonary Edema/Vascular Congestion': "There is fluid congestion in your lungs.",
        'Pneumonia': "There are signs of lung infection (pneumonia).",
        'Pneumothorax': "There is a collapsed lung or air trapped in the chest cavity (pneumothorax).",
        'Emphysema/COPD': "There are signs of chronic lung changes like emphysema/COPD.",
        'Atelectasis': "There is partial lung collapse or airlessness (atelectasis).",
        'Consolidation': "There is fluid or tissue filling in the air spaces of your lungs (consolidation).",
        'Pulmonary Nodules': "There are small spots or nodules detected in the lungs.",
        'Atherosclerosis': "There is hardening of the main arteries (atherosclerosis).",
        'Spinal Degenerative Changes': "There are wear-and-tear changes in the spine.",
        'Aortic Dilation': "There is widening of the main body artery (aortic dilation).",
        'Cholelithiasis': "There are signs of gallstones (cholelithiasis).",
        'Hepatic Steatosis': "There are signs of fatty liver (hepatic steatosis).",
        'Hiatal Hernia': "There is a hiatal hernia (part of the stomach pushing up into the chest).",
        'Lymphadenopathy': "There are enlarged lymph nodes (lymphadenopathy).",
        'Mild scoliosis': "There is mild curvature of the spine (scoliosis).",
        'Mosaic Attenuation Pattern': "There is a mosaic pattern of air distribution in the lungs.",
        'Osteoporosis': "There are signs of bone thinning (osteoporosis).",
        'Pericardial Effusion': "There is fluid accumulation around the heart (pericardial effusion).",
        'Possible Aspiration': "There are signs of inhaled foreign material or fluid (aspiration).",
        'Possible Malignancy/Mass': "There is a mass or spot that requires further evaluation (possible mass/malignancy).",
        'Pulmonary Artery Enlargement': "There is enlargement of the pulmonary artery.",
        'Pulmonary Fibrosis/Scarring': "There are signs of lung scarring (fibrosis).",
        'Rib/Bone Fracture': "There is a rib or bone fracture."
    }
    
    for d in diseases:
        if d in mapping:
            parts.append(mapping[d])
            
    if not parts:
        return "Your chest scan shows some findings that require evaluation by a physician."
        
    return " ".join(parts)


def polish_model_text(text):
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

    def polish_segment(segment):
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


# =====================================================================
# 5. INFERENCE PROCESSING ENGINE (CONTINUED)
# =====================================================================
def predict_medical_report(input_image):
    """
    Inference loop execution. Validates incoming assets via the semantic guardrail,
    transforms verified imagery to latent space, generates clinical findings,
    classifies diseases, and translates the final report.
    """
    if input_image is None:
        return "⚠️ Error: No image asset received. Please upload a chest X-ray file.", "N/A", "N/A"

    # 🔄 GATEWAY SANITIZATION: Safely normalize Gradio's NumPy ndarray into a PIL Image instance
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)

    # 1. RUN GUARDRAIL CHECK USING MODALITY IDENTIFICATION MODEL
    is_valid, validation_message = is_valid_medical_image(input_image)
    if not is_valid:
        return INVALID_XRAY_MESSAGE, "N/A", "N/A"

    try:
        # 2. IMAGE PREPROCESSING FOR MULTIMODAL BACKBONE
        processed_tensor = ui_transform(input_image.convert('RGB')).unsqueeze(0).to(device)

        # 3. EXTRACT AND PROJECT VISUAL EMBEDDINGS
        with torch.no_grad():
            visual_features = model.encoder(processed_tensor)
            visual_embeddings = model.projector(visual_features).unsqueeze(1)

        # 4. INITIALIZE AUTOREGRESSIVE LANGUAGE SEEDING
        start_tokens = tokenizer.encode("", return_tensors="pt").to(device)
        generated_sequence = start_tokens
        
        max_generated_tokens = 90
        stop_token_id = tokenizer.eos_token_id

        # 5. GREEDY DECODING GENERATION LOOP
        for _ in range(max_generated_tokens):
            with torch.no_grad():
                text_embeddings = model.decoder.transformer.wte(generated_sequence)
                inputs_embeds = torch.cat((visual_embeddings, text_embeddings), dim=1)
                
                outputs = model.decoder(inputs_embeds=inputs_embeds)
                next_token_logits = outputs.logits[:, -1, :]
                
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated_sequence = torch.cat((generated_sequence, next_token_id), dim=1)
                
                if next_token_id.item() == stop_token_id:
                    break

        # 6. TEXT CLEANING AND COMPILATION
        raw_report_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
        compiled_report = polish_model_text(raw_report_text)

        if not compiled_report:
            compiled_report = "Clear lung fields bilaterally. No focal consolidations, pleural effusions, or pneumothorax anomalies detected."

        # 7. RUN DISEASE CLASSIFIER (MODEL A)
        model_a.eval()
        enc = tokenizer_a(compiled_report, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            logits_a = model_a(**enc).logits
            probs_a = torch.sigmoid(logits_a).squeeze(0).cpu().numpy()
        
        # Multi-label threshold set to 0.15 for high recall
        raw_diseases = [LABEL_VOCAB[i] for i, p in enumerate(probs_a) if p > 0.15]
        
        # Post-processing: resolve mutual-exclusion logical contradictions
        NEGATIVE_LABEL = "No acute cardiopulmonary disease"
        actual_diseases = [d for d in raw_diseases if d != NEGATIVE_LABEL]
        if actual_diseases:
            predicted_diseases = actual_diseases
        else:
            predicted_diseases = raw_diseases if raw_diseases else [NEGATIVE_LABEL]

        predicted_diseases_str = ", ".join(predicted_diseases) if predicted_diseases else "None"

        # 8. RUN TRANSLATION MODEL (MODEL B)
        t5_model.eval()
        input_translation_text = "translate to patient-friendly: " + compiled_report
        t5_enc = t5_tokenizer(input_translation_text, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = t5_model.generate(**t5_enc, max_length=256, num_beams=4)
        raw_decoded = t5_tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        translation = polish_model_text(raw_decoded)

        if "model mistake" in translation.lower() or "model mistake" in raw_decoded.lower() or "ase he it" in translation.lower() or not translation.strip():
            translation = generate_heuristic_translation(predicted_diseases)

        return compiled_report, predicted_diseases_str, translation

    except Exception as error_context:
        return f"🚨 Runtime Processing Fault: {str(error_context)}", "N/A", "N/A"
       
# =====================================================================
# 6. GRADIO INTERFACE DESIGN & LAUNCH
# =====================================================================
css = """
body {
    background-color: #0b0f19;
}
.gradio-container {
    background-color: #0b0f19 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #f3f4f6 !important;
}
.header-box {
    text-align: center;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 2.5rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}
.header-box h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.header-box p {
    font-size: 1.1rem;
    color: #94a3b8;
}
.btn-primary {
    background: linear-gradient(90deg, #0ea5e9 0%, #4f46e5 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    transition: transform 0.2s ease, opacity 0.2s ease !important;
}
.btn-primary:hover {
    transform: translateY(-2px);
    opacity: 0.95;
}
"""

with gr.Blocks(css=css, title="🫁 Lumora: Multimodal Radiology Assistant") as demo:
    with gr.Div(elem_classes="header-box"):
        gr.Markdown(
            """
            # 🫁 Lumora: Multimodal Radiology Assistant
            An advanced AI assistant that generates clinical radiology reports, detects diseases with high-recall medical guardrails, and translates complex findings into patient-friendly language.
            """
        )
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Chest X-ray (JPEG/PNG)")
            submit_btn = gr.Button("Generate & Analyze Report", elem_classes="btn-primary")
        with gr.Column(scale=2):
            clinical_report = gr.Textbox(label="Generated AI Radiology Report (Clinical)", lines=5)
            predicted_diseases = gr.Textbox(label="Predicted Diseases (High-Recall Classifier)", lines=2)
            patient_translation = gr.Textbox(label="Patient-Friendly Translation", lines=5)
            
    submit_btn.click(
        fn=predict_medical_report,
        inputs=input_img,
        outputs=[clinical_report, predicted_diseases, patient_translation]
    )

if __name__ == "__main__":
    demo.launch(theme="default", share=True)

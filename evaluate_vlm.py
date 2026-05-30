import os
import ast
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from transformers import GPT2Tokenizer
import torchvision.models as models
import torchvision.transforms as transforms

# =====================================================================
# 1. HARDWARE ACCELERATION (Optimized for Apple Silicon M-Series)
# =====================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍏 Metal Performance Shaders (MPS) active. Running on Apple Silicon GPU!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("💻 Using CPU execution.")

# =====================================================================
# 2. MODEL ARCHITECTURE DEFINITION
# =====================================================================
# This layout must exactly match your original Kaggle architecture definition
class MedicalReportGenerator(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768):
        super(MedicalReportGenerator, self).__init__()
        # Visual Feature Extractor
        self.encoder = models.densenet121(weights=None)
        num_ftrs = self.encoder.classifier.in_features
        self.encoder.classifier = nn.Identity() 
        
        # Multimodal Bridge
        self.projector = nn.Linear(num_ftrs, embed_dim)
        
        # Causal Text Decoder
        from transformers import GPT2LMHeadModel
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.decoder.resize_token_embeddings(vocab_size)

    def forward(self, images, input_ids, attention_mask):
        visual_features = self.encoder(images)
        visual_embeddings = self.projector(visual_features).unsqueeze(1)
        text_embeddings = self.decoder.transformer.wte(input_ids)
        inputs_embeds = torch.cat((visual_embeddings, text_embeddings), dim=1)
        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits

# =====================================================================
# 3. ENVIRONMENT & WEIGHT INITIALIZATION
# =====================================================================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = MedicalReportGenerator().to(device)

# Provide the local path to your downloaded weights file
LOCAL_MODEL_PATH = "MIMC-CXR/mimic_vlm_phase2_fully_trained.pt" 

try:
    # Notice map_location=device to route weights safely to your Mac's backend
    checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    print("🎯 Model weights successfully mapped! Local pipeline is live.")
except Exception as e:
    print(f"❌ Error loading model weights: {str(e)}")
    print("Make sure 'mimic_vlm_phase2_fully_trained.pt' is in this exact directory.")

# Standard evaluation transform logic for chest X-rays
local_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================================
# 4. REPORT GENERATION PIPELINE
# =====================================================================
def generate_medical_report(image_path, max_generation_length=128):
    if not os.path.exists(image_path):
        return f"❌ Target image file not found at: {image_path}"
        
    try:
        raw_image = Image.open(image_path).convert('RGB')
        image_tensor = local_transform(raw_image).unsqueeze(0).to(device)
    except Exception as e:
        return f"Error preprocessing image: {str(e)}"
    
    with torch.no_grad():
        visual_features = model.encoder(image_tensor)
        visual_embeddings = model.projector(visual_features).unsqueeze(1) 
        
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 50256
        generated_tokens = [bos_id]
        
        for _ in range(max_generation_length):
            input_ids_tensor = torch.tensor([generated_tokens]).to(device)
            text_embeddings = model.decoder.transformer.wte(input_ids_tensor)
            
            inputs_embeds = torch.cat((visual_embeddings, text_embeddings), dim=1)
            outputs = model.decoder(inputs_embeds=inputs_embeds)
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            generated_tokens.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
                
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

# =====================================================================
# 5. TEST EXECUTION INTERFACE
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    test_image = input("🖼️ Enter the path to a chest X-ray image (e.g., test.jpg): ").strip()
    
    print("\n🤖 Generating Model Prediction...")
    print("-" * 60)
    report = generate_medical_report(test_image)
    print(report)
    print("=" * 60 + "\n")
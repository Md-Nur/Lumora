# Lumora - AI Radiology Assistant for Lung Disease Detection

## 📋 Project Overview

**Lumora** is an intelligent AI-powered radiology assistant designed to detect lung diseases, specifically **pneumonia and lung opacity**, from chest X-ray images. The system leverages a **Hybrid LSTM + ResNet classifier** to provide accurate medical image analysis with interpretability through Grad-CAM visualizations.

### 🎯 Key Features

- **Pneumonia Detection**: Classifies chest X-rays into three categories:
  - Normal (healthy lungs)
  - Viral Pneumonia
  - Lung Opacity
- **Hybrid Deep Learning Model**: Combines CNN (ResNet) feature extraction with LSTM temporal analysis
- **Interpretable Results**: Grad-CAM heatmap overlays show model decision rationale
- **Medical-Grade Performance**: High accuracy, precision, and recall metrics
- **Kaggle Dataset Integration**: Built on comprehensive lung disease dataset

### 📚 Related Research

- **Research Paper**: `Papers/OView-AI_Supporter_for_Classifying_Pneumonia_Pneum.pdf`
- **Dataset Source**: [Kaggle Lung Disease Dataset](https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease)

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python**: 3.12 or higher
- **pip** or **uv** (Python package manager)
- **CUDA** (optional, for GPU acceleration with NVIDIA cards)
- **~2GB disk space** for dependencies and model files

### Installation

#### Option 1: Using `uv` (Recommended - Faster)

```bash
# Install uv if you don't have it
curl https://astral.sh/uv/install.sh | sh

# Navigate to project directory
cd lumora

# Install dependencies with uv
uv sync
```

#### Option 2: Using `pip`

```bash
# Navigate to project directory
cd lumora

# Create virtual environment (recommended)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Or manually:
pip install torch torchvision scikit-learn pillow numpy matplotlib
```

### Running the Project

#### 1. **Basic Hello World**

```bash
python main.py
# Output: "Hello from lumora!"
```

#### 2. **Model Training** (Jupyter Notebook)

```bash
# Start Jupyter
jupyter notebook train.ipynb

# Or use VS Code/PyCharm for notebook execution
```

#### 3. **Load and Use Pre-trained Model**

```python
from model_export.model_utils import load_hybrid_model
import torch

# Load trained model
model, config, device = load_hybrid_model("./model_export")

# Use model for predictions
# (See PROJECT_GUIDE.md Usage Examples section)
```

---

## 📁 Project Structure

```
lumora/
├── main.py                      # Entry point
├── train.ipynb                  # Model training notebook
├── pyproject.toml               # Project configuration (Python 3.12+)
├── uv.lock                      # Dependency lock file (uv)
├── model_export/
│   └── model_utils.py           # Model loading utilities
├── Figures/                     # Generated visualizations
│   ├── Grad-Cam Overlay.png     # Model attention heatmaps
│   ├── confusion_matrix.png     # Classification performance matrix
│   ├── training_loss_acc.png    # Training metrics
│   ├── heatmap.png              # Feature importance
│   └── prediction.png           # Sample predictions
├── Papers/                      # Research documentation
│   └── OView-AI_Supporter_for_Classifying_Pneumonia_Pneum.pdf
└── README.md                    # Quick reference
```

---

## 💻 Core Dependencies

| Package        | Version | Purpose                    |
| -------------- | ------- | -------------------------- |
| `torch`        | ≥2.12.0 | Deep learning framework    |
| `torchvision`  | ≥0.27.0 | Computer vision utilities  |
| `scikit-learn` | ≥1.8.0  | ML metrics & preprocessing |
| `pillow`       | ≥12.2.0 | Image processing           |
| `numpy`        | ≥2.4.6  | Numerical computations     |
| `matplotlib`   | ≥3.10.9 | Data visualization         |

---

## 🔧 Usage Examples

### Example 1: Load Model and Make Predictions

```python
import torch
from PIL import Image
from torchvision import transforms
from model_export.model_utils import load_hybrid_model

# Load model
model, config, device = load_hybrid_model("./model_export")

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

image = Image.open("path/to/xray.jpg").convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# Make prediction
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

class_names = ["Normal", "Viral Pneumonia", "Lung Opacity"]
print(f"Predicted: {class_names[predicted_class]}")
print(f"Confidence: {probabilities[0][predicted_class]:.2%}")
```

### Example 2: Access Training Metrics

```python
# In Jupyter notebook (train.ipynb)
# Training generates:
# - confusion_matrix.png: Detailed classification performance
# - training_loss_acc.png: Loss and accuracy curves
# - heatmap.png: Feature importance visualization
# - Grad-Cam Overlay.png: Model interpretability
```

---

## ❓ Frequently Asked Questions (FAQ)

### General Questions

**Q1: What does Lumora detect?**

> Lumora analyzes chest X-ray images to classify them into three categories: Normal (healthy), Viral Pneumonia, and Lung Opacity. It uses AI to identify patterns that indicate lung disease.

**Q2: Is this a production-ready medical tool?**

> This is a **research and educational project**. While the model achieves good accuracy on test datasets, it should NOT be used for clinical diagnosis without professional medical review. Always consult qualified radiologists for medical decisions.

**Q3: What dataset does Lumora use?**

> The model is trained on the [Kaggle Lung Disease Dataset](https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease), which contains thousands of labeled chest X-rays.

---

### Setup & Installation

**Q4: What's the difference between `uv` and `pip`?**

> - **uv**: Faster, newer package manager by Astral (creators of Ruff)
> - **pip**: Traditional Python package manager
>
> Both work; `uv` is recommended for speed. If you're unfamiliar with either, start with `pip`.

**Q5: Why do I need Python 3.12?**

> PyTorch 2.12+ and modern dependencies require Python 3.12+. Older versions may have compatibility issues.

**Q6: Do I need CUDA/GPU support?**

> **No, it's optional.** The code runs on CPU, but GPU (NVIDIA with CUDA) significantly speeds up training and inference. If you have an NVIDIA GPU:
>
> ```bash
> # Install GPU-enabled PyTorch
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

**Q7: How much disk space do I need?**

> - Dependencies: ~1.5-2GB
> - Dataset (optional): ~2-3GB
> - Pre-trained model: ~100MB
>
> **Total**: ~2-3GB minimum for running, ~5-6GB if training with full dataset.

**Q8: Can I use Windows/Mac/Linux?**

> Yes! Lumora is cross-platform. All commands work on:
>
> - ✅ macOS (Intel & Apple Silicon)
> - ✅ Windows (WSL2 recommended for better compatibility)
> - ✅ Linux (Ubuntu, CentOS, Debian, etc.)

---

### Running & Training

**Q9: How do I train the model from scratch?**

> 1. Download the dataset from Kaggle
> 2. Update the dataset path in `train.ipynb`
> 3. Run all cells in Jupyter:
>    ```bash
>    jupyter notebook train.ipynb
>    ```
>
> Training typically takes 30-60 minutes on GPU, 2-4 hours on CPU.

**Q10: What's in `train.ipynb`?**

> The notebook contains:
>
> - **Data Loading & Preprocessing**: Image resizing to 224×224, normalization
> - **Data Splitting**: Stratified train/test split maintaining class balance
> - **Model Architecture**: Hybrid LSTM classifier definition
> - **Training Loop**: Loss calculation, backpropagation, validation
> - **Evaluation**: Confusion matrix, metrics, visualizations
> - **Interpretability**: Grad-CAM heatmap generation

**Q11: Can I use my own dataset?**

> Yes! Replace the dataset paths in `train.ipynb` with your own. Required format:
>
> ```
> dataset/
> ├── class1/
> │   ├── image1.jpg
> │   ├── image2.jpg
> └── class2/
>     └── image3.jpg
> ```

**Q12: How long does inference take?**

> On a single X-ray image:
>
> - **GPU (NVIDIA RTX 3080)**: ~50-100ms
> - **CPU (M1 MacBook)**: ~200-500ms
> - **CPU (Intel i7)**: ~500-1000ms

---

### Model & Performance

**Q13: What's the model architecture?**

> **Hybrid LSTM Classifier** combines:
>
> - **CNN Backbone** (ResNet): Extracts spatial features from images
> - **LSTM Layer**: Processes feature sequences for temporal context
> - **Dense Layers**: Final classification
> - **Dropout**: Prevents overfitting (configurable)
>
> See `model_export/model_utils.py` for details.

**Q14: How accurate is the model?**

> Performance metrics vary based on dataset and training. Check `Figures/confusion_matrix.png` for detailed metrics including:
>
> - Accuracy
> - Precision
> - Recall
> - F1-Score
>
> Always validate on your specific use case.

**Q15: What do the Grad-CAM visualizations show?**

> Grad-CAM (Gradient-weighted Class Activation Maps) highlights regions of the X-ray that influenced the model's decision. See `Figures/Grad-Cam Overlay.png` for examples.

**Q16: Can I use my own pre-trained model?**

> Yes! Replace the model in `model_export/` and update `load_hybrid_model()` to match your model's architecture.

---

### Troubleshooting

**Q17: "ModuleNotFoundError: No module named 'torch'"**

> PyTorch isn't installed. Run:
>
> ```bash
> pip install torch torchvision
> ```

**Q18: "CUDA out of memory" error**

> Your GPU doesn't have enough VRAM. Solutions:
>
> - Use CPU: `device = torch.device("cpu")`
> - Reduce batch size in training
> - Use a smaller model

**Q19: Model predictions are inconsistent**

> Ensure:
>
> - Same image preprocessing (224×224, normalization)
> - Model is in `.eval()` mode (not training mode)
> - Consistent device (CPU or GPU)

**Q20: How do I save a trained model?**

> ```python
> torch.save(model.state_dict(), "my_model.pt")
> torch.save(config, "model_config.json")  # Save config too
> ```

---

### Development & Contributing

**Q21: Can I modify the code?**

> Absolutely! The code is open for educational and research purposes. If you make improvements, consider contributing back or sharing with the community.

**Q22: How do I use this in my own project?**

> 1. Clone or download Lumora
> 2. Install dependencies: `uv sync` or `pip install -r requirements.txt`
> 3. Import utilities:
>    ```python
>    from model_export.model_utils import load_hybrid_model
>    ```
> 4. Integrate into your application

**Q23: Is there a REST API?**

> Not yet. To add one, you could create:
>
> ```python
> # api.py
> from flask import Flask, request, jsonify
> from model_export.model_utils import load_hybrid_model
>
> app = Flask(__name__)
> model, config, device = load_hybrid_model("./model_export")
>
> @app.route("/predict", methods=["POST"])
> def predict():
>     image = request.files["image"]
>     # Process and return predictions
> ```

---

### Performance & Optimization

**Q24: How can I speed up inference?**

> - Use GPU (NVIDIA CUDA)
> - Use model quantization/compression
> - Use ONNX export for optimized inference
> - Batch multiple images together

**Q25: How can I improve model accuracy?**

> - Use more training data
> - Data augmentation (rotations, flips, brightness adjustments)
> - Hyperparameter tuning (learning rate, batch size)
> - Ensemble multiple models

---

## 📖 Further Reading

1. **Research Paper**: See `Papers/` directory for the full academic paper
2. **Dataset Documentation**: https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease
3. **PyTorch Docs**: https://pytorch.org/docs/
4. **Medical AI Ethics**: Consider bias, privacy, and clinical validation

---

## ⚠️ Important Notes

### Medical Disclaimer

- ⚠️ **Not for clinical use**: This is a research project, not FDA-approved medical software
- ⚠️ **Consult professionals**: Always involve qualified radiologists for medical decisions
- ⚠️ **Data privacy**: Handle medical images according to HIPAA/GDPR regulations

### Best Practices

- Always validate model outputs with domain experts
- Use appropriate data privacy controls
- Document model limitations and biases
- Keep models updated as new data becomes available

---

## 🤝 Support & Community

- 📧 For issues: Check the project repository
- 💬 Share improvements and feedback
- 📚 Learn from the research paper and code comments

---

## 📝 License

This project is provided for educational and research purposes. Check the repository for specific license details.

---

**Last Updated**: May 2026  
**Project Version**: 0.1.0  
**Python**: 3.12+  
**Status**: Active Development

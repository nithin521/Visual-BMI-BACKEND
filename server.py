from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import timm
import torch.nn as nn

# -----------------------------
# Flask Setup
# -----------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Model
# -----------------------------
class ViTBMI(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0
        )
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

model = ViTBMI().to(DEVICE)
model.load_state_dict(torch.load("bmi_best_model.pt", map_location=DEVICE))
model.eval()

print("✅ Model loaded")

# -----------------------------
# Image Preprocessing
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# BMI Category Mapping
# -----------------------------
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# -----------------------------
# API Endpoint
# -----------------------------
@app.route("/predict-image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        bmi = model(image).item()

    return jsonify({
        "bmi": round(bmi, 2),
        "category": bmi_category(bmi)
    })

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


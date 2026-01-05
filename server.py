# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import io
# import timm
# import torch.nn as nn

# # -----------------------------
# # Flask Setup
# # -----------------------------
# app = Flask(__name__)
# CORS(app)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # -----------------------------
# # Load Model
# # -----------------------------
# class ViTBMI(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = timm.create_model(
#             "vit_base_patch16_224",
#             pretrained=False,
#             num_classes=0
#         )
#         self.head = nn.Sequential(
#             nn.Linear(768, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, x):
#         return self.head(self.backbone(x))

# model = ViTBMI().to(DEVICE)
# model.load_state_dict(torch.load("bmi_best_model.pt", map_location=DEVICE))
# model.eval()

# print("✅ Model loaded")

# # -----------------------------
# # Image Preprocessing
# # -----------------------------
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# # -----------------------------
# # BMI Category Mapping
# # -----------------------------
# def bmi_category(bmi):
#     if bmi < 18.5:
#         return "Underweight"
#     elif bmi < 25:
#         return "Normal"
#     elif bmi < 30:
#         return "Overweight"
#     else:
#         return "Obese"

# # -----------------------------
# # API Endpoint
# # -----------------------------
# @app.route("/predict-image", methods=["POST"])
# def predict_image():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     image_file = request.files["image"]
#     image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
#     image = preprocess(image).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         bmi = model(image).item()

#     return jsonify({
#         "bmi": round(bmi, 2),
#         "category": bmi_category(bmi)
#     })

# # -----------------------------
# # Run Server
# # -----------------------------
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)


import torch
import timm
import numpy as np
from PIL import Image
import gradio as gr
import torchvision.transforms as transforms

# ------------------------
# Load model
# ------------------------
DEVICE = "cpu"

class ViTBMI(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

model = ViTBMI()
model.load_state_dict(torch.load("bmi_best_model.pt", map_location="cpu"))
model.eval()

# ------------------------
# Preprocessing
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def predict_bmi(image: Image.Image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        bmi = model(image).item()

    category = (
        "Underweight" if bmi < 18.5 else
        "Normal" if bmi < 25 else
        "Overweight" if bmi < 30 else
        "Obese"
    )

    return {
        "bmi": round(bmi, 1),
        "category": category
    }

# ------------------------
# Gradio API
# ------------------------
app = gr.Interface(
    fn=predict_bmi,
    inputs=gr.Image(type="pil"),
    outputs="json",
    title="Visual BMI Prediction API",
    description="Upload a full-body image to estimate BMI"
)

app.launch()

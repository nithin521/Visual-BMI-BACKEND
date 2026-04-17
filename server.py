from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import torch
import timm
import numpy as np
from PIL import Image
import io
import torchvision.transforms as transforms
import cv2
import time

app = Flask(__name__)
CORS(app)
# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# -----------------------
# Load model
# -----------------------
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

# -----------------------
# Preprocessing
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Visual BMI backend running"})
@app.route("/predict-image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()

    # Convert to OpenCV format
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_cv is None:
        return jsonify({"error": "Invalid image"}), 400

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 🔍 Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # ❌ No face
    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400

    # ❌ Multiple faces (optional but recommended)
    if len(faces) > 1:
        return jsonify({"error": "Multiple faces detected"}), 400

    # ✅ Continue with your model
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        bmi = model(image).item() - 5

    category = (
        "Underweight" if bmi < 18.5 else
        "Normal" if bmi < 25 else
        "Overweight" if bmi < 30 else
        "Obese"
    )

    return jsonify({
        "bmi": round(bmi, 1),
        "category": category
    })

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return f"{distance:.1f} km"



def fetch_overpass(url, query):
    """Retry Overpass API safely"""
    for _ in range(2):  # 🔁 retry twice
        try:
            response = requests.get(url, params={"data": query}, timeout=10)

            if response.status_code == 200:
                return response.json()

        except Exception as e:
            print("Retrying Overpass...", str(e))

        time.sleep(1)  # small delay before retry

    return None


# @app.route("/nearby-places", methods=["GET"])
# def nearby_places():
#     try:
#         lat = request.args.get("lat")
#         lng = request.args.get("lng")
#         place_type = request.args.get("type", "gym")

#         if not lat or not lng:
#             return jsonify({"error": "Location required"}), 400

#         lat = float(lat)
#         lng = float(lng)

#         # 🔹 Map types
#         osm_tags = {
#             "gym": "leisure=fitness_centre",
#             "hospital": "amenity=hospital"
#         }

#         tag = osm_tags.get(place_type, "leisure=fitness_centre")

#         # 🔹 Smaller radius → more stable
#         query = f"""
#         [out:json][timeout:20];
#         node[{tag}](around:5000,{lat},{lng});
#         out;
#         """

#         url = "https://overpass-api.de/api/interpreter"

#         # 🔥 Use retry function
#         data = fetch_overpass(url, query)

#         # ❌ If API completely fails → DON'T crash
#         if not data:
#             return jsonify({"places": []})  # ✅ no error, safe fallback

#         elements = data.get("elements", [])

#         places = []
#         print("Fetched elements:", len(elements))
#         for element in elements:
#             place_lat = element.get("lat")
#             place_lng = element.get("lon")

#             if place_lat is None or place_lng is None:
#                 continue

#             places.append({
#                 "name": element.get("tags", {}).get("name", "Unnamed Place"),
#                 "address": element.get("tags", {}).get("addr:full", "Address not available"),
#                 "lat": place_lat,
#                 "lng": place_lng,
#                 "distance": calculate_distance(lat, lng, place_lat, place_lng)
#             })

#         # 🔥 Sort by nearest
#         places = sorted(
#             places,
#             key=lambda x: float(x["distance"].split()[0])
#         )[:5]

#         return jsonify({"places": places})

#     except Exception as e:
#         print("ERROR:", str(e))
#         return jsonify({"places": []})  # ✅ never break frontend
    



GEOAPIFY_API_KEY = "d0f4299df9714247a1c16f7742d3c71d"

@app.route("/nearby-places")
def nearby_places():
    lat = request.args.get("lat")
    lng = request.args.get("lng")
    place_type = request.args.get("type", "gym")

    # Map your types → Geoapify categories
    category_map = {
        "gym": "sport.fitness",
        "hospital": "healthcare.hospital"
    }

    category = category_map.get(place_type, "sport.fitness")

    url = f"https://api.geoapify.com/v2/places?categories={category}&filter=circle:{lng},{lat},3000&limit=10&apiKey={GEOAPIFY_API_KEY}"

    res = requests.get(url)
    data = res.json()

    places = []

    for place in data.get("features", []):
        props = place.get("properties", {})

        places.append({
            "name": props.get("name", "Unnamed Place"),
            "address": props.get("formatted", "Address not available"),
            "lat": props.get("lat"),
            "lng": props.get("lon"),

            # ✅ Geoapify place id
            "place_id": props.get("place_id"),

            # optional
            "distance": props.get("distance"),
            "phone": props.get("contact", {}).get("phone"),
            "website": props.get("website"),
            "opening_hours": props.get("opening_hours"),
            "rating": props.get("rating"),
            "city": props.get("city"),
            "state": props.get("state"),
            "postcode": props.get("postcode")
        })

    return {"places": places}
# -----------------------
# REQUIRED FOR HF DOCKER
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

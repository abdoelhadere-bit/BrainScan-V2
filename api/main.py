
from pathlib import Path
import json
import base64
from io import BytesIO

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
from ultralytics import YOLO


# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "final"

CLASSIFIER_PATH = MODEL_DIR / "brain_tumor_classifier.keras"
DETECTOR_PATH = MODEL_DIR / "best.pt"
CLASSES_PATH = MODEL_DIR / "class_names.json"


# ==================================================
# 2. LOAD MODELS
# ==================================================

with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    CLASS_NAMES = json.load(file)

classifier = load_model(
    CLASSIFIER_PATH,
    compile=False
)

detector = YOLO(str(DETECTOR_PATH))


# ==================================================
# 3. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="BrainScan V2 API",
    description="Brain MRI classification and tumor detection API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "BrainScan V2 API is running",
        "models": ["InceptionV3", "YOLOv8"]
    }


# ==================================================
# 4. IMAGE VALIDATION
# ==================================================

async def read_image(file: UploadFile):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    try:
        contents = await file.read()

        image = Image.open(BytesIO(contents))
        image = image.convert("RGB")

        return image

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )


# ==================================================
# 5. CLASSIFICATION - INCEPTIONV3
# ==================================================

@app.post("/predict/classify")
async def classify_image(file: UploadFile = File(...)):

    image = await read_image(file)

    # Resize to the model's input size
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Apply the SAME normalization used during training
    image_array = image_array / 127.5 - 1.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = classifier.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(probabilities[predicted_index])

    return {
        "task": "classification",
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {
            name: round(float(probability), 4)
            for name, probability in zip(CLASS_NAMES, probabilities)
        }
    }


# ==================================================
# 6. DETECTION - YOLOV8
# ==================================================

@app.post("/predict/detect")
async def detect_tumor(file: UploadFile = File(...)):

    image = await read_image(file)

    # Run YOLOv8 inference
    results = detector.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    detections = []

    # Extract bounding boxes
    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            coordinates = box.xyxy[0].tolist()

            detections.append({
                "class": detector.names[class_id],
                "confidence": round(confidence, 4),
                "bounding_box": {
                    "x1": round(coordinates[0], 2),
                    "y1": round(coordinates[1], 2),
                    "x2": round(coordinates[2], 2),
                    "y2": round(coordinates[3], 2)
                }
            })

    # Generate annotated image
    annotated = result.plot()

    # YOLO plot returns a BGR image
    annotated_rgb = annotated[:, :, ::-1]

    output_image = Image.fromarray(annotated_rgb)

    buffer = BytesIO()

    output_image.save(buffer, format="JPEG")

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "task": "detection",
        "number_of_detections": len(detections),
        "detections": detections,
        "annotated_image": image_base64
    }

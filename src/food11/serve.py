import io
import os

import mlflow
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5001",
)

MODEL_URI = "models:/food11@champion"

CATEGORIES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


mlflow.set_tracking_uri(TRACKING_URI)

# The model is loaded once when the API starts.
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(title="Food-11 Classification API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    input_array = TRANSFORM(image).unsqueeze(0).numpy()
    output = model.predict(input_array)

    logits = torch.as_tensor(output)

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    probabilities = torch.softmax(logits[0], dim=0)
    confidence, predicted_index = probabilities.max(dim=0)

    return {
        "category": CATEGORIES[predicted_index.item()],
        "confidence": confidence.item(),
    }

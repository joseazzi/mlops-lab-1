import io
import os
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

import mlflow
from src.food11.data import CLASSES

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5001",
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/food11@champion",
)

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(title="Food-11 Classification API")
model = None


@app.on_event("startup")
def load_model():
    """Load the champion model when the API starts."""
    global model

    if os.environ.get("SKIP_MODEL"):
        return

    model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File()],
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The model is not loaded.",
        )

    try:
        contents = await file.read()
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")
    except UnidentifiedImageError as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from error

    input_array = TRANSFORM(image).unsqueeze(0).numpy()
    output = model.predict(input_array)

    logits = torch.as_tensor(output)

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    probabilities = torch.softmax(logits[0], dim=0)
    confidence, predicted_index = probabilities.max(dim=0)

    return {
        "category": CLASSES[predicted_index.item()],
        "confidence": confidence.item(),
    }

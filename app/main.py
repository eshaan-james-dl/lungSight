"""
FastAPI service for the chest X-ray pathology model.

Run locally with:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs in a browser to test it interactively.
"""
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import onnxruntime as ort

from gradcam import load_model, generate_gradcam
from explain import generate_explanation


# ---- Config (must match training / export exactly) ----
ALL_LABELS = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
IMG_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ONNX_PATH = os.getenv("ONNX_PATH", "model.onnx")


# ---- App + model loaded once at startup ----
app = FastAPI(title="Chest X-ray Pathology Detector")

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:3000"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}


def validate_image_upload(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_IMAGE_TYPE",
                "message": "Only JPEG and PNG images are supported.",
            },
        )


def validate_image_bytes(image_bytes: bytes) -> bytes:
    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "IMAGE_TOO_LARGE",
                "message": "Image must be 10 MB or smaller.",
            },
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_IMAGE",
                "message": "Uploaded file is not a valid JPEG or PNG image.",
            },
        ) from exc

    return image_bytes

session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Separate PyTorch model, loaded once, only for Grad-CAM (ONNX Runtime can't do backward passes)
WEIGHTS_PATH = os.getenv(
    "WEIGHTS_PATH",
    "best_hybrid_model_5cls.pth"
)
torch_model = load_model(WEIGHTS_PATH)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def preprocess(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)          # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)     # add batch dimension
    return arr.astype(np.float32)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    validate_image_upload(file)
    image_bytes = validate_image_bytes(await file.read())

    try:
        input_tensor = preprocess(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "IMAGE_PREPROCESSING_FAILED",
                "message": "Could not preprocess the uploaded image.",
            },
        ) from exc

    try:
        logits = session.run(None, {input_name: input_tensor})[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "MODEL_INFERENCE_FAILED",
                "message": "Model inference failed.",
            },
        ) from exc

    probs = sigmoid(logits)[0]

    return {
        "predictions": {
            label: round(float(prob), 4)
            for label, prob in zip(ALL_LABELS, probs)
        }
    }


def resolve_class_idx(disease: str) -> int:
    """Case-insensitive lookup of a disease name against ALL_LABELS, or None if not given."""
    if disease is None:
        return None
    matches = [i for i, label in enumerate(ALL_LABELS) if label.lower() == disease.lower()]
    if not matches:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown disease '{disease}'. Choose one of: {ALL_LABELS}",
        )
    return matches[0]


async def run_gradcam(file: UploadFile, disease: str):
    """Shared logic between /gradcam and /explain."""
    validate_image_upload(file)
    class_idx = resolve_class_idx(disease)
    image_bytes = validate_image_bytes(await file.read())

    try:
        return generate_gradcam(torch_model, image_bytes, class_idx=class_idx)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "GRADCAM_FAILED",
                "message": "Could not generate the Grad-CAM heatmap.",
            },
        ) from exc


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...), disease: str = None, explain: bool = False):
    overlay_png, label, prob = await run_gradcam(file, disease)

    if not explain:
        # Default behavior, unchanged from before: just the heatmap image
        return StreamingResponse(
            io.BytesIO(overlay_png),
            media_type="image/png",
            headers={"X-Predicted-Label": label, "X-Predicted-Prob": str(round(prob, 4))},
        )

    # explain=True: also call the vision LLM and return JSON with image + explanation
    try:
        explanation_text = generate_explanation(overlay_png, label, prob)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Explanation generation failed: {e}")

    return {
        "label": label,
        "probability": round(prob, 4),
        "explanation": explanation_text,
        "heatmap_image_base64": base64.b64encode(overlay_png).decode("utf-8"),
    }
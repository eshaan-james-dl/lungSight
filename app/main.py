"""
FastAPI service for the Lungsight chest X-ray pathology model.

Models are downloaded automatically from Hugging Face at startup.

Run locally with:
    uvicorn app.main:app --reload

Then open:
    http://127.0.0.1:8000/docs
"""

import base64
import io
import os
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download

from gradcam import load_model, generate_gradcam
from explain import generate_explanation


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

ALL_LABELS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

IMG_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HF_REPO_ID = os.getenv("HF_REPO_ID", "eshaanjamesdl/lungsight")
HF_REVISION = os.getenv("HF_REVISION", "main")

ONNX_FILENAME = "model.onnx"
ONNX_DATA_FILENAME = "model.onnx.data"
WEIGHTS_FILENAME = "best_hybrid_model_5cls.pth"

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
}


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------

app = FastAPI(title="Chest X-ray Pathology Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# MODEL DOWNLOAD
# ---------------------------------------------------------

def download_models():
    """Download all model artifacts from Hugging Face into one local snapshot."""
    try:
        snapshot_path = snapshot_download(
            repo_id=HF_REPO_ID,
            revision=HF_REVISION,
            allow_patterns=[
                ONNX_FILENAME,
                ONNX_DATA_FILENAME,
                WEIGHTS_FILENAME,
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download models from Hugging Face repository "
            f"'{HF_REPO_ID}': {exc}"
        ) from exc

    onnx_path = os.path.join(snapshot_path, ONNX_FILENAME)
    onnx_data_path = os.path.join(snapshot_path, ONNX_DATA_FILENAME)
    weights_path = os.path.join(snapshot_path, WEIGHTS_FILENAME)

    missing = [
        path
        for path in (onnx_path, onnx_data_path, weights_path)
        if not os.path.isfile(path)
    ]

    if missing:
        raise RuntimeError(
            "Required model files are missing from the Hugging Face repository: "
            + ", ".join(os.path.basename(path) for path in missing)
        )

    print(f"Hugging Face model repository: {HF_REPO_ID}@{HF_REVISION}")
    print(f"ONNX model: {onnx_path}")
    print(f"ONNX external data: {onnx_data_path}")
    print(f"PyTorch weights: {weights_path}")

    return onnx_path, weights_path


# ---------------------------------------------------------
# LOAD MODELS ON STARTUP
# ---------------------------------------------------------

try:
    ONNX_PATH, WEIGHTS_PATH = download_models()

    session = ort.InferenceSession(
        ONNX_PATH,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # Separate PyTorch model for Grad-CAM.
    torch_model = load_model(WEIGHTS_PATH)

except Exception as exc:
    # Fail clearly during startup rather than producing confusing errors later.
    raise RuntimeError(f"Model initialization failed: {exc}") from exc


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def api_error(code: str, message: str):
    """Create a consistent API error payload."""
    return HTTPException(
        status_code=400,
        detail={
            "error": {
                "code": code,
                "message": message,
            }
        },
    )


def validate_image_upload(file: UploadFile):
    """Validate content type before reading the image."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise api_error(
            "UNSUPPORTED_IMAGE_TYPE",
            "Only JPEG and PNG images are supported.",
        )


def validate_image_bytes(image_bytes: bytes):
    """Validate size and ensure bytes are a real, readable image."""
    if not image_bytes:
        raise api_error("EMPTY_FILE", "The uploaded file is empty.")

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise api_error(
            "FILE_TOO_LARGE",
            "Uploaded image exceeds the 10 MB size limit.",
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
    except Exception as exc:
        raise api_error(
            "INVALID_IMAGE",
            "Uploaded file is not a valid JPEG or PNG image.",
        ) from exc


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def preprocess(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr.astype(np.float32)


def resolve_class_idx(disease: Optional[str]) -> Optional[int]:
    """Case-insensitive lookup against the five supported labels."""
    if disease is None:
        return None

    matches = [
        i for i, label in enumerate(ALL_LABELS)
        if label.lower() == disease.strip().lower()
    ]

    if not matches:
        raise api_error(
            "UNKNOWN_DISEASE",
            f"Unknown disease '{disease}'. Choose one of: {ALL_LABELS}",
        )

    return matches[0]


async def read_and_validate_image(file: UploadFile) -> bytes:
    """Validate and read an uploaded image."""
    validate_image_upload(file)
    image_bytes = await file.read()
    validate_image_bytes(image_bytes)
    return image_bytes


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "llm_configured": bool(os.getenv("LLM_API_KEY")),
    }


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await read_and_validate_image(file)

    try:
        input_tensor = preprocess(image_bytes)
        logits = session.run(None, {input_name: input_tensor})[0]
        probs = sigmoid(logits)[0]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "PREDICTION_FAILED",
                    "message": f"Prediction failed: {exc}",
                }
            },
        ) from exc

    return {
        "predictions": {
            label: round(float(prob), 4)
            for label, prob in zip(ALL_LABELS, probs)
        }
    }


# ---------------------------------------------------------
# GRAD-CAM SHARED LOGIC
# ---------------------------------------------------------

async def run_gradcam(file: UploadFile, disease: Optional[str]):
    """Validate input and run Grad-CAM."""
    image_bytes = await read_and_validate_image(file)
    class_idx = resolve_class_idx(disease)

    try:
        return generate_gradcam(
            torch_model,
            image_bytes,
            class_idx=class_idx,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "GRADCAM_FAILED",
                    "message": f"Grad-CAM generation failed: {exc}",
                }
            },
        ) from exc


# ---------------------------------------------------------
# GRAD-CAM
# ---------------------------------------------------------

@app.post("/gradcam")
async def gradcam(
    file: UploadFile = File(...),
    disease: Optional[str] = None,
    explain: bool = False,
):
    overlay_png, label, prob = await run_gradcam(file, disease)

    # -----------------------------------------------------
    # explain=false -> return actual PNG image
    # -----------------------------------------------------
    if not explain:
        return StreamingResponse(
            io.BytesIO(overlay_png),
            media_type="image/png",
            headers={
                "X-Predicted-Label": label,
                "X-Predicted-Prob": str(round(prob, 4)),
            },
        )

    # -----------------------------------------------------
    # explain=true -> return JSON containing image + explanation
    # LLM failure does NOT destroy the successful Grad-CAM.
    # -----------------------------------------------------
    explanation_text = None
    explanation_error = None

    try:
        explanation_text = generate_explanation(
            overlay_png,
            label,
            prob,
        )
    except Exception as exc:
        explanation_error = {
            "code": "LLM_EXPLANATION_FAILED",
            "message": f"Explanation generation failed: {exc}",
        }

    response = {
        "label": label,
        "probability": round(prob, 4),
        "explanation": explanation_text,
        "explanation_error": explanation_error,
        "heatmap_image_base64": base64.b64encode(overlay_png).decode("utf-8"),
    }

    return response

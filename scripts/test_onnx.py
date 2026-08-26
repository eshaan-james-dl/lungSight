"""
Quick sanity check for the exported ONNX chest X-ray model.

Usage:
    python test_onnx.py --onnx model.onnx --image sample_xray.jpg
"""

import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort

ALL_LABELS = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

IMG_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def preprocess(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype(np.float32) / 255.0          # HWC, [0,1]
    arr = (arr - MEAN) / STD                                   # normalize
    arr = arr.transpose(2, 0, 1)                                # HWC -> CHW
    arr = np.expand_dims(arr, axis=0).astype(np.float32)        # add batch dim
    return arr


def run(onnx_path: str, image_path: str):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    input_tensor = preprocess(image_path)
    logits = session.run(None, {input_name: input_tensor})[0]  # shape (1, 5)
    probs = sigmoid(logits)[0]

    print(f"\nPredictions for {image_path}:\n")
    for label, prob in sorted(zip(ALL_LABELS, probs), key=lambda x: -x[1]):
        bar = "#" * int(prob * 30)
        print(f"  {label:<20s} {prob:.3f}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="model.onnx")
    parser.add_argument("--image", type=str, required=True,
                         help="Path to a test chest X-ray image (jpg/png)")
    args = parser.parse_args()

    run(args.onnx, args.image)

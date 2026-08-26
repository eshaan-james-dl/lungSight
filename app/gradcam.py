"""
Grad-CAM for SwinConvNeXtHybrid.

Unlike the ONNX path (fast, forward-only), this loads the original PyTorch
model and hooks into the ConvNeXt branch's spatial feature map to compute
gradients — which ONNX Runtime cannot do.
"""

import io
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image

ALL_LABELS = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
IMG_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SwinConvNeXtHybrid(nn.Module):
    """Same architecture as export_onnx.py — needed here in PyTorch form for autograd."""
    def __init__(self, n_classes, pretrained=False):
        super().__init__()
        self.convnext = timm.create_model('convnext_tiny', pretrained=pretrained, features_only=True)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, features_only=True)
        self.fusion_dim = 391
        self.conv_pool = nn.AdaptiveAvgPool2d(1)
        self.swin_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.fusion_dim, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        conv_feats = self.convnext(x)[2]
        swin_feats = self.swin(x)[-1]
        conv_vec = self.conv_pool(conv_feats).flatten(1)
        swin_vec = self.swin_pool(swin_feats).flatten(1)
        fused = torch.cat([conv_vec, swin_vec], dim=1)
        return self.head(fused)


class GradCAM:
    """Hooks the ConvNeXt branch to capture its feature map and gradients."""

    def __init__(self, model: SwinConvNeXtHybrid):
        self.model = model
        self.activations = None
        # Forward hook: fires every time self.convnext runs, captures the
        # exact feature map (index 2) the model itself uses, and marks it
        # to keep its gradient after backward() (intermediate tensors
        # don't retain grad by default in PyTorch).
        self.model.convnext.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        feat = output[2]
        if feat.requires_grad:
            feat.retain_grad()
        self.activations = feat

    def generate(self, input_tensor: torch.Tensor, class_idx: int):
        self.model.zero_grad()
        logits = self.model(input_tensor)          # forward pass, hook fires here
        score = logits[0, class_idx]
        score.backward()                             # backward pass, gradients computed here

        gradients = self.activations.grad             # (1, C, H, W)
        activations = self.activations.detach()        # (1, C, H, W)

        # Global-average-pool each channel's gradient into one importance weight per channel
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of channels, then discard negative influence (ReLU)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0-1 for visualization
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, logits.detach()


def load_model(weights_path: str, device: str = "cpu") -> SwinConvNeXtHybrid:
    model = SwinConvNeXtHybrid(n_classes=len(ALL_LABELS), pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


def overlay_heatmap(cam: np.ndarray, original_image_bytes: bytes, alpha: float = 0.45) -> bytes:
    """Resizes the small CAM grid up to image size and blends it over the original X-ray."""
    base = Image.open(io.BytesIO(original_image_bytes)).convert("RGB").resize(IMG_SIZE)

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(IMG_SIZE, resample=Image.BILINEAR)
    cam_np = np.array(cam_img).astype(np.float32) / 255.0

    # Simple red-hot colormap: red channel = intensity, no external colormap dependency needed
    heatmap = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
    heatmap[..., 0] = np.uint8(255 * cam_np)                       # red channel
    heatmap[..., 1] = np.uint8(255 * np.clip(cam_np - 0.5, 0, 1) * 2)  # some green at high intensity -> yellow-hot
    heatmap_img = Image.fromarray(heatmap)

    blended = Image.blend(base, heatmap_img, alpha=alpha)

    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    return buf.getvalue()


def generate_gradcam(model: SwinConvNeXtHybrid, image_bytes: bytes, class_idx: int = None):
    """
    Runs Grad-CAM for one image. If class_idx is None, uses whichever class
    the model scored highest (the most natural default for a demo).
    Returns: (overlay_png_bytes, predicted_label, probability)
    """
    input_tensor = preprocess(image_bytes)
    cam_engine = GradCAM(model)

    if class_idx is None:
        probe_logits = model(input_tensor)
        class_idx = int(torch.argmax(probe_logits, dim=1).item())

    cam, logits = cam_engine.generate(input_tensor, class_idx)
    probs = torch.sigmoid(logits)[0]

    overlay_png = overlay_heatmap(cam, image_bytes)
    return overlay_png, ALL_LABELS[class_idx], float(probs[class_idx])
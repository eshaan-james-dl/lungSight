"""
Export SwinConvNeXtHybrid (chest X-ray, 5-class) to ONNX for deployment.

Usage:
    python export_onnx.py --weights best_hybrid_model_5cls.pth --out model.onnx

Notes:
- Uses fusion_dim=391 (fixed, hardcoded) to exactly match the architecture
  that best_hybrid_model_5cls.pth was saved with.
- No dataset needed — only the weights file and timm.
"""

import argparse
import torch
import torch.nn as nn
import timm


ALL_LABELS = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']


class SwinConvNeXtHybrid(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super().__init__()
        # pretrained=False here: we're about to load your fine-tuned weights,
        # no need to also download/init imagenet weights first.
        self.convnext = timm.create_model('convnext_tiny', pretrained=pretrained, features_only=True)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, features_only=True)

        self.fusion_dim = 391  # matches the checkpoint's saved head shape

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


def export(weights_path: str, out_path: str, img_size: int = 224, opset: int = 17):
    device = "cpu"  # export on CPU; ONNX Runtime will handle inference device later

    model = SwinConvNeXtHybrid(n_classes=len(ALL_LABELS), pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    # Sanity check: run a forward pass before export so shape errors surface here,
    # not inside onnx export internals.
    with torch.no_grad():
        test_out = model(dummy_input)
    print(f"Forward pass OK. Output shape: {tuple(test_out.shape)} "
          f"(expected (1, {len(ALL_LABELS)}))")

    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset,
    )
    print(f"Exported ONNX model to: {out_path}")
    print(f"Class order (index -> label): {list(enumerate(ALL_LABELS))}")
    print("Remember: outputs are raw logits — apply sigmoid() to get per-class probabilities "
          "(this is a multi-label model, not softmax).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best_hybrid_model_5cls.pth",
                         help="Path to the .pth state_dict file")
    parser.add_argument("--out", type=str, default="model.onnx",
                         help="Output path for the ONNX file")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export(args.weights, args.out, args.img_size, args.opset)

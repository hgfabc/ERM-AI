"""Per-stream Grad-CAM for ResNet checkpoints, over the actual model crop."""
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .data import CaseDataset
from .models import load_checkpoint, model_from_checkpoint


def _device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _target_layers(model):
    networks = list(model.backbones) if hasattr(model, "backbones") else [model.network]
    if any(not hasattr(network, "layer4") for network in networks):
        raise ValueError("Grad-CAM export currently supports ResNet models with layer4.")
    return [network.layer4[-1] for network in networks]


def export_gradcam(checkpoint_path, samples, output_dir, device="auto", max_samples=8):
    if max_samples < 1:
        raise ValueError("max_samples must be positive.")
    checkpoint = load_checkpoint(checkpoint_path)
    device = _device(device)
    model = model_from_checkpoint(checkpoint, device)
    streams = list(checkpoint["streams"])
    dataset = CaseDataset(samples[:max_samples], streams, checkpoint["preprocessing"], training=False)
    layers = _target_layers(model)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Grad-CAM output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    activations, gradients, handles = {}, {}, []

    def capture(index):
        def hook(module, inputs, output):
            activations[index] = output
            output.register_hook(lambda gradient: gradients.__setitem__(index, gradient))
        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(capture(index)))
    entries = []
    try:
        for case_index in range(len(dataset)):
            case = dataset[case_index]
            activations.clear()
            gradients.clear()
            images = case["images"].unsqueeze(0).to(device).requires_grad_(True)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = model(images)
                predicted = int(logits.argmax(dim=1).item())
                logits[0, predicted].backward()
            for stream_index, stream in enumerate(streams):
                weights = gradients[stream_index].mean(dim=(2, 3), keepdim=True)
                heat = F.relu((weights * activations[stream_index]).sum(dim=1, keepdim=True))
                heat = F.interpolate(heat, size=images.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                heat = heat.detach().cpu().numpy()
                heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-12)
                rgb = case["images"][stream_index].permute(1, 2, 0).numpy()
                prep = checkpoint["preprocessing"]
                rgb = np.clip(rgb * np.asarray(prep["std"]) + np.asarray(prep["mean"]), 0, 1)
                colors = np.stack((heat, 0.35 * (1 - np.abs(2 * heat - 1)), 1 - heat), axis=-1)
                overlay = np.clip(0.65 * rgb + 0.35 * colors, 0, 1)
                safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case["sample_id"])[:100]
                safe_stream = re.sub(r"[^A-Za-z0-9_.-]+", "_", stream)[:60]
                filename = f"{case_index:04d}_s{stream_index}_{safe_id}_{safe_stream}.png"
                Image.fromarray((overlay * 255).astype(np.uint8)).save(output_dir / filename)
                entries.append({"sample_id": case["sample_id"], "stream": stream,
                                "pred_label": predicted, "file": filename,
                                "target_layer": "layer4[-1]", "overlay_space": "model_center_crop"})
    finally:
        for handle in handles:
            handle.remove()
    (output_dir / "index.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return entries

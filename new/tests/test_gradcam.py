"""Exercise real per-stream gradients, including frozen backbones."""
import json
import numpy as np
from PIL import Image
import pytest
import torch

from multistream.gradcam import export_gradcam
from multistream.models import build_model


@pytest.mark.parametrize("streams", [["gray_inner"], ["color_inner", "color_outer"]])
def test_gradcam_exports_each_frozen_stream(tmp_path, streams):
    torch.set_num_threads(1)
    config = {"name": "resnet18", "pretrained": False, "hidden_dim": 8,
              "freeze_backbones": True, "initialization": {}}
    preprocessing = {"image_size": 32, "resize_size": 36,
                     "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    model = build_model(config, streams, 2)
    checkpoint = {"format_version": 1, "state_dict": model.state_dict(),
        "model_config": config, "streams": streams, "class_to_idx": {"bad": 0, "good": 1},
        "preprocessing": preprocessing, "epoch": 1, "fold": 0,
        "manifest_fingerprint": "a" * 64, "split_fingerprint": "b" * 64,
        "train_patient_ids": ["TRAIN"], "validation_patient_ids": ["VAL"],
        "test_patient_ids": ["TEST"]}
    checkpoint_path = tmp_path / "model.pth"
    torch.save(checkpoint, checkpoint_path)
    images = {}
    for index, stream in enumerate(streams):
        path = tmp_path / f"stream{index}.png"
        Image.fromarray(np.random.default_rng(index).integers(0, 255, (44, 48, 3), dtype=np.uint8)).save(path)
        images[stream] = str(path)
    samples = [{"sample_id": "test/case", "patient_id": "TEST", "label": None, "images": images}]
    output = tmp_path / "cams"
    entries = export_gradcam(checkpoint_path, samples, output, device="cpu", max_samples=1)
    assert [entry["stream"] for entry in entries] == streams
    assert len({entry["file"] for entry in entries}) == len(streams)
    assert len(json.loads((output / "index.json").read_text())) == len(streams)
    for entry in entries:
        assert Image.open(output / entry["file"]).size == (32, 32)
        assert entry["overlay_space"] == "model_center_crop"

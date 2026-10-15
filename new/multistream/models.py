"""Explicit stream models and strict, metadata-backed checkpoint loading.

Every model consumes images in the order declared by ``streams`` as a tensor
with shape [batch, streams, channels, height, width]. Legacy conversions are
usable for inference, but are deliberately ineligible for CV initialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import copy
import re

import timm
import torch
from torch import nn


FORMAT_VERSION = 1
CHECKPOINT_FIELDS = {
    "format_version", "state_dict", "model_config", "streams", "class_to_idx",
    "preprocessing", "epoch", "fold", "manifest_fingerprint",
    "split_fingerprint", "train_patient_ids", "validation_patient_ids",
    "test_patient_ids",
}
CONTEXT_FIELDS = (
    "fold", "manifest_fingerprint", "split_fingerprint", "train_patient_ids",
    "validation_patient_ids", "test_patient_ids",
)
PATIENT_FIELDS = CONTEXT_FIELDS[3:]


def _streams(streams):
    if not isinstance(streams, (list, tuple)) or not 1 <= len(streams) <= 8:
        raise ValueError("streams must contain between 1 and 8 names")
    if any(not isinstance(s, str) or not s.strip() for s in streams):
        raise ValueError("stream names must be nonempty strings")
    if len(set(streams)) != len(streams):
        raise ValueError("stream names must be unique and ordered explicitly")
    return tuple(streams)


def _validate_images(images, count):
    if not isinstance(images, torch.Tensor) or images.ndim != 5:
        raise ValueError("images must be a tensor shaped [B, S, C, H, W]")
    if images.shape[1] != count:
        raise ValueError("expected %s streams, received %s" % (count, images.shape[1]))
    if images.shape[0] < 1 or images.shape[2] != 3:
        raise ValueError("images must have a nonempty batch and three RGB channels")
    if images.shape[3] < 1 or images.shape[4] < 1:
        raise ValueError("images must have nonempty spatial dimensions")


def _strict_load(module, state_dict, **kwargs):
    # PyTorch's BatchNorm backwards-compatibility loader can synthesize missing
    # num_batches_tracked buffers even with strict=True. Check the complete key
    # set first so checkpoint provenance never depends on that implicit repair.
    expected, actual = set(module.state_dict()), set(state_dict)
    missing, unexpected = expected - actual, actual - expected
    if missing or unexpected:
        raise RuntimeError("Strict checkpoint key mismatch; missing=%s; unexpected=%s" %
                           (sorted(missing), sorted(unexpected)))
    return module.load_state_dict(state_dict, strict=True, **kwargs)


class SingleStreamClassifier(nn.Module):
    """A timm classifier with a five-dimensional input contract.

The public state dict stays identical to timm's state dict. This lets a legacy
single-stream classifier be converted with a strict load and no key guessing.
    """

    def __init__(self, name, stream, num_classes, pretrained=False,
                 freeze_backbones=False):
        super().__init__()
        self.streams = (stream,)
        self.network = timm.create_model(
            name, pretrained=pretrained, num_classes=num_classes
        )
        self.freeze_backbones = bool(freeze_backbones)
        if self.freeze_backbones:
            for parameter in self.network.parameters():
                parameter.requires_grad_(False)
            for parameter in self.network.get_classifier().parameters():
                parameter.requires_grad_(True)
            self.train(self.training)

    def forward(self, images):
        _validate_images(images, 1)
        return self.network(images[:, 0])

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbones:
            self.network.eval()
            self.network.get_classifier().train(mode)
        return self

    def state_dict(self, *args, **kwargs):
        return self.network.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if not strict:
            raise ValueError("partial checkpoint loading is disabled")
        return _strict_load(self.network, state_dict, **kwargs)


class MultiStreamClassifier(nn.Module):
    """Independent timm encoders followed by the original feature fusion MLP."""

    def __init__(self, name, streams, num_classes, hidden_dim=2048,
                 pretrained=False, freeze_backbones=False):
        super().__init__()
        self.streams = _streams(streams)
        self.backbones = nn.ModuleList([
            timm.create_model(name, pretrained=pretrained, num_classes=0,
                              global_pool="avg")
            for _ in self.streams
        ])
        feature_dims = [int(backbone.num_features) for backbone in self.backbones]
        self.fusion = nn.Sequential(
            nn.Linear(sum(feature_dims), hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.freeze_backbones = bool(freeze_backbones)
        if self.freeze_backbones:
            for parameter in self.backbones.parameters():
                parameter.requires_grad_(False)
            self.train(self.training)

    def forward(self, images):
        _validate_images(images, len(self.streams))
        features = [backbone(images[:, index])
                    for index, backbone in enumerate(self.backbones)]
        if any(feature.ndim != 2 for feature in features):
            raise RuntimeError("each pooled backbone must return [B, features]")
        return self.classifier(self.fusion(torch.cat(features, dim=1)))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbones:
            self.backbones.eval()
        return self


def build_model(model_config: dict, streams: list[str], num_classes: int,
                initialize: bool = True) -> nn.Module:
    """Build a classifier; ``initialize=False`` never downloads or warm starts.

For CV warm starts, ``initialization`` maps stream names to checkpoint paths.
An ``initialization_context`` containing CONTEXT_FIELDS is mandatory. The
engine may instead build without initialization and call
``initialize_from_checkpoints`` with that context explicitly.
    """
    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a dictionary")
    ordered_streams = _streams(streams)
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least 2")
    name = model_config.get("name", "resnet50")
    if not isinstance(name, str) or not name:
        raise ValueError("model name must be a nonempty string")
    hidden_dim = model_config.get("hidden_dim", 2048)
    if type(hidden_dim) is not int or hidden_dim < 1:
        raise ValueError("hidden_dim must be a positive integer")
    initialization = model_config.get("initialization") or {}
    if initialize and initialization:
        if not isinstance(initialization, dict):
            raise ValueError("initialization must map stream names to checkpoint paths")
        if set(initialization) - set(ordered_streams):
            raise ValueError("initialization contains unknown stream names")
        context = model_config.get("initialization_context")
        if not isinstance(context, dict) or set(CONTEXT_FIELDS) - context.keys():
            raise ValueError("CV warm start requires complete initialization_context")
    # Fully warm-started encoders will be replaced, so ImageNet downloads would
    # serve no purpose. Partial initialization still honors the configured prior.
    fully_warmstarted = (initialize and isinstance(initialization, dict)
                        and bool(initialization)
                        and set(initialization) == set(ordered_streams))
    pretrained = (bool(model_config.get("pretrained", True)) and initialize
                  and not fully_warmstarted)
    frozen = bool(model_config.get("freeze_backbones", False))
    if len(ordered_streams) == 1:
        model = SingleStreamClassifier(name, ordered_streams[0], num_classes,
                                       pretrained, frozen)
    else:
        model = MultiStreamClassifier(name, ordered_streams, num_classes,
                                      hidden_dim, pretrained, frozen)
    model.model_config = copy.deepcopy(model_config)
    if initialize and initialization:
        initialize_from_checkpoints(
            model, initialization,
            model_config.get("initialization_context"),
        )
    return model


def _state_dict(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError("checkpoint state_dict must be a nonempty tensor mapping")
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
           for k, v in value.items()):
        raise ValueError("state_dict must contain only string keys and tensor values")
    return value


def _classes(value):
    if not isinstance(value, dict) or len(value) < 2:
        raise ValueError("class_to_idx must describe at least two classes")
    if any(not isinstance(k, str) or not k or type(v) is not int
           for k, v in value.items()):
        raise ValueError("class_to_idx must map nonempty names to integer indices")
    if sorted(value.values()) != list(range(len(value))):
        raise ValueError("class indices must be unique and contiguous from zero")


def _patients(value, field):
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(v, str) or not v for v in value
    ):
        raise ValueError("%s must contain patient ID strings" % field)
    if len(set(value)) != len(value):
        raise ValueError("%s contains duplicate patient IDs" % field)
    return set(value)


def validate_checkpoint(checkpoint: dict) -> dict:
    """Validate schema and partition isolation before constructing any model."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a metadata-backed dictionary")
    missing = CHECKPOINT_FIELDS - checkpoint.keys()
    if missing:
        raise ValueError("checkpoint is missing metadata: %s" % ", ".join(sorted(missing)))
    if type(checkpoint["format_version"]) is not int or checkpoint["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format_version")
    _state_dict(checkpoint["state_dict"])
    _streams(checkpoint["streams"])
    _classes(checkpoint["class_to_idx"])
    if not isinstance(checkpoint["model_config"], dict):
        raise ValueError("checkpoint model_config must be a dictionary")
    if not isinstance(checkpoint["preprocessing"], dict):
        raise ValueError("checkpoint preprocessing must be a dictionary")
    patient_sets = [_patients(checkpoint[field], field) for field in PATIENT_FIELDS]
    for left in range(3):
        for right in range(left + 1, 3):
            if patient_sets[left] & patient_sets[right]:
                raise ValueError("checkpoint patient partitions overlap: %s / %s" %
                                 (PATIENT_FIELDS[left], PATIENT_FIELDS[right]))
    if checkpoint.get("legacy_no_split_metadata", False):
        if any(checkpoint[field] is not None for field in
               ("fold", "manifest_fingerprint", "split_fingerprint")):
            raise ValueError("legacy checkpoint cannot claim fold or split metadata")
        if any(patient_sets):
            raise ValueError("legacy checkpoint cannot claim patient partitions")
    else:
        if type(checkpoint["fold"]) is not int or checkpoint["fold"] < 0:
            raise ValueError("checkpoint fold must be a nonnegative integer")
        if type(checkpoint["epoch"]) is not int or checkpoint["epoch"] < 0:
            raise ValueError("checkpoint epoch must be a nonnegative integer")
        for field in ("manifest_fingerprint", "split_fingerprint"):
            if not isinstance(checkpoint[field], str) or not checkpoint[field]:
                raise ValueError("checkpoint %s must be a nonempty string" % field)
        if not patient_sets[0] or not patient_sets[1]:
            raise ValueError("CV checkpoint must identify train and validation patients")
    return checkpoint


def _safe_torch_load(path, device="cpu"):
    try:
        return torch.load(Path(path), map_location=device, weights_only=True)
    except TypeError as error:
        if "weights_only" in str(error):
            raise RuntimeError(
                "This PyTorch version does not support weights_only=True. "
                "Use a current PyTorch version; unsafe pickle loading is disabled."
            ) from error
        raise


def load_checkpoint(path, device="cpu") -> dict:
    """Safely read and validate a versioned checkpoint. Never load raw legacy weights."""
    return validate_checkpoint(_safe_torch_load(path, device))


def model_from_checkpoint(checkpoint: dict, device="cpu") -> nn.Module:
    """Reconstruct exactly from metadata, strict-load every key, and enter eval mode."""
    validate_checkpoint(checkpoint)
    model = build_model(checkpoint["model_config"], checkpoint["streams"],
                        len(checkpoint["class_to_idx"]), initialize=False)
    _strict_load(model, checkpoint["state_dict"])
    return model.to(device).eval()


def validate_warmstart_checkpoint(checkpoint, expected_context, stream=None):
    """Require the same fold, split, manifest and exact patient partitions."""
    validate_checkpoint(checkpoint)
    if checkpoint.get("legacy_no_split_metadata", False):
        raise ValueError("legacy checkpoint has no split metadata and cannot initialize CV")
    if not isinstance(expected_context, dict):
        raise ValueError("CV warm start requires initialization_context")
    missing = set(CONTEXT_FIELDS) - expected_context.keys()
    if missing:
        raise ValueError("initialization_context is missing: %s" % ", ".join(sorted(missing)))
    if stream is not None and list(checkpoint["streams"]) != [stream]:
        raise ValueError("warm start checkpoint must be single-stream for %s" % stream)
    for field in CONTEXT_FIELDS:
        actual, expected = checkpoint[field], expected_context[field]
        if field in PATIENT_FIELDS:
            actual = _patients(actual, field)
            expected = _patients(expected, field)
        if actual != expected:
            raise ValueError("warm start %s differs from the target training split" % field)
    for field in ("class_to_idx", "preprocessing"):
        if field in expected_context and checkpoint[field] != expected_context[field]:
            raise ValueError("warm start %s differs from the target experiment" % field)
    return checkpoint


def initialize_from_checkpoints(model, initialization, expected_context):
    """Strictly initialize named streams from compatible single-stream checkpoints."""
    if not isinstance(initialization, dict) or not initialization:
        raise ValueError("initialization must map stream names to checkpoint paths")
    unknown = set(initialization) - set(model.streams)
    if unknown:
        raise ValueError("initialization contains unknown streams: %s" % ", ".join(sorted(unknown)))
    loaded = []
    for stream, path in initialization.items():
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError("initialization path must be a nonempty path")
        if "{fold}" in str(path):
            raise ValueError("engine must resolve {fold} before model initialization")
        checkpoint = load_checkpoint(path)
        validate_warmstart_checkpoint(checkpoint, expected_context, stream)
        source_name = checkpoint["model_config"].get("name", "resnet50")
        target_name = model.model_config.get("name", "resnet50")
        if source_name != target_name:
            raise ValueError("warm start architecture differs from target architecture")
        source = model_from_checkpoint(checkpoint)
        if not isinstance(source, SingleStreamClassifier):
            raise ValueError("warm start requires a single-stream classifier")
        loaded.append((stream, source, checkpoint))

    # Validate all sources before changing any target parameter.
    for stream, source, checkpoint in loaded:
        if isinstance(model, SingleStreamClassifier):
            if source.network.num_classes != model.network.num_classes:
                raise ValueError("warm start classifier size differs from target")
            _strict_load(model, checkpoint["state_dict"])
        elif isinstance(model, MultiStreamClassifier):
            source.network.reset_classifier(0, global_pool="avg")
            target = model.backbones[model.streams.index(stream)]
            _strict_load(target, source.network.state_dict())
        else:
            raise TypeError("unsupported model type for stream initialization")
    return model


def _legacy_state(source):
    raw = _safe_torch_load(source)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    return _state_dict(raw)


def convert_legacy_checkpoint(source, destination, streams, model_config,
                              class_to_idx, preprocessing):
    """Convert old weights with strict key validation; mark them inference-only.

Legacy b1..bN become backbones.0..; merger.0/2 become fusion.0/2;
fc becomes classifier. Single-stream raw timm keys remain unchanged.
No patient partitions or split provenance are invented.
    """
    ordered_streams = _streams(streams)
    _classes(class_to_idx)
    if not isinstance(preprocessing, dict):
        raise ValueError("preprocessing must be a dictionary")
    raw = _legacy_state(source)
    mapped = {}
    if len(ordered_streams) == 1:
        mapped = dict(raw)
    else:
        for key, value in raw.items():
            branch = re.fullmatch(r"b([1-8])\.(.+)", key)
            if branch:
                index = int(branch.group(1)) - 1
                if index >= len(ordered_streams):
                    raise ValueError("legacy checkpoint contains an unexpected backbone")
                new_key = "backbones.%s.%s" % (index, branch.group(2))
            elif key.startswith("merger.0.") or key.startswith("merger.2."):
                new_key = "fusion." + key[len("merger."):]
            elif key.startswith("fc."):
                new_key = "classifier." + key[len("fc."):]
            else:
                raise ValueError("unsupported legacy state key: %s" % key)
            if new_key in mapped:
                raise ValueError("legacy key mapping collided: %s" % new_key)
            mapped[new_key] = value
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "state_dict": mapped,
        "model_config": copy.deepcopy(model_config),
        "streams": list(ordered_streams),
        "class_to_idx": copy.deepcopy(class_to_idx),
        "preprocessing": copy.deepcopy(preprocessing),
        "epoch": None,
        "fold": None,
        "manifest_fingerprint": None,
        "split_fingerprint": None,
        "train_patient_ids": [],
        "validation_patient_ids": [],
        "test_patient_ids": [],
        "legacy_no_split_metadata": True,
        "legacy_source": str(Path(source)),
    }
    model_from_checkpoint(checkpoint)  # Strict reconstruction before any write.
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    return checkpoint

"""Configuration defaults, inheritance and validation in one readable place."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

STREAMS = ["_vertical", "_horizontal", "color_full", "color_inner",
           "color_outer", "gray_full", "gray_inner", "gray_outer"]

DEFAULTS = {
    "data": {"root": None, "manifest": "../artifacts/manifest.json",
             "splits": "../artifacts/splits.json", "streams": ["gray_inner"],
             "cohort_streams": STREAMS, "class_to_idx": {"bad": 0, "good": 1},
             "require_all_streams": True},
    "split": {"n_folds": 10, "val_fraction": 0.2, "seed": 42},
    "preprocessing": {"image_size": 448, "resize_size": 448,
                      "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                      "crop_scale": [0.95, 1.0], "crop_ratio": [0.95, 1.02],
                      "horizontal_flip_probability": 0.5},
    "model": {"name": "resnet50", "pretrained": True, "hidden_dim": 2048,
              "freeze_backbones": False, "initialization": {}},
    "train": {"epochs": 50, "batch_size": 8, "num_workers": 0,
              "lr": 0.0007, "weight_decay": 0.01, "optimizer": "adamw",
              "momentum": 0.9, "scheduler": "cosine", "step_size": 7,
              "gamma": 0.1, "seed": 42, "device": "auto", "amp": True,
              "patience": 10, "selection_metric": "val_loss"},
    "experiment": {"name": "single_gray_inner", "output_dir": "../runs/single_gray_inner"},
}


class ConfigError(ValueError):
    """A configuration problem that can be corrected before running a model."""


def _merge(base, update):
    merged = deepcopy(base)
    for key, value in update.items():
        if key != "class_to_idx" and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read(path, ancestors=()):
    path = Path(path).resolve()
    if path in ancestors:
        raise ConfigError("Circular 'extends' in configuration files.")
    with path.open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError("The YAML configuration must be a mapping.")
    parent = config.pop("extends", None)
    inherited = _read(path.parent / parent, ancestors + (path,)) if parent else {}
    for section, keys in (("data", ("root", "manifest", "splits")),
                          ("experiment", ("output_dir",))):
        if section not in config:
            continue
        if not isinstance(config[section], dict):
            raise ConfigError(f"'{section}' must be a mapping.")
        for key in keys:
            value = config[section].get(key)
            if value is not None:
                config[section][key] = str((path.parent / str(value)).resolve())
    if "model" in config and not isinstance(config["model"], dict):
        raise ConfigError("'model' must be a mapping.")
    initializers = config.get("model", {}).get("initialization", {})
    if not isinstance(initializers, dict):
        raise ConfigError("model.initialization must map stream names to checkpoint paths.")
    for stream, checkpoint in initializers.items():
        initializers[stream] = str((path.parent / str(checkpoint)).resolve())
    return _merge(inherited, config)


def validate_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"Unknown configuration sections: {sorted(unknown)}")
    for section, defaults in DEFAULTS.items():
        if not isinstance(config[section], dict):
            raise ConfigError(f"'{section}' must be a mapping.")
        extra = set(config[section]) - set(defaults)
        if extra:
            raise ConfigError(f"Unknown settings in {section}: {sorted(extra)}")
    data, split, prep, model, train = (config[key] for key in ("data", "split", "preprocessing", "model", "train"))
    for key in ("streams", "cohort_streams"):
        streams = data[key]
        if not isinstance(streams, list) or not streams or len(streams) != len(set(streams)) or set(streams) - set(STREAMS):
            raise ConfigError(f"data.{key} must contain distinct supported stream names.")
    if not set(data["streams"]).issubset(data["cohort_streams"]):
        raise ConfigError("Training streams must be included in data.cohort_streams.")
    labels = data["class_to_idx"]
    if not isinstance(labels, dict) or len(labels) < 2 or any(not isinstance(name, str) or not name or type(index) is not int for name, index in labels.items()) or sorted(labels.values()) != list(range(len(labels))):
        raise ConfigError("class_to_idx must contain consecutive integer labels starting at zero.")
    for section, key in (("split", "n_folds"), ("preprocessing", "image_size"),
                         ("preprocessing", "resize_size"), ("model", "hidden_dim"),
                         ("train", "epochs"), ("train", "batch_size"), ("train", "step_size")):
        value = config[section][key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigError(f"{section}.{key} must be a positive integer.")
    if split["n_folds"] < 2 or not 0 < split["val_fraction"] < 1:
        raise ConfigError("Use n_folds >= 2 and 0 < val_fraction < 1.")
    for section in ("split", "train"):
        if type(config[section]["seed"]) is not int or not 0 <= config[section]["seed"] < 2**32 - split["n_folds"]:
            raise ConfigError(f"{section}.seed must be a nonnegative 32-bit integer with room for fold offsets.")
    if prep["resize_size"] < prep["image_size"]:
        raise ConfigError("resize_size must be at least image_size (to avoid padded validation crops).")
    if len(prep["mean"]) != 3 or len(prep["std"]) != 3 or any(value <= 0 for value in prep["std"]):
        raise ConfigError("RGB normalization needs three means and three positive standard deviations.")
    for key in ("crop_scale", "crop_ratio"):
        bounds = prep[key]
        if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1] or (key == "crop_scale" and bounds[1] > 1):
            raise ConfigError(f"Invalid preprocessing.{key} bounds.")
    if not 0 <= prep["horizontal_flip_probability"] <= 1:
        raise ConfigError("horizontal_flip_probability must be between zero and one.")
    for key in ("lr", "weight_decay", "momentum", "gamma"):
        value = train[key]
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ConfigError(f"train.{key} must be a finite nonnegative number.")
    if train["lr"] <= 0 or train["gamma"] <= 0:
        raise ConfigError("lr and gamma must be greater than zero.")
    if train["optimizer"] not in ("sgd", "adam", "adamw") or train["scheduler"] not in ("none", "cosine", "step"):
        raise ConfigError("Unsupported optimizer or scheduler.")
    if train["selection_metric"] not in ("val_loss", "accuracy", "balanced_accuracy", "macro_f1", "auroc"):
        raise ConfigError("Unsupported checkpoint selection_metric.")
    for key in ("num_workers", "patience"):
        if not isinstance(train[key], int) or train[key] < 0:
            raise ConfigError(f"train.{key} must be a nonnegative integer.")
    for section, key in (("model", "pretrained"), ("model", "freeze_backbones"),
                         ("train", "amp"), ("data", "require_all_streams")):
        if not isinstance(config[section][key], bool):
            raise ConfigError(f"{section}.{key} must be true or false.")
    if set(model["initialization"]) - set(data["streams"]):
        raise ConfigError("Initializers must correspond to configured training streams.")
    return config


def load_config(path):
    path = Path(path).resolve()
    defaults = deepcopy(DEFAULTS)
    for section, keys in (("data", ("manifest", "splits")), ("experiment", ("output_dir",))):
        for key in keys:
            defaults[section][key] = str((path.parent / defaults[section][key]).resolve())
    return validate_config(_merge(defaults, _read(path)))

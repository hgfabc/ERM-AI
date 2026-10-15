"""Build a resumable, patient-free collection of every legacy model and fold."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import uuid

import yaml

from export_public_model import (
    PUBLIC_LOADER, _class_mapping, _model_card, _ordered_streams,
    _public_code_url, _public_preprocessing, _safe_model_config,
    _tensor_state, _rename_with_retry, export_public_model,
)

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
RUN_NAME = re.compile(r"resnet50_runresnet50_run_(.+)@(\d+)_fold")
LEAF_FILES = {"model.safetensors", "config.json", "README.md"}
ROOT_FILES = {"catalog.json", "modeling_multistream.py", "README.md", "requirements.txt"}


def _linklike(path):
    try:
        details = Path(path).lstat()
    except FileNotFoundError:
        return False
    return Path(path).is_symlink() or bool(getattr(details, "st_file_attributes", 0) & 0x400)


def _ordinary_location(path):
    location = Path(path).absolute()
    if any(_linklike(part) for part in (location, *location.parents)):
        raise ValueError("Linked collection/source/state locations are not allowed")
    return location.resolve()


def _validate_existing_cache(output, items):
    allowed = set(ROOT_FILES) | {"catalog.json.tmp"}
    allowed.update(item["path"] + "/" + filename for item in items for filename in LEAF_FILES)
    for path in output.rglob("*"):
        if _linklike(path):
            raise ValueError("Linked files or directories in the public model cache are not allowed")
        if path.is_file() and path.relative_to(output).as_posix() not in allowed:
            raise ValueError("The public cache contains files outside the current model plan. "
                             "Rename the model-collection cache directory to preserve it, then retry with a clean cache.")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_profiles(path):
    profile = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(profile, dict) or profile.get("collection_version") != 1:
        raise ValueError("Expected collection_version: 1 in the collection profiles")
    folds = profile.get("folds", list(range(10)))
    if (not isinstance(folds, list) or not folds or any(type(value) is not int or not 0 <= value <= 9 for value in folds)
            or len(set(folds)) != len(folds)):
        raise ValueError("Profiles require distinct zero-based folds from 0 to 9")
    _safe_model_config(profile.get("model"))
    _class_mapping(profile.get("class_to_idx"))
    if not isinstance(profile.get("conditions"), dict) or not profile["conditions"]:
        raise ValueError("Profiles must explicitly declare the experiment conditions")
    pairs = set()
    for name, condition in profile["conditions"].items():
        if not SAFE_NAME.fullmatch(name) or not isinstance(condition, dict):
            raise ValueError("Invalid public condition name")
        if not isinstance(condition.get("project"), str) or not SAFE_NAME.fullmatch(condition["project"]):
            raise ValueError("Invalid records project name")
        if not isinstance(condition.get("source_prefix"), str) or not re.fullmatch(r"[A-Za-z0-9_]+", condition["source_prefix"]):
            raise ValueError("Invalid source run prefix")
        key = (condition["project"], condition["source_prefix"])
        if key in pairs:
            raise ValueError("Two conditions map to the same legacy source")
        pairs.add(key)
        _ordered_streams(condition.get("streams"))
        effective_profile(profile, name)  # Validate explicit resize settings too.
    return profile


def effective_profile(profiles, name):
    condition = profiles["conditions"][name]
    preprocessing = copy.deepcopy(profiles["preprocessing"])
    preprocessing["antialias"] = condition["antialias"]
    result = {"model": copy.deepcopy(profiles["model"]),
              "data": {"streams": list(condition["streams"]), "class_to_idx": dict(profiles["class_to_idx"])},
              "preprocessing": preprocessing}
    _public_preprocessing(preprocessing, True)
    return result


def discover_models(records_root, profiles):
    records_root = _ordinary_location(records_root)
    if not records_root.is_dir() or records_root.is_symlink():
        raise ValueError("Records root must be an existing ordinary directory")
    mapping = {(value["project"], value["source_prefix"]): name for name, value in profiles["conditions"].items()}
    items, indexed, historical = [], set(), set()
    for checkpoint in sorted(records_root.glob("**/best_model.pth")):
        if _linklike(checkpoint) or any(_linklike(parent) for parent in checkpoint.parents if parent != records_root.parent):
            raise ValueError("Linked model sources are not permitted")
        relative = checkpoint.relative_to(records_root)
        if len(relative.parts) != 3:
            raise ValueError("Unexpected legacy checkpoint location")
        project, run, _ = relative.parts
        if not SAFE_NAME.fullmatch(project):
            raise ValueError("Unsafe public historical project name")
        match = RUN_NAME.fullmatch(run)
        name = mapping.get((project, match[1])) if match else None
        if name is not None:
            fold = int(match[2])
            if fold not in profiles["folds"] or (name, fold) in indexed:
                raise ValueError("Unexpected or duplicate condition/fold source")
            indexed.add((name, fold))
            path = f"models/{name}/fold_{fold}"
            streams = list(profiles["conditions"][name]["streams"])
        else:
            if project in historical:
                raise ValueError("Multiple unidentified models in one project require explicit profiles")
            historical.add(project)
            name, fold, streams = "historical_" + project, None, None
            path = "historical/" + project
        items.append({"checkpoint": checkpoint, "condition": name, "fold": fold, "streams": streams,
                      "project": project, "path": path, "inference_ready": fold is not None,
                      "source_bytes": checkpoint.stat().st_size})
    expected = {(name, fold) for name in profiles["conditions"] for fold in profiles["folds"]}
    if indexed != expected:
        missing = sorted(expected - indexed)
        raise ValueError("Missing expected model folds: " + str(missing))
    if not items:
        raise ValueError("No model checkpoints were found")
    return items


def public_plan(items, profiles):
    ready = [item for item in items if item["inference_ready"]]
    return {"model_count": len(items), "inference_ready_count": len(ready),
            "historical_count": len(items) - len(ready), "condition_count": len(profiles["conditions"]),
            "fold_count": len(profiles["folds"]), "source_bytes": sum(item["source_bytes"] for item in items),
            "profiles": [{"condition": name, "streams": value["streams"], "antialias": value["antialias"],
                          "notes": value.get("notes", "Operator-declared inference settings")}
                         for name, value in profiles["conditions"].items()]}


def _historical_config(item):
    return {"historical_format_version": 1, "condition": item["condition"], "inference_ready": False,
            "streams": None, "class_to_idx": None, "model_config": None, "preprocessing": None,
            "source_format": "legacy_raw_state_dict", "private_training_metadata_omitted": True,
            "reason": "Original fold/stream/preprocessing metadata is unavailable; archive tensors without guessing a deployment configuration."}


def _historical_card(config):
    return ("# Historical tensor archive\n\nCondition: " + config["condition"] + "\n\n"
            "These original tensors are preserved for research provenance. No fold, stream order, "
            "label mapping or preprocessing is asserted.\n\n" + config["reason"] + "\n\n"
            "The collection inference tool rejects this entry. Recover the original experiment "
            "settings before constructing an inference model. No patient/sample metadata is included.\n")


def _ready_metadata(profiles, name):
    effective = effective_profile(profiles, name)
    return {"public_format_version": 1, "model_config": _safe_model_config(effective["model"]),
            "streams": _ordered_streams(effective["data"]["streams"]),
            "class_to_idx": _class_mapping(effective["data"]["class_to_idx"]),
            "preprocessing": _public_preprocessing(effective["preprocessing"], True),
            "artifact_kind": "research", "source_format": "legacy_state_dict",
            "settings_provenance": "operator_supplied", "private_training_metadata_omitted": True,
            "legacy_no_split_metadata": True}


def _collection_readme(items, profiles, code_url=None):
    ready = sum(item["inference_ready"] for item in items)
    first = next(iter(profiles["conditions"]))
    metadata = _ready_metadata(profiles, first)
    preprocessing = metadata["preprocessing"]
    model = metadata["model_config"]
    architecture = ("Single-stream ResNet50 maps a global pooled 2048-dimensional feature to two classes. Fusion models concatenate one feature per stream and use Linear(N*2048, 2048), ReLU, Linear(2048, 2048), and Linear(2048, 2)."
                    if model["name"] == "resnet50" and model["hidden_dim"] == 2048 and len(metadata["class_to_idx"]) == 2
                    else "Declared backbone: " + model["name"] + "; fusion hidden dimension: " + str(model["hidden_dim"]) + ". Each config.json describes its architecture.")
    lines = ["---", "tags:", "- pytorch", "- timm", "- multistream", "- research", "---", "",
             "# ERM-AI complete research model collection", "",
             f"This collection preserves {len(items)} model files: {ready} loadable model/fold entries and {len(items)-ready} historical tensor archives.",
             "A loadable entry means that its tensors match the declared architecture; it does not establish clinical validity or independent test performance.", "",
             "## Layout", "", "```text", "catalog.json", "modeling_multistream.py", "README.md", "requirements.txt",
             "models/<condition>/fold_<0-9>/{model.safetensors,config.json,README.md}",
             "historical/<project>/{model.safetensors,config.json,README.md}", "```", "",
             "## Conditions and ordered streams", "", "| Condition | Ordered streams | Tensor resize antialias | Profile evidence / limits |", "|---|---|---|---|"]
    for name, value in profiles["conditions"].items():
        note = value.get("notes", "Operator-declared settings").replace("|", "/")
        lines.append(f"| {name} | {', '.join(value['streams'])} | {value['antialias']} | {note} |")
    lines += ["", "Each condition contains the zero-based folds declared in catalog.json. Each stream has its own ResNet50 backbone; fusion concatenates pooled features and applies an MLP classifier. Single-stream models classify the pooled feature directly.",
              "", "## Architecture and preprocessing", "",
              architecture,
              f"Input tensors have shape [batch, streams, 3, {preprocessing['image_size']}, {preprocessing['image_size']}]. Class mapping: {json.dumps(metadata['class_to_idx'])}. Softmax converts logits into class probabilities.",
              "RGB resize backend: " + preprocessing["resize_backend"] + "; resize size: " + str(preprocessing["resize_size"]) + "; bilinear interpolation and center crop. Normalization mean: " + str(preprocessing["mean"]) + "; std: " + str(preprocessing["std"]) + ". Condition-specific antialias is listed above and saved in config.json.",
              "", "## Inference", "", "Clone the code repository and run from its new/ directory:", "", "```bash",
              "python -m pip install -r requirements.txt",
              "python tools/infer_published.py --repo ntuhophai/ERM-AI --list",
              "python tools/infer_published.py --repo ntuhophai/ERM-AI --condition single_gray_inner --fold 0 --root /path/to/cases --output predictions/single_gray_inner_fold0",
              "python tools/infer_published.py --repo ntuhophai/ERM-AI --condition fusion_8 --all-folds --root /path/to/cases --output predictions/fusion_8",
              "```", "",
              "Before the model PR is merged, add --revision refs/pr/<number> to every Hub command above. After merge, the default main revision can be used. Wait for all models and catalog.json to finish uploading before inference.",
              "Only selected fold weights/configurations are downloaded. Pin --revision to a model repository commit for reproducible downloads. Predictions include per-class probabilities in the saved class index order.",
              "Case folders must provide one matching image per required stream. Both class/case/image folders and unlabeled case/image folders are supported.",
              "Use --ensemble only when you explicitly want a mean of the selected models' probabilities. This is prediction on one cohort, not out-of-fold validation.", "",
              "Historical entries are tensor-only because their original fold/stream settings were not saved. They are included for completeness but are not automatically inferable.",
              "", "## Provenance and limits", "",
              "These are legacy research weights. Earlier experiments used case-level random folds, with identified patient-overlap concerns, and selected checkpoints using validation scores. Do not label those scores as patient-independent cross-validation results.",
              "Class names bad/good are recorded labels; do not infer healthy/diseased meanings without the study definition.",
              "Profiles are explicit declarations; tensor keys/shapes cannot prove historical stream semantics or preprocessing. In particular, four-/six-stream runtime versions were not retained; their antialias choice must be checked against original records.",
              "No model license was guessed. Complete the study-specific model documentation/license before treating this as a clinical release."]
    if code_url:
        lines += ["", "Public code snapshot: [code and complete usage instructions](" + code_url + "). See INFERENCE.md and MODEL_CATALOG.md in that snapshot."]
    return "\n".join(lines) + "\n"


def _collection_model_card(metadata, item, profiles, code_url=None):
    """Describe this leaf while directing users to the shared collection entry point."""
    card = _model_card(metadata, code_url)
    start, end = card.index("## Loading"), card.index("## Preprocessing and verification")
    usage = ("## Loading this condition and fold\n\n"
             "Install the code repository requirements and run from its new/ directory. "
             "The shared loader and requirements live at the collection root, not in this model directory.\n\n"
             "```bash\npython tools/infer_published.py --repo ntuhophai/ERM-AI --condition "
             + item["condition"] + " --fold " + str(item["fold"])
             + " --root /path/to/cases --output predictions/" + item["condition"]
             + "_fold" + str(item["fold"]) + "\n```\n\n"
             "Before the model PR is merged, append --revision refs/pr/<number>; after merge the default main revision can be used.\n\n"
             "Only the selected weights/config are downloaded. Provide one image for each ordered stream. "
             "The local tool uses its trusted model definition and saves class probabilities.\n\n")
    card = card[:start] + usage + card[end:]
    card += "\nCollection condition: " + item["condition"] + f"; zero-based fold: {item['fold']}.\n"
    card += "\nDeclared preprocessing:\n\n```json\n" + json.dumps(metadata["preprocessing"], indent=2) + "\n```\n"
    card += "\nProfile note: " + profiles["conditions"][item["condition"]].get("notes", "Operator-declared settings") + "\n"
    if code_url:
        documentation = code_url.replace("/tree/", "/blob/", 1).rstrip("/") + "/INFERENCE.md"
        card += "\n[Complete download, inference and output instructions](" + documentation + ")\n"
    return card


def export_collection(records_root, profiles_path, output, code_url=None, plan_only=False):
    profiles = load_profiles(profiles_path)
    items = discover_models(records_root, profiles)
    plan = public_plan(items, profiles)
    if plan_only:
        return plan
    if output is None:
        raise ValueError("--output is required when exporting")
    code_url = _public_code_url(code_url)
    output = _ordinary_location(output)
    state_root = _ordinary_location(output.parent / (output.name + ".export-state"))
    if output.is_symlink() or state_root.is_symlink():
        raise ValueError("Linked output/state roots are not allowed")
    output.mkdir(parents=True, exist_ok=True)
    _validate_existing_cache(output, items)
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "state.json"
    if _linklike(state_path):
        raise ValueError("Linked local export state is not allowed")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"version": 1, "entries": {}}
    if state.get("version") != 1 or not isinstance(state.get("entries"), dict):
        raise ValueError("Unrecognized local export progress")
    loader_hash = hashlib.sha256(PUBLIC_LOADER.encode()).hexdigest()
    entries = []
    exported = reused = 0
    for number, item in enumerate(items, 1):
        print(f"[{number}/{len(items)}] {item['condition']} / fold {item['fold']}", flush=True)
        leaf = output.joinpath(*item["path"].split("/"))
        if leaf.is_symlink() or not leaf.resolve().is_relative_to(output):
            raise ValueError("Model output escaped its collection")
        metadata = _ready_metadata(profiles, item["condition"]) if item["inference_ready"] else _historical_config(item)
        source_hash = file_sha256(item["checkpoint"])
        identity = _digest({"source_sha256": source_hash, "metadata": metadata, "loader_sha256": loader_hash})
        previous = state["entries"].get(item["path"], {})
        valid = (leaf.is_dir() and {path.name for path in leaf.iterdir()} == LEAF_FILES
                 and not any(path.is_symlink() for path in leaf.iterdir())
                 and previous.get("identity") == identity
                 and previous.get("weights_sha256") == file_sha256(leaf / "model.safetensors")
                 and previous.get("config_sha256") == file_sha256(leaf / "config.json"))
        if not valid:
            print("  Exporting missing or changed model", flush=True)
            free = shutil.disk_usage(output).free
            if free < item["source_bytes"] * 2 + 256 * 1024 * 1024:
                raise RuntimeError("Not enough disk space for the next model; retain progress and free space before retrying")
            stage_parent = state_root / "staging" / uuid.uuid4().hex
            stage_parent.mkdir(parents=True)
            stage = stage_parent / "model"
            if item["inference_ready"]:
                effective = stage_parent / "profile.yaml"
                effective.write_text(yaml.safe_dump(effective_profile(profiles, item["condition"]), sort_keys=False), encoding="utf-8")
                exported_metadata = export_public_model(item["checkpoint"], stage, legacy_config=effective, code_url=code_url)
                if exported_metadata != metadata:
                    raise RuntimeError("Export metadata did not match the declared collection profile")
                (stage / "modeling_multistream.py").unlink()
                (stage / "requirements.txt").unlink()
            else:
                import torch
                from safetensors.torch import save_file
                raw = torch.load(item["checkpoint"], map_location="cpu", weights_only=True)
                tensors = _tensor_state(raw)
                for tensor in tensors.values():
                    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                        raise ValueError("Historical model contains nonfinite tensors")
                stage.mkdir()
                save_file({key: value.detach().cpu().contiguous().clone() for key, value in tensors.items()}, str(stage / "model.safetensors"))
                _write_json(stage / "config.json", metadata)
                (stage / "README.md").write_text(_historical_card(metadata), encoding="utf-8")
                del raw, tensors
            leaf.parent.mkdir(parents=True, exist_ok=True)
            if leaf.exists():
                backup = state_root / "backups" / uuid.uuid4().hex
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not leaf.resolve().is_relative_to(output) or not backup.resolve().is_relative_to(state_root):
                    raise RuntimeError("Unsafe model cache replacement target")
                _rename_with_retry(leaf, backup)  # Preserve an earlier public export instead of deleting it.
            _rename_with_retry(stage, leaf)
            exported += 1
        else:
            print("  Reusing verified export", flush=True)
            reused += 1
        card = (_collection_model_card(metadata, item, profiles, code_url)
                if item["inference_ready"] else _historical_card(metadata))
        (leaf / "README.md").write_text(card, encoding="utf-8")
        weights_hash, config_hash = file_sha256(leaf / "model.safetensors"), file_sha256(leaf / "config.json")
        state["entries"][item["path"]] = {"identity": identity, "weights_sha256": weights_hash, "config_sha256": config_hash}
        _write_json(state_path, state)
        entries.append({"condition": item["condition"], "fold": item["fold"], "streams": item["streams"],
                        "path": item["path"], "inference_ready": item["inference_ready"],
                        "weights_sha256": weights_hash, "config_sha256": config_hash,
                        "artifact_kind": "research", "source_format": "legacy_state_dict" if item["inference_ready"] else "legacy_raw_state_dict",
                        "settings_provenance": "operator_supplied" if item["inference_ready"] else "unavailable",
                        "weights_bytes": (leaf / "model.safetensors").stat().st_size,
                        "project": item["project"], "reason": "" if item["inference_ready"] else metadata["reason"]})
    catalog = {"catalog_version": 1, "models": entries,
               "inference_ready_count": plan["inference_ready_count"], "historical_count": plan["historical_count"],
               "model_count": len(entries), "condition_count": plan["condition_count"], "fold_count": plan["fold_count"],
               "total_weight_bytes": sum(entry["weights_bytes"] for entry in entries), "private_training_metadata_omitted": True}
    (output / "modeling_multistream.py").write_text(PUBLIC_LOADER, encoding="utf-8")
    (output / "README.md").write_text(_collection_readme(items, profiles, code_url), encoding="utf-8")
    requirements = []
    for package in ("torch", "torchvision", "timm", "safetensors", "huggingface_hub"):
        requirements.append(package + "==" + importlib.metadata.version(package).split("+", 1)[0])
    (output / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    _write_json(output / "catalog.json", catalog)  # Publish the complete local catalog last.
    return {**plan, "exported": exported, "reused": reused, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", required=True)
    parser.add_argument("--profiles", default=str(Path(__file__).resolve().parents[1] / "publish_profiles" / "legacy_collection.yaml"))
    parser.add_argument("--output")
    parser.add_argument("--code-url")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = export_collection(args.records_root, args.profiles, args.output, args.code_url, args.plan_only)
    except Exception as error:
        parser.exit(2, "Collection export stopped: " + str(error) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

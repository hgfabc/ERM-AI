"""Read selected public fold models and write local, descriptive predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


ROOT_FILES = ["catalog.json", "modeling_multistream.py"]
SCOPE = {
    "evaluation_scope": "descriptive_inference_on_supplied_cohort",
    "out_of_fold": False,
    "holdout_independence_verified": False,
    "interpretation": (
        "Predictions and metrics describe the supplied cohort. Public exports omit "
        "training-patient metadata, so independence from training cannot be verified. "
        "Selecting fold models does not make these predictions out-of-fold."
    ),
}


def _project_imports():
    project = str(Path(__file__).resolve().parents[1])
    if project not in sys.path:
        sys.path.insert(0, project)
    from multistream.data import scan_dataset
    from multistream.metrics import calculate_metrics, ordered_classes
    return scan_dataset, calculate_metrics, ordered_classes


def _trusted_exporter():
    # This is the authored local exporter, never a file downloaded from the Hub.
    path = Path(__file__).resolve().with_name("export_public_model.py")
    spec = importlib.util.spec_from_file_location("_trusted_export_for_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trusted_loader():
    return _trusted_exporter()._reference_loader()


def _validated_preprocessing(value):
    if not isinstance(value, dict):
        raise ValueError("Selected config is missing preprocessing")
    if value.get("crop", "center") != "center":
        raise ValueError("The trusted loader supports only center crop")
    if value.get("resize_backend", "pil") == "pil" and value.get("antialias", True) is False:
        raise ValueError("PIL resizing always uses antialias=True; this config cannot be followed")
    # Reuse the authored validator for sizes, RGB vectors, finite values,
    # supported interpolation/backend and the actual default PIL behavior.
    return _trusted_exporter()._public_preprocessing(value, False)


def _model_path(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or "*" in value or "?" in value or "[" in value or "]" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        raise ValueError("Catalog paths must be literal repository-relative POSIX paths")
    parts = value.split("/")
    if (value.startswith("/") or any(part in ("", ".", "..") for part in parts)
            or any(":" in part for part in parts)):
        raise ValueError("Catalog model path cannot escape its repository")
    return str(PurePosixPath(value))


def load_catalog(repository_root):
    value = json.loads((Path(repository_root) / "catalog.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or type(value.get("catalog_version")) is not int or value["catalog_version"] != 1:
        raise ValueError("Unsupported catalog_version; expected 1")
    if not isinstance(value.get("models"), list) or not value["models"]:
        raise ValueError("Catalog models must be a nonempty list")
    entries, seen = [], set()
    for source in value["models"]:
        if not isinstance(source, dict):
            raise ValueError("Every catalog entry must be a mapping")
        if not {"condition", "fold", "streams", "path", "inference_ready"}.issubset(source):
            raise ValueError("Catalog entries must explicitly declare condition, fold, streams, path and inference_ready")
        condition, fold = source.get("condition"), source.get("fold")
        if (not isinstance(condition, str) or not condition.strip()
                or any(ord(character) < 32 or ord(character) == 127 for character in condition)):
            raise ValueError("Catalog condition must be a safe, nonempty identifier")
        if fold is not None and (type(fold) is not int or fold < 0):
            raise ValueError("Catalog fold must be null or a nonnegative integer")
        if type(source.get("inference_ready")) is not bool:
            raise ValueError("Catalog inference_ready must be explicitly true or false")
        streams = source.get("streams")
        if streams is not None and (
            not isinstance(streams, list) or not 1 <= len(streams) <= 8
            or any(not isinstance(stream, str) or not stream for stream in streams)
            or len(set(streams)) != len(streams)
        ):
            raise ValueError("Catalog streams must be null or unique ordered stream names")
        identity = (condition, fold)
        if identity in seen:
            raise ValueError("Catalog contains duplicate condition/fold identities")
        seen.add(identity)
        entry = {
            "condition": condition, "fold": fold, "streams": streams,
            "path": _model_path(source.get("path")),
            "inference_ready": source["inference_ready"],
        }
        for key in ("config_sha256", "weights_sha256"):
            if key in source:
                digest = source[key]
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                    raise ValueError("Catalog " + key + " must be a SHA-256 hex digest")
                entry[key] = digest.lower()
        entries.append(entry)
    return {"catalog_version": 1, "models": entries}


def select_models(catalog, condition, folds=None, all_folds=False):
    matches = [entry for entry in catalog["models"] if entry["condition"] == condition]
    if not matches:
        raise ValueError("Condition is absent from catalog; use --list")
    if not any(entry["inference_ready"] for entry in matches):
        raise ValueError("Condition is inference_ready=false history; no profile will be guessed")
    if all_folds and folds is not None:
        raise ValueError("Choose --all-folds or explicit folds, not both")
    if all_folds:
        if any(not entry["inference_ready"] for entry in matches):
            raise ValueError("Condition contains inference_ready=false history; no profile will be guessed")
        present = {entry["fold"] for entry in matches}
        if present != set(range(10)):
            raise ValueError("--all-folds requires the complete published fold set 0 through 9")
        selected = sorted(matches, key=lambda entry: entry["fold"])
    else:
        wanted = [0] if folds is None else list(folds)
        if not wanted or len(set(wanted)) != len(wanted):
            raise ValueError("Choose at least one unique fold")
        if any(fold is not None and (type(fold) is not int or fold < 0) for fold in wanted):
            raise ValueError("Fold must be null or a nonnegative integer")
        by_fold = {entry["fold"]: entry for entry in matches}
        if any(fold not in by_fold for fold in wanted):
            raise ValueError("Requested fold is absent from this condition; use --list")
        selected = [by_fold[fold] for fold in wanted]
        if any(not entry["inference_ready"] for entry in selected):
            raise ValueError("Selected model is inference_ready=false; history profiles will not be guessed")
    if any(not entry["path"].startswith("models/") for entry in selected):
        raise ValueError("Ready models must live beneath models/ in the repository")
    return selected


def _snapshot(repo, revision, patterns):
    if snapshot_download is None:
        raise RuntimeError("Install huggingface_hub to download a repository")
    return Path(snapshot_download(
        repo_id=repo, repo_type="model", revision=revision,
        allow_patterns=list(patterns),
    ))


def _pinned_revision(snapshot, requested):
    path = Path(snapshot)
    if path.parent.name == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40}", path.name):
        return path.name.lower()
    if requested is not None and re.fullmatch(r"[0-9a-fA-F]{40}", requested):
        return requested.lower()
    raise ValueError("Cannot identify an immutable Hub snapshot; supply --revision with the full commit hash")


def repository_catalog(repo=None, local_repo=None, revision=None):
    if bool(repo) == bool(local_repo):
        raise ValueError("Choose exactly one of --repo and --local-repo")
    root = Path(local_repo).resolve() if local_repo else _snapshot(repo, revision, ROOT_FILES)
    catalog = load_catalog(root)
    source = {"repo": repo, "local_repository": bool(local_repo),
              "requested_revision": revision, "resolved_revision": None}
    if not local_repo:
        source["resolved_revision"] = _pinned_revision(root, revision)
    return root, catalog, source


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_configs(root, entries):
    root = Path(root).resolve()
    results = []
    for entry in entries:
        bundle = (root / entry["path"]).resolve()
        try:
            bundle.relative_to(root)
        except ValueError as error:
            raise ValueError("Selected model directory escapes the repository") from error
        config_path, weights_path = bundle / "config.json", bundle / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError("Selected model must have config.json and model.safetensors")
        for field, path in (("config_sha256", config_path), ("weights_sha256", weights_path)):
            if field in entry and _sha256(path) != entry[field]:
                raise ValueError("Selected model " + field + " does not match catalog")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Selected config must be a mapping")
        if type(config.get("public_format_version")) is not int or config["public_format_version"] != 1:
            raise ValueError("Selected model has an unsupported public_format_version")
        streams = config.get("streams")
        if (not isinstance(streams, list) or not 1 <= len(streams) <= 8
                or any(not isinstance(stream, str) or not stream for stream in streams)
                or len(set(streams)) != len(streams)):
            raise ValueError("Selected config must declare unique ordered streams")
        if entry["streams"] is not None and entry["streams"] != streams:
            raise ValueError("Selected config stream order differs from catalog")
        if not isinstance(config.get("model_config"), dict) or not isinstance(config.get("class_to_idx"), dict):
            raise ValueError("Selected config is missing model_config or class_to_idx")
        _trusted_exporter()._safe_model_config(config["model_config"])
        if (isinstance(config.get("preprocessing"), dict)
                and config["preprocessing"].get("resize_backend") == "tensor"
                and type(config["preprocessing"].get("antialias")) is not bool):
            raise ValueError("Tensor preprocessing requires an explicit antialias boolean")
        config["preprocessing"] = _validated_preprocessing(config.get("preprocessing"))
        results.append((entry, bundle, config))
    return results


def _device(value):
    import torch
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _predict(model, samples, streams, preprocessing, batch_size, device, loader):
    import numpy as np
    import torch
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            cases = samples[start:start + batch_size]
            tensors = [
                loader["preprocess_streams"](
                    [sample["images"][stream] for stream in streams], preprocessing,
                )[0]
                for sample in cases
            ]
            logits = model(torch.stack(tensors).to(device))
            if logits.ndim != 2 or logits.shape[0] != len(cases) or not torch.isfinite(logits).all():
                raise ValueError("Model produced invalid or nonfinite logits")
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def _metrics(samples, probabilities, classes, calculate_metrics):
    selected = [index for index, sample in enumerate(samples) if sample["label"] is not None]
    if not selected:
        return None
    return calculate_metrics(
        [samples[index]["label"] for index in selected], probabilities[selected], classes,
    )


def _fold_label(fold):
    return "none" if fold is None else str(fold)


def _output_paths(output, folds, ensemble):
    path = Path(output).resolve()
    if path.suffix.casefold() == ".csv":
        fold_paths = {
            fold: (path if len(folds) == 1 and not ensemble else
                   path.with_name(path.stem + ".fold_" + _fold_label(fold) + ".csv"))
            for fold in folds
        }
        ensemble_path = path if ensemble else None
        summary = path.with_suffix(".summary.json")
    else:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError("Choose a new or empty inference output directory")
        fold_paths = {fold: path / ("predictions_fold_" + _fold_label(fold) + ".csv")
                      for fold in folds}
        ensemble_path = path / "predictions_ensemble.csv" if ensemble else None
        summary = path / "summary.json"
    targets = list(fold_paths.values()) + [summary]
    if ensemble_path is not None:
        targets.append(ensemble_path)
    if len(set(targets)) != len(targets) or any(target.exists() for target in targets):
        raise FileExistsError("Inference output files must be distinct and must not already exist")
    return fold_paths, ensemble_path, summary


def _write_predictions(path, samples, probabilities, classes, class_order,
                       condition, fold, include_patient_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id"]
    if include_patient_id:
        fields.append("patient_id")
    fields += ["condition", "fold", "true_label", "true_class", "pred_label", "pred_class"]
    fields += ["prob_%d_%s" % (index, name) for index, name in enumerate(class_order)]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample, probability in zip(samples, probabilities):
            label, predicted = sample["label"], int(probability.argmax())
            row = {"sample_id": sample["sample_id"], "condition": condition, "fold": fold,
                   "true_label": "" if label is None else label,
                   "true_class": "" if label is None else class_order[label],
                   "pred_label": predicted, "pred_class": class_order[predicted]}
            if include_patient_id:
                row["patient_id"] = sample["patient_id"]
            row.update({"prob_%d_%s" % (index, name): float(probability[index])
                        for index, name in enumerate(class_order)})
            writer.writerow(row)


def run_inference(repo=None, local_repo=None, revision=None, condition=None,
                  folds=None, all_folds=False, root=None, output=None,
                  batch_size=8, device="auto", ensemble=False,
                  include_patient_id=False):
    """Run selected models. No training independence or OOF claim is inferred."""
    if not condition or root is None or output is None:
        raise ValueError("Inference requires --condition, --root and --output")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    repository, catalog, source = repository_catalog(repo, local_repo, revision)
    entries = select_models(catalog, condition, folds, all_folds)
    if not local_repo:
        patterns = ROOT_FILES + [
            entry["path"] + "/" + filename
            for entry in entries for filename in ("config.json", "model.safetensors")
        ]
        repository = _snapshot(repo, source["resolved_revision"], patterns)
    configurations = _selected_configs(repository, entries)
    scan_dataset, calculate_metrics, ordered_classes = _project_imports()
    if ensemble:
        expected_classes = configurations[0][2].get("class_to_idx")
        expected_streams = configurations[0][2]["streams"]
        for _, _, config in configurations[1:]:
            if config.get("class_to_idx") != expected_classes:
                raise ValueError("Ensemble requires identical class mappings across selected models")
            if config["streams"] != expected_streams:
                raise ValueError("Ensemble requires identical ordered streams across selected models")
    chosen_folds = [entry["fold"] for entry in entries]
    fold_paths, ensemble_path, summary_path = _output_paths(output, chosen_folds, ensemble)
    actual_device, loader = _device(device), _trusted_loader()
    results, arrays, cohorts = [], [], []
    for entry, bundle, config in configurations:
        classes = config.get("class_to_idx")
        class_order = ordered_classes(classes)
        manifest = scan_dataset(root, config["streams"], classes,
                                require_all_streams=True, allow_unlabeled=True)
        samples = manifest["samples"]
        model = loader["load_public_model"](bundle, device=actual_device)
        probability = _predict(model, samples, config["streams"], config["preprocessing"],
                               batch_size, actual_device, loader)
        if probability.shape[1] != len(class_order):
            raise ValueError("Model probability columns differ from selected class mapping")
        results.append({
            "fold": entry["fold"], "streams": config["streams"], "class_to_idx": classes,
            "sample_count": len(samples),
            "labeled_count": sum(sample["label"] is not None for sample in samples),
            "descriptive_metrics": _metrics(samples, probability, classes, calculate_metrics),
            "prediction_file": fold_paths[entry["fold"]].name,
            "preprocessing": {
                key: config["preprocessing"][key]
                for key in ("image_size", "resize_size", "mean", "std",
                            "resize_backend", "interpolation", "antialias", "crop")
                if key in config["preprocessing"]
            },
            "artifact_kind": (config.get("artifact_kind")
                              if config.get("artifact_kind") in ("demo", "research") else None),
            "source_format": (config.get("source_format")
                              if config.get("source_format") in
                              ("legacy_state_dict", "converted_legacy", "versioned_checkpoint")
                              else None), **SCOPE,
        })
        arrays.append(probability)
        cohorts.append((samples, classes, class_order))
        del model
    ensemble_result, mean = None, None
    if ensemble:
        import numpy as np
        reference = [(sample["sample_id"], sample["label"]) for sample in cohorts[0][0]]
        if any([(sample["sample_id"], sample["label"]) for sample in samples] != reference
               for samples, _, _ in cohorts[1:]):
            raise ValueError("Ensemble models did not predict the same samples in the same order")
        mean = np.stack(arrays).mean(axis=0)
        ensemble_result = {
            "method": "arithmetic_mean_of_selected_model_probabilities",
            "folds": chosen_folds, "sample_count": len(reference),
            "descriptive_metrics": _metrics(cohorts[0][0], mean, cohorts[0][1], calculate_metrics),
            "prediction_file": ensemble_path.name, **SCOPE,
        }
    summary = {
        "condition": condition, "folds": chosen_folds, "source": source,
        "device": str(actual_device), "batch_size": batch_size,
        "loader_source": "trusted_local_export_template",
        "downloaded_repository_code_executed": False,
        "patient_ids_in_local_csv": bool(include_patient_id),
        "models": results, "ensemble": ensemble_result, **SCOPE,
    }
    # Inference and validation complete before creating any output files.
    for entry, probability, (samples, classes, class_order) in zip(entries, arrays, cohorts):
        _write_predictions(fold_paths[entry["fold"]], samples, probability, classes,
                           class_order, condition, entry["fold"], include_patient_id)
    if ensemble:
        _write_predictions(ensemble_path, cohorts[0][0], mean, cohorts[0][1], cohorts[0][2],
                           condition, "ensemble", include_patient_id)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return summary


def _fold_arg(value):
    if value.casefold() in ("none", "null"):
        return None
    try:
        fold = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("fold must be an integer or none") from error
    if fold < 0:
        raise argparse.ArgumentTypeError("fold must be nonnegative")
    return fold


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="Hugging Face model repository, e.g. ntuhophai/ERM-AI")
    source.add_argument("--local-repo", help="Offline checkout/snapshot of the same catalog layout")
    parser.add_argument("--revision", help="Pin a Hugging Face commit revision")
    parser.add_argument("--condition")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--fold", type=_fold_arg, default=argparse.SUPPRESS)
    selection.add_argument("--folds", nargs="+", type=_fold_arg)
    selection.add_argument("--all-folds", action="store_true", help="Require published folds 0 through 9")
    parser.add_argument("--root", help="Labeled class/case folders or flat unlabeled case folders")
    parser.add_argument("--output", "--outputCSV", "--output-csv", dest="output",
                        help="New output folder or CSV filename/prefix")
    parser.add_argument("--list", action="store_true", help="List catalog entries without downloading weights")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ensemble", action="store_true", help="Average probabilities; this is not OOF")
    parser.add_argument("--include-patient-id", action="store_true", help="Include inferred IDs in local CSV only")
    args = parser.parse_args(argv)
    try:
        if args.list:
            _, catalog, source_info = repository_catalog(args.repo, args.local_repo, args.revision)
            entries = catalog["models"]
            if args.condition:
                entries = [entry for entry in entries if entry["condition"] == args.condition]
                if not entries:
                    raise ValueError("Condition is absent from catalog")
            result = {"source": source_info, "models": entries}
        else:
            folds = args.folds
            if hasattr(args, "fold"):
                folds = [args.fold]
            result = run_inference(
                repo=args.repo, local_repo=args.local_repo, revision=args.revision,
                condition=args.condition, folds=folds, all_folds=args.all_folds,
                root=args.root, output=args.output, batch_size=args.batch_size,
                device=args.device, ensemble=args.ensemble,
                include_patient_id=args.include_patient_id,
            )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, "Inference failed: " + str(error) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

"""Patient-isolated cross-validation training and checkpoint evaluation."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform
import random
from pathlib import Path

import numpy as np
import torch

from .metrics import calculate_metrics, ordered_classes


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _patients(samples):
    if any(not sample.get("patient_id") for sample in samples):
        raise ValueError("Every sample must contain a patient_id")
    return sorted({str(sample["patient_id"]) for sample in samples})


def _check_partitions(partitions):
    for name, samples in partitions.items():
        if not samples:
            raise ValueError("The %s partition is empty" % name)
        if any(sample.get("label") is None or sample.get("label", -1) < 0 for sample in samples):
            raise ValueError("Training, validation, and test partitions must be labeled")
        ids = [sample["sample_id"] for sample in samples]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample IDs in the %s partition" % name)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(_patients(partitions[left])) & set(_patients(partitions[right])):
            raise ValueError("Patient overlap between %s and %s" % (left, right))
        if {sample["sample_id"] for sample in partitions[left]} & {sample["sample_id"] for sample in partitions[right]}:
            raise ValueError("Sample overlap between %s and %s" % (left, right))


def predict(model, loader, device, class_to_idx):
    """Produce stable per-sample predictions, and loss for labeled samples."""
    names = ordered_classes(class_to_idx)
    device = _device(device) if isinstance(device, str) else device
    rows = []
    loss_total = 0.0
    labeled_count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            if logits.ndim != 2 or logits.shape[1] != len(names):
                raise ValueError("Model output does not match the saved class mapping")
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced nonfinite logits")
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            labeled = labels >= 0
            if bool(labeled.any()):
                loss_total += float(torch.nn.functional.cross_entropy(logits[labeled], labels[labeled], reduction="sum").item())
                labeled_count += int(labeled.sum().item())
            for index, probability in enumerate(probabilities):
                label = int(labels[index].item())
                predicted = int(probability.argmax())
                row = {"sample_id": str(batch["sample_id"][index]),
                       "patient_id": str(batch["patient_id"][index]),
                       "true_label": label if label >= 0 else None,
                       "pred_label": predicted, "pred_class": names[predicted]}
                row.update({"prob_%s_%s" % (class_index, name): float(probability[class_index])
                            for class_index, name in enumerate(names)})
                rows.append(row)
    if not rows:
        raise ValueError("Prediction loader is empty")
    return rows, loss_total / labeled_count if labeled_count else None


def _prediction_metrics(rows, class_to_idx):
    names = ordered_classes(class_to_idx)
    labeled = [row for row in rows if row["true_label"] is not None]
    if not labeled:
        return None
    metrics = calculate_metrics([row["true_label"] for row in labeled],
                                [[row["prob_%s_%s" % (index, name)] for index, name in enumerate(names)] for row in labeled],
                                class_to_idx)
    metrics["prediction_count"] = len(rows)
    metrics["unlabeled_count"] = len(rows) - len(labeled)
    return metrics


def _prediction_fields(class_to_idx, fold=False):
    fields = ["sample_id", "patient_id", "true_label", "pred_label", "pred_class"]
    fields += ["prob_%s_%s" % (index, name) for index, name in enumerate(ordered_classes(class_to_idx))]
    return (["fold"] + fields) if fold else fields


def _selection(metric, loss, metrics):
    if metric == "val_loss":
        return loss, "min"
    key = {"val_accuracy": "accuracy", "accuracy": "accuracy",
           "val_balanced_accuracy": "balanced_accuracy", "balanced_accuracy": "balanced_accuracy",
           "val_macro_f1": "macro_f1", "macro_f1": "macro_f1",
           "val_auroc_macro": "auroc_macro", "auroc": "auroc_macro"}.get(metric)
    if key is None:
        raise ValueError("Unknown selection_metric: %s" % metric)
    value = metrics[key]
    if value is None:
        raise ValueError("The requested selection metric is undefined for this validation partition")
    return value, "max"


def _resolved_model_config(model_config, context):
    result = copy.deepcopy(model_config)
    initialization = result.get("initialization") or {}
    if not isinstance(initialization, dict):
        raise ValueError("model.initialization must map stream names to checkpoint paths")
    result["initialization"] = {stream: str(path).replace("{fold}", str(context["fold"]))
                                for stream, path in initialization.items()}
    result["initialization_context"] = context
    return result


def _make_optimizer(model, config):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    kwargs = {"lr": float(config.get("lr", 0.001)), "weight_decay": float(config.get("weight_decay", 0))}
    if kwargs["lr"] <= 0 or kwargs["weight_decay"] < 0:
        raise ValueError("lr must be positive and weight_decay nonnegative")
    name = str(config.get("optimizer", "adamw")).lower()
    if name == "sgd":
        return torch.optim.SGD(parameters, momentum=float(config.get("momentum", 0.9)), **kwargs)
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    raise ValueError("Unknown optimizer: %s" % name)


def _make_scheduler(optimizer, config, epochs):
    name = str(config.get("scheduler", "none")).lower()
    if name in ("none", "null"):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.get("step_size", 7)), gamma=float(config.get("gamma", 0.1)))
    raise ValueError("Unknown scheduler: %s" % name)


def _train_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    loss_total = 0.0
    correct = 0
    count = 0
    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        size = int(labels.shape[0])
        loss_total += float(loss.item()) * size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        count += size
    if not count:
        raise ValueError("Training loader is empty")
    return loss_total / count, correct / count


def _save_fold_outputs(fold_dir, partition, rows, loss, class_to_idx, fold):
    metrics = _prediction_metrics(rows, class_to_idx)
    if metrics is None:
        raise ValueError("CV partitions must contain labels")
    metrics.update({"loss": loss, "fold": fold, "evaluation_role": "selection_validation" if partition == "validation" else "held_out_outer_test"})
    _write_json(fold_dir / (partition + "_metrics.json"), metrics)
    _write_csv(fold_dir / (partition + "_predictions.csv"), rows, _prediction_fields(class_to_idx))
    return metrics


def _write_summary(output_dir, metadata):
    folds = []
    oof_rows = []
    for fold_dir in sorted(output_dir.glob("fold_*")):
        if not fold_dir.is_dir() or not (fold_dir / "summary.json").exists():
            continue
        result = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
        folds.append(result)
        with (fold_dir / "test_predictions.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["fold"] = result["fold"]
                row["true_label"] = int(row["true_label"])
                row["pred_label"] = int(row["pred_label"])
                for key in row:
                    if key.startswith("prob_"):
                        row[key] = float(row[key])
                oof_rows.append(row)
    folds.sort(key=lambda value: value["fold"])
    ids = [row["sample_id"] for row in oof_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("A sample appears in multiple outer test folds")
    summary = dict(metadata)
    summary["completed_folds"] = [value["fold"] for value in folds]
    summary["folds"] = folds
    summary["complete_cross_validation"] = summary["completed_folds"] == metadata["planned_folds"]
    summary["completion_status"] = "complete" if summary["complete_cross_validation"] else "partial"
    summary["completed_fold_count"] = len(folds)
    summary["planned_fold_count"] = len(metadata["planned_folds"])
    summary["out_of_fold_sample_count"] = len(oof_rows)
    summary["outer_test_coverage_fraction"] = len(oof_rows) / metadata["manifest_sample_count"]
    aggregate = {}
    for key in ("loss", "accuracy", "balanced_accuracy", "macro_f1", "auroc_macro"):
        values = [value["test_metrics"][key] for value in folds if value["test_metrics"][key] is not None]
        aggregate[key] = {"mean": float(np.mean(values)) if values else None,
                          "sample_sd": float(np.std(values, ddof=1)) if len(values) >= 2 else None,
                          "available_fold_count": len(values)}
    summary["test_metrics_aggregate"] = aggregate
    summary["out_of_fold_metrics"] = _prediction_metrics(oof_rows, metadata["class_to_idx"]) if oof_rows else None
    fields = ["fold", "best_epoch", "epochs_completed", "validation_loss", "validation_accuracy", "test_loss", "test_accuracy", "test_balanced_accuracy", "test_macro_f1", "test_auroc_macro"]
    rows = [{"fold": value["fold"], "best_epoch": value["best_epoch"], "epochs_completed": value["epochs_completed"],
             "validation_loss": value["validation_metrics"]["loss"], "validation_accuracy": value["validation_metrics"]["accuracy"],
             "test_loss": value["test_metrics"]["loss"], "test_accuracy": value["test_metrics"]["accuracy"],
             "test_balanced_accuracy": value["test_metrics"]["balanced_accuracy"], "test_macro_f1": value["test_metrics"]["macro_f1"],
             "test_auroc_macro": value["test_metrics"]["auroc_macro"]} for value in folds]
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "summary.csv", rows, fields)
    _write_csv(output_dir / "out_of_fold_predictions.csv", oof_rows, _prediction_fields(metadata["class_to_idx"], fold=True))
    return summary


def train(config, manifest, splits, folds=None, resume=None):
    """Train fresh folds, select only on validation, then test each best model once."""
    from .data import make_loader, select_samples

    if resume is not None:
        raise NotImplementedError("Checkpoint resume is not supported; completed folds can be added with folds=")
    config = copy.deepcopy(config)
    fingerprint = manifest.get("fingerprint")
    if not fingerprint or splits.get("manifest_fingerprint") != fingerprint:
        raise ValueError("Splits do not belong to this manifest")
    split_fingerprint = splits.get("fingerprint") or _fingerprint(splits)
    streams = config["data"]["streams"]
    if not streams or len(set(streams)) != len(streams):
        raise ValueError("data.streams must contain distinct stream names")
    if not set(streams).issubset(manifest.get("streams", [])):
        raise ValueError("Requested streams are absent from the manifest")
    class_to_idx = manifest["class_to_idx"]
    ordered_classes(class_to_idx)
    train_config = config["train"]
    epochs = int(train_config.get("epochs", 50))
    batch_size = int(train_config.get("batch_size", 8))
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    selection_metric = train_config.get("selection_metric", "val_loss")
    _selection(selection_metric, 1.0, {"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0, "auroc_macro": 1.0})
    patience = train_config.get("patience")
    if patience is not None and int(patience) < 0:
        raise ValueError("patience must be nonnegative or null")
    patience = int(patience) if patience else None
    device = _device(train_config.get("device", "auto"))
    use_amp = bool(train_config.get("amp", False)) and device.type == "cuda"
    seed = int(train_config.get("seed", 42))
    fold_definitions = {int(value["fold"]): value for value in splits["folds"]}
    if len(fold_definitions) != len(splits["folds"]):
        raise ValueError("Duplicate fold IDs")
    chosen = sorted(fold_definitions) if folds is None else [int(fold) for fold in folds]
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(fold_definitions):
        raise ValueError("Requested folds must be distinct known fold IDs")
    output_dir = Path(config["experiment"]["output_dir"])
    if output_dir.exists():
        if folds is None and any(output_dir.iterdir()):
            raise FileExistsError("Output directory contains records; specify uncompleted folds explicitly")
        if any((output_dir / ("fold_%s" % fold)).exists() for fold in chosen):
            raise FileExistsError("Refusing to overwrite an existing fold")
        if any(output_dir.iterdir()) and not (output_dir / "experiment.json").exists():
            raise FileExistsError("Existing directory is not a compatible experiment")
    partitions_by_fold = {}
    all_test_ids = set()
    all_test_patients = set()
    manifest_ids = {sample["sample_id"] for sample in manifest["samples"]}
    if len(manifest_ids) != len(manifest["samples"]):
        raise ValueError("Manifest sample IDs must be unique")
    for fold, definition in fold_definitions.items():
        partitions = {partition: select_samples(manifest, definition[partition]) for partition in ("train", "validation", "test")}
        _check_partitions(partitions)
        partition_ids = {sample["sample_id"] for samples in partitions.values() for sample in samples}
        if partition_ids != manifest_ids:
            raise ValueError("Each fold must partition every manifest sample")
        if any(sample["label"] not in class_to_idx.values() for samples in partitions.values() for sample in samples):
            raise ValueError("A partition contains an unknown class label")
        test_ids = {sample["sample_id"] for sample in partitions["test"]}
        test_patients = set(_patients(partitions["test"]))
        if test_ids & all_test_ids:
            raise ValueError("A sample appears in multiple outer test folds")
        if test_patients & all_test_patients:
            raise ValueError("A patient appears in multiple outer test folds")
        all_test_ids.update(test_ids)
        all_test_patients.update(test_patients)
        partitions_by_fold[fold] = partitions
    if all_test_ids != manifest_ids:
        raise ValueError("Outer test folds must partition every manifest sample")
    metadata = {"format_version": 1, "name": config["experiment"].get("name", output_dir.name),
                "manifest_fingerprint": fingerprint, "split_fingerprint": split_fingerprint,
                "config_fingerprint": _fingerprint(config), "config": config,
                "streams": list(streams), "class_to_idx": class_to_idx,
                "selection_metric": selection_metric, "planned_folds": sorted(fold_definitions),
                "manifest_sample_count": len(manifest["samples"]),
                "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "numpy": str(np.__version__), "device": str(device), "amp": use_amp}}
    if (output_dir / "experiment.json").exists():
        previous = json.loads((output_dir / "experiment.json").read_text(encoding="utf-8"))
        for key in ("manifest_fingerprint", "split_fingerprint", "config_fingerprint"):
            if previous.get(key) != metadata[key]:
                raise ValueError("Existing experiment has a different %s" % key)
    from .models import build_model, load_checkpoint, model_from_checkpoint

    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "experiment.json").exists():
        _write_json(output_dir / "experiment.json", metadata)
        _write_json(output_dir / "config.json", config)
        _write_json(output_dir / "splits.json", splits)
    for fold in chosen:
        fold_dir = output_dir / ("fold_%s" % fold)
        fold_dir.mkdir()
        _seed(seed + fold)
        partitions = partitions_by_fold[fold]
        context = {"fold": fold, "manifest_fingerprint": fingerprint, "split_fingerprint": split_fingerprint,
                   "train_patient_ids": _patients(partitions["train"]),
                   "validation_patient_ids": _patients(partitions["validation"]),
                   "test_patient_ids": _patients(partitions["test"]),
                   "class_to_idx": copy.deepcopy(class_to_idx),
                   "preprocessing": copy.deepcopy(config["preprocessing"])}
        model_config = _resolved_model_config(config["model"], context)
        model = build_model(model_config, streams, len(class_to_idx), initialize=True).to(device)
        optimizer = _make_optimizer(model, train_config)
        scheduler = _make_scheduler(optimizer, train_config, epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        loader_kwargs = {"streams": streams, "preprocessing": config["preprocessing"], "batch_size": batch_size,
                         "num_workers": int(train_config.get("num_workers", 0)), "seed": seed + fold}
        train_loader = make_loader(partitions["train"], training=True, **loader_kwargs)
        validation_loader = make_loader(partitions["validation"], training=False, **loader_kwargs)
        history = []
        best_value = None
        best_epoch = None
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_loss, train_accuracy = _train_epoch(model, train_loader, optimizer, scaler, device, use_amp)
            validation_rows, validation_loss = predict(model, validation_loader, device, class_to_idx)
            validation_metrics = _prediction_metrics(validation_rows, class_to_idx)
            value, direction = _selection(selection_metric, validation_loss, validation_metrics)
            if value is None or not math.isfinite(value):
                raise ValueError("Selection metric is not finite")
            history.append({"epoch": epoch, "lr": learning_rate, "train_loss": train_loss,
                            "train_accuracy": train_accuracy, "val_loss": validation_loss,
                            "val_accuracy": validation_metrics["accuracy"],
                            "val_balanced_accuracy": validation_metrics["balanced_accuracy"],
                            "val_macro_f1": validation_metrics["macro_f1"],
                            "val_auroc_macro": validation_metrics["auroc_macro"]})
            print("Fold %s | epoch %s/%s | train loss %.5f | val loss %.5f | val accuracy %.2f%%" %
                  (fold, epoch, epochs, train_loss, validation_loss, validation_metrics["accuracy"] * 100), flush=True)
            improved = best_value is None or (value < best_value if direction == "min" else value > best_value)
            if improved:
                best_value, best_epoch, stale_epochs = value, epoch, 0
                checkpoint = {"format_version": 1, "state_dict": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                              "model_config": model_config, "streams": list(streams), "class_to_idx": dict(class_to_idx),
                              "preprocessing": copy.deepcopy(config["preprocessing"]), "epoch": epoch,
                              "config": config, "selection_metric": selection_metric, "selection_value": value,
                              "validation_metrics": validation_metrics, "validation_loss": validation_loss,
                              "train_sample_ids": list(fold_definitions[fold]["train"]),
                              "validation_sample_ids": list(fold_definitions[fold]["validation"]),
                              "test_sample_ids": list(fold_definitions[fold]["test"]),
                              "seed": seed + fold, "runtime": metadata["runtime"], **context}
                temporary_checkpoint = fold_dir / "best_model.pth.tmp"
                torch.save(checkpoint, temporary_checkpoint)
                temporary_checkpoint.replace(fold_dir / "best_model.pth")
            else:
                stale_epochs += 1
            _write_csv(fold_dir / "epoch_metrics.csv", history, list(history[0]))
            if scheduler is not None:
                scheduler.step()
            if patience is not None and stale_epochs >= int(patience):
                break
        del model, optimizer, scaler, scheduler
        checkpoint = load_checkpoint(fold_dir / "best_model.pth", device="cpu")
        best_model = model_from_checkpoint(checkpoint, device=device)
        validation_rows, validation_loss = predict(best_model, validation_loader, device, class_to_idx)
        validation_metrics = _save_fold_outputs(fold_dir, "validation", validation_rows, validation_loss, class_to_idx, fold)
        test_loader = make_loader(partitions["test"], training=False, **loader_kwargs)
        test_rows, test_loss = predict(best_model, test_loader, device, class_to_idx)
        test_metrics = _save_fold_outputs(fold_dir, "test", test_rows, test_loss, class_to_idx, fold)
        result = {"fold": fold, "best_epoch": best_epoch, "epochs_completed": len(history),
                  "selection_metric": selection_metric, "selection_value": best_value,
                  "validation_metrics": validation_metrics, "test_metrics": test_metrics,
                  "checkpoint": str((fold_dir / "best_model.pth").resolve())}
        _write_json(fold_dir / "summary.json", result)
        _write_summary(output_dir, metadata)
        del best_model
    return _write_summary(output_dir, metadata)


def evaluate_checkpoint(checkpoint_path, samples, output_dir, device="auto", batch_size=8,
                        num_workers=0, allow_patient_overlap=False, check_patient_overlap=True,
                        evaluation_role=None):
    """Evaluate using saved preprocessing; external evaluations reject known patients.

    A caller may provide a saved-subset role only after verifying its manifest
    and split fingerprints. Sample names alone never establish held-out scope.
    """
    from .data import make_loader
    from .models import load_checkpoint, model_from_checkpoint

    if evaluation_role not in (None, "external_validation", "held_out_outer_test", "selection_validation", "inference"):
        raise ValueError("Unsupported evaluation_role")
    requested_role = evaluation_role
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    samples = list(samples)
    if not samples:
        raise ValueError("Evaluation samples are empty")
    if len({sample["sample_id"] for sample in samples}) != len(samples):
        raise ValueError("Evaluation sample IDs must be unique")
    patient_ids = set(_patients(samples))
    overlap = []
    provenance_verified = not checkpoint.get("legacy_no_split_metadata", False) and all(
        field in checkpoint for field in ("train_patient_ids", "validation_patient_ids", "test_patient_ids")) and bool(
            checkpoint.get("manifest_fingerprint") and checkpoint.get("split_fingerprint"))
    overlap_check_completed = bool(check_patient_overlap and provenance_verified)
    if check_patient_overlap:
        if not provenance_verified and not allow_patient_overlap:
            raise ValueError("Checkpoint lacks patient provenance; external overlap cannot be verified")
        if provenance_verified:
            known_patients = set(map(str, checkpoint["train_patient_ids"])) | set(map(str, checkpoint["validation_patient_ids"]))
            overlap = sorted(patient_ids & known_patients)
            if overlap and not allow_patient_overlap:
                raise ValueError("Evaluation overlaps %s training/validation patients" % len(overlap))
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite evaluation records")
    actual_device = _device(device)
    model = model_from_checkpoint(checkpoint, device=actual_device)
    loader = make_loader(samples, checkpoint["streams"], checkpoint["preprocessing"],
                         batch_size, num_workers, False, int(checkpoint.get("seed", 42)))
    rows, loss = predict(model, loader, actual_device, checkpoint["class_to_idx"])
    metrics = _prediction_metrics(rows, checkpoint["class_to_idx"])
    if metrics is None or requested_role == "inference":
        evaluation_role = "inference"
    elif not provenance_verified:
        evaluation_role = "exploratory_validation_unverified_provenance"
    elif overlap:
        evaluation_role = "exploratory_validation_patient_overlap"
    elif requested_role is not None:
        evaluation_role = requested_role
    else:
        evaluation_role = "external_validation" if check_patient_overlap else "unchecked_validation"
    result = {"metrics": metrics, "loss": loss, "prediction_count": len(rows),
              "class_to_idx": checkpoint["class_to_idx"], "streams": checkpoint["streams"],
              "checkpoint": str(Path(checkpoint_path).resolve()), "epoch": checkpoint.get("epoch"),
              "fold": checkpoint.get("fold"), "patient_overlap_check_requested": bool(check_patient_overlap),
              "patient_overlap_check": overlap_check_completed, "provenance_verified": provenance_verified,
              "patient_overlap_allowed": bool(allow_patient_overlap), "overlapping_patient_count": len(overlap),
              "evaluation_role": evaluation_role,
              "independent_validation": bool(metrics is not None and overlap_check_completed and not overlap and
                                             evaluation_role in ("external_validation", "held_out_outer_test"))}
    if metrics is not None:
        result.update(metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "predictions.csv", rows, _prediction_fields(checkpoint["class_to_idx"]))
    _write_json(output_dir / "metrics.json", result)
    return result

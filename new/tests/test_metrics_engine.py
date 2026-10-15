"""Metrics and end-to-end CPU checks for patient-isolated fold training."""
import copy
import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from multistream import engine
from multistream.metrics import calculate_metrics


CLASS_MAP = {"bad": 0, "good": 1}


def test_metrics_class_mapping_and_undefined_auc():
    metrics = calculate_metrics([0, 0, 1, 1], [[.9, .1], [.4, .6], [.3, .7], [.2, .8]], CLASS_MAP)
    assert metrics["class_order"] == ["bad", "good"]
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == .75
    assert metrics["balanced_accuracy"] == .75
    assert metrics["per_class"]["bad"]["recall"] == .5
    assert metrics["per_class"]["good"]["precision"] == pytest.approx(2 / 3)
    assert metrics["auroc_macro"] == 1
    one_class = calculate_metrics([0, 0], [[.9, .1], [.8, .2]], CLASS_MAP)
    assert one_class["auroc_macro"] is None
    assert one_class["auroc_one_vs_rest"] == {"bad": None, "good": None}
    assert one_class["confusion_matrix"] == [[2, 0], [0, 0]]
    assert one_class["balanced_accuracy"] == 1


def test_metrics_reject_bad_probability_columns_and_labels():
    with pytest.raises(ValueError, match="columns"):
        calculate_metrics([0], [[1]], CLASS_MAP)
    with pytest.raises(ValueError, match="sum"):
        calculate_metrics([0], [[.2, .2]], CLASS_MAP)
    with pytest.raises(ValueError, match="unknown"):
        calculate_metrics([2], [[.2, .8]], CLASS_MAP)


def _fixture(tmp_path, streams):
    all_streams = ["color_inner", "gray_inner"]
    samples = []
    for patient in range(6):
        for label in range(2):
            sample_id = "patient_%s_sample_%s" % (patient, label)
            images = {}
            for stream in all_streams:
                path = tmp_path / (sample_id + "_" + stream + ".png")
                array = np.full((36, 36, 3), 30 + label * 190, dtype=np.uint8)
                array[patient:patient + 4, :, 0] = 100
                Image.fromarray(array).save(path)
                images[stream] = str(path)
            samples.append({"sample_id": sample_id, "patient_id": "patient_%s" % patient,
                            "label": label, "images": images})
    manifest = {"fingerprint": "fixture_manifest", "class_to_idx": CLASS_MAP,
                "streams": all_streams, "samples": samples}
    def ids(patients):
        return [sample["sample_id"] for sample in samples if int(sample["patient_id"].split("_")[-1]) in patients]
    splits = {"manifest_fingerprint": manifest["fingerprint"], "fingerprint": "fixture_splits", "folds": [
        {"fold": 0, "train": ids([3, 4]), "validation": ids([5]), "test": ids([0, 1, 2])},
        {"fold": 1, "train": ids([0, 1]), "validation": ids([2]), "test": ids([3, 4, 5])},
    ]}
    config = {"data": {"streams": streams},
              "preprocessing": {"image_size": 32, "resize_size": 36, "mean": [.5] * 3, "std": [.5] * 3,
                                "horizontal_flip_probability": 0, "crop_scale": [.9, 1], "crop_ratio": [1, 1]},
              "model": {"name": "resnet18", "pretrained": False, "hidden_dim": 8,
                        "freeze_backbones": False, "initialization": {}},
              "train": {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": .001,
                        "weight_decay": .001, "optimizer": "adamw", "scheduler": "none",
                        "seed": 42, "device": "cpu", "amp": False, "patience": 0, "selection_metric": "val_loss"},
              "experiment": {"output_dir": str(tmp_path / "run"), "name": "fixture"}}
    return config, manifest, splits


@pytest.mark.parametrize("streams", [["color_inner"], ["color_inner", "gray_inner"]])
def test_cpu_two_fold_single_and_multistream_pipeline(tmp_path, streams):
    from multistream.models import load_checkpoint
    config, manifest, splits = _fixture(tmp_path, streams)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first = engine.train(config, manifest, splits, folds=[0])
        assert first["completed_folds"] == [0]
        assert first["complete_cross_validation"] is False
        assert first["completion_status"] == "partial"
        assert first["test_metrics_aggregate"]["accuracy"]["sample_sd"] is None
        summary = engine.train(config, manifest, splits, folds=[1])
        assert summary["completed_folds"] == [0, 1]
        assert summary["complete_cross_validation"] is True
        assert summary["test_metrics_aggregate"]["accuracy"]["available_fold_count"] == 2
        assert summary["test_metrics_aggregate"]["accuracy"]["sample_sd"] is not None
        assert summary["out_of_fold_metrics"]["sample_count"] == 12
        output = Path(config["experiment"]["output_dir"])
        checkpoint = load_checkpoint(output / "fold_0" / "best_model.pth")
        assert checkpoint["streams"] == streams
        assert checkpoint["fold"] == 0
        assert checkpoint["train_patient_ids"] == ["patient_3", "patient_4"]
        assert checkpoint["validation_patient_ids"] == ["patient_5"]
        assert checkpoint["test_patient_ids"] == ["patient_0", "patient_1", "patient_2"]
        assert checkpoint["preprocessing"] == config["preprocessing"]
        for fold in (0, 1):
            for filename in ("best_model.pth", "epoch_metrics.csv", "validation_metrics.json",
                             "validation_predictions.csv", "test_metrics.json", "test_predictions.csv", "summary.json"):
                assert (output / ("fold_%s" % fold) / filename).exists()
        with (output / "out_of_fold_predictions.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 12
        assert len({row["sample_id"] for row in rows}) == 12
        with pytest.raises(FileExistsError, match="overwrite"):
            engine.train(config, manifest, splits, folds=[0])
        with pytest.raises(ValueError, match="overlaps"):
            engine.evaluate_checkpoint(output / "fold_0" / "best_model.pth", [manifest["samples"][6]], tmp_path / "overlap", device="cpu")
        unlabeled = copy.deepcopy(manifest["samples"][:2])
        for sample in unlabeled:
            sample["label"] = None
        inference = engine.evaluate_checkpoint(output / "fold_0" / "best_model.pth", unlabeled,
                                               tmp_path / "inference", device="cpu", check_patient_overlap=False)
        assert inference["metrics"] is None
        assert inference["loss"] is None
        assert inference["evaluation_role"] == "inference"
        assert (tmp_path / "inference" / "predictions.csv").exists()
    finally:
        torch.set_num_threads(previous_threads)


def test_validation_selects_best_loss_and_test_is_used_once(tmp_path, monkeypatch):
    from multistream import models
    config, manifest, splits = _fixture(tmp_path, ["color_inner"])
    config["train"]["epochs"] = 3
    state = {"epoch": 0, "test_calls": 0, "model_builds": 0}
    def build(*args, **kwargs):
        state["model_builds"] += 1
        return torch.nn.Linear(1, 2)
    def train_epoch(model, *args, **kwargs):
        state["epoch"] += 1
        return .5, .5
    def predictions(model, loader, device, class_to_idx):
        partition_ids = {str(batch_id) for batch in loader for batch_id in batch["sample_id"]}
        is_test = partition_ids == set(splits["folds"][0]["test"])
        if is_test:
            state["test_calls"] += 1
        rows = [{"sample_id": sample_id, "patient_id": "p", "true_label": index % 2,
                 "pred_label": index % 2, "pred_class": "bad" if index % 2 == 0 else "good",
                 "prob_0_bad": .8 if index % 2 == 0 else .2,
                 "prob_1_good": .2 if index % 2 == 0 else .8} for index, sample_id in enumerate(sorted(partition_ids))]
        return rows, [0, .3, .1, .2][getattr(model, "loaded_epoch", state["epoch"])] if not is_test else .4
    def reconstruct(checkpoint, device):
        model = torch.nn.Linear(1, 2)
        model.loaded_epoch = checkpoint["epoch"]
        return model
    monkeypatch.setattr(models, "build_model", build)
    monkeypatch.setattr(models, "model_from_checkpoint", reconstruct)
    monkeypatch.setattr(engine, "_train_epoch", train_epoch)
    monkeypatch.setattr(engine, "predict", predictions)
    summary = engine.train(config, manifest, splits, folds=[0])
    assert summary["folds"][0]["best_epoch"] == 2
    assert summary["folds"][0]["validation_metrics"]["loss"] == .1
    assert state["model_builds"] == 1
    assert state["test_calls"] == 1


def test_partition_patient_overlap_is_rejected_before_output(tmp_path):
    config, manifest, splits = _fixture(tmp_path, ["color_inner"])
    manifest["samples"][0]["patient_id"] = "patient_5"
    with pytest.raises(ValueError, match="Patient overlap"):
        engine.train(config, manifest, splits, folds=[0])
    assert not Path(config["experiment"]["output_dir"]).exists()


def test_legacy_external_evaluation_requires_explicit_exploratory_opt_in(tmp_path, monkeypatch):
    from multistream import models
    config, manifest, _ = _fixture(tmp_path, ["color_inner"])
    checkpoint = {"legacy_no_split_metadata": True, "class_to_idx": CLASS_MAP,
                  "streams": ["color_inner"], "preprocessing": config["preprocessing"]}
    monkeypatch.setattr(models, "load_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(models, "model_from_checkpoint", lambda *args, **kwargs: torch.nn.Linear(1, 2))
    rows = [{"sample_id": "external", "patient_id": "external", "true_label": 0,
             "pred_label": 0, "pred_class": "bad", "prob_0_bad": .8, "prob_1_good": .2}]
    monkeypatch.setattr(engine, "predict", lambda *args: (rows, .2))
    with pytest.raises(ValueError, match="provenance"):
        engine.evaluate_checkpoint("legacy.pth", manifest["samples"][:1], tmp_path / "strict", device="cpu")
    result = engine.evaluate_checkpoint("legacy.pth", manifest["samples"][:1], tmp_path / "exploratory",
                                        device="cpu", allow_patient_overlap=True, evaluation_role="external_validation")
    assert result["provenance_verified"] is False
    assert result["patient_overlap_check"] is False
    assert result["independent_validation"] is False
    assert result["evaluation_role"] == "exploratory_validation_unverified_provenance"


@pytest.mark.parametrize("requested_role,check_overlap,expected_role,independent", [
    (None, True, "external_validation", True),
    (None, False, "unchecked_validation", False),
    ("inference", False, "inference", False),
    ("held_out_outer_test", True, "held_out_outer_test", True),
    ("selection_validation", False, "selection_validation", False),
])
def test_evaluation_scope_needs_verified_caller_context_not_matching_names(
        tmp_path, monkeypatch, requested_role, check_overlap, expected_role, independent):
    from multistream import models
    config, manifest, _ = _fixture(tmp_path, ["color_inner"])
    sample = manifest["samples"][0]
    checkpoint = {"class_to_idx": CLASS_MAP, "streams": ["color_inner"],
                  "preprocessing": config["preprocessing"], "manifest_fingerprint": "original_manifest",
                  "split_fingerprint": "original_splits", "train_patient_ids": ["patient_5"],
                  "validation_patient_ids": ["patient_4"], "test_patient_ids": ["patient_0"],
                  "test_sample_ids": [sample["sample_id"]]}
    monkeypatch.setattr(models, "load_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(models, "model_from_checkpoint", lambda *args, **kwargs: torch.nn.Linear(1, 2))
    row = {"sample_id": sample["sample_id"], "patient_id": "patient_0", "true_label": 0,
           "pred_label": 0, "pred_class": "bad", "prob_0_bad": .8, "prob_1_good": .2}
    monkeypatch.setattr(engine, "predict", lambda *args: ([row], .2))
    result = engine.evaluate_checkpoint("model.pth", [sample], tmp_path / "evaluation", device="cpu",
                                        check_patient_overlap=check_overlap, evaluation_role=requested_role)
    assert result["evaluation_role"] == expected_role
    assert result["independent_validation"] is independent

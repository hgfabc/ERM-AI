"""Offline inference tests: generated CPU models, local images and mocked Hub reads."""
import contextlib
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from multistream.models import build_model


TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


infer = _module("published_inference_under_test", "infer_published.py")
exporter = _module("public_export_for_inference_test", "export_public_model.py")
COMMIT = "a" * 40
CONDITION = "single_gray_inner"
CLASSES = {"bad": 0, "good": 1}
PREPROCESSING = {"image_size": 32, "resize_size": 32,
                 "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class PublishedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.workspace = tempfile.TemporaryDirectory()
        cls.base = Path(cls.workspace.name)
        cls.repository = cls.base / "cache" / "snapshots" / COMMIT
        cls.repository.mkdir(parents=True)
        # Executing downloaded code would fail every inference test.
        (cls.repository / "modeling_multistream.py").write_text(
            "raise AssertionError('Hub code must not execute')\n", encoding="utf-8",
        )
        entries = []
        for fold in (0, 1):
            torch.manual_seed(55 + fold)
            model = build_model(
                {"name": "resnet18", "pretrained": False, "hidden_dim": 8},
                ["gray_inner"], 2,
            ).eval()
            checkpoint = cls.base / ("synthetic_source_%d.pth" % fold)
            torch.save({
                "format_version": 1, "state_dict": model.state_dict(),
                "model_config": {"name": "resnet18", "pretrained": False, "hidden_dim": 8},
                "streams": ["gray_inner"], "class_to_idx": CLASSES,
                "preprocessing": PREPROCESSING,
            }, checkpoint)
            relative = "models/%s/fold_%d" % (CONDITION, fold)
            bundle = cls.repository / relative
            exporter.export_public_model(checkpoint, bundle, kind="demo")
            # Batch repository layout keeps the loader and requirements at root.
            (bundle / "modeling_multistream.py").unlink()
            (bundle / "requirements.txt").unlink()
            entries.append({
                "condition": CONDITION, "fold": fold, "streams": ["gray_inner"],
                "path": relative, "inference_ready": True,
                "config_sha256": infer._sha256(bundle / "config.json"),
                "weights_sha256": infer._sha256(bundle / "model.safetensors"),
            })
            del model
        entries.extend([
            {"condition": "fusion_8", "fold": 0,
             "streams": ["_vertical", "_horizontal", "color_full", "color_inner",
                         "color_outer", "gray_full", "gray_inner", "gray_outer"],
             "path": "models/fusion_8/fold_0", "inference_ready": True},
            {"condition": "historical_unverified", "fold": None, "streams": None,
             "path": "historical/unverified", "inference_ready": False},
        ])
        (cls.repository / "catalog.json").write_text(
            json.dumps({"catalog_version": 1, "models": entries, "unused_extra": "ignored"}),
            encoding="utf-8",
        )
        cls.labeled = cls.base / "labeled"
        cls.unlabeled = cls.base / "unlabeled"
        for root, labeled in ((cls.labeled, True), (cls.unlabeled, False)):
            for index, (label, color) in enumerate((("bad", (20, 40, 80)), ("good", (210, 160, 100)))):
                case = root / label / ("case_%d" % index) if labeled else root / ("case_%d" % index)
                case.mkdir(parents=True)
                # Only the selected stream is present; all-eight completeness is not required.
                Image.new("RGB", (40, 32), color).save(case / "synthetic_gray_inner.png")

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()
        torch.set_num_threads(cls.old_threads)

    def test_local_labeled_fold_predictions_are_descriptive_not_oof(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result"
            with patch.object(infer, "snapshot_download") as downloaded:
                summary = infer.run_inference(
                    local_repo=self.repository, condition=CONDITION, folds=[0],
                    root=self.labeled, output=output, batch_size=2, device="cpu",
                )
            downloaded.assert_not_called()
            rows = _read_csv(output / "predictions_fold_0.csv")
            self.assertEqual(len(rows), 2)
            self.assertNotIn("patient_id", rows[0])
            self.assertEqual(set(row["true_class"] for row in rows), {"bad", "good"})
            self.assertIn("prob_0_bad", rows[0])
            self.assertIn("prob_1_good", rows[0])
            for row in rows:
                self.assertAlmostEqual(float(row["prob_0_bad"]) + float(row["prob_1_good"]), 1., places=6)
            self.assertFalse(summary["out_of_fold"])
            self.assertFalse(summary["holdout_independence_verified"])
            self.assertFalse(summary["downloaded_repository_code_executed"])
            self.assertEqual(summary["models"][0]["descriptive_metrics"]["sample_count"], 2)

    def test_unlabeled_csv_alias_and_optional_local_patient_column(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "predictions.csv"
            summary = infer.run_inference(
                local_repo=self.repository, condition=CONDITION, root=self.unlabeled,
                output=output, device="cpu", include_patient_id=True,
            )
            rows = _read_csv(output)
            self.assertEqual(len(rows), 2)
            self.assertIn("patient_id", rows[0])
            self.assertTrue(all(row["true_label"] == "" for row in rows))
            self.assertIsNone(summary["models"][0]["descriptive_metrics"])
            self.assertTrue(output.with_suffix(".summary.json").is_file())

    def test_selected_fold_ensemble_is_probability_mean_and_not_oof(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "ensemble"
            summary = infer.run_inference(
                local_repo=self.repository, condition=CONDITION, folds=[0, 1],
                root=self.labeled, output=output, device="cpu", ensemble=True,
            )
            zero = _read_csv(output / "predictions_fold_0.csv")
            one = _read_csv(output / "predictions_fold_1.csv")
            mean = _read_csv(output / "predictions_ensemble.csv")
            for first, second, averaged in zip(zero, one, mean):
                self.assertEqual(first["sample_id"], averaged["sample_id"])
                for field in ("prob_0_bad", "prob_1_good"):
                    self.assertAlmostEqual(float(averaged[field]),
                                           (float(first[field]) + float(second[field])) / 2, places=7)
            self.assertFalse(summary["ensemble"]["out_of_fold"])
            self.assertFalse(summary["ensemble"]["holdout_independence_verified"])

    def test_history_and_incomplete_all_folds_are_explicitly_rejected(self):
        catalog = infer.load_catalog(self.repository)
        with self.assertRaisesRegex(ValueError, "inference_ready=false"):
            infer.select_models(catalog, "historical_unverified")
        with self.assertRaisesRegex(ValueError, "0 through 9"):
            infer.select_models(catalog, CONDITION, all_folds=True)
        complete = copy.deepcopy(catalog)
        prototype = next(entry for entry in complete["models"] if entry["condition"] == CONDITION)
        complete["models"] += [{**prototype, "fold": fold,
                               "path": "models/%s/fold_%d" % (CONDITION, fold)}
                              for fold in range(2, 10)]
        selected = infer.select_models(complete, CONDITION, all_folds=True)
        self.assertEqual([entry["fold"] for entry in selected], list(range(10)))
        self.assertTrue(all(entry["condition"] == CONDITION for entry in selected))

    def test_hub_download_is_mocked_selected_only_and_pinned_after_catalog(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(infer, "snapshot_download", return_value=str(self.repository)) as download:
                summary = infer.run_inference(
                    repo="example/synthetic-models", revision="refs/pr/42", condition=CONDITION,
                    folds=[0], root=self.unlabeled, output=Path(temp) / "result", device="cpu",
                )
            self.assertEqual(download.call_count, 2)
            first, second = [call.kwargs for call in download.call_args_list]
            self.assertEqual(first["allow_patterns"], infer.ROOT_FILES)
            self.assertEqual(first["revision"], "refs/pr/42")
            self.assertEqual(second["revision"], COMMIT)
            self.assertEqual(set(second["allow_patterns"]), {
                "catalog.json", "modeling_multistream.py",
                "models/%s/fold_0/config.json" % CONDITION,
                "models/%s/fold_0/model.safetensors" % CONDITION,
            })
            self.assertFalse(any("fusion_8" in path or "fold_1" in path for path in second["allow_patterns"]))
            self.assertEqual(summary["source"]["resolved_revision"], COMMIT)

    def test_list_needs_no_images_and_does_not_download_model_weights(self):
        with patch.object(infer, "snapshot_download", return_value=str(self.repository)) as download:
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                infer.main(["--repo", "example/synthetic-models", "--list"])
            result = json.loads(captured.getvalue())
            self.assertEqual(len(result["models"]), 4)
            self.assertEqual(download.call_count, 1)
            self.assertEqual(download.call_args.kwargs["allow_patterns"], infer.ROOT_FILES)

    def test_ensemble_rejects_different_class_mapping_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            repository = Path(temp) / "repository"
            shutil.copytree(self.repository, repository)
            config_path = repository / ("models/%s/fold_1/config.json" % CONDITION)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["class_to_idx"] = {"bad": 1, "good": 0}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            catalog_path = repository / "catalog.json"
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            for entry in catalog["models"]:
                if entry["condition"] == CONDITION and entry["fold"] == 1:
                    entry["config_sha256"] = infer._sha256(config_path)
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            output = Path(temp) / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "identical class mappings"):
                infer.run_inference(
                    local_repo=repository, condition=CONDITION, folds=[0, 1],
                    root=self.labeled, output=output, device="cpu", ensemble=True,
                )
            self.assertFalse(output.exists())

    def test_catalog_rejects_path_traversal_and_duplicate_identities(self):
        source = json.loads((self.repository / "catalog.json").read_text(encoding="utf-8"))
        for mutation in ("traversal", "glob", "duplicate", "missing-fold", "missing-streams"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                catalog = copy.deepcopy(source)
                if mutation == "traversal":
                    catalog["models"][0]["path"] = "models/../../private"
                elif mutation == "glob":
                    catalog["models"][0]["path"] = "models/*"
                elif mutation == "duplicate":
                    catalog["models"].append(copy.deepcopy(catalog["models"][0]))
                elif mutation == "missing-fold":
                    del catalog["models"][0]["fold"]
                else:
                    del catalog["models"][0]["streams"]
                (Path(temp) / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
                with self.assertRaises(ValueError):
                    infer.load_catalog(temp)

    def test_catalog_digest_mismatch_rejects_weights_before_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            repository = Path(temp) / "repository"
            shutil.copytree(self.repository, repository)
            catalog_path = repository / "catalog.json"
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            catalog["models"][0]["weights_sha256"] = "0" * 64
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "weights_sha256"):
                infer.run_inference(
                    local_repo=repository, condition=CONDITION, root=self.unlabeled,
                    output=Path(temp) / "result", device="cpu",
                )

    def test_unsupported_or_invalid_preprocessing_is_not_silently_substituted(self):
        base = exporter._public_preprocessing(PREPROCESSING, False)
        invalid = [
            {"interpolation": "nearest"}, {"crop": "random"},
            {"resize_backend": "pil", "antialias": False},
            {"resize_size": 0}, {"image_size": True},
            {"mean": [float("nan"), 0., 0.]}, {"std": [0., 1., 1.]},
        ]
        for change in invalid:
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    infer._validated_preprocessing({**base, **change})
        valid = infer._validated_preprocessing(
            {**base, "resize_backend": "tensor", "antialias": False},
        )
        self.assertEqual(valid["resize_backend"], "tensor")
        self.assertFalse(valid["antialias"])

    def test_fold_none_equals_syntax_is_not_replaced_with_default_zero(self):
        with patch.object(infer, "run_inference", return_value={"checked": True}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                infer.main(["--local-repo", str(self.repository), "--condition", "unused",
                            "--fold=none", "--root", "unused", "--outputCSV", "unused.csv"])
            self.assertEqual(run.call_args.kwargs["folds"], [None])
            self.assertEqual(run.call_args.kwargs["output"], "unused.csv")


if __name__ == "__main__":
    unittest.main()

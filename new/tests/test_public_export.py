"""Public export regression tests use generated local weights and images only."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import numpy as np
from PIL import Image
import yaml

from multistream.data import CaseDataset
from multistream.models import build_model


EXPORT_PATH = Path(__file__).resolve().parents[1] / "tools" / "export_public_model.py"
SPEC = importlib.util.spec_from_file_location("public_export_under_test", EXPORT_PATH)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)

MODEL_CONFIG = {"name": "resnet18", "pretrained": False, "hidden_dim": 8,
                "freeze_backbones": False}
PREPROCESSING = {"image_size": 32, "resize_size": 48,
                 "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
CLASSES = {"bad": 0, "good": 1}
PUBLIC_FILES = {"model.safetensors", "config.json", "modeling_multistream.py",
                "README.md", "requirements.txt"}


def _load_public_module(bundle):
    spec = importlib.util.spec_from_file_location(
        "standalone_loader_under_test", bundle / "modeling_multistream.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(model, streams):
    return {
        "format_version": 1, "state_dict": model.state_dict(),
        "model_config": {
            **MODEL_CONFIG,
            "initialization": {streams[0]: "PRIVATE_CHECKPOINT_ORIGIN"},
            "initialization_context": {"train_patient_ids": ["PRIVATE_ID_MARKER"]},
        },
        "streams": list(streams), "class_to_idx": dict(CLASSES),
        "preprocessing": {**PREPROCESSING, "private_notes": "PRIVATE_PREPROCESS_MARKER"},
        "epoch": 1, "fold": 0,
        "train_patient_ids": ["PRIVATE_ID_MARKER"],
        "validation_patient_ids": ["PRIVATE_VALIDATION_MARKER"],
        "test_patient_ids": ["PRIVATE_TEST_MARKER"],
        "train_sample_ids": ["PRIVATE_SAMPLE_MARKER"],
        "validation_sample_ids": ["PRIVATE_VALIDATION_SAMPLE"],
        "test_sample_ids": ["PRIVATE_TEST_SAMPLE"],
        "manifest_fingerprint": "PRIVATE_MANIFEST_MARKER",
        "split_fingerprint": "PRIVATE_SPLIT_MARKER",
        "legacy_source": "PRIVATE_LEGACY_PATH",
        "config": {"data": {"root": "PRIVATE_DATASET_PATH"},
                   "experiment": {"output_dir": "PRIVATE_OUTPUT_PATH"}},
    }


def _legacy_state(model):
    result = {}
    for key, value in model.state_dict().items():
        if key.startswith("backbones."):
            _, index, remainder = key.split(".", 2)
            key = "b%d.%s" % (int(index) + 1, remainder)
        elif key.startswith("fusion."):
            key = "merger." + key[len("fusion."):]
        elif key.startswith("classifier."):
            key = "fc." + key[len("classifier."):]
        result[key] = value
    return result


def _windows_lock(code=5):
    error = PermissionError(13, "simulated transient Windows file lock")
    error.winerror = code
    return error


class ModelCardLanguageTests(unittest.TestCase):
    def test_model_card_languages_keep_all_scope_and_provenance_branches(self):
        for kind in ("research", "demo"):
            for legacy in (False, True):
                with self.subTest(kind=kind, legacy=legacy):
                    metadata = {"model_config": MODEL_CONFIG, "streams": ["gray_inner", "_vertical"],
                                "class_to_idx": CLASSES, "artifact_kind": kind,
                                "legacy_no_split_metadata": legacy}
                    code_url = "https://github.com/example/ERM-AI/tree/public/new"
                    card = exporter._model_card(metadata, code_url)
                    _, front_matter, body = card.split("---", 2)
                    parsed = yaml.safe_load(front_matter)
                    self.assertEqual(parsed["language"], ["en", "zh", "ja"])
                    self.assertIn(kind, parsed["tags"])
                    navigation = body.strip().splitlines()[0]
                    self.assertEqual(navigation, "[English](#english) | [繁體中文](#traditional-chinese) | [日本語](#japanese)")
                    self.assertEqual(re.findall(r"\]\(#([^)]*)\)", navigation),
                                     re.findall(r'<a id="([^"]+)"></a>', body))
                    sections = re.split(r'<a id="[^"]+"></a>\n\n', body)[1:]
                    self.assertEqual(len(sections), 3)
                    scope_phrases = (("demo/smoke-test", "示範／冒煙測試", "デモ／スモークテスト") if kind == "demo"
                                     else ("This is a research artifact", "這是研究用產物", "これは研究用の成果物"))
                    provenance_phrases = (("case-level random folds", "病例層級的隨機 fold", "症例単位のランダムな fold 分割") if legacy
                                          else ("Inference settings come from the source checkpoint", "推論設定來自來源 checkpoint", "推論設定は元のチェックポイントに由来"))
                    for section, scope_phrase, provenance_phrase in zip(sections, scope_phrases, provenance_phrases):
                        self.assertIn(scope_phrase, section)
                        self.assertIn(provenance_phrase, section)
                        self.assertIn('`["gray_inner", "_vertical"]`', section)
                        self.assertIn('`{"bad": 0, "good": 1}`', section)
                        self.assertIn("resnet18", section)
                        self.assertIn("8", section)
                        self.assertIn(code_url, section)
                        self.assertIn("`config.json`", section)
                    examples = [re.findall(r"```python\n(.*?)\n```", section, re.DOTALL) for section in sections]
                    self.assertEqual(len(examples[0]), 1)
                    self.assertEqual(examples[0], examples[1])
                    self.assertEqual(examples[1], examples[2])
                    compile(examples[0][0], "model-card-example", "exec")


class RenameCommitTests(unittest.TestCase):
    def _stage(self, directory):
        source = directory / "stage"
        source.mkdir()
        (source / "weights.txt").write_text("preserve this completed output")
        return source, directory / "committed"

    def test_transient_windows_lock_retries_without_losing_output(self):
        for code in (5, 32, 33):
            with self.subTest(winerror=code), tempfile.TemporaryDirectory() as temp:
                source, target = self._stage(Path(temp))
                original = Path.rename
                calls = []

                def locked_then_released(path, destination):
                    calls.append(path)
                    if len(calls) < 3:
                        raise _windows_lock(code)
                    return original(path, destination)

                with mock.patch.object(Path, "rename", locked_then_released), \
                        mock.patch.object(exporter.time, "sleep") as wait:
                    exporter._rename_with_retry(source, target)
                self.assertEqual(len(calls), 3)
                self.assertEqual(wait.call_args_list, [mock.call(0.2), mock.call(0.4)])
                self.assertFalse(source.exists())
                self.assertEqual((target / "weights.txt").read_text(), "preserve this completed output")

    def test_permanent_windows_lock_is_bounded_and_preserves_original_error(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self._stage(Path(temp))
            error = _windows_lock()
            with mock.patch.object(Path, "rename", side_effect=error) as rename, \
                    mock.patch.object(exporter.time, "sleep") as wait:
                with self.assertRaises(OSError) as caught:
                    exporter._rename_with_retry(source, target)
            self.assertIs(caught.exception, error)
            self.assertEqual(rename.call_count, 10)
            self.assertEqual(wait.call_count, 9)
            self.assertLess(sum(call.args[0] for call in wait.call_args_list), 10)
            self.assertEqual((source / "weights.txt").read_text(), "preserve this completed output")
            self.assertFalse(target.exists())

    def test_existing_or_appearing_destination_is_never_overwritten(self):
        for appears_during_retry in (False, True):
            with self.subTest(during_retry=appears_during_retry), tempfile.TemporaryDirectory() as temp:
                source, target = self._stage(Path(temp))

                def competing_writer(_delay=None):
                    target.mkdir()
                    (target / "existing.txt").write_text("another completed export")

                if not appears_during_retry:
                    competing_writer()
                with mock.patch.object(Path, "rename", side_effect=_windows_lock()) as rename, \
                        mock.patch.object(exporter.time, "sleep", side_effect=competing_writer):
                    with self.assertRaises(FileExistsError):
                        exporter._rename_with_retry(source, target)
                self.assertEqual(rename.call_count, int(appears_during_retry))
                self.assertEqual((target / "existing.txt").read_text(), "another completed export")
                self.assertTrue((source / "weights.txt").is_file())

    def test_other_os_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self._stage(Path(temp))
            error = PermissionError(13, "permanent permissions problem without Windows lock code")
            with mock.patch.object(Path, "rename", side_effect=error) as rename, \
                    mock.patch.object(exporter.time, "sleep") as wait:
                with self.assertRaises(OSError) as caught:
                    exporter._rename_with_retry(source, target)
            self.assertIs(caught.exception, error)
            self.assertEqual(rename.call_count, 1)
            wait.assert_not_called()
            self.assertTrue(source.is_dir())

    def test_reparse_source_is_rejected_before_rename(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self._stage(Path(temp))
            original = Path.lstat

            def reparse_source(path):
                details = original(path)
                return (SimpleNamespace(st_mode=details.st_mode, st_file_attributes=0x400)
                        if path == source else details)

            with mock.patch.object(Path, "lstat", reparse_source), \
                    mock.patch.object(Path, "rename") as rename:
                with self.assertRaisesRegex(ValueError, "Linked rename"):
                    exporter._rename_with_retry(source, target)
            rename.assert_not_called()
            self.assertTrue(source.is_dir())

    def test_complete_public_staging_is_preserved_after_permanent_lock(self):
        metadata = {
            "public_format_version": 1, "model_config": exporter._safe_model_config(MODEL_CONFIG),
            "streams": ["_vertical"], "class_to_idx": CLASSES,
            "preprocessing": exporter._public_preprocessing(PREPROCESSING, False),
            "artifact_kind": "research", "source_format": "versioned_checkpoint",
            "settings_provenance": "checkpoint_metadata", "legacy_no_split_metadata": False,
            "private_training_metadata_omitted": True,
        }
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            target = directory / "public"
            error = _windows_lock()
            with mock.patch.object(exporter, "read_public_source", return_value=({"weight": torch.ones(1)}, metadata)), \
                    mock.patch.object(exporter, "_validate_state"), \
                    mock.patch.object(Path, "rename", side_effect=error), \
                    mock.patch.object(exporter.time, "sleep"):
                with self.assertRaises(OSError) as caught:
                    exporter.export_public_model("unused-source.pth", target)
            self.assertIs(caught.exception, error)
            self.assertFalse(target.exists())
            stages = list(directory.glob(".public-model-*"))
            self.assertEqual(len(stages), 1)
            self.assertEqual({path.name for path in stages[0].iterdir()}, PUBLIC_FILES)
            from safetensors.torch import load_file
            torch.testing.assert_close(load_file(str(stages[0] / "model.safetensors"))["weight"], torch.ones(1))

    @unittest.skipUnless(os.name == "nt", "Requires Windows file-sharing semantics")
    def test_real_windows_file_handle_lock_releases_before_retry(self):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as temp:
            source, target = self._stage(Path(temp))
            # Permit reads/writes while withholding FILE_SHARE_DELETE, as a
            # temporary scanner lock can do. The parent rename then fails.
            handle = kernel.CreateFileW(str(source / "weights.txt"), 0x80000000, 3,
                                        None, 3, 0x80, None)
            self.assertNotIn(handle, (None, ctypes.c_void_p(-1).value))
            released = threading.Event()

            def release():
                kernel.CloseHandle(handle)
                released.set()

            timer = None
            try:
                with self.assertRaises(OSError) as caught:
                    source.rename(target)
                self.assertEqual(caught.exception.winerror, 5)
                timer = threading.Timer(0.3, release)
                timer.start()
                exporter._rename_with_retry(source, target)
                self.assertTrue(released.is_set())
                self.assertEqual((target / "weights.txt").read_text(), "preserve this completed output")
            finally:
                if timer is not None:
                    timer.cancel()
                    timer.join()
                if not released.is_set():
                    release()


class PublicExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(412)
        cls.single = build_model(MODEL_CONFIG, ["_vertical"], 2).eval()
        cls.fusion = build_model(MODEL_CONFIG, ["_vertical", "color_full"], 2).eval()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_new_single_and_fusion_roundtrip_and_private_metadata_removed(self):
        for model in (self.single, self.fusion):
            with self.subTest(streams=len(model.streams)), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                source = directory / "source.pth"
                bundle = directory / "public"
                torch.save(_source(model, model.streams), source)
                metadata = exporter.export_public_model(source, bundle)
                self.assertEqual({p.name for p in bundle.iterdir()}, PUBLIC_FILES)
                self.assertEqual(set(metadata["model_config"]),
                                 {"name", "hidden_dim", "freeze_backbones", "pretrained"})
                self.assertFalse(metadata["model_config"]["pretrained"])
                self.assertEqual(metadata["artifact_kind"], "research")  # Epoch 1 is not a demo rule.
                for filename in PUBLIC_FILES - {"model.safetensors"}:
                    self.assertNotIn("PRIVATE_", (bundle / filename).read_text(encoding="utf-8"))
                loader = _load_public_module(bundle)
                restored = loader.load_public_model(bundle)
                self.assertEqual(set(restored.state_dict()), set(model.state_dict()))
                self.assertIn(
                    "bn1.num_batches_tracked" if len(model.streams) == 1
                    else "backbones.0.bn1.num_batches_tracked",
                    restored.state_dict(),
                )
                images = torch.randn(2, len(model.streams), 3, 32, 32)
                with torch.no_grad():
                    torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)

    def test_yaml_extends_relative_paths_and_class_mapping_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            child_dir = directory / "variants"
            child_dir.mkdir()
            base = {
                "data": {"streams": ["old"], "class_to_idx": {"old-negative": 0, "old-positive": 1}},
                "model": MODEL_CONFIG, "preprocessing": PREPROCESSING,
                "experiment": {"output_dir": "PRIVATE_CONFIG_PATH"},
            }
            (directory / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
            child = {"extends": "../base.yaml",
                     "data": {"streams": ["_vertical", "color_full"], "class_to_idx": CLASSES}}
            path = child_dir / "fusion.yaml"
            path.write_text(yaml.safe_dump(child), encoding="utf-8")
            resolved = exporter.resolve_legacy_config(path)
            self.assertEqual(resolved["data"]["class_to_idx"], CLASSES)
            self.assertEqual(resolved["model"]["name"], "resnet18")
            self.assertEqual(resolved["preprocessing"], PREPROCESSING)

    def test_yaml_extends_cycle_and_depth_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
            (directory / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cycle"):
                exporter.resolve_legacy_config(directory / "a.yaml")
            for index in range(4):
                content = ("extends: chain%d.yaml\n" % (index + 1)) if index < 3 else "model: {}\n"
                (directory / ("chain%d.yaml" % index)).write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "depth"):
                exporter.resolve_legacy_config(directory / "chain0.yaml", max_depth=3)

    def test_raw_legacy_requires_config_and_extends_config_exports_strictly(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "legacy.pth"
            torch.save(_legacy_state(self.fusion), source)
            with self.assertRaisesRegex(ValueError, "legacy-config"):
                exporter.export_public_model(source, directory / "no-config")
            base = {"model": MODEL_CONFIG,
                    "preprocessing": {**PREPROCESSING, "resize_backend": "tensor", "antialias": False},
                    "data": {"class_to_idx": CLASSES}}
            (directory / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
            config_path = directory / "fusion.yaml"
            config_path.write_text(
                "extends: base.yaml\ndata:\n  streams: [_vertical, color_full]\n",
                encoding="utf-8",
            )
            bundle = directory / "public"
            metadata = exporter.export_public_model(source, bundle, config_path)
            self.assertTrue(metadata["legacy_no_split_metadata"])
            self.assertEqual(metadata["settings_provenance"], "operator_supplied")
            self.assertEqual(metadata["preprocessing"]["resize_backend"], "tensor")
            self.assertFalse(metadata["preprocessing"]["antialias"])
            restored = _load_public_module(bundle).load_public_model(bundle)
            images = torch.randn(1, 2, 3, 32, 32)
            with torch.no_grad():
                torch.testing.assert_close(self.fusion(images), restored(images), rtol=0, atol=0)
            card = (bundle / "README.md").read_text(encoding="utf-8")
            self.assertIn("case-level random folds", card)
            self.assertIn("operator", card)

    def test_eighth_legacy_branch_and_missing_batchnorm_buffer(self):
        translated = exporter._legacy_keys(
            {"b1.bn1.num_batches_tracked": torch.tensor(3),
             "b8.bn1.num_batches_tracked": torch.tensor(7),
             "merger.2.bias": torch.zeros(8), "fc.bias": torch.zeros(2)}, 8,
        )
        self.assertEqual(set(translated),
                         {"backbones.0.bn1.num_batches_tracked",
                          "backbones.7.bn1.num_batches_tracked",
                          "fusion.2.bias", "classifier.bias"})
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "missing-buffer.pth"
            bad = _source(self.single, self.single.streams)
            del bad["state_dict"]["bn1.num_batches_tracked"]
            torch.save(bad, source)
            output = Path(temp) / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "strictly match"):
                exporter.export_public_model(source, output)
            self.assertFalse(output.exists())

    def test_converted_legacy_is_not_remapped_again(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "converted.pth"
            converted = _source(self.fusion, self.fusion.streams)
            converted["legacy_no_split_metadata"] = True
            converted["fold"] = None
            converted["train_patient_ids"] = []
            converted["validation_patient_ids"] = []
            converted["test_patient_ids"] = []
            torch.save(converted, source)
            output = Path(temp) / "public"
            metadata = exporter.export_public_model(source, output)
            self.assertEqual(metadata["source_format"], "converted_legacy")
            self.assertEqual(metadata["preprocessing"]["resize_backend"], "pil")
            self.assertTrue(metadata["preprocessing"]["antialias"])
            restored = _load_public_module(output).load_public_model(output)
            self.assertEqual(set(restored.state_dict()), set(self.fusion.state_dict()))

    def test_smoke_path_forces_demo_and_preview_needs_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "SMOKE_OUTPUT"
            directory.mkdir()
            source = directory / "source.pth"
            torch.save(_source(self.single, self.single.streams), source)
            metadata = exporter.export_public_model(source, None, kind="research", preview=True)
            self.assertEqual(metadata["artifact_kind"], "demo")
            self.assertEqual({p.name for p in directory.iterdir()}, {"source.pth"})

    def test_public_pil_preprocessing_matches_new_case_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            image_path = directory / "image.png"
            Image.new("RGB", (60, 48), (55, 120, 200)).save(image_path)
            sample = {"sample_id": "synthetic-case", "patient_id": "synthetic-group",
                      "label": 0, "images": {"_vertical": str(image_path)}}
            expected = CaseDataset([sample], ["_vertical"], PREPROCESSING, training=False)[0]["images"]
            loader = exporter._reference_loader()
            preprocessing = exporter._public_preprocessing(PREPROCESSING, legacy=False)
            actual = loader["preprocess_streams"]([image_path], preprocessing)[0]
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_raw_legacy_requires_explicit_backend_and_antialias(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "legacy.pth"
            torch.save(_legacy_state(self.fusion), source)
            config = {"model": MODEL_CONFIG, "preprocessing": PREPROCESSING,
                      "data": {"streams": list(self.fusion.streams), "class_to_idx": CLASSES}}
            config_path = directory / "profile.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicit resize_backend and antialias"):
                exporter.export_public_model(source, directory / "missing-both", config_path)
            config["preprocessing"] = {**PREPROCESSING, "resize_backend": "tensor"}
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicit resize_backend and antialias"):
                exporter.export_public_model(source, directory / "missing-antialias", config_path)

    def test_public_tensor_preprocessing_respects_both_antialias_versions(self):
        from torchvision.transforms import functional as F
        from torchvision.transforms import InterpolationMode
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pattern.png"
            pixels = np.random.default_rng(19).integers(0, 256, (81, 107, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            loader = exporter._reference_loader()
            results = []
            for antialias in (False, True):
                preprocessing = exporter._public_preprocessing(
                    {**PREPROCESSING, "resize_backend": "tensor", "antialias": antialias}, True,
                )
                with Image.open(path) as image:
                    expected = F.resize(
                        F.to_tensor(image.convert("RGB")), PREPROCESSING["resize_size"],
                        interpolation=InterpolationMode.BILINEAR, antialias=antialias,
                    )
                expected = F.center_crop(expected, [PREPROCESSING["image_size"]] * 2)
                expected = F.normalize(expected, PREPROCESSING["mean"], PREPROCESSING["std"])
                actual = loader["preprocess_streams"]([path], preprocessing)[0, 0]
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                results.append(actual)
            self.assertFalse(torch.allclose(results[0], results[1]))


if __name__ == "__main__":
    unittest.main()

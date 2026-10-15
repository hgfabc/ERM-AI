"""Offline model, strict checkpoint and split-provenance regression tests."""

import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from multistream.models import (
    MultiStreamClassifier, SingleStreamClassifier, build_model,
    convert_legacy_checkpoint, initialize_from_checkpoints, load_checkpoint,
    model_from_checkpoint, validate_checkpoint, validate_warmstart_checkpoint,
)


CONFIG = {"name": "resnet18", "pretrained": False, "hidden_dim": 8}
CONTEXT = {
    "fold": 0,
    "manifest_fingerprint": "manifest-v1",
    "split_fingerprint": "split-v1",
    "train_patient_ids": ["patient-a", "patient-b"],
    "validation_patient_ids": ["patient-c"],
    "test_patient_ids": ["patient-d"],
}


def checkpoint_for(model, streams):
    return {
        "format_version": 1,
        "state_dict": model.state_dict(),
        "model_config": copy.deepcopy(CONFIG),
        "streams": list(streams),
        "class_to_idx": {"bad": 0, "good": 1},
        "preprocessing": {"image_size": 32, "resize_size": 32},
        "epoch": 2,
        **copy.deepcopy(CONTEXT),
    }


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_single_and_arbitrary_stream_counts(self):
        for count in (1, 2, 3, 4, 5, 6, 7, 8):
            with self.subTest(count=count):
                streams = ["stream-%s" % i for i in range(count)]
                model = build_model(CONFIG, streams, 2).eval()
                with torch.no_grad():
                    result = model(torch.randn(1, count, 3, 32, 32))
                self.assertEqual(tuple(result.shape), (1, 2))
                self.assertEqual(model.streams, tuple(streams))
                if count == 1:
                    self.assertIsInstance(model, SingleStreamClassifier)
                    self.assertIn("fc.weight", model.state_dict())
                else:
                    self.assertIsInstance(model, MultiStreamClassifier)
                    self.assertEqual(len(model.backbones), count)
                    self.assertEqual(model.fusion[0].in_features, count * 512)
                    self.assertIsNot(model.backbones[0], model.backbones[1])

    def test_input_contract_rejects_wrong_stream_count_and_rank(self):
        model = build_model(CONFIG, ["a", "b"], 2)
        for images in (torch.zeros(1, 3, 32, 32),
                       torch.zeros(1, 1, 3, 32, 32),
                       torch.zeros(1, 2, 1, 32, 32)):
            with self.assertRaises(ValueError):
                model(images)
        for streams in ([], ["a", "a"], ["a"] * 9, [""]):
            with self.assertRaises(ValueError):
                build_model(CONFIG, streams, 2)

    def test_backbones_follow_declared_stream_order(self):
        class MeanFeature(nn.Module):
            def forward(self, image):
                return image.mean(dim=(1, 2, 3)).unsqueeze(1)

        model = build_model(CONFIG, ["outer", "inner"], 2)
        model.backbones = nn.ModuleList([MeanFeature(), MeanFeature()])
        model.fusion = nn.Identity()
        model.classifier = nn.Identity()
        images = torch.stack((torch.ones(1, 3, 8, 8) * 7,
                              torch.ones(1, 3, 8, 8) * 2), dim=1)
        self.assertTrue(torch.equal(model(images), torch.tensor([[7., 2.]])))

    def test_frozen_backbones_keep_batchnorm_statistics_and_train_head(self):
        cfg = {**CONFIG, "freeze_backbones": True}
        model = build_model(cfg, ["a", "b"], 2).train()
        batchnorm = next(m for m in model.backbones[0].modules()
                         if isinstance(m, nn.BatchNorm2d))
        before_mean = batchnorm.running_mean.clone()
        before_count = batchnorm.num_batches_tracked.clone()
        head_before = model.classifier.weight.detach().clone()
        optimizer = torch.optim.SGD(
            (p for p in model.parameters() if p.requires_grad), lr=0.1
        )
        result = model(torch.randn(2, 2, 3, 32, 32))
        nn.functional.cross_entropy(result, torch.tensor([0, 1])).backward()
        optimizer.step()
        self.assertTrue(torch.equal(before_mean, batchnorm.running_mean))
        self.assertTrue(torch.equal(before_count, batchnorm.num_batches_tracked))
        self.assertFalse(torch.equal(head_before, model.classifier.weight))
        self.assertFalse(any(p.requires_grad for p in model.backbones.parameters()))
        self.assertTrue(model.fusion.training)
        model.eval().train()
        self.assertFalse(batchnorm.training)

    def test_unfrozen_backbones_train_and_model_instances_are_independent(self):
        first = build_model(CONFIG, ["a", "b"], 2).train()
        second = build_model(CONFIG, ["a", "b"], 2)
        self.assertIsNot(first.backbones[0].conv1.weight, second.backbones[0].conv1.weight)
        batchnorm = first.backbones[0].bn1
        count = batchnorm.num_batches_tracked.clone()
        first(torch.randn(2, 2, 3, 32, 32))
        self.assertEqual(int(batchnorm.num_batches_tracked), int(count) + 1)
        self.assertTrue(all(p.requires_grad for p in first.backbones.parameters()))

    def test_frozen_single_classifier_keeps_head_trainable_and_batchnorm_fixed(self):
        model = build_model({**CONFIG, "freeze_backbones": True}, ["a"], 2).train()
        before = model.network.bn1.num_batches_tracked.clone()
        output = model(torch.randn(2, 1, 3, 32, 32))
        output.sum().backward()
        self.assertTrue(torch.equal(before, model.network.bn1.num_batches_tracked))
        self.assertFalse(model.network.bn1.training)
        self.assertFalse(model.network.conv1.weight.requires_grad)
        self.assertTrue(model.network.fc.weight.requires_grad)
        self.assertIsNotNone(model.network.fc.weight.grad)

    def test_strict_reconstruction_uses_metadata_without_initialization(self):
        model = build_model(CONFIG, ["a", "b"], 2).eval()
        checkpoint = checkpoint_for(model, ["a", "b"])
        checkpoint["model_config"]["pretrained"] = True
        checkpoint["model_config"]["initialization"] = {"a": "nonexistent.pth"}
        restored = model_from_checkpoint(checkpoint)
        self.assertFalse(restored.training)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))
        del checkpoint["state_dict"]["classifier.bias"]
        with self.assertRaises(RuntimeError):
            model_from_checkpoint(checkpoint)

    def test_batchnorm_missing_buffer_is_not_silently_repaired(self):
        checkpoint = checkpoint_for(build_model(CONFIG, ["a"], 2), ["a"])
        del checkpoint["state_dict"]["bn1.num_batches_tracked"]
        with self.assertRaisesRegex(RuntimeError, "bn1.num_batches_tracked"):
            model_from_checkpoint(checkpoint)

    def test_schema_rejects_missing_metadata_and_patient_overlap(self):
        model = build_model(CONFIG, ["a"], 2)
        checkpoint = checkpoint_for(model, ["a"])
        validate_checkpoint(checkpoint)
        missing = dict(checkpoint)
        del missing["split_fingerprint"]
        with self.assertRaisesRegex(ValueError, "missing metadata"):
            validate_checkpoint(missing)
        overlapping = dict(checkpoint)
        overlapping["validation_patient_ids"] = ["patient-a"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_checkpoint(overlapping)

    def test_warmstart_rejects_wrong_fold_split_patients_and_stream(self):
        checkpoint = checkpoint_for(build_model(CONFIG, ["a"], 2), ["a"])
        validate_warmstart_checkpoint(checkpoint, CONTEXT, "a")
        for field, wrong in (("fold", 1), ("manifest_fingerprint", "other"),
                             ("split_fingerprint", "other"),
                             ("train_patient_ids", ["different-patient"])):
            bad_context = {**CONTEXT, field: wrong}
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_warmstart_checkpoint(checkpoint, bad_context, "a")
        with self.assertRaises(ValueError):
            validate_warmstart_checkpoint(checkpoint, CONTEXT, "b")
        with self.assertRaises(ValueError):
            validate_warmstart_checkpoint(checkpoint, None, "a")

    def test_warmstart_loads_named_branch_and_requires_context(self):
        source = build_model(CONFIG, ["b"], 2)
        with torch.no_grad():
            source.network.conv1.weight.fill_(0.125)
        checkpoint = checkpoint_for(source, ["b"])
        target = build_model(CONFIG, ["a", "b"], 2)
        with patch("multistream.models.load_checkpoint", return_value=checkpoint):
            initialize_from_checkpoints(target, {"b": "source.pth"}, CONTEXT)
        self.assertTrue(torch.equal(target.backbones[1].conv1.weight,
                                    source.network.conv1.weight))
        self.assertFalse(torch.equal(target.backbones[0].conv1.weight,
                                     source.network.conv1.weight))
        with patch("multistream.models.load_checkpoint", return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, "initialization_context"):
                build_model({**CONFIG, "initialization": {"b": "source.pth"}},
                            ["a", "b"], 2)

    def test_warmstart_rejects_label_semantics_crop_and_architecture_changes(self):
        checkpoint = checkpoint_for(build_model(CONFIG, ["a"], 2), ["a"])
        for field, wrong in (("class_to_idx", {"good": 0, "bad": 1}),
                             ("preprocessing", {"image_size": 64, "resize_size": 64})):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_warmstart_checkpoint(checkpoint, {**CONTEXT, field: wrong}, "a")
        checkpoint["model_config"] = {**CONFIG, "name": "resnet34"}
        target = build_model(CONFIG, ["a", "b"], 2)
        with patch("multistream.models.load_checkpoint", return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, "architecture"):
                initialize_from_checkpoints(target, {"a": "source.pth"}, CONTEXT)

    def test_checkpoint_load_explicitly_requests_weights_only(self):
        checkpoint = checkpoint_for(build_model(CONFIG, ["a"], 2), ["a"])
        with patch("multistream.models.torch.load", return_value=checkpoint) as mocked:
            self.assertIs(load_checkpoint("fake.pth"), checkpoint)
        self.assertTrue(mocked.call_args.kwargs["weights_only"])
        with patch("multistream.models.torch.load",
                   side_effect=TypeError("unexpected keyword argument 'weights_only'")):
            with self.assertRaisesRegex(RuntimeError, "unsafe pickle loading is disabled"):
                load_checkpoint("fake.pth")

    @unittest.skipUnless("weights_only" in inspect.signature(torch.load).parameters,
                         "requires PyTorch safe weights-only loading")
    def test_real_safe_checkpoint_roundtrip(self):
        checkpoint = checkpoint_for(build_model(CONFIG, ["a"], 2), ["a"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save(checkpoint, path)
            restored = model_from_checkpoint(load_checkpoint(path))
            self.assertEqual(restored.streams, ("a",))

    def test_legacy_conversion_is_strict_and_cannot_warmstart(self):
        model = build_model(CONFIG, ["a", "b"], 2)
        legacy = {}
        for key, tensor in model.state_dict().items():
            if key.startswith("backbones."):
                _, index, remainder = key.split(".", 2)
                legacy["b%s.%s" % (int(index) + 1, remainder)] = tensor
            elif key.startswith("fusion."):
                legacy["merger." + key[len("fusion."):]] = tensor
            elif key.startswith("classifier."):
                legacy["fc." + key[len("classifier."):]] = tensor
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "converted.pth"
            with patch("multistream.models._safe_torch_load", return_value=legacy):
                converted = convert_legacy_checkpoint(
                    "legacy.pth", path, ["a", "b"], CONFIG,
                    {"bad": 0, "good": 1}, {"image_size": 32},
                )
            self.assertTrue(path.is_file())
            self.assertTrue(converted["legacy_no_split_metadata"])
            with self.assertRaisesRegex(ValueError, "cannot initialize CV"):
                validate_warmstart_checkpoint(converted, CONTEXT, "a")
            del legacy["fc.bias"]
            with patch("multistream.models._safe_torch_load", return_value=legacy):
                with self.assertRaises(RuntimeError):
                    convert_legacy_checkpoint(
                        "legacy.pth", Path(directory) / "bad.pth", ["a", "b"],
                        CONFIG, {"bad": 0, "good": 1}, {"image_size": 32},
                    )
            self.assertFalse((Path(directory) / "bad.pth").exists())

    def test_legacy_single_keeps_exact_timm_keys(self):
        source = build_model(CONFIG, ["a"], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single.pth"
            with patch("multistream.models._safe_torch_load", return_value=source.state_dict()):
                converted = convert_legacy_checkpoint(
                    "single-legacy.pth", path, ["a"], CONFIG,
                    {"bad": 0, "good": 1}, {"image_size": 32},
                )
            restored = model_from_checkpoint(converted)
            self.assertEqual(set(source.state_dict()), set(restored.state_dict()))
            self.assertTrue(torch.equal(source.network.fc.weight, restored.network.fc.weight))


if __name__ == "__main__":
    unittest.main()

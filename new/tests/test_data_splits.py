"""Data-integrity, patient-isolation, and preprocessing regression tests."""

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from multistream.data import (
    CaseDataset, load_manifest, make_loader, manifest_fingerprint,
    save_manifest, scan_dataset, select_samples, validate_manifest,
)
from multistream.splits import make_splits, save_splits, load_splits, validate_splits


def image(path, size=(48, 32), gradient=False):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, (50, 100, 150))
    if gradient:
        img.putdata([(x * 5 % 256, y * 7 % 256, (x + y) * 3 % 256)
                     for y in range(size[1]) for x in range(size[0])])
    img.save(path)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def case(self, name="P001_visit_OS", label="bad", streams=("gray_outer",), gradient=False):
        folder = self.root / label / name
        for stream in streams:
            image(folder / ("image" + stream + ".png"), gradient=gradient)
        return folder

    def test_sorted_scan_and_stable_fingerprint(self):
        self.case("P002_visit_OD", "good")
        self.case("P001_visit_OS", "bad")
        manifest = scan_dataset(self.root, ["gray_outer"])
        self.assertEqual([s["sample_id"] for s in manifest["samples"]], ["P001_visit_OS", "P002_visit_OD"])
        self.assertEqual(manifest["samples"][0]["patient_id"], "P001")
        self.assertEqual(manifest["samples"][0]["label"], 0)
        self.assertTrue(Path(manifest["samples"][0]["images"]["gray_outer"]).is_absolute())
        self.assertEqual(manifest["fingerprint"], scan_dataset(self.root, ["gray_outer"])["fingerprint"])

    def test_missing_stream_refuses_or_reports_explicitly(self):
        self.case("P001_visit_OS", streams=("gray_outer",))
        self.case("P002_visit_OS", streams=("gray_outer", "gray_inner"))
        with self.assertRaisesRegex(ValueError, "missing selected streams"):
            scan_dataset(self.root, ["gray_outer", "gray_inner"])
        manifest = scan_dataset(self.root, ["gray_outer", "gray_inner"], require_all_streams=False)
        self.assertEqual(len(manifest["samples"]), 1)
        self.assertEqual(manifest["scan_report"]["incomplete_cases"], 1)
        self.assertEqual(manifest["scan_report"]["skipped_cases"][0]["missing_streams"], ["gray_inner"])
        (self.root / "bad" / "P002_visit_OS" / "imagegray_inner.png").unlink()
        with self.assertRaisesRegex(ValueError, "no complete cases.*2 incomplete"):
            scan_dataset(self.root, ["gray_outer", "gray_inner"], require_all_streams=False)

    def test_non_images_ignored_and_ambiguity_refused(self):
        folder = self.case()
        (folder / "notes_gray_outer.txt").write_text("not an image")
        self.assertEqual(len(scan_dataset(self.root, ["gray_outer"])["samples"]), 1)
        image(folder / "second_gray_outer.jpg")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            scan_dataset(self.root, ["gray_outer"])

    def test_conflicting_class_copy_refused(self):
        self.case(label="bad")
        self.case(label="good")
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            scan_dataset(self.root, ["gray_outer"])

    def test_flat_unlabeled_cases_require_opt_in(self):
        image(self.root / "P001_visit_OS" / "image_gray_outer.png")
        with self.assertRaises(ValueError):
            scan_dataset(self.root, ["gray_outer"])
        manifest = scan_dataset(self.root, ["gray_outer"], allow_unlabeled=True)
        self.assertIsNone(manifest["samples"][0]["label"])

    def test_relative_paths_formal_patient_id_and_round_trip(self):
        folder = self.case()
        manifest = scan_dataset(self.root, ["gray_outer"])
        manifest.pop("fingerprint")
        manifest["samples"][0]["patient_id"] = "verified-subject-A"
        manifest["samples"][0]["images"]["gray_outer"] = str((folder / "imagegray_outer.png").relative_to(self.root))
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = load_manifest(path)
        self.assertEqual(loaded["samples"][0]["patient_id"], "verified-subject-A")
        self.assertTrue(Path(loaded["samples"][0]["images"]["gray_outer"]).is_absolute())
        saved = save_manifest(loaded, self.root / "saved.json")
        self.assertEqual(saved["fingerprint"], load_manifest(self.root / "saved.json")["fingerprint"])

    def test_metadata_tamper_missing_path_duplicate_id_and_bool_label_refused(self):
        self.case()
        manifest = scan_dataset(self.root, ["gray_outer"])
        changed = copy.deepcopy(manifest)
        changed["samples"][0]["patient_id"] = "other"
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            validate_manifest(changed)
        changed = copy.deepcopy(manifest)
        changed["samples"].append(copy.deepcopy(changed["samples"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            validate_manifest(changed)
        changed = copy.deepcopy(manifest)
        changed["samples"][0]["label"] = True
        with self.assertRaisesRegex(ValueError, "known integer class"):
            validate_manifest(changed)
        Path(manifest["samples"][0]["images"]["gray_outer"]).unlink()
        with self.assertRaisesRegex(ValueError, "does not exist"):
            validate_manifest(manifest)

    def test_selection_order_and_unknown_or_duplicate_rejection(self):
        self.case("P001_OS")
        self.case("P002_OS")
        manifest = scan_dataset(self.root, ["gray_outer"])
        ids = [s["sample_id"] for s in reversed(manifest["samples"])]
        self.assertEqual([s["sample_id"] for s in select_samples(manifest, ids)], ids)
        with self.assertRaisesRegex(ValueError, "unknown"):
            select_samples(manifest, ["unknown"])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            select_samples(manifest, [ids[0], ids[0]])

    def test_shared_image_cannot_be_hidden_under_different_sample_ids(self):
        self.case("P001_OS")
        self.case("P002_OS")
        manifest = scan_dataset(self.root, ["gray_outer"])
        manifest.pop("fingerprint")
        manifest["samples"][1]["images"] = dict(manifest["samples"][0]["images"])
        with self.assertRaisesRegex(ValueError, "reused across multiple samples"):
            validate_manifest(manifest, require_fingerprint=False)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        samples = []
        for pid in range(60):
            for eye in ("OS", "OD"):
                fixture = self.root / "P{:03d}_{}_gray_outer.png".format(pid, eye)
                image(fixture)
                samples.append({"sample_id": "P{:03d}_visit_{}".format(pid, eye),
                                "patient_id": "P{:03d}".format(pid), "label": pid % 2,
                                "images": {"gray_outer": str(fixture)}})
        self.manifest = validate_manifest({"version": 1, "class_to_idx": {"bad": 0, "good": 1},
                                           "streams": ["gray_outer"], "samples": samples},
                                          require_fingerprint=False)
        self.splits = make_splits(self.manifest, n_folds=5, val_fraction=0.25, seed=42)

    def tearDown(self):
        self.temp.cleanup()

    def test_nested_patient_isolation_and_outer_test_exactly_once(self):
        mapping = {s["sample_id"]: s for s in self.manifest["samples"]}
        seen = Counter()
        for fold in self.splits["folds"]:
            sets = [{mapping[s]["patient_id"] for s in fold[name]} for name in ("train", "validation", "test")]
            self.assertFalse(sets[0] & sets[1])
            self.assertFalse(sets[0] & sets[2])
            self.assertFalse(sets[1] & sets[2])
            for name in ("train", "validation", "test"):
                self.assertEqual({mapping[s]["label"] for s in fold[name]}, {0, 1})
            seen.update(fold["test"])
        self.assertEqual(set(seen), set(mapping))
        self.assertEqual(set(seen.values()), {1})

    def test_seeded_repeatability_and_round_trip(self):
        self.assertEqual(self.splits, make_splits(self.manifest, n_folds=5, val_fraction=0.25, seed=42))
        path = self.root / "splits.json"
        save_splits(self.splits, path)
        self.assertEqual(self.splits, load_splits(path, self.manifest))

    def test_same_patient_across_different_samples_rejected(self):
        changed = copy.deepcopy(self.splits)
        fold = changed["folds"][0]
        sid = fold["train"][0]
        mate = sid[:-2] + ("OS" if sid.endswith("OD") else "OD")
        fold["train"].remove(mate)
        fold["validation"].append(mate)
        changed.pop("fingerprint")
        with self.assertRaisesRegex(ValueError, "patient overlap"):
            validate_splits(self.manifest, changed)

    def test_invalid_coverage_duplicate_fold_wrong_manifest_and_hash_rejected(self):
        changed = copy.deepcopy(self.splits)
        changed["folds"][0]["train"].pop()
        with self.assertRaisesRegex(ValueError, "partition all"):
            validate_splits(self.manifest, changed)
        changed = copy.deepcopy(self.splits)
        changed["folds"][1]["fold"] = 0
        with self.assertRaisesRegex(ValueError, "unique and contiguous"):
            validate_splits(self.manifest, changed)
        changed = copy.deepcopy(self.splits)
        changed["manifest_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "different manifest"):
            validate_splits(self.manifest, changed)
        changed = copy.deepcopy(self.splits)
        changed["folds"][0]["train"].reverse()
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            validate_splits(self.manifest, changed)

    def test_too_few_patients_and_unlabeled_data_refused(self):
        with self.assertRaisesRegex(ValueError, "number of patients"):
            make_splits(self.manifest, n_folds=61)
        with self.assertRaisesRegex(ValueError, "distinct patients"):
            make_splits(self.manifest, n_folds=31)
        changed = copy.deepcopy(self.manifest)
        changed.pop("fingerprint")
        changed["samples"][0]["label"] = None
        changed["fingerprint"] = manifest_fingerprint(changed)
        with self.assertRaisesRegex(ValueError, "labeled"):
            make_splits(changed, n_folds=5)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        paths = {}
        for stream in ("gray_outer", "gray_inner"):
            path = self.root / (stream + ".png")
            image(path, gradient=True)
            paths[stream] = str(path)
        self.sample = {"sample_id": "case", "patient_id": "subject", "label": None, "images": paths}
        self.streams = ["gray_outer", "gray_inner"]
        self.preprocessing = {"resize_size": 40, "image_size": 24,
                              "mean": [0, 0, 0], "std": [1, 1, 1]}

    def tearDown(self):
        self.temp.cleanup()

    def test_validation_is_deterministic_rgb_and_uses_image_size(self):
        import torch
        dataset = CaseDataset([self.sample], self.streams, self.preprocessing)
        first, second = dataset[0], dataset[0]
        self.assertEqual(tuple(first["images"].shape), (2, 3, 24, 24))
        self.assertTrue(torch.equal(first["images"], second["images"]))
        self.assertEqual(first["label"], -1)

    def test_training_geometry_is_shared_and_loader_seed_repeatable(self):
        import torch
        first = make_loader([self.sample] * 3, self.streams, self.preprocessing, 2, 0, True, 19)
        second = make_loader([self.sample] * 3, self.streams, self.preprocessing, 2, 0, True, 19)
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a["images"], b["images"]))
            self.assertTrue(torch.equal(a["images"][:, 0], a["images"][:, 1]))


if __name__ == "__main__":
    unittest.main()

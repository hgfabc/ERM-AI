"""Validated case manifests and synchronized multi-stream image loading."""

import copy
import hashlib
import json
import math
import random
from pathlib import Path

STREAMS = [
    "_vertical", "_horizontal", "color_full", "color_inner",
    "color_outer", "gray_full", "gray_inner", "gray_outer",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CLASSES = {"bad": 0, "good": 1}


def _digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_streams(streams):
    if not isinstance(streams, (list, tuple)) or not streams:
        raise ValueError("streams must be a nonempty list")
    if any(not isinstance(s, str) or s not in STREAMS for s in streams):
        raise ValueError("streams contain an unsupported modality")
    if len(set(streams)) != len(streams):
        raise ValueError("streams must be unique")
    return list(streams)


def _canonical_manifest(manifest, base_dir=None, check_paths=True):
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("manifest version must be 1")
    streams = _validate_streams(manifest.get("streams"))
    classes = manifest.get("class_to_idx")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("class_to_idx must be a nonempty mapping")
    if any(not isinstance(k, str) or not k or type(v) is not int or v < 0
           for k, v in classes.items()):
        raise ValueError("class_to_idx requires nonempty names and nonnegative integer labels")
    if sorted(classes.values()) != list(range(len(classes))):
        raise ValueError("class_to_idx labels must be unique and contiguous from 0")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest samples must be a nonempty list")
    seen, seen_image_paths = set(), set()
    canonical_samples = []
    base = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("each sample must be a mapping")
        sid, pid, label = sample.get("sample_id"), sample.get("patient_id"), sample.get("label")
        if not isinstance(sid, str) or not sid.strip() or not isinstance(pid, str) or not pid.strip():
            raise ValueError("sample_id and patient_id must be nonempty strings")
        if sid in seen:
            raise ValueError("duplicate sample_id; resolve duplicate or conflicting case annotations")
        seen.add(sid)
        if "label" not in sample or (label is not None and (type(label) is not int or label not in classes.values())):
            raise ValueError("sample label must be null or a known integer class")
        images = sample.get("images")
        if not isinstance(images, dict) or set(images) != set(streams):
            raise ValueError("every sample must provide exactly the manifest streams")
        resolved = {}
        for stream in streams:
            value = images[stream]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("image paths must be nonempty strings")
            path = Path(value)
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError("image paths must use a supported raster image extension")
            if check_paths and not path.is_file():
                raise ValueError("manifest references an image that does not exist: {}".format(path))
            resolved[stream] = str(path)
        if len(set(resolved.values())) != len(streams):
            raise ValueError("a sample reuses one image path for multiple streams")
        if seen_image_paths.intersection(resolved.values()):
            raise ValueError("an image path is reused across multiple samples")
        seen_image_paths.update(resolved.values())
        canonical_samples.append({"sample_id": sid, "patient_id": pid,
                                  "label": label, "images": resolved})
    return {"version": 1, "class_to_idx": dict(sorted(classes.items())),
            "streams": streams,
            "samples": sorted(canonical_samples, key=lambda s: s["sample_id"])}


def manifest_fingerprint(manifest, base_dir=None):
    """Hash canonical metadata, including resolved image paths; not image bytes."""
    return _digest(_canonical_manifest(manifest, base_dir, check_paths=False))


def validate_manifest(manifest, base_dir=None, require_fingerprint=True):
    canonical = _canonical_manifest(manifest, base_dir, check_paths=True)
    fingerprint = _digest(canonical)
    provided = manifest.get("fingerprint")
    if provided is None and require_fingerprint:
        raise ValueError("manifest fingerprint is missing")
    if provided is not None and provided != fingerprint:
        raise ValueError("manifest fingerprint mismatch; metadata changed")
    canonical["fingerprint"] = fingerprint
    if "scan_report" in manifest:
        canonical["scan_report"] = copy.deepcopy(manifest["scan_report"])
    return canonical


def scan_dataset(root, streams, class_to_idx=None, require_all_streams=True,
                 allow_unlabeled=False):
    """Scan class/case/image folders in sorted order.

    The case directory name is sample_id. Its first underscore-delimited
    token is the inferred patient_id. A manually authored manifest can replace
    that inference with a verified patient identifier. With require_all_streams
    False, incomplete cases are explicitly reported and omitted. Flat case
    folders are supported only with allow_unlabeled=True.
    """
    streams = _validate_streams(streams)
    classes = dict(DEFAULT_CLASSES if class_to_idx is None else class_to_idx)
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("dataset root must be an existing directory")
    dirs = sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
                  key=lambda p: p.name)
    class_layout = any(p.name in classes for p in dirs)
    cases = []
    if class_layout:
        unknown = [p for p in dirs if p.name not in classes]
        if unknown:
            raise ValueError("dataset contains unknown class directories")
        for cls in dirs:
            cases.extend((case, classes[cls.name]) for case in sorted(cls.iterdir(), key=lambda p: p.name)
                         if case.is_dir() and not case.name.startswith("."))
    elif allow_unlabeled:
        cases = [(case, None) for case in dirs]
    else:
        raise ValueError("expected class/case/image directories; flat unlabeled data requires allow_unlabeled")
    samples, incomplete, seen = [], [], {}
    for case, label in cases:
        if case.name in seen:
            if seen[case.name] != label:
                raise ValueError("same sample_id appears under conflicting labels")
            raise ValueError("duplicate sample_id")
        seen[case.name] = label
        files = sorted((p for p in case.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                       key=lambda p: p.name)
        images, missing = {}, []
        for stream in streams:
            candidates = [p for p in files if stream.casefold() in p.stem.casefold()]
            if len(candidates) > 1:
                raise ValueError("ambiguous stream image: multiple candidates for {}".format(stream))
            if not candidates:
                missing.append(stream)
            else:
                images[stream] = str(candidates[0].resolve())
        if len(set(images.values())) != len(images):
            raise ValueError("one image matches multiple selected streams")
        if missing:
            incomplete.append({"sample_id": case.name, "missing_streams": missing})
            continue
        patient = case.name.split("_", 1)[0]
        if not patient:
            raise ValueError("cannot infer a patient_id from the case directory")
        samples.append({"sample_id": case.name, "patient_id": patient,
                        "label": label, "images": images})
    if incomplete and require_all_streams:
        missing_counts = {s: sum(s in c["missing_streams"] for c in incomplete) for s in streams}
        raise ValueError("missing selected streams in {} case(s): {}".format(len(incomplete), missing_counts))
    if not samples:
        raise ValueError("dataset contains no complete cases for the selected streams ({} incomplete cases)".format(len(incomplete)))
    manifest = {"version": 1, "class_to_idx": classes, "streams": streams, "samples": samples,
                "scan_report": {"incomplete_cases": len(incomplete), "skipped_cases": incomplete}}
    return validate_manifest(manifest, require_fingerprint=False)


def load_manifest(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    # A hand-authored manifest may omit the hash. An existing hash is always
    # checked, so intentional metadata edits must explicitly clear it first.
    return validate_manifest(manifest, base_dir=path.parent, require_fingerprint=False)


def save_manifest(manifest, path):
    path = Path(path).resolve()
    normalized = validate_manifest(manifest, base_dir=path.parent, require_fingerprint=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return normalized


def select_samples(manifest, sample_ids):
    if not isinstance(sample_ids, (list, tuple)) or any(not isinstance(s, str) for s in sample_ids):
        raise ValueError("sample_ids must be a list of strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids contain duplicates")
    mapping = {s["sample_id"]: s for s in manifest["samples"]}
    if any(s not in mapping for s in sample_ids):
        raise ValueError("sample_ids contain unknown samples")
    return [mapping[s] for s in sample_ids]


class CaseDataset:
    """Duck-typed torch Dataset; importing this module does not import torch."""

    def __init__(self, samples, streams, preprocessing, training=False):
        self.samples = list(samples)
        self.streams = _validate_streams(streams)
        self.preprocessing = dict(preprocessing or {})
        self.training = bool(training)
        self.image_size = self.preprocessing.get("image_size", 224)
        self.resize_size = self.preprocessing.get("resize_size", 256)
        if (type(self.image_size) is not int or type(self.resize_size) is not int
                or self.image_size < 1 or self.resize_size < self.image_size):
            raise ValueError("resize_size and image_size must be positive integers, resize_size >= image_size")
        self.mean = list(self.preprocessing.get("mean", [0.485, 0.456, 0.406]))
        self.std = list(self.preprocessing.get("std", [0.229, 0.224, 0.225]))
        if len(self.mean) != 3 or len(self.std) != 3 or any(not math.isfinite(float(x)) for x in self.mean + self.std) or any(x <= 0 for x in self.std):
            raise ValueError("RGB mean/std must each have three finite values and std must be positive")
        self.scale = self.preprocessing.get("crop_scale", [0.8, 1.0])
        self.ratio = self.preprocessing.get("crop_ratio", [0.9, 1.1])
        self.flip_probability = self.preprocessing.get("horizontal_flip_probability", 0.5)
        if (len(self.scale) != 2 or not 0 < self.scale[0] <= self.scale[1] <= 1
                or len(self.ratio) != 2 or not 0 < self.ratio[0] <= self.ratio[1]
                or not 0 <= self.flip_probability <= 1):
            raise ValueError("invalid crop_scale, crop_ratio, or flip probability")
        self.seed = 0
        self._random = None
        for sample in self.samples:
            if any(stream not in sample.get("images", {}) for stream in self.streams):
                raise ValueError("dataset sample is missing a selected stream")

    def __len__(self):
        return len(self.samples)

    def _rng(self):
        if self._random is None:
            from torch.utils.data import get_worker_info
            worker = get_worker_info()
            self._random = random.Random(self.seed if worker is None else worker.seed)
        return self._random

    def _crop(self, width, height):
        rng = self._rng()
        area = width * height
        for _ in range(10):
            target = area * rng.uniform(*self.scale)
            aspect = math.exp(rng.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            w, h = round(math.sqrt(target * aspect)), round(math.sqrt(target / aspect))
            if 0 < w <= width and 0 < h <= height:
                return rng.randint(0, height - h), rng.randint(0, width - w), h, w
        aspect = width / height
        if aspect < self.ratio[0]:
            w, h = width, round(width / self.ratio[0])
        elif aspect > self.ratio[1]:
            h, w = height, round(height * self.ratio[1])
        else:
            w, h = width, height
        return (height - h) // 2, (width - w) // 2, h, w

    def __getitem__(self, index):
        import torch
        from PIL import Image
        from torchvision.transforms import functional as F
        sample = self.samples[index]
        images = []
        for stream in self.streams:
            with Image.open(sample["images"][stream]) as image:
                images.append(F.resize(image.convert("RGB"), self.resize_size))
        if self.training:
            ref_w, ref_h = images[0].size
            top, left, height, width = self._crop(ref_w, ref_h)
            flip = self._rng().random() < self.flip_probability
            transformed = []
            for image in images:
                w, h = image.size
                y, x = round(top / ref_h * h), round(left / ref_w * w)
                ch = min(h - y, max(1, round(height / ref_h * h)))
                cw = min(w - x, max(1, round(width / ref_w * w)))
                image = F.resized_crop(image, y, x, ch, cw, [self.image_size, self.image_size])
                transformed.append(F.hflip(image) if flip else image)
            images = transformed
        else:
            images = [F.center_crop(image, [self.image_size, self.image_size]) for image in images]
        tensors = [F.normalize(F.to_tensor(image), self.mean, self.std) for image in images]
        label = sample.get("label")
        return {"images": torch.stack(tensors), "label": -1 if label is None else int(label),
                "sample_id": sample["sample_id"], "patient_id": sample["patient_id"]}


def _seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(samples, streams, preprocessing, batch_size=16, num_workers=0,
                training=False, seed=42):
    import torch
    from torch.utils.data import DataLoader
    if type(batch_size) is not int or batch_size < 1 or type(num_workers) is not int or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    dataset = CaseDataset(samples, streams, preprocessing, training)
    dataset.seed = int(seed)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=bool(training),
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=_seed_worker, generator=generator)

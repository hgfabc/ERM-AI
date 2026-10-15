"""Frozen nested cross-validation with patient isolation in every partition."""

import copy
import json
from collections import Counter
from pathlib import Path

from .data import _digest, validate_manifest


def _canonical_splits(splits):
    return {"version": splits.get("version"),
            "manifest_fingerprint": splits.get("manifest_fingerprint"),
            "folds": splits.get("folds")}


def split_fingerprint(splits):
    return _digest(_canonical_splits(splits))


def validate_splits(manifest, splits):
    manifest = validate_manifest(manifest)
    if not isinstance(splits, dict) or type(splits.get("version")) is not int or splits["version"] != 1:
        raise ValueError("split version must be 1")
    if splits.get("manifest_fingerprint") != manifest["fingerprint"]:
        raise ValueError("splits belong to a different manifest fingerprint")
    folds = splits.get("folds")
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("splits require at least two outer folds")
    samples = {s["sample_id"]: s for s in manifest["samples"]}
    if any(s["label"] is None for s in samples.values()):
        raise ValueError("cross-validation requires labeled samples")
    all_ids = set(samples)
    test_counts, fold_numbers = Counter(), []
    for fold in folds:
        if not isinstance(fold, dict) or type(fold.get("fold")) is not int:
            raise ValueError("fold identifiers must be integers")
        fold_numbers.append(fold["fold"])
        partitions, patients = {}, {}
        for name in ("train", "validation", "test"):
            ids = fold.get(name)
            if not isinstance(ids, list) or not ids or any(not isinstance(s, str) for s in ids):
                raise ValueError("every partition must contain a nonempty list of sample IDs")
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate sample IDs within a partition")
            partitions[name] = set(ids)
            if partitions[name] - all_ids:
                raise ValueError("split contains unknown sample IDs")
            patients[name] = {samples[s]["patient_id"] for s in ids}
            if {samples[s]["label"] for s in ids} != set(manifest["class_to_idx"].values()):
                raise ValueError("every train/validation/test partition must represent every class")
        names = list(partitions)
        for i, name in enumerate(names):
            for other in names[i + 1:]:
                if partitions[name] & partitions[other]:
                    raise ValueError("sample overlap between partitions")
                if patients[name] & patients[other]:
                    raise ValueError("patient overlap between partitions")
        if set.union(*partitions.values()) != all_ids:
            raise ValueError("every fold must partition all manifest samples exactly once")
        test_counts.update(fold["test"])
    if sorted(fold_numbers) != list(range(len(folds))):
        raise ValueError("fold identifiers must be unique and contiguous from 0")
    if any(test_counts[s] != 1 for s in all_ids):
        raise ValueError("each sample must appear in outer test exactly once")
    provided = splits.get("fingerprint")
    expected = split_fingerprint(splits)
    if provided is not None and provided != expected:
        raise ValueError("split fingerprint mismatch; partitions changed")
    normalized = copy.deepcopy(splits)
    normalized["fingerprint"] = expected
    return normalized


def _inner_holdout(indices, labels, groups, val_fraction, seed):
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit
    local_y, local_groups = labels[indices], groups[indices]
    if len(set(local_groups)) < 2:
        raise ValueError("outer training pool needs at least two patients for inner validation")
    classes = sorted(set(labels))
    population = np.array([np.mean(local_y == label) for label in classes])
    best = None
    splitter = GroupShuffleSplit(n_splits=256, test_size=val_fraction, random_state=seed)
    for train, validation in splitter.split(indices, local_y, local_groups):
        # Missing classes have priority over small differences in fraction.
        missing = sum(not np.any(local_y[part] == label)
                      for part in (train, validation) for label in classes)
        fraction_error = abs(len(validation) / len(indices) - val_fraction)
        balance_error = sum(abs(np.mean(local_y[validation] == label) - population[i])
                            for i, label in enumerate(classes))
        score = (missing, fraction_error + balance_error, tuple(sorted(set(local_groups[validation]))))
        if best is None or score < best[0]:
            best = (score, train, validation)
    if best is None:
        raise ValueError("unable to construct an inner patient holdout")
    if best[0][0]:
        raise ValueError("inner train/validation cannot represent every class; add patients or revise fold/fraction settings")
    return indices[best[1]], indices[best[2]]


def make_splits(manifest, n_folds=10, val_fraction=0.2, seed=42):
    import numpy as np
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as error:
        raise ImportError("nested splits require scikit-learn >= 1.1 with StratifiedGroupKFold") from error
    manifest = validate_manifest(manifest)
    if type(n_folds) is not int or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2")
    if not isinstance(val_fraction, (int, float)) or isinstance(val_fraction, bool) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must lie strictly between 0 and 1")
    samples = manifest["samples"]
    if any(s["label"] is None for s in samples):
        raise ValueError("cross-validation requires labeled samples")
    labels = np.array([s["label"] for s in samples], dtype=int)
    groups = np.array([s["patient_id"] for s in samples])
    if set(labels) != set(manifest["class_to_idx"].values()):
        raise ValueError("cross-validation requires examples for every declared class")
    if n_folds > len(set(groups)):
        raise ValueError("n_folds exceeds the number of patients")
    for label in set(labels):
        if len(set(groups[labels == label])) < n_folds:
            raise ValueError("each class needs at least n_folds distinct patients")
    indices = np.arange(len(samples))
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    folds = []
    for number, (pool, test) in enumerate(outer.split(indices, labels, groups)):
        if set(labels[test]) != set(labels):
            raise ValueError("outer test fold omits a class; revise grouping or fold settings")
        train, validation = _inner_holdout(pool, labels, groups, val_fraction, int(seed) + number + 1)
        folds.append({"fold": number,
                      "train": sorted(samples[i]["sample_id"] for i in train),
                      "validation": sorted(samples[i]["sample_id"] for i in validation),
                      "test": sorted(samples[i]["sample_id"] for i in test)})
    splits = {"version": 1, "manifest_fingerprint": manifest["fingerprint"], "folds": folds}
    splits["fingerprint"] = split_fingerprint(splits)
    return validate_splits(manifest, splits)


def load_splits(path, manifest):
    with Path(path).open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    return validate_splits(manifest, splits)


def save_splits(splits, path):
    if not isinstance(splits, dict):
        raise ValueError("splits must be a mapping")
    expected = split_fingerprint(splits)
    if splits.get("fingerprint") is not None and splits["fingerprint"] != expected:
        raise ValueError("split fingerprint mismatch")
    normalized = copy.deepcopy(splits)
    normalized["fingerprint"] = expected
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return normalized

"""Classification metrics with explicit, stable class ordering."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def ordered_classes(class_to_idx):
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("class_to_idx must be a nonempty mapping")
    pairs = sorted(class_to_idx.items(), key=lambda pair: pair[1])
    if any(not isinstance(name, str) or not isinstance(index, int) or isinstance(index, bool)
           for name, index in pairs):
        raise ValueError("Class names must be strings and class indices must be integers")
    if [index for _, index in pairs] != list(range(len(pairs))):
        raise ValueError("Class indices must be unique and contiguous from zero")
    return [name for name, _ in pairs]


def calculate_metrics(labels, probabilities, class_to_idx):
    """Return metrics; unavailable one-vs-rest AUROC values are JSON null.

    balanced_accuracy averages recall over observed true classes. macro_f1
    averages F1 over every configured class, including absent classes.
    """
    names = ordered_classes(class_to_idx)
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Metrics require a nonempty one-dimensional label array")
    if probabilities.shape != (len(labels), len(names)):
        raise ValueError("Probability columns must match class_to_idx in index order")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be finite and between zero and one")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=1e-5):
        raise ValueError("Each probability row must sum to one")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be integer class indices")
    labels = labels.astype(int)
    if np.any(labels < 0) or np.any(labels >= len(names)):
        raise ValueError("Labels contain an unknown class index")
    predicted = probabilities.argmax(axis=1)
    indices = list(range(len(names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=indices, zero_division=0)
    per_class = {}
    aucs = {}
    for index, name in enumerate(names):
        binary_labels = (labels == index).astype(int)
        auc = float(roc_auc_score(binary_labels, probabilities[:, index])) if len(np.unique(binary_labels)) == 2 else None
        aucs[name] = auc
        per_class[name] = {
            "class_index": index, "support": int(support[index]),
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "auroc_one_vs_rest": auc,
        }
    available_aucs = [value for value in aucs.values() if value is not None]
    return {
        "sample_count": len(labels), "class_to_idx": dict(class_to_idx),
        "class_order": names,
        "accuracy": float((labels == predicted).mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1.mean()), "per_class": per_class,
        "auroc_one_vs_rest": aucs,
        "auroc_macro": float(np.mean(available_aucs)) if available_aucs else None,
        "auroc_macro_available_classes": len(available_aucs),
        "auroc_macro_definition": "Mean one-vs-rest AUROC over classes with positive and negative examples",
        "confusion_matrix": confusion_matrix(labels, predicted, labels=indices).tolist(),
        "confusion_matrix_rows": "true_label",
        "confusion_matrix_columns": "predicted_label",
        "balanced_accuracy_definition": "Mean recall over observed true classes",
        "macro_f1_definition": "Mean F1 over every configured class",
    }

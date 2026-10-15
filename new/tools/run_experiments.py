"""Run selected single/fusion conditions on the exact same saved patient splits."""
import argparse
import csv
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONDITIONS = {
    **{f"single_{stream.lstrip('_')}": [stream] for stream in (
        "_vertical", "_horizontal", "color_full", "color_inner", "color_outer",
        "gray_full", "gray_inner", "gray_outer")},
    "fusion_2_cross": ["_vertical", "_horizontal"],
    "fusion_2_color": ["color_inner", "color_outer"],
    "fusion_2_gray": ["gray_inner", "gray_outer"],
    "fusion_3_color": ["color_inner", "color_outer", "color_full"],
    "fusion_3_gray": ["gray_inner", "gray_outer", "gray_full"],
    "fusion_4": ["_vertical", "_horizontal", "color_full", "gray_full"],
    "fusion_6": ["color_inner", "color_outer", "color_full", "gray_inner", "gray_outer", "gray_full"],
    "fusion_8": ["_vertical", "_horizontal", "color_full", "color_inner", "color_outer", "gray_full", "gray_inner", "gray_outer"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--output", help="Parent directory for condition outputs.")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for name, streams in CONDITIONS.items():
            print(f"{name}: {', '.join(streams)}")
        return
    from multistream.config import load_config
    from multistream.data import load_manifest
    from multistream.splits import load_splits
    from multistream.engine import train

    base = load_config(args.config)
    if base["model"]["initialization"]:
        parser.error("The condition sweep uses ImageNet/random initialization; use individual configs for per-stream warm starts.")
    manifest = load_manifest(base["data"]["manifest"])
    splits = load_splits(base["data"]["splits"], manifest)
    destination = Path(args.output).resolve() if args.output else Path(base["experiment"]["output_dir"]).parent
    comparison_path = destination / "comparison.csv"
    if comparison_path.exists():
        parser.error("comparison.csv exists; choose a new --output destination.")
    for name in args.conditions:
        if (destination / name).exists():
            parser.error(f"Condition output already exists: {name}. Use individual train --folds to add unfinished folds.")
    rows = []
    for name in args.conditions:
        config = deepcopy(base)
        config["data"]["streams"] = CONDITIONS[name]
        config["experiment"] = {"name": name, "output_dir": str(destination / name)}
        print(f"Running {name} on shared split {splits['fingerprint'][:12]}", flush=True)
        result = train(config, manifest, splits, folds=args.folds)
        metrics = result["out_of_fold_metrics"]
        row = {"condition": name, "streams": "+".join(CONDITIONS[name]),
               "completed_folds": ",".join(map(str, result["completed_folds"])),
               "planned_folds": len(result["planned_folds"]), "test_samples": metrics["sample_count"],
               "test_accuracy": metrics["accuracy"], "test_balanced_accuracy": metrics["balanced_accuracy"],
               "test_macro_f1": metrics["macro_f1"], "test_auroc_macro": metrics["auroc_macro"]}
        rows.append(row)
        with comparison_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Comparison: {comparison_path}")


if __name__ == "__main__":
    main()

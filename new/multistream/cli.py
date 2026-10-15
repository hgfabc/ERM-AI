"""Small commands that connect the reusable data, model and training modules."""
import argparse
import json
from pathlib import Path

from .config import load_config


def _print(value):
    print(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False))


def _check_new_outputs(*paths):
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f"Output already exists: {path}. Choose a new path to preserve previous results.")


def prepare(args):
    from .data import load_manifest, save_manifest, scan_dataset
    from .splits import make_splits, save_splits

    config = load_config(args.config)
    data = config["data"]
    manifest_path = Path(args.manifest or data["manifest"]).resolve()
    splits_path = Path(args.splits or data["splits"]).resolve()
    if manifest_path == splits_path:
        raise ValueError("Manifest and splits must be different files.")
    _check_new_outputs(manifest_path, splits_path)
    if args.from_manifest:
        manifest = load_manifest(args.from_manifest)
    else:
        root = args.root or data["root"]
        if not root:
            raise ValueError("Set data.root in your config or pass --root.")
        manifest = scan_dataset(root, data["cohort_streams"], data["class_to_idx"],
                                require_all_streams=data["require_all_streams"])
    if manifest["class_to_idx"] != data["class_to_idx"]:
        raise ValueError("Manifest class mapping differs from config data.class_to_idx.")
    if not set(data["cohort_streams"]).issubset(manifest["streams"]):
        raise ValueError("Manifest is missing configured cohort streams.")
    splits = make_splits(manifest, **config["split"])
    save_manifest(manifest, manifest_path)
    save_splits(splits, splits_path)
    _print({"manifest": str(manifest_path), "splits": str(splits_path),
            "samples": len(manifest["samples"]),
            "patients": len({sample["patient_id"] for sample in manifest["samples"]}),
            "folds": len(splits["folds"]), "class_to_idx": manifest["class_to_idx"]})


def train_command(args):
    from .data import load_manifest
    from .splits import load_splits
    from .engine import train

    config = load_config(args.config)
    if args.output:
        config["experiment"]["output_dir"] = str(Path(args.output).resolve())
    if args.device:
        config["train"]["device"] = args.device
    manifest = load_manifest(args.manifest or config["data"]["manifest"])
    splits = load_splits(args.splits or config["data"]["splits"], manifest)
    if manifest["class_to_idx"] != config["data"]["class_to_idx"]:
        raise ValueError("Config and manifest use different class mappings.")
    result = train(config, manifest, splits, folds=args.folds)
    _print({"output": config["experiment"]["output_dir"],
            "completed_folds": result["completed_folds"],
            "complete_cross_validation": result["complete_cross_validation"],
            "test_metrics_aggregate": result["test_metrics_aggregate"],
            "out_of_fold_metrics": result["out_of_fold_metrics"]})


def _evaluation_samples(args, checkpoint, unlabeled):
    from .data import load_manifest, scan_dataset, select_samples
    from .splits import load_splits

    if args.root:
        if args.subset != "external":
            raise ValueError("Use --manifest and --splits for a saved test/validation subset.")
        manifest = scan_dataset(args.root, list(checkpoint["streams"]),
                                checkpoint["class_to_idx"], require_all_streams=True,
                                allow_unlabeled=unlabeled)
    else:
        manifest = load_manifest(args.manifest)
    if manifest["class_to_idx"] != checkpoint["class_to_idx"]:
        raise ValueError("Evaluation class mapping differs from the saved checkpoint.")
    if not set(checkpoint["streams"]).issubset(manifest["streams"]):
        raise ValueError("Evaluation manifest is missing checkpoint streams.")
    samples = manifest["samples"]
    if args.subset != "external":
        if not args.splits:
            raise ValueError("--splits is required to evaluate a saved test/validation subset.")
        splits = load_splits(args.splits, manifest)
        if checkpoint["manifest_fingerprint"] != manifest["fingerprint"] or checkpoint["split_fingerprint"] != splits["fingerprint"]:
            raise ValueError("Checkpoint, manifest and split fingerprints do not match.")
        fold = next((row for row in splits["folds"] if row["fold"] == checkpoint["fold"]), None)
        if fold is None:
            raise ValueError("Checkpoint fold does not exist in the supplied split file.")
        samples = select_samples(manifest, fold[args.subset])
    return samples


def evaluate_command(args):
    from .models import load_checkpoint
    from .engine import evaluate_checkpoint

    checkpoint = load_checkpoint(args.checkpoint)
    infer = args.command == "infer"
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers must be nonnegative.")
    if getattr(args, "gradcam", False) and args.max_cams < 1:
        raise ValueError("max-cams must be positive.")
    if checkpoint.get("legacy_no_split_metadata") and not infer and not args.allow_patient_overlap:
        raise ValueError("Legacy weights have no training-patient metadata. Use infer for predictions, or explicitly allow overlap for exploratory validation.")
    samples = _evaluation_samples(args, checkpoint, unlabeled=infer)
    if not infer and any(sample["label"] is None for sample in samples):
        raise ValueError("Validation requires a ground-truth label for every sample.")
    output = Path(args.output).resolve()
    _check_new_outputs(output)
    check_overlap = not infer and args.subset != "validation"
    requested_role = "inference" if infer else {
        "test": "held_out_outer_test", "validation": "selection_validation",
        "external": "external_validation",
    }[args.subset]
    result = evaluate_checkpoint(
        args.checkpoint, samples, output, device=args.device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        check_patient_overlap=check_overlap,
        allow_patient_overlap=args.allow_patient_overlap,
        evaluation_role=requested_role,
    )
    interpretation = {
        "selection_validation": "checkpoint_selection_validation",
        "inference": "predictions_only_not_independent_validation",
        "held_out_outer_test": "held_out_test",
        "external_validation": "external_validation",
        "exploratory_validation": "exploratory_validation_independence_not_established",
        "exploratory_validation_unverified_provenance": "exploratory_validation_independence_not_established",
        "exploratory_validation_patient_overlap": "exploratory_validation_independence_not_established",
        "unchecked_validation": "validation_independence_not_checked",
    }[result["evaluation_role"]]
    (output / "evaluation_scope.json").write_text(json.dumps({"interpretation": interpretation,
        "subset": args.subset, "checkpoint": str(Path(args.checkpoint).resolve()),
        "legacy_no_split_metadata": checkpoint.get("legacy_no_split_metadata", False),
        "evaluation_role": result["evaluation_role"],
        "independent_validation": result.get("independent_validation", False),
        "provenance_verified": result.get("provenance_verified", False)}, indent=2), encoding="utf-8")
    if getattr(args, "gradcam", False):
        from .gradcam import export_gradcam
        export_gradcam(args.checkpoint, samples, output / "gradcam", device=args.device,
                       max_samples=args.max_cams)
    _print({"output": str(output), "interpretation": interpretation, "results": result})


def convert_legacy(args):
    from .models import convert_legacy_checkpoint

    config = load_config(args.config)
    _check_new_outputs(args.output)
    converted = convert_legacy_checkpoint(
        args.checkpoint, args.output, config["data"]["streams"], config["model"],
        config["data"]["class_to_idx"], config["preprocessing"],
    )
    _print({"checkpoint": str(Path(args.output).resolve()),
            "scope": "legacy predictions; ineligible for new cross-validation warm start"})


def make_parser():
    parser = argparse.ArgumentParser(description="OCT experiments: one pipeline for single-stream and fusion models.")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Scan images and create immutable patient-grouped splits.")
    prep.add_argument("--config", required=True)
    prep.add_argument("--root", help="Override the image root (relative to your current directory).")
    prep.add_argument("--from-manifest", help="Use an existing manifest with verified patient IDs instead of scanning folders.")
    prep.add_argument("--manifest", help="Destination manifest path.")
    prep.add_argument("--splits", help="Destination split path.")
    prep.set_defaults(handler=prepare)

    training = commands.add_parser("train", help="Train, select on inner validation and score on outer test.")
    training.add_argument("--config", required=True)
    training.add_argument("--manifest")
    training.add_argument("--splits")
    training.add_argument("--folds", type=int, nargs="+", help="Only these zero-based folds; omit to run all.")
    training.add_argument("--output", help="Override experiment output directory.")
    training.add_argument("--device", help="auto, cpu, cuda or cuda:0.")
    training.set_defaults(handler=train_command)

    for command, help_text in (("validate", "Score labeled data using saved settings and check patient overlap."),
                               ("infer", "Predict labeled or unlabeled cases using saved settings.")):
        evaluation = commands.add_parser(command, help=help_text)
        evaluation.add_argument("--checkpoint", required=True)
        source = evaluation.add_mutually_exclusive_group(required=True)
        source.add_argument("--root", help="Image folder root.")
        source.add_argument("--manifest", help="Prepared manifest.")
        evaluation.add_argument("--splits", help="Required for a saved test/validation subset.")
        evaluation.add_argument("--subset", choices=("external", "test", "validation"), default="external")
        evaluation.add_argument("--output", required=True)
        evaluation.add_argument("--device", default="auto")
        evaluation.add_argument("--batch-size", type=int, default=8)
        evaluation.add_argument("--num-workers", type=int, default=0)
        evaluation.add_argument("--allow-patient-overlap", action="store_true", help="Explicit exploratory validation; independence is not established.")
        if command == "infer":
            evaluation.add_argument("--gradcam", action="store_true", help="Also export per-stream Grad-CAM for ResNet models.")
            evaluation.add_argument("--max-cams", type=int, default=8)
        evaluation.set_defaults(handler=evaluate_command)

    legacy = commands.add_parser("convert-legacy", help="Strictly convert old weights for inference, with explicit stream/settings metadata.")
    legacy.add_argument("--checkpoint", required=True)
    legacy.add_argument("--config", required=True, help="Explicit original model, stream order, labels and preprocessing.")
    legacy.add_argument("--output", required=True)
    legacy.set_defaults(handler=convert_legacy)
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")

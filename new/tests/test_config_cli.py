"""Configuration/path safeguards and command-line contracts."""
from pathlib import Path
import argparse
import json
import yaml
import pytest

from multistream.config import ConfigError, load_config
from multistream.cli import make_parser


def test_inherited_paths_stay_relative_to_defining_file(tmp_path):
    base = tmp_path / "shared" / "base.yaml"
    base.parent.mkdir()
    base.write_text("data:\n  root: ../../images\nexperiment:\n  output_dir: ../runs/base\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("extends: shared/base.yaml\ndata:\n  streams: [color_inner, color_outer]\n", encoding="utf-8")
    config = load_config(child)
    assert config["data"]["root"] == str((base.parent / "../../images").resolve())
    assert config["experiment"]["output_dir"] == str((base.parent / "../runs/base").resolve())
    assert config["data"]["streams"] == ["color_inner", "color_outer"]


@pytest.mark.parametrize("update", [
    {"preprocessing": {"image_size": 500, "resize_size": 448}},
    {"data": {"streams": ["gray_inner", "gray_inner"]}},
    {"train": {"lern_rate": 0.01}},
    {"model": {"pretrained": "false"}},
    {"data": {"class_to_idx": {"bad": 0, "good": 2}}},
])
def test_invalid_experiments_fail_before_training(tmp_path, update):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(update), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_circular_inheritance_is_rejected(tmp_path):
    config = tmp_path / "cycle.yaml"
    config.write_text("extends: cycle.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Circular"):
        load_config(config)


def test_every_example_config_is_loadable():
    for path in (Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"):
        assert load_config(path)["train"]["epochs"] > 0


def test_custom_class_mapping_replaces_default_mapping(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text("data:\n  class_to_idx: {reject: 0, accept: 1}\n", encoding="utf-8")
    assert load_config(path)["data"]["class_to_idx"] == {"reject": 0, "accept": 1}


def test_cli_source_is_required_and_mutually_exclusive():
    parser = make_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["infer", "--checkpoint", "model.pt", "--output", "out"])
    with pytest.raises(SystemExit):
        parser.parse_args(["infer", "--checkpoint", "model.pt", "--output", "out", "--root", "images", "--manifest", "manifest.json"])
    args = parser.parse_args(["train", "--config", "config.yaml", "--folds", "0", "3"])
    assert args.folds == [0, 3]


@pytest.mark.parametrize("role", ["exploratory_validation_patient_overlap", "exploratory_validation_unverified_provenance"])
def test_cli_preserves_exploratory_scope_reasons(tmp_path, monkeypatch, role):
    from multistream import cli, engine, models
    checkpoint = {"streams": ["gray_inner"], "class_to_idx": {"bad": 0, "good": 1}}
    monkeypatch.setattr(models, "load_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(cli, "_evaluation_samples", lambda *args, **kwargs: [{"label": 0}])
    def fake_evaluate(checkpoint_path, samples, output_dir, **kwargs):
        output_dir.mkdir()
        return {"evaluation_role": role, "independent_validation": False, "provenance_verified": False}
    monkeypatch.setattr(engine, "evaluate_checkpoint", fake_evaluate)
    destination = tmp_path / "evaluation"
    args = argparse.Namespace(checkpoint="model.pth", command="validate", batch_size=1,
        num_workers=0, subset="external", output=str(destination), device="cpu", allow_patient_overlap=True)
    cli.evaluate_command(args)
    scope = json.loads((destination / "evaluation_scope.json").read_text())
    assert scope["interpretation"] == "exploratory_validation_independence_not_established"
    assert scope["evaluation_role"] == role
    assert scope["independent_validation"] is False

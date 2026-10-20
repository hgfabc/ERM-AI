"""Preserve every fold and unidentified model without inventing private metadata."""
import importlib.util
import json
from pathlib import Path
import re

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _exporter():
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("_collection_export", ROOT / "tools" / "export_model_collection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def sources(tmp_path):
    torch.set_num_threads(1)
    from multistream.models import build_model
    records = tmp_path / "records"
    settings = {"name": "resnet18", "hidden_dim": 8, "pretrained": False, "freeze_backbones": False}
    conditions = {
        "single_gray_inner": {"project": "image-classification-7", "source_prefix": "gray_inner", "streams": ["gray_inner"], "antialias": True},
        "fusion_2_color": {"project": "image-classification-ms-2", "source_prefix": "color_inner", "streams": ["color_inner", "color_outer"], "antialias": True},
    }
    for name, condition in conditions.items():
        model = build_model(settings, condition["streams"], 2)
        raw = {}
        for key, tensor in model.state_dict().items():
            if key.startswith("backbones."):
                _, branch, tail = key.split(".", 2)
                key = "b" + str(int(branch) + 1) + "." + tail
            elif key.startswith("fusion."):
                key = "merger." + key[len("fusion."):]
            elif key.startswith("classifier."):
                key = "fc." + key[len("classifier."):]
            raw[key] = tensor
        for fold in (0, 1):
            path = records / condition["project"] / ("resnet50_runresnet50_run_" + condition["source_prefix"] + f"@{fold}_fold") / "best_model.pth"
            path.parent.mkdir(parents=True)
            torch.save(raw, path)
    extra = records / "unidentified-project" / "resnet50_run" / "best_model.pth"
    extra.parent.mkdir(parents=True)
    torch.save({"unknown_legacy_conv.weight": torch.ones(3, 4, 1, 1)}, extra)
    profiles = {"collection_version": 1, "folds": [0, 1], "model": settings,
                "class_to_idx": {"bad": 0, "good": 1},
                "preprocessing": {"image_size": 32, "resize_size": 36, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "resize_backend": "tensor"},
                "conditions": conditions}
    profile_path = tmp_path / "profiles.yaml"
    profile_path.write_text(yaml.safe_dump(profiles), encoding="utf-8")
    return records, profile_path, tmp_path / "collection"


def test_plan_covers_every_fold_and_historical_extra_without_export(sources):
    module = _exporter()
    records, profiles, destination = sources
    plan = module.export_collection(records, profiles, destination, plan_only=True)
    assert plan["model_count"] == 5
    assert plan["inference_ready_count"] == 4
    assert plan["historical_count"] == 1
    assert not destination.exists()


def test_collection_readme_languages_preserve_navigation_and_model_details():
    module = _exporter()
    profiles = module.load_profiles(ROOT / "publish_profiles" / "legacy_collection.yaml")
    items = [{"inference_ready": True}] * 4 + [{"inference_ready": False}]
    code_url = "https://github.com/example/ERM-AI/tree/documentation/new"
    card = module._collection_readme(items, profiles, code_url)
    _, front_matter, body = card.split("---", 2)
    assert yaml.safe_load(front_matter)["language"] == ["en", "zh", "ja"]
    navigation = body.strip().splitlines()[0]
    assert navigation == "[English](#english) | [繁體中文](#traditional-chinese) | [日本語](#japanese)"
    targets = re.findall(r"\]\(#([^)]*)\)", navigation)
    assert targets == re.findall(r'<a id="([^"]+)"></a>', body)
    sections = re.split(r'<a id="[^"]+"></a>\n\n', body)[1:]
    assert len(sections) == 3
    summaries = (
        "5 model weight files: 4 loadable model/fold entries and 1 historical tensor archive",
        "5 個模型權重檔案，包含 4 個可載入的模型／fold 項目，以及 1 個歷史張量封存項目",
        "5 個のモデル重みファイルを保存しています。内訳は、読み込み可能なモデル／fold が 4 件、過去のテンソルアーカイブが 1 件",
    )
    for section, summary in zip(sections, summaries):
        assert summary in section
        assert "[batch, streams, 3, 448, 448]" in section
        assert '{"bad": 0, "good": 1}' in section
        assert "`--revision refs/pr/<number>`" in section
        assert "`historical/`" in section
        assert code_url in section
        assert "Linear(N*2048, 2048)" in section
        assert "torchvision 0.17.1" in section and "torchvision 0.16.2" in section
        assert "config2.yaml" in section
        assert ("antialias=true" if section != sections[0] else "antialias true") in section
        for name, condition in profiles["conditions"].items():
            assert f"| {name} | {', '.join(condition['streams'])} | {condition['antialias']} |" in section
    for condition in profiles["conditions"].values():
        assert module._collection_profile_note(condition["notes"], "zh") != condition["notes"]
        assert module._collection_profile_note(condition["notes"], "ja") != condition["notes"]


def _language_sections(card):
    _, front_matter, body = card.split("---", 2)
    assert yaml.safe_load(front_matter)["language"] == ["en", "zh", "ja"]
    navigation = body.strip().splitlines()[0]
    assert navigation == "[English](#english) | [繁體中文](#traditional-chinese) | [日本語](#japanese)"
    assert re.findall(r"\]\(#([^)]*)\)", navigation) == re.findall(r'<a id="([^"]+)"></a>', body)
    sections = re.split(r'<a id="[^"]+"></a>\n\n', body)[1:]
    assert len(sections) == 3
    return sections


def test_collection_leaf_languages_keep_shared_loader_and_matching_commands():
    module = _exporter()
    profiles = module.load_profiles(ROOT / "publish_profiles" / "legacy_collection.yaml")
    metadata = module._ready_metadata(profiles, "fusion_8")
    item = {"condition": "fusion_8", "fold": 3}
    code_url = "https://github.com/example/ERM-AI/tree/public/new"
    card = module._collection_model_card(metadata, item, profiles, code_url)
    sections = _language_sections(card)
    assert "from modeling_multistream" not in card and "downloaded_model" not in card
    assert "shared loader" in sections[0]
    for section in sections:
        assert "--condition fusion_8 --fold 3" in section
        assert "`--revision refs/pr/<number>`" in section
        assert json.dumps(metadata["streams"]) in section
        assert "`fusion_8`" in section and "`3`" in section
        assert "https://github.com/example/ERM-AI/blob/public/new/INFERENCE.md" in section
        assert "torchvision 0.16.2" in section
    commands = [re.findall(r"```bash\n(.*?)\n```", section, re.DOTALL) for section in sections]
    assert len(commands[0]) == 1 and commands[0] == commands[1] == commands[2]
    settings = [re.findall(r"```json\n(.*?)\n```", section, re.DOTALL) for section in sections]
    assert len(settings[0]) == 1 and settings[0] == settings[1] == settings[2]
    assert json.loads(settings[0][0]) == metadata["preprocessing"]


def test_historical_card_languages_limit_inference_claims_to_archive_entries():
    module = _exporter()
    config = module._historical_config({"condition": "unidentified-project"})
    sections = _language_sections(module._historical_card(config))
    for section in sections:
        assert "`unidentified-project`" in section
        assert "`historical/`" in section
        assert "fold" in section
        assert "```" not in section and "--condition" not in section
    for section, scope in zip(sections, ("No fold, stream order", "未宣告其 fold、輸入流順序", "fold、ストリームの順序")):
        assert scope in section


def test_full_collection_and_resume_preserve_weights(sources):
    module = _exporter()
    records, profiles, destination = sources
    result = module.export_collection(records, profiles, destination, code_url="https://github.com/user/ERM-AI")
    assert result["exported"] == 5
    catalog = json.loads((destination / "catalog.json").read_text())
    assert catalog["model_count"] == 5
    assert {entry["fold"] for entry in catalog["models"] if entry["inference_ready"]} == {0, 1}
    assert {entry["path"] for entry in catalog["models"] if not entry["inference_ready"]} == {"historical/unidentified-project"}
    hashes = {entry["path"]: entry["weights_sha256"] for entry in catalog["models"]}
    for entry in catalog["models"]:
        leaf = destination / entry["path"]
        assert {path.name for path in leaf.iterdir()} == module.LEAF_FILES
        _language_sections((leaf / "README.md").read_text(encoding="utf-8"))
    historical = json.loads((destination / "historical/unidentified-project/config.json").read_text())
    assert historical["streams"] is None and historical["inference_ready"] is False
    assert historical["model_config"] is None
    resumed = module.export_collection(records, profiles, destination, code_url="https://github.com/user/ERM-AI/tree/new-version")
    assert resumed["reused"] == 5 and resumed["exported"] == 0
    second = json.loads((destination / "catalog.json").read_text())
    assert hashes == {entry["path"]: entry["weights_sha256"] for entry in second["models"]}
    assert "new-version" in (destination / "models/single_gray_inner/fold_0/README.md").read_text(encoding="utf-8")
    assert not (destination / "state.json").exists()


def test_missing_fold_stops_before_export(sources):
    module = _exporter()
    records, profiles, destination = sources
    missing = records / "image-classification-7/resnet50_runresnet50_run_gray_inner@1_fold/best_model.pth"
    missing.unlink()
    with pytest.raises(ValueError, match="Missing"):
        module.export_collection(records, profiles, destination)
    assert not destination.exists()


def test_public_collection_uploader_accepts_exporter_output(sources):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from upload_model_collection import upload_collection
    module = _exporter()
    records, profiles, destination = sources
    module.export_collection(records, profiles, destination)
    result = upload_collection(destination, verify_only=True)
    assert result["model_count"] == 5


def test_exported_collection_runs_selected_folds_and_ensemble(sources):
    """Exercise the real export-to-inference contract, including legacy key remapping."""
    import csv
    from PIL import Image
    from infer_published import run_inference
    module = _exporter()
    records, profiles, destination = sources
    module.export_collection(records, profiles, destination)
    cohort = destination.parent / "cohort"
    for label, case, intensity in (("bad", "CASE001_OD", 64), ("good", "CASE002_OS", 192)):
        folder = cohort / label / case
        folder.mkdir(parents=True)
        for stream in ("color_inner", "color_outer"):
            Image.new("RGB", (40, 38), (intensity, 80, 120)).save(folder / (case + "_" + stream + ".png"))
    prediction_folder = destination.parent / "predictions"
    summary = run_inference(local_repo=destination, condition="fusion_2_color", folds=[0, 1],
                            root=cohort, output=prediction_folder, device="cpu", batch_size=2, ensemble=True)
    assert summary["folds"] == [0, 1]
    assert summary["ensemble"]["sample_count"] == 2
    assert summary["downloaded_repository_code_executed"] is False
    probability_rows = []
    for name in ("predictions_fold_0.csv", "predictions_fold_1.csv", "predictions_ensemble.csv"):
        with (prediction_folder / name).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        probability_rows.append([{key: value for key, value in row.items() if key.startswith("prob_")}
                                 for row in rows])
    assert probability_rows[0] == probability_rows[1] == probability_rows[2]
    card = (destination / "models/fusion_2_color/fold_1/README.md").read_text(encoding="utf-8")
    assert "--condition fusion_2_color --fold 1" in card
    assert "shared loader" in card

def test_unexpected_cached_model_stops_before_exporting(sources, monkeypatch):
    module = _exporter()
    records, profiles, destination = sources
    stale = destination / "models/removed_condition/fold_0"
    stale.mkdir(parents=True)
    (stale / "model.safetensors").write_bytes(b"preserve this old cache")
    def unexpected_export(*args, **kwargs):
        pytest.fail("Weights must not be exported before an invalid cache is rejected")
    monkeypatch.setattr(module, "export_public_model", unexpected_export)
    with pytest.raises(ValueError, match="outside the current model plan"):
        module.export_collection(records, profiles, destination)
    assert (stale / "model.safetensors").read_bytes() == b"preserve this old cache"

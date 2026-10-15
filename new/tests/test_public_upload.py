"""Keep direct Hub publishing bounded; every network operation is mocked."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exported_bundle(tmp_path_factory):
    torch.set_num_threads(1)
    from multistream.models import build_model
    exporter = _module("export_public_model")
    root = tmp_path_factory.mktemp("public-upload")
    settings = {"name": "resnet18", "hidden_dim": 8, "freeze_backbones": False, "pretrained": False}
    model = build_model(settings, ["gray_inner"], 2)
    source = root / "source.pth"
    torch.save({"format_version": 1, "state_dict": model.state_dict(), "model_config": settings,
                "streams": ["gray_inner"], "class_to_idx": {"bad": 0, "good": 1},
                "preprocessing": {"image_size": 32, "resize_size": 36,
                                  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "train_patient_ids": ["PRIVATE_FIXTURE"], "config": {"private_dataset": "fixture"}}, source)
    bundle = root / "bundle"
    exporter.export_public_model(source, bundle, kind="demo")
    return bundle


def _copy_bundle(source, destination):
    import shutil
    shutil.copytree(source, destination)
    return destination


def test_upload_rejects_extra_private_file(exported_bundle, tmp_path):
    upload = _module("upload_public_model")
    bundle = _copy_bundle(exported_bundle, tmp_path / "extra")
    (bundle / "patient_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="exactly"):
        upload.validate_bundle(bundle)


@pytest.mark.parametrize("nested", [False, True])
def test_upload_rejects_private_metadata_even_inside_model_config(exported_bundle, tmp_path, nested):
    upload = _module("upload_public_model")
    bundle = _copy_bundle(exported_bundle, tmp_path / "private")
    path = bundle / "config.json"
    config = json.loads(path.read_text())
    if nested:
        config["model_config"]["initialization_context"] = {"patient_ids": ["PRIVATE_FIXTURE"]}
    else:
        config["train_patient_ids"] = ["PRIVATE_FIXTURE"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        upload.validate_bundle(bundle)


def test_verify_only_reloads_weights_without_auth_or_network(exported_bundle, monkeypatch):
    upload = _module("upload_public_model")
    import huggingface_hub
    def forbidden(*args, **kwargs):
        raise AssertionError("Verification must not authenticate or access the Hub")
    monkeypatch.setattr(huggingface_hub, "HfApi", forbidden)
    assert upload.upload_model(exported_bundle, "ntuhophai/ERM-AI", verify_only=True) is None


def test_direct_upload_sends_only_five_files_and_does_not_create_pr(exported_bundle, monkeypatch):
    upload = _module("upload_public_model")
    import huggingface_hub
    calls = []
    class FakeApi:
        def whoami(self):
            return {"name": "ntuhophai", "auth": {"accessToken": {"role": "write"}}, "orgs": []}
        def repo_info(self, **kwargs):
            calls.append(("info", kwargs))
        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs))
            return SimpleNamespace(commit_url="https://huggingface.co/ntuhophai/ERM-AI/commit/test")
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    upload.upload_model(exported_bundle, "ntuhophai/ERM-AI")
    options = next(value for kind, value in calls if kind == "upload")
    assert set(options["allow_patterns"]) == upload.ALLOWED_FILES
    assert options["repo_id"] == "ntuhophai/ERM-AI"
    assert options["create_pr"] is False
    assert "delete_patterns" not in options


def test_modified_public_loader_is_not_executed_or_uploaded(exported_bundle, tmp_path, monkeypatch):
    upload = _module("upload_public_model")
    bundle = _copy_bundle(exported_bundle, tmp_path / "edited")
    (bundle / "modeling_multistream.py").write_text("raise AssertionError('must never execute')")
    with pytest.raises(ValueError, match="modified"):
        upload.upload_model(bundle, "ntuhophai/ERM-AI", verify_only=True)


def test_unknown_hf_namespace_is_not_treated_as_write_access():
    upload = _module("upload_public_model")
    assert not upload._known_write_account({"name": "other", "auth": {"accessToken": {"role": "write"}}, "orgs": []}, "ntuhophai")
    assert not upload._known_write_account({"name": "ntuhophai", "auth": {"accessToken": {"role": "read"}}}, "ntuhophai")

"""Collection uploads stay bounded and resumable; all Hub calls are mocked."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def uploader():
    spec = importlib.util.spec_from_file_location("collection_upload_test", ROOT / "tools" / "upload_model_collection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _safetensors(path, metadata=None):
    header = {"network.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + struct.pack("<f", 1.0))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _blob(path):
    data = path.read_bytes()
    return hashlib.sha1(("blob %d\0" % len(data)).encode("ascii") + data).hexdigest()


@pytest.fixture
def collection(tmp_path, uploader):
    folder = tmp_path / "collection"
    folder.mkdir()
    (folder / "modeling_multistream.py").write_text(uploader._exporter.PUBLIC_LOADER, encoding="utf-8")
    (folder / "README.md").write_text("# Public collection\n", encoding="utf-8")
    (folder / "requirements.txt").write_text("torch==2.2.2\ntorchvision==0.17.2\ntimm==1.0.0\nsafetensors==0.5.0\n", encoding="utf-8")
    entries = []
    for fold in range(2):
        relative = "models/gray_inner/fold_%d" % fold
        directory = folder / relative
        directory.mkdir(parents=True)
        config = {
            "public_format_version": 1,
            "model_config": {"name": "resnet18", "hidden_dim": 8, "freeze_backbones": False, "pretrained": False},
            "streams": ["gray_inner"], "class_to_idx": {"bad": 0, "good": 1},
            "preprocessing": {"image_size": 32, "resize_size": 36, "mean": [0.485, 0.456, 0.406],
                              "std": [0.229, 0.224, 0.225], "resize_backend": "tensor",
                              "interpolation": "bilinear", "antialias": True, "crop": "center"},
            "artifact_kind": "research", "source_format": "legacy_state_dict",
            "settings_provenance": "declared_profiles", "private_training_metadata_omitted": True,
            "legacy_no_split_metadata": True,
        }
        _write_json(directory / "config.json", config)
        _safetensors(directory / "model.safetensors")
        (directory / "README.md").write_text("# Public gray model\n", encoding="utf-8")
        entries.append({"condition": "gray_inner", "fold": fold, "streams": ["gray_inner"], "path": relative,
                        "inference_ready": True, "weights_sha256": _sha(directory / "model.safetensors"),
                        "config_sha256": _sha(directory / "config.json"),
                        "weights_bytes": (directory / "model.safetensors").stat().st_size})
    relative = "historical/legacy_run"
    directory = folder / relative
    directory.mkdir(parents=True)
    _safetensors(directory / "model.safetensors")
    historical = {"historical_format_version": 1, "condition": "historical_legacy_run", "inference_ready": False,
                  "streams": None, "class_to_idx": None, "model_config": None, "preprocessing": None,
                  "source_format": "legacy_raw_state_dict", "private_training_metadata_omitted": True,
                  "reason": "Unknown historical inference settings"}
    _write_json(directory / "config.json", historical)
    (directory / "README.md").write_text("# Historical tensors\n", encoding="utf-8")
    entries.append({"condition": "historical_legacy_run", "fold": None, "streams": None, "path": relative,
                    "inference_ready": False, "weights_sha256": _sha(directory / "model.safetensors"),
                    "config_sha256": _sha(directory / "config.json"), "project": "legacy_run",
                    "source_format": "legacy_raw_state_dict", "reason": historical["reason"],
                    "weights_bytes": (directory / "model.safetensors").stat().st_size})
    catalog = {"catalog_version": 1, "models": entries, "model_count": 3, "inference_ready_count": 2,
               "historical_count": 1, "condition_count": 1, "fold_count": 2,
               "total_weight_bytes": sum(entry["weights_bytes"] for entry in entries),
               "private_training_metadata_omitted": True}
    _write_json(folder / "catalog.json", catalog)
    return folder


class FakeApi:
    def __init__(self):
        self.calls = []
        self.remote = {}
        self.fail_path = None
        self.read_calls = 0
        self.tamper_after_reads = None

    def whoami(self):
        self.calls.append(("identity", {}))
        return {"name": "ntuhophai", "auth": {"accessToken": {"role": "write"}}, "orgs": []}

    def repo_info(self, **kwargs):
        self.calls.append(("info", kwargs))

    def auth_check(self, **kwargs):
        self.calls.append(("auth_check", kwargs))

    def get_paths_info(self, **kwargs):
        self.calls.append(("paths", kwargs))
        self.read_calls += 1
        if self.read_calls == self.tamper_after_reads:
            self.remote["models/gray_inner/fold_0/config.json"].blob_id = "tampered"
        return [self.remote[path] for path in kwargs["paths"] if path in self.remote]

    def upload_folder(self, **kwargs):
        self.calls.append(("upload", kwargs))
        if kwargs["path_in_repo"] == self.fail_path:
            raise RuntimeError("A mock failure with a credential that must not be printed: hf_" + "A" * 40)
        directory = Path(kwargs["folder_path"])
        prefix = kwargs["path_in_repo"]
        for name in kwargs["allow_patterns"]:
            file = directory / name
            path = prefix + "/" + name if prefix else name
            self.remote[path] = SimpleNamespace(path=path, size=file.stat().st_size, blob_id=_blob(file),
                                                lfs=SimpleNamespace(sha256=_sha(file)) if name == "model.safetensors" else None)
        return SimpleNamespace(oid="mock-commit")


def _hub(monkeypatch, api):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api))


def _uploads(api):
    return [options for kind, options in api.calls if kind == "upload"]


def test_verify_only_has_no_auth_network_or_state(collection, uploader, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Verification must not authenticate or access the Hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=forbidden))
    monkeypatch.setattr(uploader, "_local_login", forbidden)
    result = uploader.upload_collection(collection, verify_only=True)
    assert result["model_count"] == 3 and result["verified_only"] is True
    assert result["uploaded"] == result["skipped"] == 0
    assert not (collection.parent / ".upload-state.json").exists()


@pytest.mark.parametrize("where", ["private_manifest.json", "models/gray_inner/fold_0/private_manifest.json"])
def test_rejects_extra_private_files(collection, uploader, where):
    (collection / where).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        uploader.validate_collection(collection)


@pytest.mark.parametrize("where", ["catalog", "config", "nested_config", "nested_catalog"])
def test_rejects_private_metadata(collection, uploader, where):
    path = collection / ("catalog.json" if "catalog" in where else "models/gray_inner/fold_0/config.json")
    metadata = json.loads(path.read_text())
    target = metadata
    if where == "nested_config":
        target = metadata["model_config"]
    elif where == "nested_catalog":
        target = metadata["models"][0]
    target["private_training_context"] = {"private_identifier": "NEVER_UPLOAD"}
    _write_json(path, metadata)
    with pytest.raises(ValueError):
        uploader.validate_collection(collection)


def test_rejects_nested_values_in_catalog_public_fields(collection, uploader):
    path = collection / "catalog.json"
    metadata = json.loads(path.read_text())
    metadata["models"][0]["project"] = {"private_identifier": "NEVER_UPLOAD"}
    _write_json(path, metadata)
    with pytest.raises(ValueError, match="plain text"):
        uploader.validate_collection(collection)


def test_rejects_links_without_following_them(collection, uploader):
    path = collection / "models/gray_inner/fold_0/README.md"
    target = collection.parent / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("The local account cannot create symlinks")
    with pytest.raises(ValueError, match="Symlinks"):
        uploader.validate_collection(collection)


def test_rejects_modified_loader_and_safetensors_metadata(collection, uploader):
    (collection / "modeling_multistream.py").write_text("raise AssertionError('must not run')", encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        uploader.validate_collection(collection)
    (collection / "modeling_multistream.py").write_text(uploader._exporter.PUBLIC_LOADER, encoding="utf-8")
    _safetensors(collection / "models/gray_inner/fold_0/model.safetensors", {"private_identifier": "NEVER_UPLOAD"})
    with pytest.raises(ValueError, match="non-tensor"):
        uploader.validate_collection(collection)


def test_rejects_bad_hashes_counts_and_paths(collection, uploader):
    path = collection / "catalog.json"
    original = json.loads(path.read_text())
    for change in (lambda c: c["models"][0].update(weights_sha256="0" * 64),
                   lambda c: c.update(model_count=999),
                   lambda c: c["models"][0].update(path="../private")):
        metadata = json.loads(json.dumps(original))
        change(metadata)
        _write_json(path, metadata)
        with pytest.raises(ValueError):
            uploader.validate_collection(collection)


def test_underscored_legacy_stream_names_keep_the_declared_order(collection, uploader):
    path = collection / "models/gray_inner/fold_0/config.json"
    config = json.loads(path.read_text())
    config["streams"] = ["_vertical", "_horizontal"]
    _write_json(path, config)
    catalog_path = collection / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["models"][0]["streams"] = config["streams"]
    catalog["models"][0]["config_sha256"] = _sha(path)
    _write_json(catalog_path, catalog)
    _, _, prepared, _ = uploader.validate_collection(collection)
    assert prepared[0]["entry"]["streams"] == ["_vertical", "_horizontal"]


def test_direct_upload_commits_each_model_then_root(collection, uploader, monkeypatch):
    api = FakeApi()
    _hub(monkeypatch, api)
    result = uploader.upload_collection(collection)
    options = _uploads(api)
    assert [call["path_in_repo"] for call in options] == ["models/gray_inner/fold_0", "models/gray_inner/fold_1",
                                                        "historical/legacy_run", ""]
    assert all(set(call["allow_patterns"]) == uploader.MODEL_FILES for call in options[:-1])
    assert set(options[-1]["allow_patterns"]) == uploader.ROOT_FILES
    assert all(call["create_pr"] is False and "delete_patterns" not in call for call in options)
    assert result["uploaded"] == result["weights_uploaded"] == 3 and result["root_published"] is True
    state = json.loads((collection.parent / ".upload-state.json").read_text())
    assert len(state["entries"]) == 3
    assert all("weights_sha256" in entry and "config_sha256" in entry for entry in state["entries"].values())
    assert not any("state" in name for call in options for name in call["allow_patterns"])


def test_partial_failure_keeps_receipt_and_resume_rechecks_remote(collection, uploader, monkeypatch):
    api = FakeApi()
    api.fail_path = "models/gray_inner/fold_1"
    _hub(monkeypatch, api)
    with pytest.raises(RuntimeError, match="uploaded=1, skipped=0, root_published=false") as failure:
        uploader.upload_collection(collection)
    assert "hf_" not in str(failure.value)
    assert not any(call["path_in_repo"] == "" for call in _uploads(api))
    state = json.loads((collection.parent / ".upload-state.json").read_text())
    assert list(state["entries"]) == ["models/gray_inner/fold_0"]
    api.fail_path = None
    api.calls.clear()
    result = uploader.upload_collection(collection)
    assert result["skipped"] == 1 and result["uploaded"] == 2 and result["root_published"] is True
    assert _uploads(api)[-1]["path_in_repo"] == ""


def test_remote_config_tampering_is_not_hidden_by_local_receipt(collection, uploader, monkeypatch):
    api = FakeApi()
    _hub(monkeypatch, api)
    uploader.upload_collection(collection)
    api.remote["models/gray_inner/fold_0/config.json"].blob_id = "tampered"
    api.calls.clear()
    result = uploader.upload_collection(collection)
    assert result["uploaded"] == 1 and result["skipped"] == 2
    assert set(_uploads(api)[0]["allow_patterns"]) == uploader.MODEL_FILES


def test_remote_lfs_hash_tampering_forces_a_model_commit(collection, uploader, monkeypatch):
    api = FakeApi()
    _hub(monkeypatch, api)
    uploader.upload_collection(collection)
    api.remote["models/gray_inner/fold_0/model.safetensors"].lfs.sha256 = "0" * 64
    api.calls.clear()
    result = uploader.upload_collection(collection)
    assert result["weights_uploaded"] == 1 and result["skipped"] == 2


def test_remote_matches_allow_resume_without_a_local_receipt(collection, uploader, monkeypatch):
    api = FakeApi()
    _hub(monkeypatch, api)
    uploader.upload_collection(collection)
    (collection.parent / ".upload-state.json").unlink()
    api.calls.clear()
    result = uploader.upload_collection(collection)
    assert result["skipped"] == 3 and result["uploaded"] == 0
    assert [call["path_in_repo"] for call in _uploads(api)] == [""]


def test_local_source_change_before_upload_stops_without_catalog(collection, uploader, monkeypatch):
    api = FakeApi()
    original = api.get_paths_info
    def change_source(**kwargs):
        result = original(**kwargs)
        if api.read_calls == 1:
            (collection / "models/gray_inner/fold_0/README.md").write_text("changed after validation", encoding="utf-8")
        return result
    api.get_paths_info = change_source
    _hub(monkeypatch, api)
    with pytest.raises(RuntimeError, match="uploaded=0"):
        uploader.upload_collection(collection)
    assert not _uploads(api)


def test_readme_change_updates_only_the_small_file(collection, uploader, monkeypatch):
    api = FakeApi()
    _hub(monkeypatch, api)
    uploader.upload_collection(collection)
    (collection / "models/gray_inner/fold_0/README.md").write_text("# Updated code URL\n", encoding="utf-8")
    api.calls.clear()
    result = uploader.upload_collection(collection)
    assert result["uploaded"] == result["readmes_updated"] == 1
    assert result["weights_uploaded"] == 0 and result["skipped"] == 2
    assert _uploads(api)[0]["allow_patterns"] == ["README.md"]


def test_final_remote_tampering_withholds_catalog(collection, uploader, monkeypatch):
    api = FakeApi()
    api.tamper_after_reads = 4
    _hub(monkeypatch, api)
    with pytest.raises(RuntimeError, match="final remote model verification"):
        uploader.upload_collection(collection)
    assert not any(call["path_in_repo"] == "" for call in _uploads(api))


def test_state_must_stay_outside_collection(collection, uploader):
    with pytest.raises(ValueError, match="outside"):
        uploader.upload_collection(collection, verify_only=True, state_file=collection / "state.json")


def test_broken_state_link_is_rejected_without_creating_its_target(collection, uploader):
    state = collection.parent / "linked-state.json"
    target = collection.parent / "never-created.json"
    try:
        state.symlink_to(target)
    except OSError:
        pytest.skip("The local account cannot create symlinks")
    with pytest.raises(ValueError, match="Linked"):
        uploader.upload_collection(collection, verify_only=True, state_file=state)
    assert not target.exists()


def test_linked_collection_ancestor_is_rejected(collection, uploader):
    alias = collection.parent / "linked-parent"
    try:
        alias.symlink_to(collection.parent, target_is_directory=True)
    except OSError:
        pytest.skip("The local account cannot create symlinks")
    with pytest.raises(ValueError, match="Linked"):
        uploader.validate_collection(alias / collection.name)


def test_login_only_checks_write_access_without_upload_or_folder(uploader, monkeypatch):
    api = FakeApi()
    _hub(monkeypatch, api)
    uploader.main(["--login-only"])
    assert not _uploads(api)
    assert any(kind == "auth_check" and kwargs["write"] is True for kind, kwargs in api.calls)


class HubFailure(RuntimeError):
    def __init__(self, status):
        super().__init__("PRIVATE_AUTH_DETAILS hf_" + "A" * 40)
        self.response = SimpleNamespace(status_code=status) if status is not None else None


@pytest.mark.parametrize("role", ["fineGrained", None, "read"])
@pytest.mark.parametrize("orgs", [[], [{"name": "ntuhophai", "roleInOrg": "contributor"}]])
def test_repo_specific_write_grants_do_not_require_namespace_roles(uploader, monkeypatch, role, orgs):
    api = FakeApi()
    api.whoami = lambda: {"name": "Billy12345", "orgs": orgs, "auth": {"accessToken": {"role": role}}}
    _hub(monkeypatch, api)
    monkeypatch.setattr(uploader, "_local_login", lambda: pytest.fail("Server-authorized credentials need no new login"))
    assert uploader.ensure_auth() is api
    assert api.calls == [("auth_check", {"repo_id": "ntuhophai/ERM-AI", "repo_type": "model", "write": True})]
    assert not _uploads(api)


@pytest.mark.parametrize("status", [401, 403])
def test_denied_repository_write_rechecks_after_one_local_login(uploader, monkeypatch, status):
    denied, allowed = FakeApi(), FakeApi()
    def fail(**kwargs):
        denied.calls.append(("auth_check", kwargs))
        raise HubFailure(status)
    denied.auth_check = fail
    accounts = iter([denied, allowed])
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: next(accounts)))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    logins = []
    monkeypatch.setattr(uploader, "_local_login", lambda: logins.append(True))
    assert uploader.ensure_auth() is allowed
    assert logins == [True]
    assert len([call for call in denied.calls if call[0] == "auth_check"]) == 1
    assert len([call for call in allowed.calls if call[0] == "auth_check"]) == 1
    assert not _uploads(denied) and not _uploads(allowed)


def test_missing_local_token_can_login_once(uploader, monkeypatch):
    LocalTokenNotFoundError = type("LocalTokenNotFoundError", (RuntimeError,), {})
    missing, allowed = FakeApi(), FakeApi()
    def fail(**kwargs):
        raise LocalTokenNotFoundError("PRIVATE_AUTH_DETAILS")
    missing.auth_check = fail
    accounts = iter([missing, allowed])
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: next(accounts)))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    logins = []
    monkeypatch.setattr(uploader, "_local_login", lambda: logins.append(True))
    assert uploader.ensure_auth() is allowed
    assert logins == [True]


@pytest.mark.parametrize("status", [401, 403])
def test_denial_after_login_is_sanitized_and_does_not_loop(uploader, monkeypatch, capsys, status):
    api = FakeApi()
    api.whoami = lambda: {"name": "Billy12345", "orgs": [], "auth": {"accessToken": {"role": None}}}
    def fail(**kwargs):
        api.calls.append(("auth_check", kwargs))
        raise HubFailure(status)
    api.auth_check = fail
    _hub(monkeypatch, api)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    logins = []
    monkeypatch.setattr(uploader, "_local_login", lambda: logins.append(True))
    with pytest.raises(RuntimeError, match="denied write access") as failure:
        uploader.ensure_auth()
    assert logins == [True] and not _uploads(api)
    assert len([call for call in api.calls if call[0] == "auth_check"]) == 2
    assert "ntuhophai/ERM-AI for account Billy12345" in str(failure.value)
    assert "token/OAuth scope" in str(failure.value) and "administrator" in str(failure.value)
    assert "PRIVATE_AUTH_DETAILS" not in str(failure.value) + capsys.readouterr().out
    assert "hf_" not in str(failure.value)


def test_token_override_does_not_trigger_login(uploader, monkeypatch):
    api = FakeApi()
    def fail(**kwargs):
        raise HubFailure(403)
    api.auth_check = fail
    _hub(monkeypatch, api)
    monkeypatch.setenv("HF_TOKEN", "fixture-never-a-real-token")
    monkeypatch.setattr(uploader, "_local_login", lambda: pytest.fail("Must not overwrite token override"))
    with pytest.raises(RuntimeError, match="HF_TOKEN overrides"):
        uploader.ensure_auth()


def test_authorized_token_override_is_accepted(uploader, monkeypatch):
    api = FakeApi()
    api.whoami = lambda: {"name": "other", "orgs": []}
    _hub(monkeypatch, api)
    monkeypatch.setenv("HF_TOKEN", "fixture-never-a-real-token")
    monkeypatch.setattr(uploader, "_local_login", lambda: pytest.fail("Must not overwrite token override"))
    assert uploader.ensure_auth() is api


@pytest.mark.parametrize("status, message", [(404, "not found or is inaccessible"), (503, "Unable to reach"),
                                              (429, "Unable to reach"), (None, "Unable to reach")])
def test_non_auth_failure_does_not_trigger_login(uploader, monkeypatch, capsys, status, message):
    api = FakeApi()
    def fail(**kwargs):
        raise HubFailure(status)
    api.auth_check = fail
    _hub(monkeypatch, api)
    monkeypatch.setattr(uploader, "_local_login", lambda: pytest.fail("A repository/connection failure must not trigger login"))
    with pytest.raises(RuntimeError, match=message) as failure:
        uploader.ensure_auth()
    assert "PRIVATE_AUTH_DETAILS" not in str(failure.value) + capsys.readouterr().out
    assert "hf_" not in str(failure.value) and not _uploads(api)


@pytest.mark.parametrize("unsupported", [None, lambda repo_id, repo_type: None])
def test_old_sdk_without_write_check_requires_upgrade(uploader, monkeypatch, unsupported):
    api = FakeApi()
    api.auth_check = unsupported
    _hub(monkeypatch, api)
    monkeypatch.setattr(uploader, "_local_login", lambda: pytest.fail("An unsupported SDK must not trigger login"))
    with pytest.raises(RuntimeError, match="Upgrade huggingface_hub"):
        uploader.ensure_auth()
    assert not _uploads(api)


def test_identity_display_failure_does_not_override_server_write_success(uploader, monkeypatch, capsys):
    api = FakeApi()
    def fail():
        raise HubFailure(503)
    api.whoami = fail
    _hub(monkeypatch, api)
    monkeypatch.setattr(uploader, "_local_login", lambda: pytest.fail("Server-authorized credentials need no new login"))
    assert uploader.ensure_auth() is api
    output = capsys.readouterr().out
    assert "repository write access verified" in output and "PRIVATE_AUTH_DETAILS" not in output


def test_local_login_failure_is_sanitized(uploader, monkeypatch, capsys):
    api = FakeApi()
    def fail(**kwargs):
        raise HubFailure(401)
    def login_failure():
        raise RuntimeError("PRIVATE_AUTH_DETAILS hf_" + "A" * 40)
    api.auth_check = fail
    _hub(monkeypatch, api)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(uploader, "_local_login", login_failure)
    with pytest.raises(RuntimeError, match="login did not complete") as failure:
        uploader.ensure_auth()
    assert "PRIVATE_AUTH_DETAILS" not in str(failure.value) + capsys.readouterr().out
    assert "hf_" not in str(failure.value)


def test_missing_remote_hash_api_never_trusts_a_filename_listing(collection, uploader, monkeypatch):
    api = FakeApi()
    api.get_paths_info = None
    api.list_repo_files = lambda **kwargs: ["models/gray_inner/fold_0/model.safetensors"]
    _hub(monkeypatch, api)
    with pytest.raises(RuntimeError, match="uploaded=0"):
        uploader.upload_collection(collection)
    assert not _uploads(api)


class FakePrApi(FakeApi):
    def __init__(self):
        super().__init__()
        self.account = "Billy12345"
        self.pr_num = 7
        self.pr_status = "draft"
        self.pr_author = self.account
        self.pr_repo = "ntuhophai/ERM-AI"
        self.pr_type = "model"
        self.pr_target = "main"
        self.is_pr = True
        self.pr_merge = None
        self.create_failure = None
        self.create_attempts = 0
        self.main_remote = {}
        self.root_tamper = False

    def whoami(self):
        self.calls.append(("identity", {}))
        return {"name": self.account, "orgs": [], "auth": {"accessToken": {"role": "read"}}}

    def _details(self):
        return SimpleNamespace(repo_id=self.pr_repo, repo_type=self.pr_type, author=self.pr_author,
                               num=self.pr_num, status=self.pr_status, is_pull_request=self.is_pr,
                               target_branch=self.pr_target, merge_commit_oid=self.pr_merge,
                               git_reference="refs/pr/%d" % self.pr_num,
                               url="https://huggingface.co/%s/discussions/%d" % (self.pr_repo, self.pr_num))

    def create_pull_request(self, **kwargs):
        self.calls.append(("create_pr", kwargs))
        self.create_attempts += 1
        if self.create_failure is not None:
            raise self.create_failure
        return self._details()

    def get_discussion_details(self, **kwargs):
        self.calls.append(("discussion", kwargs))
        assert kwargs["repo_id"] == "ntuhophai/ERM-AI" and kwargs["repo_type"] == "model"
        assert kwargs["discussion_num"] == self.pr_num
        return self._details()

    def get_paths_info(self, **kwargs):
        assert kwargs["revision"] == "refs/pr/%d" % self.pr_num
        return super().get_paths_info(**kwargs)

    def upload_folder(self, **kwargs):
        assert kwargs["revision"] == "refs/pr/%d" % self.pr_num and kwargs["create_pr"] is False
        result = super().upload_folder(**kwargs)
        if self.root_tamper and kwargs["path_in_repo"] == "":
            self.remote["catalog.json"].blob_id = "tampered"
        return result

    def change_discussion_status(self, **kwargs):
        self.calls.append(("status", kwargs))
        assert kwargs["new_status"] == "open"
        self.pr_status = "open"


def _creation_denial(status):
    error = HubFailure(status)
    error.response.request = SimpleNamespace(method="POST", url="https://huggingface.co/api/models/ntuhophai/ERM-AI/discussions")
    return error


def _prepare(uploader, api, monkeypatch, state):
    _hub(monkeypatch, api)
    return uploader.prepare_pr(state_file=state)


def test_pr_login_only_needs_identity_and_readable_repo_without_main_write(uploader, monkeypatch):
    api = FakePrApi()
    _hub(monkeypatch, api)
    uploader.main(["--mode", "pr", "--login-only"])
    assert any(kind == "info" for kind, _ in api.calls)
    assert not any(kind in ("auth_check", "create_pr", "upload", "status") for kind, _ in api.calls)


def test_pr_prepare_creates_one_draft_receipt_before_any_weights(uploader, monkeypatch, tmp_path):
    api = FakePrApi()
    state_file = tmp_path / "pr-state.json"
    state = _prepare(uploader, api, monkeypatch, state_file)
    assert state["account"] == "Billy12345" and state["mode"] == "pr"
    assert state["pr_num"] == 7 and state["pr_ref"] == "refs/pr/7" and state["target_branch"] == "main"
    assert state["pr_url"] == "https://huggingface.co/ntuhophai/ERM-AI/discussions/7"
    assert state["pr_status"] == "draft" and state["entries"] == {}
    assert json.loads(state_file.read_text()) == state
    assert api.create_attempts == 1 and not _uploads(api)
    assert not any(kind in ("auth_check", "status") for kind, _ in api.calls)
    again = uploader.prepare_pr(state_file=state_file)
    assert again["pr_ref"] == state["pr_ref"] and api.create_attempts == 1


def test_pr_cli_prepare_allows_no_folder(uploader, monkeypatch, tmp_path):
    api = FakePrApi()
    _hub(monkeypatch, api)
    state = tmp_path / "pr-state.json"
    uploader.main(["--mode", "pr", "--prepare-pr", "--state-file", str(state)])
    assert state.exists() and api.create_attempts == 1 and not _uploads(api)


def test_pr_code_description_accepts_only_a_public_github_snapshot(uploader, monkeypatch, tmp_path):
    api = FakePrApi()
    _hub(monkeypatch, api)
    code = "https://github.com/chou141253/ERM-AI/tree/" + "a" * 40 + "/new"
    uploader.prepare_pr(state_file=tmp_path / "pr-state.json", code_url=code)
    description = [kwargs["description"] for kind, kwargs in api.calls if kind == "create_pr"][0]
    assert code in description and "PRIVATE" not in description
    with pytest.raises(ValueError, match="snapshot URL"):
        uploader.prepare_pr(state_file=tmp_path / "other-state.json", code_url="https://example.com/secret")
    assert not (tmp_path / "other-state.json").exists()


def test_pr_upload_pins_every_check_and_commit_then_verifies_and_opens(collection, uploader, monkeypatch):
    api = FakePrApi()
    state_file = collection.parent / "pr-state.json"
    _prepare(uploader, api, monkeypatch, state_file)
    result = uploader.upload_collection(collection, mode="pr", state_file=state_file)
    calls = _uploads(api)
    assert len(calls) == 4 and calls[-1]["path_in_repo"] == ""
    assert all(call["revision"] == "refs/pr/7" and call["create_pr"] is False for call in calls)
    assert all(kwargs["revision"] == "refs/pr/7" for kind, kwargs in api.calls if kind == "paths")
    assert result["pr_ref"] == "refs/pr/7" and result["pr_status"] == "open"
    state = json.loads(state_file.read_text())
    assert state["collection_complete"] is True and state["pr_status"] == "open" and len(state["entries"]) == 3
    status_index = next(index for index, call in enumerate(api.calls) if call[0] == "status")
    assert api.calls[status_index - 1][0] == "discussion"
    assert any(kind == "paths" and "catalog.json" in kwargs["paths"] for kind, kwargs in api.calls[:status_index])
    assert api.create_attempts == 1 and not api.main_remote
    assert not any(kind == "auth_check" for kind, _ in api.calls)


def test_pr_resume_uses_existing_branch_and_does_not_create_or_open_again(collection, uploader, monkeypatch):
    api = FakePrApi()
    state = collection.parent / "pr-state.json"
    _prepare(uploader, api, monkeypatch, state)
    uploader.upload_collection(collection, mode="pr", state_file=state)
    api.calls.clear()
    result = uploader.upload_collection(collection, mode="pr", state_file=state)
    assert result["skipped"] == 3 and result["uploaded"] == 0 and result["pr_status"] == "open"
    assert len(_uploads(api)) == 1 and _uploads(api)[0]["path_in_repo"] == ""
    assert not any(kind in ("create_pr", "status") for kind, _ in api.calls)


def test_pr_partial_failure_stays_draft_and_resumes_same_pr(collection, uploader, monkeypatch):
    api = FakePrApi()
    state = collection.parent / "pr-state.json"
    _prepare(uploader, api, monkeypatch, state)
    api.fail_path = "models/gray_inner/fold_1"
    with pytest.raises(RuntimeError, match="uploaded=1"):
        uploader.upload_collection(collection, mode="pr", state_file=state)
    assert api.pr_status == "draft" and api.create_attempts == 1
    assert json.loads(state.read_text())["pr_ref"] == "refs/pr/7"
    assert not any(call["path_in_repo"] == "" for call in _uploads(api))
    api.fail_path = None
    result = uploader.upload_collection(collection, mode="pr", state_file=state)
    assert result["skipped"] == 1 and result["uploaded"] == 2 and api.create_attempts == 1
    assert api.pr_status == "open"


def test_pr_root_hash_failure_does_not_request_review(collection, uploader, monkeypatch):
    api = FakePrApi()
    api.root_tamper = True
    state = collection.parent / "pr-state.json"
    _prepare(uploader, api, monkeypatch, state)
    with pytest.raises(RuntimeError, match="final remote root verification"):
        uploader.upload_collection(collection, mode="pr", state_file=state)
    assert api.pr_status == "draft" and not any(kind == "status" for kind, _ in api.calls)
    assert json.loads(state.read_text())["pr_status"] == "draft"


@pytest.mark.parametrize("field,value", [("pr_author", "SomeoneElse"), ("pr_repo", "someone/other"),
                                       ("pr_type", "dataset"), ("pr_target", "other"), ("is_pr", False),
                                       ("pr_status", "closed"), ("pr_status", "merged"), ("pr_merge", "abc")])
def test_pr_receipt_rejects_changed_author_repo_type_target_or_closed(collection, uploader, monkeypatch, field, value):
    api = FakePrApi()
    state = collection.parent / "pr-state.json"
    _prepare(uploader, api, monkeypatch, state)
    setattr(api, field, value)
    with pytest.raises(RuntimeError):
        uploader.upload_collection(collection, mode="pr", state_file=state)
    assert not _uploads(api) and api.create_attempts == 1


def test_pr_state_cannot_be_reused_by_another_account(collection, uploader, monkeypatch):
    api = FakePrApi()
    state = collection.parent / "pr-state.json"
    _prepare(uploader, api, monkeypatch, state)
    api.account = "AnotherUser"
    with pytest.raises(ValueError, match="different repository/account"):
        uploader.upload_collection(collection, mode="pr", state_file=state)
    assert not _uploads(api) and api.create_attempts == 1


def test_pr_upload_without_receipt_never_creates_a_pr(collection, uploader, monkeypatch):
    api = FakePrApi()
    _hub(monkeypatch, api)
    with pytest.raises(ValueError, match="prepare-pr"):
        uploader.upload_collection(collection, mode="pr", state_file=collection.parent / "missing.json")
    assert api.calls == []


def test_pr_verify_only_has_no_auth_network_create_or_state(collection, uploader, monkeypatch):
    api = FakePrApi()
    _hub(monkeypatch, api)
    state = collection.parent / "nonexistent-pr-state.json"
    result = uploader.upload_collection(collection, mode="pr", verify_only=True, state_file=state)
    assert result["verified_only"] is True and api.calls == [] and not state.exists()


def test_pr_unknown_identity_does_not_create_or_write_intent(uploader, monkeypatch, tmp_path):
    api = FakePrApi()
    api.whoami = lambda: {"orgs": []}
    _hub(monkeypatch, api)
    state = tmp_path / "pr-state.json"
    with pytest.raises(RuntimeError, match="username could not be verified"):
        uploader.prepare_pr(state_file=state)
    assert not state.exists() and api.create_attempts == 0


@pytest.mark.parametrize("status", [401, 403])
def test_pr_creation_auth_denial_retries_login_once_then_clears_only_intent(uploader, monkeypatch, tmp_path, status):
    api = FakePrApi()
    api.create_failure = _creation_denial(status)
    _hub(monkeypatch, api)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    logins = []
    monkeypatch.setattr(uploader, "_local_login", lambda: logins.append(True))
    state = tmp_path / "pr-state.json"
    with pytest.raises(RuntimeError, match="write-discussions") as failure:
        uploader.prepare_pr(state_file=state)
    assert logins == [True] and api.create_attempts == 2 and not state.exists()
    assert "PRIVATE_AUTH_DETAILS" not in str(failure.value) and "hf_" not in str(failure.value)


def test_pr_creation_can_succeed_after_local_scope_reauthorization(uploader, monkeypatch, tmp_path):
    api = FakePrApi()
    api.create_failure = _creation_denial(403)
    _hub(monkeypatch, api)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    def authorize():
        api.create_failure = None
    monkeypatch.setattr(uploader, "_local_login", authorize)
    state = uploader.prepare_pr(state_file=tmp_path / "pr-state.json")
    assert state["pr_num"] == 7 and api.create_attempts == 2


def test_pr_creation_token_override_is_not_replaced(uploader, monkeypatch, tmp_path):
    api = FakePrApi()
    api.create_failure = _creation_denial(403)
    _hub(monkeypatch, api)
    monkeypatch.setenv("HF_TOKEN", "fixture-never-a-real-token")
    monkeypatch.setattr(uploader, "_local_login", lambda: pytest.fail("Must not replace an explicit token"))
    state = tmp_path / "pr-state.json"
    with pytest.raises(RuntimeError, match="HF_TOKEN overrides"):
        uploader.prepare_pr(state_file=state)
    assert api.create_attempts == 1 and not state.exists()


def test_pr_unknown_creation_outcome_preserves_intent_and_never_duplicates(uploader, monkeypatch, tmp_path):
    api = FakePrApi()
    api.create_failure = HubFailure(503)
    _hub(monkeypatch, api)
    state = tmp_path / "pr-state.json"
    with pytest.raises(RuntimeError, match="no confirmed outcome"):
        uploader.prepare_pr(state_file=state)
    assert json.loads(state.read_text())["preparation_status"] == "creating" and api.create_attempts == 1
    api.create_failure = None
    with pytest.raises(RuntimeError, match="no confirmed receipt"):
        uploader.prepare_pr(state_file=state)
    assert api.create_attempts == 1 and not _uploads(api)


def test_pr_direct_receipt_is_rejected_without_creating_a_new_pr(uploader, monkeypatch, tmp_path):
    api = FakePrApi()
    _hub(monkeypatch, api)
    state = tmp_path / "pr-state.json"
    _write_json(state, {"state_version": 1, "repo": "ntuhophai/ERM-AI", "entries": {}})
    with pytest.raises(ValueError, match="different repository/account"):
        uploader.prepare_pr(state_file=state)
    assert api.create_attempts == 0 and json.loads(state.read_text())["state_version"] == 1


@pytest.mark.parametrize("http_request", [None, SimpleNamespace(method="GET", url="https://huggingface.co/api/models/ntuhophai/ERM-AI/discussions/7"),
                                    SimpleNamespace(method="POST", url="https://huggingface.co/api/models/another/repo/discussions")])
def test_pr_followup_get_denial_or_unknown_endpoint_never_creates_twice(uploader, monkeypatch, tmp_path, http_request):
    api = FakePrApi()
    error = HubFailure(403)
    error.response.request = http_request
    api.create_failure = error
    _hub(monkeypatch, api)
    monkeypatch.setattr(uploader, "_local_login", lambda: pytest.fail("Unknown creation must not automatically reauthorize/recreate"))
    state = tmp_path / "pr-state.json"
    with pytest.raises(RuntimeError, match="no confirmed outcome"):
        uploader.prepare_pr(state_file=state)
    assert json.loads(state.read_text())["preparation_status"] == "creating" and api.create_attempts == 1
    api.create_failure = None
    with pytest.raises(RuntimeError, match="no confirmed receipt"):
        uploader.prepare_pr(state_file=state)
    assert api.create_attempts == 1


def test_pr_identity_failure_after_access_check_is_sanitized(uploader, monkeypatch, tmp_path, capsys):
    api = FakePrApi()
    count = 0
    def identity():
        nonlocal count
        count += 1
        if count == 1:
            return {"name": "Billy12345"}
        raise HubFailure(503)
    api.whoami = identity
    _hub(monkeypatch, api)
    with pytest.raises(RuntimeError, match="username could not be verified") as failure:
        uploader.prepare_pr(state_file=tmp_path / "pr-state.json")
    assert api.create_attempts == 0 and "PRIVATE_AUTH_DETAILS" not in str(failure.value) + capsys.readouterr().out
    assert "hf_" not in str(failure.value)


def test_pr_qualified_main_target_uses_real_sdk_details_and_same_pr(collection, uploader, monkeypatch):
    from datetime import datetime, timezone
    from huggingface_hub.community import DiscussionWithDetails
    api = FakePrApi()
    def details():
        return DiscussionWithDetails(
            title="Publish complete OCT stream/fold model collection", num=api.pr_num,
            author=api.account, status=api.pr_status, created_at=datetime.now(timezone.utc),
            repo_id=api.pr_repo, repo_type="model", is_pull_request=True,
            endpoint="https://huggingface.co", events=[], conflicting_files=[], target_branch="refs/heads/main",
            merge_commit_oid=None, diff=None,
        )
    monkeypatch.setattr(api, "_details", details)
    state_path = collection.parent / "qualified-pr-state.json"
    state = _prepare(uploader, api, monkeypatch, state_path)
    assert state["target_branch"] == "main" and state["pr_ref"] == "refs/pr/7"
    result = uploader.upload_collection(collection, mode="pr", state_file=state_path)
    assert result["pr_status"] == "open" and api.create_attempts == 1
    assert all(call["revision"] == "refs/pr/7" for call in _uploads(api))
    uploader.prepare_pr(state_file=state_path)
    assert api.create_attempts == 1


@pytest.mark.parametrize("target", [None, "refs/heads/other"])
def test_pr_ambiguous_or_different_target_is_rejected(collection, uploader, monkeypatch, target):
    api = FakePrApi()
    state_path = collection.parent / "invalid-target-pr-state.json"
    _prepare(uploader, api, monkeypatch, state_path)
    api.pr_target = target
    with pytest.raises(RuntimeError, match="does not match"):
        uploader.upload_collection(collection, mode="pr", state_file=state_path)
    assert not _uploads(api) and api.create_attempts == 1

"""Validate and publish a public model collection in resumable model commits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile


ROOT_FILES = frozenset({"catalog.json", "modeling_multistream.py", "README.md", "requirements.txt"})
MODEL_FILES = frozenset({"model.safetensors", "config.json", "README.md"})
CATALOG_FIELDS = frozenset({"catalog_version", "models", "inference_ready_count", "historical_count",
                          "model_count", "condition_count", "fold_count", "total_weight_bytes",
                          "private_training_metadata_omitted"})
ENTRY_REQUIRED = frozenset({"condition", "fold", "streams", "path", "inference_ready",
                            "weights_sha256", "config_sha256"})
ENTRY_OPTIONAL = frozenset({"artifact_kind", "source_format", "settings_provenance", "weights_bytes",
                            "project", "reason"})
HISTORICAL_FIELDS = frozenset({"historical_format_version", "condition", "inference_ready", "streams",
                              "class_to_idx", "model_config", "preprocessing", "source_format",
                              "private_training_metadata_omitted", "reason"})
SAFE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
DTYPE_BYTES = {"BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4,
               "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
               "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}


def _tool(name):
    spec = importlib.util.spec_from_file_location("_collection_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_public = _tool("upload_public_model")
_exporter = _tool("export_public_model")
_local_login = _public._local_login
_identity = _public._identity
_known_write_account = _public._known_write_account


def _name(value):
    return isinstance(value, str) and SAFE_NAME.fullmatch(value) is not None and ".." not in value


def _linklike(path):
    # Windows directory junctions are reparse points, including on Python 3.11.
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    return path.is_symlink() or bool(getattr(details, "st_file_attributes", 0) & 0x400)


def _ordinary_location(path):
    location = Path(path).absolute()
    if any(_linklike(part) for part in (location, *location.parents)):
        raise ValueError("Linked collection and external state locations are forbidden")
    return location.resolve()


def _json(path, limit=4 * 1024 * 1024):
    _signature(path)
    if path.stat().st_size > limit:
        raise ValueError("Public JSON exceeds the supported size")
    text = path.read_text(encoding="utf-8")
    if _public.TOKEN_PATTERN.search(text):
        raise ValueError("Credential-like text found in public JSON")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate public JSON key")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite public JSON")))


def _signature(path):
    if _linklike(path) or not path.is_file():
        raise ValueError("Public collection files must be regular files without links")
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _hashes(path):
    before = _signature(path)
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(("blob %d\0" % before[0]).encode("ascii"))
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            blob.update(chunk)
    if before != _signature(path):
        raise ValueError("A public file changed during validation")
    return {"sha256": sha256.hexdigest(), "blob_id": blob.hexdigest(), "size": before[0], "signature": before}


def _header(path):
    """Check the safetensors structure without unpickling or allocating model weights."""
    before = _signature(path)
    size = path.stat().st_size
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("Invalid safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(16 * 1024 * 1024, size - 8):
            raise ValueError("Invalid safetensors header length")
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate safetensors header key")
                result[key] = value
            return result
        header = json.loads(source.read(length), object_pairs_hook=unique_pairs)
    if not isinstance(header, dict) or ("__metadata__" in header and header["__metadata__"] != {}):
        raise ValueError("Safetensors must not contain private non-tensor metadata")
    offsets = []
    for key, tensor in header.items():
        if key == "__metadata__":
            continue
        if (not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9_.]+", key) is None
                or not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}):
            raise ValueError("Invalid safetensors tensor header")
        shape, bounds = tensor["shape"], tensor["data_offsets"]
        if (tensor["dtype"] not in DTYPE_BYTES or not isinstance(shape, list)
                or any(type(dim) is not int or dim < 0 for dim in shape)
                or not isinstance(bounds, list) or len(bounds) != 2
                or any(type(offset) is not int or offset < 0 for offset in bounds)
                or bounds[1] < bounds[0]
                or bounds[1] - bounds[0] != math.prod(shape) * DTYPE_BYTES[tensor["dtype"]]):
            raise ValueError("Invalid safetensors tensor sizes")
        offsets.append(tuple(bounds))
    if not offsets:
        raise ValueError("Safetensors must contain tensors")
    end = 0
    for start, stop in sorted(offsets):
        if start != end:
            raise ValueError("Safetensors data offsets overlap or contain holes")
        end = stop
    if end != size - 8 - length:
        raise ValueError("Safetensors data length does not match its header")
    if before != _signature(path):
        raise ValueError("Safetensors changed during header validation")
    return before


def _metadata(config, entry):
    if entry["inference_ready"]:
        if (not isinstance(config, dict) or set(config) != _public.PUBLIC_FIELDS
                or type(config["public_format_version"]) is not int or config["public_format_version"] != 1
                or config["private_training_metadata_omitted"] is not True
                or type(config["legacy_no_split_metadata"]) is not bool):
            raise ValueError("Unexpected public inference metadata")
        if (not isinstance(config["model_config"], dict) or set(config["model_config"]) != _public.MODEL_FIELDS
                or not _name(config["model_config"]["name"])
                or config["model_config"]["pretrained"] is not False
                or config["model_config"] != _exporter._safe_model_config(config["model_config"])):
            raise ValueError("Unexpected nested model metadata")
        if (not isinstance(config["preprocessing"], dict)
                or set(config["preprocessing"]) != _public.PREPROCESSING_FIELDS
                or config["preprocessing"] != _exporter._public_preprocessing(config["preprocessing"], True)):
            raise ValueError("Unexpected nested preprocessing metadata")
        _exporter._class_mapping(config["class_to_idx"])
        if any(not _name(name) for name in config["class_to_idx"]):
            raise ValueError("Public class names must be safe labels")
        if config["streams"] != entry["streams"]:
            raise ValueError("Catalog and config stream order differ")
        if config["artifact_kind"] not in ("research", "demo"):
            raise ValueError("Unsupported public artifact kind")
        if config["settings_provenance"] not in ("checkpoint_metadata", "operator_supplied", "declared_profiles"):
            raise ValueError("Unsupported public settings provenance")
        if config["source_format"] not in ("versioned_checkpoint", "converted_legacy", "legacy_state_dict"):
            raise ValueError("Unsupported public source format")
    else:
        if (not isinstance(config, dict) or set(config) != HISTORICAL_FIELDS
                or type(config["historical_format_version"]) is not int or config["historical_format_version"] != 1
                or config["condition"] != entry["condition"] or config["inference_ready"] is not False
                or any(config[field] is not None for field in ("streams", "class_to_idx", "model_config", "preprocessing"))
                or config["source_format"] != "legacy_raw_state_dict"
                or config["private_training_metadata_omitted"] is not True
                or not isinstance(config["reason"], str) or not config["reason"]):
            raise ValueError("Unexpected historical metadata; unknown inference settings must remain null")
    for field in ("artifact_kind", "source_format", "settings_provenance", "reason"):
        if field in entry and field in config and entry[field] != config[field]:
            raise ValueError("Catalog and config public provenance differ")


def validate_collection(folder):
    folder = _ordinary_location(folder)
    if not folder.is_dir():
        raise ValueError("The collection folder must be a directory without links")
    catalog = _json(folder / "catalog.json")
    if (not isinstance(catalog, dict) or set(catalog) != CATALOG_FIELDS
            or type(catalog["catalog_version"]) is not int or catalog["catalog_version"] != 1
            or catalog["private_training_metadata_omitted"] is not True
            or not isinstance(catalog["models"], list) or not catalog["models"]):
        raise ValueError("Unexpected public catalog schema")
    paths, expected_files, expected_dirs, prepared = set(), set(ROOT_FILES), set(), []
    for entry in catalog["models"]:
        if (not isinstance(entry, dict) or not ENTRY_REQUIRED <= set(entry)
                or not set(entry) <= ENTRY_REQUIRED | ENTRY_OPTIONAL
                or not _name(entry["condition"]) or type(entry["inference_ready"]) is not bool
                or any(not isinstance(entry[key], str) or not SHA256.fullmatch(entry[key])
                       for key in ("weights_sha256", "config_sha256"))):
            raise ValueError("Unexpected public catalog entry")
        if any(not isinstance(entry[field], str) for field in ENTRY_OPTIONAL - {"weights_bytes"} if field in entry):
            raise ValueError("Optional public provenance fields must be plain text")
        if entry["inference_ready"]:
            streams, fold = entry["streams"], entry["fold"]
            if (type(fold) is not int or not 0 <= fold <= 9 or not isinstance(streams, list)
                    or not 1 <= len(streams) <= 8 or any(not _name(stream) for stream in streams)
                    or len(set(streams)) != len(streams)):
                raise ValueError("Ready entries require an ordered stream list and fold 0 through 9")
            expected_path = "models/%s/fold_%d" % (entry["condition"], fold)
        else:
            project = entry.get("project")
            if (not _name(project) or entry["fold"] is not None or entry["streams"] is not None
                    or entry["condition"] != "historical_" + project):
                raise ValueError("Historical entries require a safe project and unknown stream/fold settings")
            expected_path = "historical/" + project
        if entry["path"] != expected_path or expected_path in paths:
            raise ValueError("Catalog contains an invalid or duplicate model path")
        paths.add(expected_path)
        parts = expected_path.split("/")
        expected_dirs.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
        expected_files.update(expected_path + "/" + name for name in MODEL_FILES)
    actual_files, actual_dirs, stack = set(), set(), [folder]
    while stack:
        for item in stack.pop().iterdir():
            if _linklike(item):
                raise ValueError("Symlinks and directory junctions are forbidden in public collections")
            relative = item.relative_to(folder).as_posix()
            if item.is_dir():
                actual_dirs.add(relative)
                stack.append(item)
            elif item.is_file():
                actual_files.add(relative)
            else:
                raise ValueError("Unexpected non-file artifact in public collection")
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise ValueError("Collection requires exactly the public root files and three files per catalog entry")
    if (folder / "modeling_multistream.py").read_text(encoding="utf-8") != _exporter.PUBLIC_LOADER:
        raise ValueError("The collection public loader was modified")
    for relative in actual_files:
        if relative.endswith("model.safetensors"):
            continue
        file = folder / relative
        if file.stat().st_size > 4 * 1024 * 1024 or _public.TOKEN_PATTERN.search(file.read_text(encoding="utf-8")):
            raise ValueError("Unexpected large text or credential-like text in public collection")
    allowed_packages = {"torch", "torchvision", "timm", "safetensors", "pillow", "numpy", "huggingface-hub"}
    for line in (folder / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([A-Za-z0-9_-]+)==([A-Za-z0-9][A-Za-z0-9.+!-]*)", line.strip())
        if not match or match[1].lower().replace("_", "-") not in allowed_packages:
            raise ValueError("Public requirements must contain only pinned inference packages")
    for entry in catalog["models"]:
        directory = folder / entry["path"]
        config = _json(directory / "config.json")
        _metadata(config, entry)
        header_signature = _header(directory / "model.safetensors")
        hashes = {name: _hashes(directory / name) for name in MODEL_FILES}
        if hashes["model.safetensors"]["signature"] != header_signature:
            raise ValueError("Safetensors changed between header and content validation")
        if (hashes["model.safetensors"]["sha256"] != entry["weights_sha256"]
                or hashes["config.json"]["sha256"] != entry["config_sha256"]
                or ("weights_bytes" in entry and (type(entry["weights_bytes"]) is not int
                    or entry["weights_bytes"] != hashes["model.safetensors"]["size"]))):
            raise ValueError("Catalog weights/config hash or byte count differs from local files")
        prepared.append({"entry": entry, "folder": directory, "hashes": hashes, "root": folder})
    ready = [item["entry"] for item in prepared if item["entry"]["inference_ready"]]
    counts = {"model_count": len(prepared), "inference_ready_count": len(ready),
              "historical_count": len(prepared) - len(ready), "condition_count": len({e["condition"] for e in ready}),
              "fold_count": len({e["fold"] for e in ready}),
              "total_weight_bytes": sum(item["hashes"]["model.safetensors"]["size"] for item in prepared)}
    if any(type(catalog[key]) is not int or catalog[key] != count for key, count in counts.items()):
        raise ValueError("Catalog aggregate counts differ from its model entries")
    return folder, catalog, prepared, {name: _hashes(folder / name) for name in ROOT_FILES}


def _value(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _remote_status(api, repo_id, prepared, revision=None):
    method = getattr(api, "get_paths_info", None)
    if method is None:
        raise RuntimeError("Upgrade huggingface_hub: remote file hashes are required for safe resume")
    prefix, hashes = prepared["entry"]["path"], prepared["hashes"]
    paths = [(prefix + "/" if prefix else "") + name for name in sorted(hashes)]
    options = {"revision": revision} if revision is not None else {}
    try:
        remote = method(repo_id=repo_id, repo_type="model", paths=paths, **options)
    except Exception as error:
        if getattr(getattr(error, "response", None), "status_code", None) == 404:
            return {name: False for name in hashes}
        raise RuntimeError("Unable to verify remote model content for safe resume") from None
    indexed = {_value(file, "path"): file for file in remote}
    status = {}
    for name, local in hashes.items():
        file = indexed.get((prefix + "/" if prefix else "") + name)
        if file is None or _value(file, "size", local["size"]) != local["size"]:
            status[name] = False
            continue
        lfs = _value(file, "lfs")
        if lfs:
            status[name] = _value(lfs, "sha256") == local["sha256"]
        else:
            status[name] = _value(file, "blob_id") == local["blob_id"]
    return status


def _remote_matches(api, repo_id, prepared, revision=None):
    return all(_remote_status(api, repo_id, prepared, revision).values())


def _state_path(folder, requested):
    path = Path(requested).absolute() if requested else folder.parent / ".upload-state.json"
    path = _ordinary_location(path)
    if path == folder or folder in path.parents:
        raise ValueError("Upload state must be outside the public collection")
    if not path.parent.is_dir():
        raise ValueError("The upload state parent directory does not exist")
    if path.exists() and not path.is_file():
        raise ValueError("Upload state must be a regular file")
    return path


def _save_state(path, state):
    _ordinary_location(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".collection-state-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(state, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _unchanged(prepared):
    directory = prepared["folder"]
    root = prepared.get("root", directory)
    while True:
        if _linklike(directory) or not directory.is_dir():
            raise ValueError("Public collection directories changed or became links after validation")
        if directory == root:
            break
        directory = directory.parent
    for name, expected in prepared["hashes"].items():
        if _signature(prepared["folder"] / name) != expected["signature"]:
            raise ValueError("Public collection changed after validation; validate again")


def _validate_repo(repo_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise ValueError("Expected a Hugging Face owner/model repository ID")


def _account_name(api):
    info, _ = _identity(api)
    name = info.get("name") if isinstance(info, dict) else None
    return name if _name(name) and not _public.TOKEN_PATTERN.search(name) else None


def _validate_mode(mode):
    if mode not in ("direct", "pr"):
        raise ValueError("The publishing mode must be direct or pr")


def _pr_account(api):
    info = api.whoami()
    name = _value(info, "name")
    if not _name(name) or name.lower() in ("unknown", "deleted") or _public.TOKEN_PATTERN.search(name):
        raise RuntimeError("The authenticated Hugging Face username could not be verified; no PR was created")
    return name


def _confirmed_pr_account(api):
    try:
        return _pr_account(api)
    except Exception:
        raise RuntimeError("The authenticated Hugging Face username could not be verified; "
                           "the existing PR/state are preserved") from None


def _auth_status(error):
    return getattr(getattr(error, "response", None), "status_code", None)


def _login_for_pr():
    if os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN overrides local login. Provide a token authorized for pull requests locally "
                           "or unset the override before browser login. Never paste tokens into scripts/chat") from None
    print("Log in locally for Hugging Face pull requests. Browser authorization needs write-discussions; "
          "a fine-grained token must allow discussions/PRs on the target repository.", flush=True)
    try:
        _local_login()
    except Exception:
        raise RuntimeError("Local Hugging Face login did not complete. Retry the Hugging Face launcher") from None


def _ensure_pr_auth(repo_id):
    """Check a real signed-in contributor and readable repo; PRs need no main-branch write access."""
    from huggingface_hub import HfApi
    for attempt in range(2):
        api = HfApi()
        try:
            account = _pr_account(api)
            api.repo_info(repo_id=repo_id, repo_type="model")
        except Exception as error:
            status = _auth_status(error)
            missing = type(error).__name__ == "LocalTokenNotFoundError"
            if status in (401, 403) or missing:
                if attempt == 0:
                    _login_for_pr()
                    continue
                raise RuntimeError("Hugging Face could not authorize this contributor for the readable target repository. "
                                   "Check local login, repository visibility, and token/OAuth permissions") from None
            if status == 404:
                raise RuntimeError("The target Hugging Face repository was not found or is inaccessible. "
                                   "Check the repository ID and visibility") from None
            if isinstance(error, RuntimeError) and str(error).startswith("The authenticated Hugging Face username"):
                raise error from None
            raise RuntimeError("Unable to verify the authenticated Hugging Face contributor and target repository. "
                               "Check the connection and installed huggingface_hub version") from None
        print("Using local Hugging Face account: " + account, flush=True)
        print("Hugging Face target is readable for pull-request publishing: " + repo_id, flush=True)
        return api


def _pr_state_path(requested):
    if not requested:
        raise ValueError("PR publishing requires an explicit external --state-file")
    path = _ordinary_location(Path(requested).absolute())
    if not path.parent.is_dir() or (path.exists() and not path.is_file()):
        raise ValueError("PR state needs an existing parent directory and a regular file")
    return path


def _read_pr_state(path, repo_id, account):
    try:
        state = _json(path)
    except Exception:
        raise ValueError("Invalid PR state; preserve it and choose a separate state file only for a new PR") from None
    if (not isinstance(state, dict) or state.get("state_version") != 2 or state.get("mode") != "pr"
            or state.get("repo") != repo_id or state.get("account") != account
            or not isinstance(state.get("entries"), dict)):
        raise ValueError("PR state belongs to a different repository/account or publishing mode; no upload performed")
    if state.get("preparation_status") == "creating":
        raise RuntimeError("The previous PR creation has no confirmed receipt. Inspect the repository Discussions "
                           "for an existing draft and preserve this state. Share only its PR URL to recover "
                           "the receipt; do not rename the state and blindly create another PR")
    number = state.get("pr_num")
    if (type(number) is not int or number < 1 or state.get("pr_ref") != "refs/pr/%d" % number
            or state.get("pr_url") != "https://huggingface.co/%s/discussions/%d" % (repo_id, number)
            or state.get("target_branch") != "main" or state.get("pr_status") not in ("draft", "open")):
        raise ValueError("Invalid PR number/ref/URL/target/status in local state; no upload performed")
    return state


def _canonical_pr_target(value):
    # Hub responses can return the fully qualified Git ref for the main branch.
    return "main" if value in ("main", "refs/heads/main") else value


def _validate_pr_details(details, state, account):
    if (_value(details, "repo_id") != state["repo"] or _value(details, "repo_type") != "model"
            or _value(details, "author") != account or _value(details, "is_pull_request") is not True
            or _value(details, "num") != state["pr_num"] or _value(details, "git_reference") != state["pr_ref"]
            or _canonical_pr_target(_value(details, "target_branch")) != state["target_branch"]
            or _value(details, "url") != state["pr_url"]):
        raise RuntimeError("The saved PR does not match this repository, contributor, or branch; upload stopped")
    status = _value(details, "status")
    if status not in ("draft", "open"):
        raise RuntimeError("The saved Hugging Face PR is closed or merged. It cannot receive more model commits")
    if _value(details, "merge_commit_oid") is not None:
        raise RuntimeError("The saved Hugging Face PR has already been merged; upload stopped")
    state["pr_status"] = status
    return details


def _check_pr(api, state, account):
    try:
        details = api.get_discussion_details(repo_id=state["repo"], repo_type="model", discussion_num=state["pr_num"])
    except Exception:
        raise RuntimeError("Unable to verify the saved Hugging Face PR. Check login, permissions, and connection; "
                           "the existing PR/state are preserved") from None
    return _validate_pr_details(details, state, account)


def _pr_description(code_url):
    if code_url and re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/tree/[0-9a-f]{40}/new/?", code_url) is None:
        raise ValueError("--code-url must be a GitHub code snapshot URL")
    description = ("Publishes the complete public OCT model collection with ordered stream/fold configuration, "
                   "tensor-only historical checkpoints, and inference documentation. Models are uploaded to this "
                   "single PR branch; the catalog is published only after all model hashes are verified. "
                   "Legacy preprocessing and evaluation limitations are documented in the model cards.")
    if code_url:
        description += "\n\nCode and usage instructions: " + code_url
    return description


def _reserve_pr_intent(path, state):
    _ordinary_location(path)
    try:
        with path.open("x", encoding="utf-8") as file:
            json.dump(state, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError:
        raise RuntimeError("A PR preparation state already exists. Retry to verify its existing receipt") from None


def _clear_rejected_intent(path, state):
    """Only a definite server denial permits retrying creation with no saved PR number."""
    _ordinary_location(path)
    if (state.get("preparation_status") == "creating" and not state.get("entries") and "pr_num" not in state
            and path.is_file() and _json(path) == state):
        path.unlink()


def _creation_was_denied(error, repo_id):
    # The SDK performs a GET after POSTing a PR. A GET denial does not prove
    # that creation failed and must never trigger a second creation request.
    request = getattr(getattr(error, "response", None), "request", None)
    return (_auth_status(error) in (401, 403) and getattr(request, "method", None) == "POST"
            and str(getattr(request, "url", "")) == "https://huggingface.co/api/models/" + repo_id + "/discussions")


def prepare_pr(repo_id="ntuhophai/ERM-AI", state_file=None, code_url=None):
    """Create/reuse exactly one draft PR before exporting large local model files."""
    _validate_repo(repo_id)
    path = _pr_state_path(state_file)
    description = _pr_description(code_url)
    api = ensure_auth(repo_id, mode="pr")
    account = _confirmed_pr_account(api)
    if path.exists():
        state = _read_pr_state(path, repo_id, account)
        _check_pr(api, state, account)
        _save_state(path, state)
        print(json.dumps({"repo": repo_id, "pr_url": state["pr_url"], "pr_ref": state["pr_ref"],
                          "pr_status": state["pr_status"], "reused": True}), flush=True)
        return state
    # Store the intent first: an unknown API outcome must not create another PR on retry.
    state = {"state_version": 2, "mode": "pr", "repo": repo_id, "account": account,
             "entries": {}, "preparation_status": "creating"}
    _reserve_pr_intent(path, state)
    for attempt in range(2):
        try:
            details = api.create_pull_request(repo_id=repo_id, repo_type="model",
                                             title="Publish complete OCT stream/fold model collection",
                                             description=description)
            break
        except Exception as error:
            denied = _creation_was_denied(error, repo_id)
            if denied and attempt == 0:
                try:
                    _login_for_pr()
                    api = ensure_auth(repo_id, mode="pr")
                    if _confirmed_pr_account(api) != account:
                        raise RuntimeError("The account changed during authorization; no PR/model upload was performed")
                except Exception:
                    _clear_rejected_intent(path, state)
                    raise
                continue
            if denied:
                _clear_rejected_intent(path, state)
            message = ("Hugging Face denied PR creation. Browser OAuth needs write-discussions; a local fine-grained "
                       "token must authorize discussions/PRs on this repository. " if denied
                       else "Hugging Face PR creation has no confirmed outcome. ")
            recovery = ("Correct local permissions and retry the launcher; no PR was created. "
                        if denied else
                        "Inspect the repository Discussions for an existing draft and preserve this state. "
                        "Share only its PR URL to recover the receipt; do not rename the state and blindly "
                        "create another PR. ")
            raise RuntimeError(message + recovery) from None
    number = _value(details, "num")
    if type(number) is not int or number < 1:
        raise RuntimeError("Hugging Face returned no valid PR number. Preserve the intent and inspect Discussions")
    state.update(pr_num=number, pr_ref="refs/pr/%d" % number,
                 pr_url="https://huggingface.co/%s/discussions/%d" % (repo_id, number),
                 target_branch="main", pr_status="draft", preparation_status="prepared")
    _validate_pr_details(details, state, account)
    _save_state(path, state)
    _check_pr(api, state, account)
    _save_state(path, state)
    print(json.dumps({"repo": repo_id, "pr_url": state["pr_url"], "pr_ref": state["pr_ref"],
                      "pr_status": state["pr_status"], "reused": False}), flush=True)
    return state


def ensure_auth(repo_id="ntuhophai/ERM-AI", mode="direct"):
    """Verify repository write access; namespace membership is not a permission check."""
    _validate_repo(repo_id)
    _validate_mode(mode)
    if mode == "pr":
        return _ensure_pr_auth(repo_id)
    from huggingface_hub import HfApi
    for attempt in range(2):
        api = HfApi()
        method = getattr(api, "auth_check", None)
        if not callable(method):
            raise RuntimeError("Upgrade huggingface_hub: repository auth_check(write=True) is required. No files uploaded")
        try:
            # The Hub checks both the user's repository permission and the active
            # credential's scope, including OAuth and fine-grained repository grants.
            method(repo_id=repo_id, repo_type="model", write=True)
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            missing_login = type(error).__name__ == "LocalTokenNotFoundError"
            if status in (401, 403) or missing_login:
                if os.environ.get("HF_TOKEN"):
                    raise RuntimeError("Hugging Face denied repository write access or no usable login was found. "
                                       "HF_TOKEN overrides local login; unset it or provide an authorized token locally. "
                                       "Never paste tokens into scripts/chat. No files uploaded") from None
                if attempt == 0:
                    print("Log in locally with an account and credential authorized to write to " + repo_id + ".", flush=True)
                    print("For browser login, grant the app access to the " + repo_id.split("/", 1)[0]
                          + " organization if listed. Alternatively, choose 'Paste an access token' and use "
                          "a repository-scoped write token locally.", flush=True)
                    try:
                        _local_login()
                    except Exception:
                        raise RuntimeError("Local Hugging Face login did not complete. Retry the Hugging Face launcher. No files uploaded") from None
                    continue
                account = _account_name(api)
                identity = " for account " + account if account else ""
                raise RuntimeError("Hugging Face denied write access to " + repo_id + identity + ". "
                                   "Check organization membership, OAuth organization access, and token/OAuth scope; "
                                   "one of these does not permit this repository write. "
                                   "Ask the repository/organization administrator to grant access, or log in locally "
                                   "with authorized credentials. No files uploaded") from None
            if status == 404:
                raise RuntimeError("The target Hugging Face repository was not found or is inaccessible. "
                                   "Check the repository ID and visibility. No files uploaded") from None
            if isinstance(error, (TypeError, AttributeError, NotImplementedError)):
                raise RuntimeError("Upgrade huggingface_hub: repository auth_check(write=True) is required. No files uploaded") from None
            if status is None or status >= 500 or status in (408, 429):
                raise RuntimeError("Unable to reach Hugging Face to verify repository write access. "
                                   "Check the connection or service availability and retry. No files uploaded") from None
            raise RuntimeError("Hugging Face could not verify repository write access. "
                               "Check the target repository and installed huggingface_hub version. No files uploaded") from None
        name = _account_name(api)
        if name:
            print("Using local Hugging Face account: " + name, flush=True)
        print("Hugging Face repository write access verified: " + repo_id, flush=True)
        return api


def upload_collection(folder, repo_id="ntuhophai/ERM-AI", verify_only=False, state_file=None, mode="direct"):
    _validate_repo(repo_id)
    _validate_mode(mode)
    folder, catalog, models, root_hashes = validate_collection(folder)
    state_path = _state_path(folder, state_file)
    result = {"repo": repo_id, "model_count": len(models), "uploaded": 0, "skipped": 0,
              "weights_uploaded": 0, "readmes_updated": 0, "root_published": False,
              "verified_only": verify_only, "mode": mode}
    if verify_only:
        print(json.dumps(result, indent=2), flush=True)
        return result
    if mode == "pr":
        if not state_file or not state_path.exists():
            raise ValueError("PR upload requires an existing external receipt. Run --mode pr --prepare-pr first")
        api = ensure_auth(repo_id, mode="pr")
        account = _confirmed_pr_account(api)
        state = _read_pr_state(state_path, repo_id, account)
        _check_pr(api, state, account)
        state["collection_complete"] = False
        _save_state(state_path, state)
        revision = state["pr_ref"]
        result.update(pr_url=state["pr_url"], pr_ref=revision, pr_status=state["pr_status"])
    else:
        state = {"state_version": 1, "repo": repo_id, "entries": {}}
        revision = None
    if mode == "direct" and state_path.exists():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            if (previous.get("state_version") == 1 and previous.get("repo") == repo_id
                    and isinstance(previous.get("entries"), dict)):
                state["entries"] = previous["entries"]
        except (ValueError, AttributeError):
            raise ValueError("Invalid local upload state; choose a new state file") from None
    if mode == "direct":
        api = ensure_auth(repo_id)
    revision_options = {"revision": revision} if revision is not None else {}
    stage = "repository access"
    try:
        for prepared in models:
            entry = prepared["entry"]
            stage = entry["path"]
            if mode == "pr":
                _check_pr(api, state, account)
            _unchanged(prepared)
            remote_status = _remote_status(api, repo_id, prepared, revision)
            if all(remote_status.values()):
                result["skipped"] += 1
                receipt = None
            else:
                docs_only = remote_status["model.safetensors"] and remote_status["config.json"]
                allowed = ["README.md"] if docs_only else sorted(MODEL_FILES)
                if mode == "pr":
                    _check_pr(api, state, account)
                _unchanged(prepared)
                receipt = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(prepared["folder"]),
                                            path_in_repo=entry["path"], allow_patterns=allowed,
                                            create_pr=False, commit_message="Publish public model " + entry["path"],
                                            **revision_options)
                result["uploaded"] += 1
                result["readmes_updated" if docs_only else "weights_uploaded"] += 1
            state["entries"][entry["path"]] = {
                "weights_sha256": entry["weights_sha256"], "config_sha256": entry["config_sha256"],
                "readme_sha256": prepared["hashes"]["README.md"]["sha256"],
                "commit": str(getattr(receipt, "oid", "")) if receipt is not None else "remote_verified",
            }
            _save_state(state_path, state)
            print(json.dumps({"model": entry["path"], "uploaded": result["uploaded"], "skipped": result["skipped"]}), flush=True)
        stage = "final remote model verification"
        if mode == "pr":
            _check_pr(api, state, account)
        for prepared in models:
            _unchanged(prepared)
            if not _remote_matches(api, repo_id, prepared, revision):
                raise RuntimeError("A remote model does not match the validated collection")
        stage = "root catalog/documentation commit"
        if mode == "pr":
            _check_pr(api, state, account)
        _unchanged({"folder": folder, "hashes": root_hashes})
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder), path_in_repo="",
                          allow_patterns=sorted(ROOT_FILES), create_pr=False,
                          commit_message="Publish complete public model catalog and shared inference documentation",
                          **revision_options)
        result["root_published"] = True
        state["catalog_sha256"] = root_hashes["catalog.json"]["sha256"]
        _save_state(state_path, state)
        if mode == "pr":
            stage = "final remote root verification"
            _unchanged({"folder": folder, "hashes": root_hashes})
            if not _remote_matches(api, repo_id, {"entry": {"path": ""}, "hashes": root_hashes}, revision):
                raise RuntimeError("The remote PR catalog or shared documentation differs from the local collection")
            stage = "submit completed PR for review"
            _check_pr(api, state, account)
            if state["pr_status"] == "draft":
                api.change_discussion_status(repo_id=repo_id, repo_type="model", discussion_num=state["pr_num"],
                                             new_status="open")
                _check_pr(api, state, account)
                if state["pr_status"] != "open":
                    raise RuntimeError("The completed PR did not enter open status")
            result["pr_status"] = state["pr_status"]
            state["collection_complete"] = True
            _save_state(state_path, state)
    except Exception:
        raise RuntimeError("Collection upload stopped at %s: uploaded=%d, skipped=%d, root_published=%s. "
                           "Keep the collection and external state; retry resumes after verifying remote content. "
                           "No prior models were deleted." % (stage, result["uploaded"], result["skipped"],
                                                                str(result["root_published"]).lower())) from None
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", help="Root of the prepared public collection")
    parser.add_argument("--repo", default="ntuhophai/ERM-AI")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-only", action="store_true", help="Validate local files and hashes without login/network/upload")
    modes.add_argument("--login-only", action="store_true", help="Authenticate locally and check the existing target before preparing weights")
    modes.add_argument("--prepare-pr", action="store_true", help="Create/reuse one draft PR and save its external receipt before exporting weights")
    parser.add_argument("--mode", choices=("direct", "pr"), default="direct", help="Write to the main branch or one contributor PR")
    parser.add_argument("--state-file", help="Resume state path outside the public collection")
    parser.add_argument("--code-url", help="GitHub code snapshot URL to include when preparing a PR")
    args = parser.parse_args(argv)
    if not args.login_only and not args.prepare_pr and not args.folder:
        parser.error("--folder is required unless --login-only or --prepare-pr is used")
    if args.prepare_pr and args.mode != "pr":
        parser.error("--prepare-pr requires --mode pr")
    try:
        if args.login_only:
            ensure_auth(args.repo, args.mode)
            print(json.dumps({"repo": args.repo, "login_verified": True, "mode": args.mode}), flush=True)
        elif args.prepare_pr:
            prepare_pr(args.repo, args.state_file, args.code_url)
        else:
            upload_collection(args.folder, args.repo, args.verify_only, args.state_file, args.mode)
    except Exception as error:
        parser.exit(2, "Collection upload stopped: " + str(error) + "\n")


if __name__ == "__main__":
    main()

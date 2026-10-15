"""Authenticate locally and upload exactly the five public inference files."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

os.environ.pop("HF_HUB_DEBUG", None)
os.environ["HF_HUB_VERBOSITY"] = "warning"

ALLOWED_FILES = frozenset({"model.safetensors", "config.json", "modeling_multistream.py", "README.md", "requirements.txt"})
PUBLIC_FIELDS = frozenset({"public_format_version", "model_config", "streams", "class_to_idx", "preprocessing",
                          "artifact_kind", "source_format", "settings_provenance", "private_training_metadata_omitted",
                          "legacy_no_split_metadata"})
MODEL_FIELDS = frozenset({"name", "hidden_dim", "freeze_backbones", "pretrained"})
PREPROCESSING_FIELDS = frozenset({"image_size", "resize_size", "mean", "std", "resize_backend", "interpolation", "antialias", "crop"})
TOKEN_PATTERN = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def validate_bundle(folder):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError("The public model folder does not exist")
    entries = list(folder.iterdir())
    if {entry.name for entry in entries} != ALLOWED_FILES or any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise ValueError("Upload requires exactly the five exported public inference files")
    metadata = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if set(metadata) != PUBLIC_FIELDS or metadata["public_format_version"] != 1 or metadata["private_training_metadata_omitted"] is not True:
        raise ValueError("Unexpected public metadata fields; private checkpoint metadata must never be uploaded")
    if not isinstance(metadata["model_config"], dict) or set(metadata["model_config"]) != MODEL_FIELDS:
        raise ValueError("Unexpected nested model metadata")
    if not isinstance(metadata["preprocessing"], dict) or set(metadata["preprocessing"]) != PREPROCESSING_FIELDS:
        raise ValueError("Unexpected nested preprocessing metadata")
    for entry in entries:
        if entry.name != "model.safetensors":
            if TOKEN_PATTERN.search(entry.read_text(encoding="utf-8")):
                raise ValueError("Credential-like text found in public artifact " + entry.name)
    return folder, metadata


def _cli_path():
    candidate = Path(sys.executable).parent / ("hf.exe" if os.name == "nt" else "hf")
    if candidate.is_file():
        return str(candidate)
    installed = shutil.which("hf")
    if installed:
        return installed
    raise RuntimeError("The hf CLI is unavailable. Install huggingface_hub in the selected environment")


def _local_login():
    # Authentication is performed by the official CLI, never via a script argument containing a token.
    cli = _cli_path()
    help_result = subprocess.run([cli, "auth", "login", "--help"], capture_output=True, text=True, check=True)
    command = [cli, "auth", "login"]
    if "--force" in help_result.stdout:
        command.append("--force")
    if "--format" in help_result.stdout:
        command += ["--format", "human"]
    subprocess.run(command, check=True)  # Inherit the visible console for browser/device or masked token login.


def _identity(api):
    try:
        info = api.whoami()
        return info, None
    except Exception:
        # Do not print full credential/authentication payloads or request headers.
        return None, "No usable Hugging Face login was found"


def _known_write_account(info, owner):
    if not isinstance(info, dict):
        return False
    namespaces = {info.get("name")}
    namespaces.update(org.get("name") for org in info.get("orgs", []) if org.get("roleInOrg") in ("write", "admin"))
    role = info.get("auth", {}).get("accessToken", {}).get("role")
    return owner in namespaces and role != "read"


def upload_model(folder, repo_id, verify_only=False):
    folder, metadata = validate_bundle(folder)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise ValueError("Expected a Hugging Face owner/model repository ID")
    # Reload through the trusted local template. Never execute an arbitrary loader from the selected folder.
    exporter_path = Path(__file__).with_name("export_public_model.py")
    spec = importlib.util.spec_from_file_location("_erm_public_exporter", exporter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = module._reference_loader()
    if (folder / "modeling_multistream.py").read_text(encoding="utf-8") != module.PUBLIC_LOADER:
        raise ValueError("The exported loader was modified; re-export from the trusted local tool before upload")
    from safetensors import safe_open
    with safe_open(str(folder / "model.safetensors"), framework="pt", device="cpu") as tensors:
        if tensors.metadata():
            raise ValueError("Unexpected non-tensor metadata in safetensors; re-export before upload")
    loaded = namespace["load_public_model"](folder, device="cpu")
    del loaded
    print(json.dumps({"repo": repo_id, "files": sorted(ALLOWED_FILES), "streams": metadata["streams"],
                      "artifact_kind": metadata["artifact_kind"], "verified_only": verify_only}, indent=2), flush=True)
    if verify_only:
        return None
    from huggingface_hub import HfApi
    api = HfApi()
    info, error = _identity(api)
    owner = repo_id.split("/", 1)[0]
    if not _known_write_account(info, owner):
        if os.environ.get("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN overrides the local login. Unset it in this terminal or supply a write-scoped token there; do not paste it into scripts/chat")
        print("Log in with a Hugging Face account authorized to write to " + repo_id + ".", flush=True)
        _local_login()
        api = HfApi()
        info, error = _identity(api)
    if not _known_write_account(info, owner):
        raise RuntimeError("The local Hugging Face identity/token does not establish write access to the target namespace. No model files were uploaded")
    print("Using local Hugging Face account: " + str(info.get("name")), flush=True)
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")  # Existing target only; never create a different repository.
        receipt = api.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=str(folder),
            allow_patterns=sorted(ALLOWED_FILES), create_pr=False,
            commit_message="Publish sanitized multistream inference weights and configuration",
        )
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status in (401, 403):
            raise RuntimeError("Hugging Face denied the direct push. Check namespace membership and token write scope; GitHub changes remain intact") from None
        raise RuntimeError("Hugging Face upload did not complete. Keep the public bundle and retry the Hugging Face launcher; no credentials are printed") from None
    print("Model commit: " + str(getattr(receipt, "commit_url", "https://huggingface.co/" + repo_id)), flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo", default="ntuhophai/ERM-AI")
    parser.add_argument("--verify-only", action="store_true", help="Validate files and reload weights, without login/network/upload")
    args = parser.parse_args(argv)
    try:
        upload_model(args.folder, args.repo, verify_only=args.verify_only)
    except Exception as error:
        parser.exit(2, "Upload stopped: " + str(error) + "\n")


if __name__ == "__main__":
    main()

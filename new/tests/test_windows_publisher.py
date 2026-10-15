"""Exercise the actual Windows launcher helpers without any remote operations."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe")


@pytest.mark.skipif(os.name != "nt" or not POWERSHELL, reason="Windows PowerShell check")
@pytest.mark.parametrize("case", ["native-exit", "source-list", "wrong-fork", "github-flow", "github-reuse", "model-flow", "model-direct", "model-no-github"])
def test_windows_publishing_guards_without_network(case):
    completed = subprocess.run([
        POWERSHELL, "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(ROOT / "tools" / "check_publish_flow.ps1"), "-Case", case, "-Python", sys.executable,
    ], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", timeout=45)
    assert completed.returncode == 0, completed.stdout + completed.stderr

# ERM-AI

Single-stream and multistream OCT research code lives in [`new/`](new/).

- [Usage and experiment workflow](new/README.md)
- [Windows publishing launcher](new/PUBLISHING.md)
- [Published model download and inference](new/INFERENCE.md)
- [All stream/fold models and model architecture](new/MODEL_CATALOG.md)
- Model repository: https://huggingface.co/ntuhophai/ERM-AI

Windows model publishing defaults to one Hugging Face pull request for the complete collection.
The repository maintainer reviews and merges it; the launcher never merges the PR.

```powershell
cd new
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe main.py --help
```

Use Python 3.10+ and install a matching PyTorch/torchvision pair for your CPU or CUDA environment.
Real datasets, old checkpoints, patient manifests, predictions and local environments are excluded.
Synthetic smoke tests validate the software workflow and do not establish clinical model performance.
The original artifacts remain in the source computer's local `old/` archive.
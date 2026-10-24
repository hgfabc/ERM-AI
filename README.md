[English](README.md) | [&#32321;&#39636;&#20013;&#25991;](new/README.zh-TW.md) | [&#26085;&#26412;&#35486;](new/README.ja.md)

# ERM-AI

Single-stream and multistream optical coherence tomography (OCT) image classification research code lives in [`new/`](new/). The shared workflow supports one to eight input streams.

- [Usage and experiment workflow](new/README.md)
- [Windows publishing launcher](new/PUBLISHING.md)
- [Published model download and inference](new/INFERENCE.md)
- [All stream/fold models and model architecture](new/MODEL_CATALOG.md)
- [Hugging Face model repository](https://huggingface.co/ntuhophai/ERM-AI)

The current framework uses patient-grouped outer test folds and inner validation sets for checkpoint selection. Legacy peak validation accuracy is based on a different evaluation protocol and must not be reported as patient-independent test performance.

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
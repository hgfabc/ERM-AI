# 從 Hugging Face 下載並推論

這份說明適用整套 `ntuhophai/ERM-AI` 模型 collection。完整 collection 已發布於 Hugging Face 的預設 `main` 版本，以下指令可直接使用；若需重現結果，請見第 7 節固定模型版本。舊版只有根目錄單一模型時，尚不適用 collection 工具。

## 1. 安裝程式環境

從 GitHub 取得程式，在 `new/` 目錄使用 Python 3.10 以上：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

若已用本機整理好的 `.venv/`，直接用它。GPU 環境請先依 PyTorch 官方選擇符合 CUDA 的 torch／torchvision；`--device auto` 有可用 CUDA 才使用 GPU。

## 2. 查看全部條件與 fold

```powershell
.\.venv\Scripts\python.exe tools/infer_published.py --repo ntuhophai/ERM-AI --list
.\.venv\Scripts\python.exe tools/infer_published.py --repo ntuhophai/ERM-AI --list --condition fusion_8
```

這一步只下載小型索引，不下載全部權重。名稱對照見 [MODEL_CATALOG.md](MODEL_CATALOG.md)。歷史權重會列出，但不能自動推論。

## 3. 準備輸入影像

有標籤資料：

```text
dataset/
├── bad/
│   └── CASE001_OD/
│       ├── CASE001_OD_vertical.jpg
│       ├── CASE001_OD_horizontal.jpg
│       ├── CASE001_OD_color_full.jpg
│       ├── CASE001_OD_color_inner.jpg
│       ├── CASE001_OD_color_outer.jpg
│       ├── CASE001_OD_gray_full.jpg
│       ├── CASE001_OD_gray_inner.jpg
│       └── CASE001_OD_gray_outer.jpg
└── good/
    └── CASE002_OS/
        └── ...
```

無標籤資料可以直接是 `dataset/CASE001_OD/影像`、`dataset/CASE002_OS/影像`。檔名需包含模型要求的 stream 名稱；大小寫不敏感，每個 stream 只能有一個匹配影像。

單流 `single_gray_inner` **只需要 gray_inner 影像**；八流則需要完整八張。缺少要求影像或同 stream 有多個候選時，程式會停止提示，避免挑錯圖。

資料僅在本機讀取。工具下載權重，不把你的影像上傳 Hugging Face。

## 4. 單模型／單 fold 推論

```powershell
.\.venv\Scripts\python.exe tools/infer_published.py --repo ntuhophai/ERM-AI --condition single_gray_inner --fold 0 --root "D:/my_cases" --output "predictions/gray_inner_fold0"
.\.venv\Scripts\python.exe tools/infer_published.py --repo ntuhophai/ERM-AI --condition fusion_8 --fold 3 --root "D:/my_cases" --output "predictions/fusion8_fold3" --device cuda --batch-size 4
```

下載只限指定 fold 的 `config.json` 與 `model.safetensors`；架構、stream 順序、標籤與 resize/crop/normalization 由 config 讀取。使用本機可信模型定義，不執行 Hub 下載的 Python 程式。

## 5. 指定多個 fold 或全部十折模型

```powershell
.\.venv\Scripts\python.exe tools/infer_published.py --repo ntuhophai/ERM-AI --condition fusion_3_color --folds 0 2 5 --root "D:/my_cases" --output "predictions/color_selected"
.\.venv\Scripts\python.exe tools/infer_published.py --repo ntuhophai/ERM-AI --condition fusion_8 --all-folds --root "D:/my_cases" --output "predictions/fusion8_all"
```

每個 fold 分別保存預測，不會把十個模型的同一病例預測當成十個獨立病例。`--all-folds` 要求索引中有完整 0–9。

若明確要機率平均的 ensemble：

```powershell
.\.venv\Scripts\python.exe tools/infer_published.py --repo ntuhophai/ERM-AI --condition fusion_8 --all-folds --ensemble --root "D:/my_cases" --output "predictions/fusion8_ensemble"
```

ensemble 是同一批影像經選定模型推論後的平均機率；**不是內部 OOF，也不證明病例／病人與訓練資料獨立**。不同 class mapping 或 stream 順序不能混在同一 ensemble。

## 6. 輸出與評估

輸出目錄含各 fold 的 `predictions_fold_<n>.csv` 與 `summary.json`；加 ensemble 則另有 `predictions_ensemble.csv`。CSV 保存 sample_id、真值（有標籤時）、預測類別、每類機率，`bad=0/good=1` 的意義依原研究定義。

有標籤資料會計算描述性 Accuracy、各類 recall/F1、AUROC 與混淆矩陣；無標籤不生成虛構準確率。這些是推論資料的結果，不能冒稱未見病人的驗證。若需要新版訓練過程中的 held-out outer test，請用 `main.py train/validate` 與固定 manifest/splits。

預設不把推定 patient_id 加進 CSV；如研究工作需要，可用 `--include-patient-id`，它只寫本機結果。輸出資料夾已存在結果時不覆蓋，請指定新路徑。

## 7. 固定模型版本、離線使用

以 `--revision <Hugging-Face-commit-SHA>` 固定模型版本：

```powershell
.\.venv\Scripts\python.exe tools/infer_published.py --repo ntuhophai/ERM-AI --revision <commit-sha> --condition fusion_8 --fold 0 --root "D:/my_cases" --output "predictions/fixed_revision"
```

工具先下載索引取得實際 commit，再從同一 commit 下載指定權重，避免索引與模型版本混用。權重／config 的 SHA256 亦與索引核對。

若已有完整 collection 的本機副本：

```powershell
.\.venv\Scripts\python.exe tools/infer_published.py --local-repo .publish-work/model-collection --condition single_gray_inner --fold 0 --root "D:/my_cases" --output "predictions/offline"
```

這時不需 Hub 網路。也可從你的已下載 snapshot 目錄執行。


## 8. 維護者：合併前測試新的 Hugging Face PR

一般使用者不需要此節：目前完整 collection 已發布於 `main`。本節僅供維護者日後發布新版本時，在合併前測試 Hub PR。

BAT 完成後會顯示 Hugging Face PR 網址，例如 `https://huggingface.co/ntuhophai/ERM-AI/discussions/42`。網址末尾數字就是 PR 編號；以下 `42` 只是範例，請換成你的實際編號。

```powershell
.\.venv\Scripts\python.exe tools/infer_published.py --repo ntuhophai/ERM-AI --revision refs/pr/42 --list
.\.venv\Scripts\python.exe tools/infer_published.py --repo ntuhophai/ERM-AI --revision refs/pr/42 --condition single_gray_inner --fold 0 --root "D:/my_cases" --output "predictions/pr_gray_inner_fold0"
.\.venv\Scripts\python.exe tools/infer_published.py --repo ntuhophai/ERM-AI --revision refs/pr/42 --condition fusion_8 --all-folds --root "D:/my_cases" --output "predictions/pr_fusion8_all"
```

程式仍會把 PR revision 解析成實際 commit，固定從同一版本下載索引與權重，並核對雜湊。PR 合併後可省略 `--revision`，或用正式版本的 commit SHA 固定下載。不要在上傳中途使用尚未完成的 collection。

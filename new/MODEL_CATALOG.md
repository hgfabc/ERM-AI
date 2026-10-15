# 全部模型、stream 與 fold

Hugging Face 目的地：[ntuhophai/ERM-AI](https://huggingface.co/ntuhophai/ERM-AI)。新版 BAT 將整套模型提交到同一個 Hugging Face PR。管理者合併後才會出現在正式版本；合併前可指定 PR revision 推論。

## 包含哪些模型

| 類型 | 數量 | 用途 |
|---|---:|---|
| 8 種單流，每組 fold 0–9 | 80 | 指定 stream／fold 推論 |
| 3 種雙流，每組 fold 0–9 | 30 | 指定融合模型／fold 推論 |
| 2 種三流，每組 fold 0–9 | 20 | 指定融合模型／fold 推論 |
| 四流、六流、八流，各 fold 0–9 | 30 | 指定融合模型／fold 推論 |
| 未編號的歷史模型 | 7 | 保存原始 tensors；缺少設定，不自動推論 |
| **合計** | **167** | 160 可依宣告設定載入，7 歷史權重 |

原始 checkpoint 總大小 **43,429,382,301 bytes（約 43.4 GB／40.45 GiB）**。`old/output/best_model.pth` 與八流 fold 9 完全相同，不重複上傳。`new/smoke_output/` 是程式測試資料與模型，不混入這套 legacy 研究模型。

## 模型名稱與固定輸入順序

| condition | stream 數 | 順序 |
|---|---:|---|
| `single_vertical` | 1 | `_vertical` |
| `single_horizontal` | 1 | `_horizontal` |
| `single_color_full` | 1 | `color_full` |
| `single_color_inner` | 1 | `color_inner` |
| `single_color_outer` | 1 | `color_outer` |
| `single_gray_full` | 1 | `gray_full` |
| `single_gray_inner` | 1 | `gray_inner` |
| `single_gray_outer` | 1 | `gray_outer` |
| `fusion_2_cross` | 2 | `_vertical`, `_horizontal` |
| `fusion_2_color` | 2 | `color_inner`, `color_outer` |
| `fusion_2_gray` | 2 | `gray_inner`, `gray_outer` |
| `fusion_3_color` | 3 | `color_inner`, `color_outer`, `color_full` |
| `fusion_3_gray` | 3 | `gray_inner`, `gray_outer`, `gray_full` |
| `fusion_4` | 4 | `_vertical`, `_horizontal`, `color_full`, `gray_full` |
| `fusion_6` | 6 | `color_inner`, `color_outer`, `color_full`, `gray_inner`, `gray_outer`, `gray_full` |
| `fusion_8` | 8 | `_vertical`, `_horizontal`, `color_full`, `color_inner`, `color_outer`, `gray_full`, `gray_inner`, `gray_outer` |

fold 編號為 **0 到 9**。每個模型保留獨立 `model.safetensors`、`config.json`、`README.md`，不會彼此覆寫。

```text
ntuhophai/ERM-AI (Hugging Face)
├── catalog.json                         # 全167模型索引、條件、fold與SHA256
├── modeling_multistream.py              # 共用公開模型定義
├── README.md
├── requirements.txt
├── models/
│   ├── single_gray_inner/
│   │   ├── fold_0/{model.safetensors,config.json,README.md}
│   │   └── ... fold_9/
│   ├── ...其餘單流與融合條件...
│   └── fusion_8/
│       ├── fold_0/{model.safetensors,config.json,README.md}
│       └── ... fold_9/
└── historical/
    └── <project>/{model.safetensors,config.json,README.md}
```

## 架構與輸出

單流為 **ResNet50 → global pooled 2048 維特徵 → 2 類分類器**。多流是每個輸入各有獨立 ResNet50，將特徵串接，經 `N×2048 → 2048 → 2048 → 2` 的 MLP 分類；第一層後使用 ReLU。

輸入 tensor 為 `[batch, stream_count, 3, height, width]`，stream 必須遵守 config 中的順序。輸出 logits 經 softmax 得每類機率。

目前標籤索引為 **`bad=0`、`good=1`**；這是原資料的標籤名稱，不能僅依名稱推論「疾病／健康」或特定臨床定義。請依原研究標註規則解讀。

## 前處理與來源限制

已宣告的 profile 在 [publish_profiles/legacy_collection.yaml](publish_profiles/legacy_collection.yaml)。共同參數是 RGB、tensor resize、448×448 center crop、ImageNet mean/std。

- 單流／雙流／三流保存的執行環境為 torchvision 0.17.1，宣告 antialias=true。
- 八流保存的環境為 torchvision 0.16.2，宣告 antialias=false。
- **四流／六流未保存直接 runtime snapshot**；目前明確宣告 true，屬於要核對的 operator choice。部分 stream 順序依歷史推論目錄命名，無法由權重本身證明。

`inference_ready=true` 表示權重鍵／尺寸符合宣告的模型設定，**不表示已建立臨床有效性或病人獨立驗證**。這批是舊版病例隨機切折並以 validation 選模的研究權重；先前稽核有病人跨折重疊問題。新版病人分組訓練框架需另外重新訓練，不能把這批權重的成績改稱新版 CV 結果。

7 個歷史模型會標示 `inference_ready=false`、stream／fold 未知；包括一個舊 Conv merger 模型與一個三分支模型。其原始 tensors 保存完整，但不猜架構／設定來執行推論。

## 如何使用

完整下載、單 fold、多 fold、ensemble 與輸出格式，見 [INFERENCE.md](INFERENCE.md)。只下載你選取的模型，不必先下載全部約43GB。

# 雙擊發布程式與全部模型

在 Windows 檔案總管進入 `new/`，雙擊 **`publish_erm_ai.bat`**。GitHub 會發布程式與文件；Hugging Face 會把每個 stream 條件、每個 fold 的模型提交到同一個 PR。

如果舊 BAT 仍停在「Select exactly one model checkpoint」或檔案選擇視窗，先取消並關閉舊視窗，再執行更新後的 BAT。它不再要求你挑一個 checkpoint。

## 1. GitHub：fork、更新分支，由你建立 PR

1. 先跑本機測試，核對公開程式清單。
2. 若未安裝 GitHub CLI，下載官方可攜版到 `.publish-work/` 並核對 SHA256。
3. 尚未登入時，依官方瀏覽器提示登入你的個人 GitHub 帳號。
4. 建立或重用 `<登入帳號>/ERM-AI` fork，確認它來自 `ntuhophai/ERM-AI`。同名無關 repo 會使流程停止。
5. 提交新版程式、設定、測試與文件。上次發布分支仍符合本機紀錄，且 PR 尚未關閉／合併時，會接續更新該分支；已有 PR 會顯示更新內容。若分支遭其他操作修改，或 PR 已關閉／合併，則建立新分支保留既有修改。
6. 成功後開啟比較頁。尚未建立 PR 時，由你按 **Create pull request**；已有開啟的 PR 則到該 PR 檢查更新。

腳本不 force push，也不替你合併 PR。登入帳號以本機實際授權為準，不寫死個人帳號。若使用原 repo 擁有者 `ntuhophai`，請改用要建立 fork 的個人帳號。

公開程式位於 repo 的 `new/`，包含訓練、驗證、模型目錄與 Hugging Face 推論工具；根目錄另有導覽 README。`old/`、真實影像、病人 manifests、原始 checkpoint、預測結果、環境與暫存不包含在 source 提交中。

GitHub 首頁以英文顯示，頁首提供 **English／繁體中文／日本語** 連結。完整使用說明分別為 `README.md`、`README.zh-TW.md` 與 `README.ja.md`，發布工具會一併提交三個版本。Hugging Face 首頁、各條件／fold 模型及歷史張量封存項目的 README 同樣以英文開頭，頁首連結可跳至同一份說明中的繁體中文或日文段落。單一模型匯出的 README 亦使用相同的三語格式。

更新這些文件後，先執行 `publish_github.bat` 更新 GitHub 發布分支，再執行 `publish_huggingface.bat` 更新 Hugging Face 模型說明；也可執行 `publish_erm_ai.bat` 依序處理兩者。只修改本機檔案不會改變線上內容；GitHub 與 Hugging Face 的 PR 皆須由維護者合併，正式首頁才會顯示更新。Hugging Face 階段仍會核對既有模型快取與遠端內容，相同權重會重用。

## 2. Hugging Face：全部 stream 與 fold，提交同一個 PR

目的地：[ntuhophai/ERM-AI](https://huggingface.co/ntuhophai/ERM-AI)。Hugging Face PR 不需 fork；模型先放在該 PR 的 `refs/pr/<number>`，由管理者合併到正式版本。

1. 自動核對 `old/records/`：**16 組條件 × fold 0–9 = 160 個模型**，另有 **7 個歷史模型**，合計 167 個。缺 fold、重複 fold 或來源格式異常就停止。
2. 顯示 stream 順序及前處理。先閱讀 [模型說明](MODEL_CATALOG.md) 與 `publish_profiles/legacy_collection.yaml`，確認符合原實驗後，在本機提示輸入 `YES`。
3. 透過官方本機登入，確認可讀目的 repo，然後建立一個 **Draft PR**，或接續已保存且由同一帳號建立的 PR。這一步發生在大型匯出之前。
4. 匯出純 tensors 的 `model.safetensors`，檢查架構、鍵與尺寸；公開包不帶病人／病例編號、內部資料路徑或切分指紋。
5. 全部模型逐一提交到同一 PR，依條件／fold 保存在獨立目錄。核對遠端內容後，最後才提交完整 `catalog.json` 與共用文件。
6. 全部核對成功後，將 Draft PR 發布為 Open，開啟 PR 網頁並顯示 **HUGGING FACE PR READY**。由管理者審查及合併，腳本不自動合併。

```text
ntuhophai/ERM-AI
├── catalog.json
├── modeling_multistream.py
├── README.md
├── requirements.txt
├── models/
│   ├── single_gray_inner/fold_0/{model.safetensors,config.json,README.md}
│   ├── ...所有16組，各fold_0到fold_9...
│   └── fusion_8/fold_9/{model.safetensors,config.json,README.md}
└── historical/
    └── <project>/{model.safetensors,config.json,README.md}
```

**模型不會互相覆寫。** 每個模型有獨立設定與說明。既有根目錄 `model.safetensors`／`config.json` 等單模型檔案不會刪除；新版推論依 `catalog.json` 選取模型。

7 個歷史模型缺少 fold／stream／前處理資訊，會保存原始 tensors 並標示 `inference_ready=false`。完整十折的 160 個模型可依宣告設定載入；這不表示其效能或歷史設定已獲獨立驗證。四流／六流的原 runtime snapshot 未保留，profile 中的 antialias=true 是待原始紀錄核對的明確選擇。詳見 [MODEL_CATALOG.md](MODEL_CATALOG.md)。

`old/output/best_model.pth` 是八流 fold 9 的相同副本，不重複上傳。合成 smoke 模型也不列入此批研究權重。

## 3. 大小、時間與失敗重試

原始模型共 **43,429,382,301 bytes，約 43.4 GB**。第一次需要額外約同等空間保存公開權重包，以及少量暫存；匯出、檢查與網路上傳會花時間，視窗會逐模型顯示進度。請保持電腦與網路連線。

穩定的公開包保存在 `new/.publish-work/model-collection/`，匯出快取與上傳紀錄另存在 `new/.publish-work/`，不提交到公開 repo。

- 匯出失敗：保留已完成模型；重跑後核對來源與輸出雜湊，符合才重用。若提示快取包含本次清單外的檔案，先將 `new/.publish-work/model-collection/` 改名保存，再重跑產生新的快取。
- 上傳失敗：保留已提交模型與同一 PR。重跑會核對 PR 分支上的權重／設定／文件，符合才略過。只有文件改動時只更新文件。
- GitHub 已完成、Hugging Face 尚未完成：雙擊 **`publish_huggingface.bat`** 接續模型流程，無需重傳程式。
- 程式或文件又改動：雙擊 **`publish_erm_ai.bat`**，更新 GitHub 與模型文件；已相同的模型權重會重用。

腳本不刪除或搬動原始實驗，也不撤回已成功的遠端提交。最終完整 catalog 在全部模型通過遠端核對後才提交；首次上傳中途還不能使用完整 collection 推論。PR 尚未合併時，推論需要指定 `--revision refs/pr/<number>`。

PR 紀錄保存在 `new/.publish-work/upload-collection-pr-state.json`，含 PR 編號／分支／建立者與完成項目；不要刪除，續傳時會驗證目前登入帳號與 PR 建立者一致。若 PR 已關閉／合併，腳本會停止，不會默默改寫正式分支或另建 PR。請先由管理者處理原 PR；若確定要另開新的 PR，將該紀錄改名保存後再執行。

若出現「PR creation has no confirmed outcome」或「previous PR creation has no confirmed receipt」，伺服器可能已建立草稿但後續回應失敗。請保留該紀錄，查看 Hugging Face 的 Community 頁面是否已有自己的草稿 PR，提供 PR 網址以便恢復；不要直接把紀錄改名後盲目重開。

若 Windows 在移動暫存模型目錄時出現 `WinError 5`／`32`／`33`，可能是檔案仍被掃描或其他程式開啟。新版會做有限次重試，持續失敗則保留完整暫存與先前完成的模型。關閉正在預覽模型快取檔案的視窗後可重跑；前面顯示 `Reusing verified export` 的項目會直接重用，無需刪除資料或 PR 紀錄。

## 4. 啟動檔與環境

| 檔案 | 動作 |
|---|---|
| `publish_erm_ai.bat` | 更新 GitHub 程式／PR分支，接著提交 Hugging Face 全167模型 PR |
| `publish_github.bat` | 只發布 GitHub 程式與文件 |
| `publish_huggingface.bat` | 只建立／接續 Hugging Face 模型 PR |
| `publish_preview.bat` | 列出公開程式清單；不登入、不 fork、不上傳 |

優先使用 `new/.venv/`。若不存在，嘗試以已安裝的 Python 3.10 以上在 `.publish-work/runtime/` 建立隔離環境；首次安裝 PyTorch 等依賴需時間與磁碟空間。Python 未安裝或版本太舊時會停止。

需要直接 push 的維護者仍可顯式使用 Direct 模式（從 `new/` 執行）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/launch_publish.ps1 -Target HuggingFace -ModelPublishMode Direct
```

模型使用方式見 [INFERENCE.md](INFERENCE.md)：列出條件、只下載指定 fold、跑全部 fold、選擇 ensemble、讀取機率 CSV，以及固定模型版本。重新訓練與驗證見 [README.md](README.md)。

## 5. 本機驗證範圍

2026-10-12 更新後完整測試：**186 passed、33 subtests passed**。正式160個fold模型的權重鍵、尺寸與型別亦全部符合宣告架構；完整模型包已在本機匯出並通過上傳前檢查；真正遠端傳輸由 BAT 執行。

測試涵蓋完整性核對、純權重匯出與載回、快取重用、分 fold 推論、機率平均、版本與雜湊核對、GitHub 分支更新，以及 Hugging Face 逐模型提交與續傳。遠端寫入測試使用模擬 API；測試本身不登入、不 fork、不上傳。

實際帳號權限、Hub 容量限制與傳輸成功仍以你執行 BAT 的結果為準。腳本會先確認目的 repo／登入與 PR 是否可用，再匯出整套模型。


## 登入成功但 PR／寫入權限檢查失敗

`Login successful` 表示已驗證帳號；寫入 `ntuhophai/ERM-AI` 還需要對該 repo 的有效授權。GitHub 的協作者權限不會自動成為 Hugging Face 權限。

預設 PR 模式確認 repo 可讀與有效登入，並使用伺服器允許的 PR 建立／更新流程；不要求正式分支的直接寫入權限。Direct 模式才檢查 `auth_check(write=True)`。兩種模式都不依帳號名稱或組織角色猜測授權。

若 PR 模式出現 403：

1. 確認登入帳號能讀取目的 repo，且 repo 開啟社群 PR 功能。對公開 repo 提交 PR 可以使用貢獻者帳號；合併仍需管理者權限。
2. 瀏覽器／OAuth 授權需要允許討論與 PR（`write-discussions`）；私人 repo 另需可讀權限。登入成功本身不保證這些權限。
3. 若使用 Token，可在 [Hugging Face Token 設定](https://huggingface.co/settings/tokens) 建立允許對目的 repo 的討論／PR 與所需模型檔案操作的憑證。官方登入選單選 **Paste an access token**，只貼到本機遮蔽提示。
4. 授權調整後重跑 `publish_huggingface.bat`，保留原 PR 紀錄。若首次建立 PR 已成功但模型上傳遭拒，草稿與紀錄會留下，不會再開另一個 PR。

若選 Direct 模式，帳號與憑證必須具有目的 repo 的直接寫入權限；可由 repo／組織管理者授權。GitHub 的協作者權限不會自動套用到 Hugging Face。

官方說明：[OAuth 組織授權](https://huggingface.co/docs/hub/oauth#accessing-organization-resources)、[組織權限](https://huggingface.co/docs/hub/organizations-security)、[Token 權限](https://huggingface.co/docs/hub/security-tokens)。

## 參考

- [GitHub CLI 登入](https://cli.github.com/manual/gh_auth_login)
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [Hugging Face 上傳](https://huggingface.co/docs/huggingface_hub/guides/upload)

"""Build a resumable, patient-free collection of every legacy model and fold."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import uuid

import yaml

from export_public_model import (
    PUBLIC_LOADER, _class_mapping, _model_card, _ordered_streams,
    _public_code_url, _public_preprocessing, _safe_model_config,
    _tensor_state, _rename_with_retry, export_public_model,
)

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
RUN_NAME = re.compile(r"resnet50_runresnet50_run_(.+)@(\d+)_fold")
LEAF_FILES = {"model.safetensors", "config.json", "README.md"}
ROOT_FILES = {"catalog.json", "modeling_multistream.py", "README.md", "requirements.txt"}


def _linklike(path):
    try:
        details = Path(path).lstat()
    except FileNotFoundError:
        return False
    return Path(path).is_symlink() or bool(getattr(details, "st_file_attributes", 0) & 0x400)


def _ordinary_location(path):
    location = Path(path).absolute()
    if any(_linklike(part) for part in (location, *location.parents)):
        raise ValueError("Linked collection/source/state locations are not allowed")
    return location.resolve()


def _validate_existing_cache(output, items):
    allowed = set(ROOT_FILES) | {"catalog.json.tmp"}
    allowed.update(item["path"] + "/" + filename for item in items for filename in LEAF_FILES)
    for path in output.rglob("*"):
        if _linklike(path):
            raise ValueError("Linked files or directories in the public model cache are not allowed")
        if path.is_file() and path.relative_to(output).as_posix() not in allowed:
            raise ValueError("The public cache contains files outside the current model plan. "
                             "Rename the model-collection cache directory to preserve it, then retry with a clean cache.")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_profiles(path):
    profile = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(profile, dict) or profile.get("collection_version") != 1:
        raise ValueError("Expected collection_version: 1 in the collection profiles")
    folds = profile.get("folds", list(range(10)))
    if (not isinstance(folds, list) or not folds or any(type(value) is not int or not 0 <= value <= 9 for value in folds)
            or len(set(folds)) != len(folds)):
        raise ValueError("Profiles require distinct zero-based folds from 0 to 9")
    _safe_model_config(profile.get("model"))
    _class_mapping(profile.get("class_to_idx"))
    if not isinstance(profile.get("conditions"), dict) or not profile["conditions"]:
        raise ValueError("Profiles must explicitly declare the experiment conditions")
    pairs = set()
    for name, condition in profile["conditions"].items():
        if not SAFE_NAME.fullmatch(name) or not isinstance(condition, dict):
            raise ValueError("Invalid public condition name")
        if not isinstance(condition.get("project"), str) or not SAFE_NAME.fullmatch(condition["project"]):
            raise ValueError("Invalid records project name")
        if not isinstance(condition.get("source_prefix"), str) or not re.fullmatch(r"[A-Za-z0-9_]+", condition["source_prefix"]):
            raise ValueError("Invalid source run prefix")
        key = (condition["project"], condition["source_prefix"])
        if key in pairs:
            raise ValueError("Two conditions map to the same legacy source")
        pairs.add(key)
        _ordered_streams(condition.get("streams"))
        effective_profile(profile, name)  # Validate explicit resize settings too.
    return profile


def effective_profile(profiles, name):
    condition = profiles["conditions"][name]
    preprocessing = copy.deepcopy(profiles["preprocessing"])
    preprocessing["antialias"] = condition["antialias"]
    result = {"model": copy.deepcopy(profiles["model"]),
              "data": {"streams": list(condition["streams"]), "class_to_idx": dict(profiles["class_to_idx"])},
              "preprocessing": preprocessing}
    _public_preprocessing(preprocessing, True)
    return result


def discover_models(records_root, profiles):
    records_root = _ordinary_location(records_root)
    if not records_root.is_dir() or records_root.is_symlink():
        raise ValueError("Records root must be an existing ordinary directory")
    mapping = {(value["project"], value["source_prefix"]): name for name, value in profiles["conditions"].items()}
    items, indexed, historical = [], set(), set()
    for checkpoint in sorted(records_root.glob("**/best_model.pth")):
        if _linklike(checkpoint) or any(_linklike(parent) for parent in checkpoint.parents if parent != records_root.parent):
            raise ValueError("Linked model sources are not permitted")
        relative = checkpoint.relative_to(records_root)
        if len(relative.parts) != 3:
            raise ValueError("Unexpected legacy checkpoint location")
        project, run, _ = relative.parts
        if not SAFE_NAME.fullmatch(project):
            raise ValueError("Unsafe public historical project name")
        match = RUN_NAME.fullmatch(run)
        name = mapping.get((project, match[1])) if match else None
        if name is not None:
            fold = int(match[2])
            if fold not in profiles["folds"] or (name, fold) in indexed:
                raise ValueError("Unexpected or duplicate condition/fold source")
            indexed.add((name, fold))
            path = f"models/{name}/fold_{fold}"
            streams = list(profiles["conditions"][name]["streams"])
        else:
            if project in historical:
                raise ValueError("Multiple unidentified models in one project require explicit profiles")
            historical.add(project)
            name, fold, streams = "historical_" + project, None, None
            path = "historical/" + project
        items.append({"checkpoint": checkpoint, "condition": name, "fold": fold, "streams": streams,
                      "project": project, "path": path, "inference_ready": fold is not None,
                      "source_bytes": checkpoint.stat().st_size})
    expected = {(name, fold) for name in profiles["conditions"] for fold in profiles["folds"]}
    if indexed != expected:
        missing = sorted(expected - indexed)
        raise ValueError("Missing expected model folds: " + str(missing))
    if not items:
        raise ValueError("No model checkpoints were found")
    return items


def public_plan(items, profiles):
    ready = [item for item in items if item["inference_ready"]]
    return {"model_count": len(items), "inference_ready_count": len(ready),
            "historical_count": len(items) - len(ready), "condition_count": len(profiles["conditions"]),
            "fold_count": len(profiles["folds"]), "source_bytes": sum(item["source_bytes"] for item in items),
            "profiles": [{"condition": name, "streams": value["streams"], "antialias": value["antialias"],
                          "notes": value.get("notes", "Operator-declared inference settings")}
                         for name, value in profiles["conditions"].items()]}


def _historical_config(item):
    return {"historical_format_version": 1, "condition": item["condition"], "inference_ready": False,
            "streams": None, "class_to_idx": None, "model_config": None, "preprocessing": None,
            "source_format": "legacy_raw_state_dict", "private_training_metadata_omitted": True,
            "reason": "Original fold/stream/preprocessing metadata is unavailable; archive tensors without guessing a deployment configuration."}


def _historical_card(config):
    return ("# Historical tensor archive\n\nCondition: " + config["condition"] + "\n\n"
            "These original tensors are preserved for research provenance. No fold, stream order, "
            "label mapping or preprocessing is asserted.\n\n" + config["reason"] + "\n\n"
            "The collection inference tool rejects this entry. Recover the original experiment "
            "settings before constructing an inference model. No patient/sample metadata is included.\n")


def _ready_metadata(profiles, name):
    effective = effective_profile(profiles, name)
    return {"public_format_version": 1, "model_config": _safe_model_config(effective["model"]),
            "streams": _ordered_streams(effective["data"]["streams"]),
            "class_to_idx": _class_mapping(effective["data"]["class_to_idx"]),
            "preprocessing": _public_preprocessing(effective["preprocessing"], True),
            "artifact_kind": "research", "source_format": "legacy_state_dict",
            "settings_provenance": "operator_supplied", "private_training_metadata_omitted": True,
            "legacy_no_split_metadata": True}


def _collection_profile_note(note, language):
    """Translate documented profile evidence without reinterpreting unknown notes."""
    if not note:
        return {"en": "Operator-declared settings", "zh": "由操作者明確宣告的設定", "ja": "操作者が明示的に宣言した設定"}[language]
    if language == "en":
        return note
    translations = {
        "Single-stream environment records use torchvision 0.17.1; exact per-fold environment snapshots were not retained.": (
            "單流環境紀錄使用 torchvision 0.17.1；未保留各 fold 的完整環境快照。",
            "単一ストリームの環境記録では torchvision 0.17.1 を使用しています。fold ごとの正確な環境スナップショットは保存されていません。"),
        "Order follows historical inference directory naming; original YAML is missing. Two-stream runtime records use torchvision 0.17.1.": (
            "順序依歷史推論目錄名稱判定；原始 YAML 已缺失。雙流執行環境紀錄使用 torchvision 0.17.1。",
            "順序は過去の推論ディレクトリ名に基づきます。元の YAML は残っていません。2 ストリームの実行環境記録では torchvision 0.17.1 を使用しています。"),
        "Order follows saved config2.yaml; two-stream runtime records use torchvision 0.17.1.": (
            "順序依保存的 config2.yaml；雙流執行環境紀錄使用 torchvision 0.17.1。",
            "順序は保存された config2.yaml に基づきます。2 ストリームの実行環境記録では torchvision 0.17.1 を使用しています。"),
        "Order follows saved config3.yaml; two-stream runtime records use torchvision 0.17.1.": (
            "順序依保存的 config3.yaml；雙流執行環境紀錄使用 torchvision 0.17.1。",
            "順序は保存された config3.yaml に基づきます。2 ストリームの実行環境記録では torchvision 0.17.1 を使用しています。"),
        "Order follows saved config4.yaml; three-stream runtime records use torchvision 0.17.1.": (
            "順序依保存的 config4.yaml；三流執行環境紀錄使用 torchvision 0.17.1。",
            "順序は保存された config4.yaml に基づきます。3 ストリームの実行環境記録では torchvision 0.17.1 を使用しています。"),
        "Order follows saved config5.yaml; three-stream runtime records use torchvision 0.17.1.": (
            "順序依保存的 config5.yaml；三流執行環境紀錄使用 torchvision 0.17.1。",
            "順序は保存された config5.yaml に基づきます。3 ストリームの実行環境記録では torchvision 0.17.1 を使用しています。"),
        "Order follows historical inference directory naming. Original runtime version is unrecorded; antialias true is an explicit operator choice to verify.": (
            "順序依歷史推論目錄名稱判定。原始執行環境版本未記錄；antialias=true 是操作者明確指定、仍須核對的設定。",
            "順序は過去の推論ディレクトリ名に基づきます。元の実行環境のバージョンは記録されていません。antialias=true は操作者が明示的に選択した設定であり、照合が必要です。"),
        "Order follows saved config.yaml; saved eight-stream environment uses torchvision 0.16.2, whose Tensor Resize default was not antialiased.": (
            "順序依保存的 config.yaml；保存的八流環境使用 torchvision 0.16.2，其張量 Resize 預設不啟用抗鋸齒。",
            "順序は保存された config.yaml に基づきます。保存された 8 ストリーム環境では torchvision 0.16.2 を使用しており、テンソルの Resize は既定でアンチエイリアスを適用しません。"),
    }
    if note in translations:
        return translations[note][0 if language == "zh" else 1]
    prefix = {"zh": "原始設定註記（保留原文）：", "ja": "元の設定注記（原文を保持）："}[language]
    return prefix + note


def _collection_readme_section(items, profiles, code_url=None, language="en"):
    def text(english, chinese, japanese):
        return {"en": english, "zh": chinese, "ja": japanese}[language]

    ready = sum(item["inference_ready"] for item in items)
    historical = len(items) - ready
    first = next(iter(profiles["conditions"]))
    metadata = _ready_metadata(profiles, first)
    preprocessing = metadata["preprocessing"]
    model = metadata["model_config"]
    architecture = (text(
        "Single-stream ResNet50 maps a global pooled 2048-dimensional feature to two classes. Fusion models concatenate one feature per stream and use Linear(N*2048, 2048), ReLU, Linear(2048, 2048), and Linear(2048, 2).",
        "單流 ResNet50 將全域池化後的 2048 維特徵映射至兩個類別。融合模型串接各輸入流的特徵，再依序使用 Linear(N*2048, 2048)、ReLU、Linear(2048, 2048) 與 Linear(2048, 2)。",
        "単一ストリームの ResNet50 は、グローバルプーリング後の 2048 次元特徴を 2 クラスに写像します。融合モデルは各ストリームの特徴を連結し、Linear(N*2048, 2048)、ReLU、Linear(2048, 2048)、Linear(2048, 2) を順に適用します。")
        if model["name"] == "resnet50" and model["hidden_dim"] == 2048 and len(metadata["class_to_idx"]) == 2
        else text(
            f"Declared backbone: {model['name']}; fusion hidden dimension: {model['hidden_dim']}. Each `config.json` describes its architecture.",
            f"宣告的骨幹網路：{model['name']}；融合層隱藏維度：{model['hidden_dim']}。各 `config.json` 記錄對應的模型架構。",
            f"宣言されたバックボーン：{model['name']}。融合層の隠れ次元数：{model['hidden_dim']}。各 `config.json` に対応するモデル構造が記録されています。"))
    lines = [text("# ERM-AI complete research model collection", "# ERM-AI 完整研究模型集合", "# ERM-AI 研究モデルの完全コレクション"), "",
             text(f"This collection preserves {len(items)} model weight {'file' if len(items) == 1 else 'files'}: {ready} loadable model/fold {'entry' if ready == 1 else 'entries'} and {historical} historical tensor {'archive' if historical == 1 else 'archives'}.",
                  f"本集合保存 {len(items)} 個模型權重檔案，包含 {ready} 個可載入的模型／fold 項目，以及 {len(items)-ready} 個歷史張量封存項目。",
                  f"本コレクションには {len(items)} 個のモデル重みファイルを保存しています。内訳は、読み込み可能なモデル／fold が {ready} 件、過去のテンソルアーカイブが {len(items)-ready} 件です。"),
             text("A loadable entry means that its tensors match the declared architecture; it does not establish clinical validity or independent test performance.",
                  "「可載入」僅表示張量符合宣告的模型架構，不代表已證實臨床有效性或獨立測試效能。",
                  "「読み込み可能」とは、テンソルが宣言されたモデル構造に適合することを意味します。臨床的妥当性や独立したテストでの性能が確認されたことを意味するものではありません。"), "",
             text("## Layout", "## 目錄結構", "## ディレクトリ構成"), "", "```text", "catalog.json", "modeling_multistream.py", "README.md", "requirements.txt",
             "models/<condition>/fold_<0-9>/{model.safetensors,config.json,README.md}",
             "historical/<project>/{model.safetensors,config.json,README.md}", "```", "",
             text("## Conditions and ordered streams", "## 實驗條件與輸入流順序", "## 実験条件とストリームの順序"), "",
             text("| Condition | Ordered streams | Tensor resize antialias | Profile evidence / limits |",
                  "| 條件 | 輸入流順序 | 張量縮放抗鋸齒 | 設定依據／限制 |",
                  "| 条件 | ストリームの順序 | テンソルのリサイズ時のアンチエイリアス | 設定の根拠／制約 |"), "|---|---|---|---|"]
    for name, value in profiles["conditions"].items():
        note = _collection_profile_note(value.get("notes"), language).replace("|", "/")
        lines.append(f"| {name} | {', '.join(value['streams'])} | {value['antialias']} | {note} |")
    lines += ["", text(
              "Each condition contains the zero-based folds declared in `catalog.json`. Each stream has its own backbone; fusion concatenates pooled features and applies an MLP classifier. Single-stream models classify the pooled feature directly.",
              "各條件包含 `catalog.json` 宣告的 fold，編號從 0 開始。各輸入流使用獨立的骨幹網路；融合模型串接池化後的特徵，再以多層感知器（MLP）分類。單流模型則直接以池化後的特徵分類。",
              "各条件には、`catalog.json` に宣言された 0 始まりの fold が含まれます。各ストリームは独立したバックボーンを持ち、融合モデルはプーリング後の特徴を連結して多層パーセプトロン（MLP）で分類します。単一ストリームモデルはプーリング後の特徴を直接分類します。"),
              "", text("## Architecture and preprocessing", "## 模型架構與前處理", "## モデル構造と前処理"), "",
              architecture,
              text(f"Input tensors have shape [batch, streams, 3, {preprocessing['image_size']}, {preprocessing['image_size']}]. Class mapping: {json.dumps(metadata['class_to_idx'])}. Softmax converts logits into class probabilities.",
                   f"輸入張量形狀為 [batch, streams, 3, {preprocessing['image_size']}, {preprocessing['image_size']}]。類別對應：{json.dumps(metadata['class_to_idx'])}。使用 softmax 將 logits 轉為各類別機率。",
                   f"入力テンソルの形状は [batch, streams, 3, {preprocessing['image_size']}, {preprocessing['image_size']}] です。クラスの対応関係：{json.dumps(metadata['class_to_idx'])}。softmax で logits を各クラスの確率に変換します。"),
              text(f"RGB resize backend: {preprocessing['resize_backend']}; resize size: {preprocessing['resize_size']}; bilinear interpolation and center crop. Normalization mean: {preprocessing['mean']}; std: {preprocessing['std']}. Condition-specific antialias is listed above and saved in `config.json`.",
                   f"RGB 影像縮放後端：{preprocessing['resize_backend']}；縮放尺寸：{preprocessing['resize_size']}；使用雙線性插值與中心裁切。正規化平均值：{preprocessing['mean']}；標準差：{preprocessing['std']}。各條件的抗鋸齒設定列於上表，並保存在 `config.json`。",
                   f"RGB 画像のリサイズ処理：{preprocessing['resize_backend']}。リサイズサイズ：{preprocessing['resize_size']}。双線形補間と中央クロップを使用します。正規化の平均値：{preprocessing['mean']}。標準偏差：{preprocessing['std']}。条件ごとのアンチエイリアス設定は上表と `config.json` に記録されています。"),
              "", text("## Inference", "## 推論", "## 推論"), "",
              text("Clone the code repository and run from its `new/` directory:", "複製程式碼儲存庫後，在其中的 `new/` 目錄執行：", "コードリポジトリをクローンし、`new/` ディレクトリで実行してください。"), "", "```bash",
              "python -m pip install -r requirements.txt",
              "python tools/infer_published.py --repo ntuhophai/ERM-AI --list",
              "python tools/infer_published.py --repo ntuhophai/ERM-AI --condition single_gray_inner --fold 0 --root /path/to/cases --output predictions/single_gray_inner_fold0",
              "python tools/infer_published.py --repo ntuhophai/ERM-AI --condition fusion_8 --all-folds --root /path/to/cases --output predictions/fusion_8",
              "```", "",
              text("Before the model PR is merged, add `--revision refs/pr/<number>` to every Hub command above. After merge, the default `main` revision can be used. Wait for all models and `catalog.json` to finish uploading before inference.",
                   "模型 PR 合併前，請在上述每個存取 Hub 的指令加上 `--revision refs/pr/<number>`；合併後可使用預設的 `main` 版本。請等所有模型與 `catalog.json` 上傳完成後再執行推論。",
                   "モデルの PR がマージされる前は、上記の Hub にアクセスする各コマンドに `--revision refs/pr/<number>` を追加してください。マージ後は既定の `main` リビジョンを使用できます。すべてのモデルと `catalog.json` のアップロードが完了してから推論を実行してください。"),
              text("Only selected fold weights/configurations are downloaded. Pin `--revision` to a model repository commit for reproducible downloads. Predictions include per-class probabilities in the saved class index order.",
                   "僅下載所選 fold 的權重與設定。為確保下載結果可重現，請使用 `--revision` 指定模型儲存庫的 commit。預測結果依保存的類別索引順序包含各類別機率。",
                   "選択した fold の重みと設定のみをダウンロードします。再現可能なダウンロードのため、`--revision` でモデルリポジトリのコミットを指定してください。予測結果には、保存されたクラスインデックス順の各クラスの確率が含まれます。"),
              text("Case folders must provide one matching image per required stream. Both class/case/image folders and unlabeled case/image folders are supported.",
                   "每個病例目錄須為各必要輸入流提供一張對應影像。同時支援 class/case/image 與不含類別標籤的 case/image 目錄結構。",
                   "各症例フォルダーには、必要なストリームごとに対応する画像を 1 枚用意してください。class/case/image 構成と、クラスラベルのない case/image 構成の両方に対応しています。"),
              text("Use `--ensemble` only when you explicitly want a mean of the selected models' probabilities. This is prediction on one cohort, not out-of-fold validation.",
                   "若要平均所選模型的預測機率，再使用 `--ensemble`。這是在同一群體上進行預測，並非 out-of-fold 驗證。",
                   "選択したモデルの予測確率を平均する場合に `--ensemble` を使用してください。同一コホートに対する予測であり、out-of-fold 検証ではありません。"), "",
              text("Archive entries under `historical/` are tensor-only because their original fold/stream settings were not saved. They are included for completeness but are not automatically inferable.",
                   "`historical/` 下的封存項目因未保存原始 fold／輸入流設定，因此僅封存張量，以保留完整紀錄；無法自動用於推論。",
                   "`historical/` 配下のアーカイブ項目は、元の fold／ストリーム設定が保存されていないため、テンソルのみを収録しています。記録の完全性を保つためのものであり、自動推論には対応していません。"),
              "", text("## Provenance and limits", "## 來源與限制", "## 来歴と制約"), "",
              text("These are legacy research weights. Earlier experiments used case-level random folds, with identified patient-overlap concerns, and selected checkpoints using validation scores. Do not label those scores as patient-independent cross-validation results.",
                   "這些權重來自歷史研究實驗。早期實驗採用病例層級的隨機 fold，已發現跨 fold 病人重疊的疑慮，並依驗證分數選取 checkpoint。不得將這些分數稱為病人互斥的交叉驗證結果。",
                   "これらは過去の研究で得られた重みです。初期の実験では症例単位のランダムな fold 分割を用い、fold 間で患者が重複する懸念が確認されており、検証スコアに基づいてチェックポイントを選択していました。これらのスコアを、患者単位で独立した交差検証の結果と表記しないでください。"),
              text("Class names bad/good are recorded labels; do not infer healthy/diseased meanings without the study definition.",
                   "bad／good 是原始紀錄中的類別標籤；未取得研究定義前，不得將其直接解讀為健康／患病。",
                   "bad／good は記録されたクラスラベルです。研究上の定義を確認せずに、健康／疾患を意味すると解釈しないでください。"),
              text("Profiles are explicit declarations; tensor keys/shapes cannot prove historical stream semantics or preprocessing. In particular, four-/six-stream runtime versions were not retained; their antialias choice must be checked against original records.",
                   "Profile 是明確宣告的設定；張量的鍵與形狀無法證實歷史輸入流語意或前處理方式。尤其四流／六流實驗未保留執行環境版本，其抗鋸齒選擇須依原始紀錄核對。",
                   "プロファイルは明示的に宣言された設定です。テンソルのキーや形状から、過去のストリームの意味や前処理を証明することはできません。特に 4／6 ストリーム実験では実行環境のバージョンが保存されていないため、アンチエイリアスの選択を元の記録と照合する必要があります。"),
              text("No model license was guessed. Complete the study-specific model documentation/license before treating this as a clinical release.",
                   "未推定或擅自指定模型授權。將本集合視為臨床用途的發布版本前，須完成該研究的模型文件與授權說明。",
                   "モデルのライセンスは推測に基づいて指定していません。臨床用途の公開版として扱う前に、本研究固有のモデル文書とライセンスを整備してください。")]
    if code_url:
        lines += ["", text(
            "Public code snapshot: [code and complete usage instructions](" + code_url + "). See `INFERENCE.md` and `MODEL_CATALOG.md` in that snapshot.",
            "公開程式碼快照：[程式碼與完整使用說明](" + code_url + ")。另請參閱該快照中的 `INFERENCE.md` 與 `MODEL_CATALOG.md`。",
            "公開コードのスナップショット：[コードと完全な利用手順](" + code_url + ")。スナップショット内の `INFERENCE.md` と `MODEL_CATALOG.md` も参照してください。")]
    return "\n".join(lines) + "\n"


def _collection_readme(items, profiles, code_url=None):
    """Keep all language versions in the model card accepted by the Hub uploader."""
    header = ("---\nlanguage:\n- en\n- zh\n- ja\ntags:\n- pytorch\n- timm\n"
              "- multistream\n- research\n---\n\n"
              "[English](#english) | [繁體中文](#traditional-chinese) | [日本語](#japanese)\n\n")
    sections = []
    for language, anchor in (("en", "english"), ("zh", "traditional-chinese"), ("ja", "japanese")):
        sections.append('<a id="' + anchor + '"></a>\n\n'
                        + _collection_readme_section(items, profiles, code_url, language))
    return header + "\n---\n\n".join(sections)


def _collection_model_card(metadata, item, profiles, code_url=None):
    """Describe this leaf while directing users to the shared collection entry point."""
    card = _model_card(metadata, code_url)
    start, end = card.index("## Loading"), card.index("## Preprocessing and verification")
    usage = ("## Loading this condition and fold\n\n"
             "Install the code repository requirements and run from its new/ directory. "
             "The shared loader and requirements live at the collection root, not in this model directory.\n\n"
             "```bash\npython tools/infer_published.py --repo ntuhophai/ERM-AI --condition "
             + item["condition"] + " --fold " + str(item["fold"])
             + " --root /path/to/cases --output predictions/" + item["condition"]
             + "_fold" + str(item["fold"]) + "\n```\n\n"
             "Before the model PR is merged, append --revision refs/pr/<number>; after merge the default main revision can be used.\n\n"
             "Only the selected weights/config are downloaded. Provide one image for each ordered stream. "
             "The local tool uses its trusted model definition and saves class probabilities.\n\n")
    card = card[:start] + usage + card[end:]
    card += "\nCollection condition: " + item["condition"] + f"; zero-based fold: {item['fold']}.\n"
    card += "\nDeclared preprocessing:\n\n```json\n" + json.dumps(metadata["preprocessing"], indent=2) + "\n```\n"
    card += "\nProfile note: " + profiles["conditions"][item["condition"]].get("notes", "Operator-declared settings") + "\n"
    if code_url:
        documentation = code_url.replace("/tree/", "/blob/", 1).rstrip("/") + "/INFERENCE.md"
        card += "\n[Complete download, inference and output instructions](" + documentation + ")\n"
    return card


def export_collection(records_root, profiles_path, output, code_url=None, plan_only=False):
    profiles = load_profiles(profiles_path)
    items = discover_models(records_root, profiles)
    plan = public_plan(items, profiles)
    if plan_only:
        return plan
    if output is None:
        raise ValueError("--output is required when exporting")
    code_url = _public_code_url(code_url)
    output = _ordinary_location(output)
    state_root = _ordinary_location(output.parent / (output.name + ".export-state"))
    if output.is_symlink() or state_root.is_symlink():
        raise ValueError("Linked output/state roots are not allowed")
    output.mkdir(parents=True, exist_ok=True)
    _validate_existing_cache(output, items)
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "state.json"
    if _linklike(state_path):
        raise ValueError("Linked local export state is not allowed")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"version": 1, "entries": {}}
    if state.get("version") != 1 or not isinstance(state.get("entries"), dict):
        raise ValueError("Unrecognized local export progress")
    loader_hash = hashlib.sha256(PUBLIC_LOADER.encode()).hexdigest()
    entries = []
    exported = reused = 0
    for number, item in enumerate(items, 1):
        print(f"[{number}/{len(items)}] {item['condition']} / fold {item['fold']}", flush=True)
        leaf = output.joinpath(*item["path"].split("/"))
        if leaf.is_symlink() or not leaf.resolve().is_relative_to(output):
            raise ValueError("Model output escaped its collection")
        metadata = _ready_metadata(profiles, item["condition"]) if item["inference_ready"] else _historical_config(item)
        source_hash = file_sha256(item["checkpoint"])
        identity = _digest({"source_sha256": source_hash, "metadata": metadata, "loader_sha256": loader_hash})
        previous = state["entries"].get(item["path"], {})
        valid = (leaf.is_dir() and {path.name for path in leaf.iterdir()} == LEAF_FILES
                 and not any(path.is_symlink() for path in leaf.iterdir())
                 and previous.get("identity") == identity
                 and previous.get("weights_sha256") == file_sha256(leaf / "model.safetensors")
                 and previous.get("config_sha256") == file_sha256(leaf / "config.json"))
        if not valid:
            print("  Exporting missing or changed model", flush=True)
            free = shutil.disk_usage(output).free
            if free < item["source_bytes"] * 2 + 256 * 1024 * 1024:
                raise RuntimeError("Not enough disk space for the next model; retain progress and free space before retrying")
            stage_parent = state_root / "staging" / uuid.uuid4().hex
            stage_parent.mkdir(parents=True)
            stage = stage_parent / "model"
            if item["inference_ready"]:
                effective = stage_parent / "profile.yaml"
                effective.write_text(yaml.safe_dump(effective_profile(profiles, item["condition"]), sort_keys=False), encoding="utf-8")
                exported_metadata = export_public_model(item["checkpoint"], stage, legacy_config=effective, code_url=code_url)
                if exported_metadata != metadata:
                    raise RuntimeError("Export metadata did not match the declared collection profile")
                (stage / "modeling_multistream.py").unlink()
                (stage / "requirements.txt").unlink()
            else:
                import torch
                from safetensors.torch import save_file
                raw = torch.load(item["checkpoint"], map_location="cpu", weights_only=True)
                tensors = _tensor_state(raw)
                for tensor in tensors.values():
                    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                        raise ValueError("Historical model contains nonfinite tensors")
                stage.mkdir()
                save_file({key: value.detach().cpu().contiguous().clone() for key, value in tensors.items()}, str(stage / "model.safetensors"))
                _write_json(stage / "config.json", metadata)
                (stage / "README.md").write_text(_historical_card(metadata), encoding="utf-8")
                del raw, tensors
            leaf.parent.mkdir(parents=True, exist_ok=True)
            if leaf.exists():
                backup = state_root / "backups" / uuid.uuid4().hex
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not leaf.resolve().is_relative_to(output) or not backup.resolve().is_relative_to(state_root):
                    raise RuntimeError("Unsafe model cache replacement target")
                _rename_with_retry(leaf, backup)  # Preserve an earlier public export instead of deleting it.
            _rename_with_retry(stage, leaf)
            exported += 1
        else:
            print("  Reusing verified export", flush=True)
            reused += 1
        card = (_collection_model_card(metadata, item, profiles, code_url)
                if item["inference_ready"] else _historical_card(metadata))
        (leaf / "README.md").write_text(card, encoding="utf-8")
        weights_hash, config_hash = file_sha256(leaf / "model.safetensors"), file_sha256(leaf / "config.json")
        state["entries"][item["path"]] = {"identity": identity, "weights_sha256": weights_hash, "config_sha256": config_hash}
        _write_json(state_path, state)
        entries.append({"condition": item["condition"], "fold": item["fold"], "streams": item["streams"],
                        "path": item["path"], "inference_ready": item["inference_ready"],
                        "weights_sha256": weights_hash, "config_sha256": config_hash,
                        "artifact_kind": "research", "source_format": "legacy_state_dict" if item["inference_ready"] else "legacy_raw_state_dict",
                        "settings_provenance": "operator_supplied" if item["inference_ready"] else "unavailable",
                        "weights_bytes": (leaf / "model.safetensors").stat().st_size,
                        "project": item["project"], "reason": "" if item["inference_ready"] else metadata["reason"]})
    catalog = {"catalog_version": 1, "models": entries,
               "inference_ready_count": plan["inference_ready_count"], "historical_count": plan["historical_count"],
               "model_count": len(entries), "condition_count": plan["condition_count"], "fold_count": plan["fold_count"],
               "total_weight_bytes": sum(entry["weights_bytes"] for entry in entries), "private_training_metadata_omitted": True}
    (output / "modeling_multistream.py").write_text(PUBLIC_LOADER, encoding="utf-8")
    (output / "README.md").write_text(_collection_readme(items, profiles, code_url), encoding="utf-8")
    requirements = []
    for package in ("torch", "torchvision", "timm", "safetensors", "huggingface_hub"):
        requirements.append(package + "==" + importlib.metadata.version(package).split("+", 1)[0])
    (output / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    _write_json(output / "catalog.json", catalog)  # Publish the complete local catalog last.
    return {**plan, "exported": exported, "reused": reused, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", required=True)
    parser.add_argument("--profiles", default=str(Path(__file__).resolve().parents[1] / "publish_profiles" / "legacy_collection.yaml"))
    parser.add_argument("--output")
    parser.add_argument("--code-url")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = export_collection(args.records_root, args.profiles, args.output, args.code_url, args.plan_only)
    except Exception as error:
        parser.exit(2, "Collection export stopped: " + str(error) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

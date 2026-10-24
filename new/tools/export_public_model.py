"""Export model tensors with public inference metadata only."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import re
import stat
import tempfile
import time
from urllib.parse import urlparse


# This trusted template is also the public loader. It does not import the
# private training engine and never fabricates patient/split metadata.
PUBLIC_LOADER = r'''"""Load public multistream weights without private training metadata."""
from pathlib import Path
import json

import timm
import torch
from torch import nn
from safetensors.torch import load_file


def _strict_load(model, state):
    expected = model.state_dict()
    if set(expected) != set(state):
        missing, extra = set(expected) - set(state), set(state) - set(expected)
        raise ValueError("Weight keys differ: missing=%s, extra=%s" %
                         (sorted(missing), sorted(extra)))
    for key, value in state.items():
        if tuple(value.shape) != tuple(expected[key].shape):
            raise ValueError("Weight shape differs for " + key)
    model.load_state_dict(state, strict=True)


class SingleStreamClassifier(nn.Module):
    def __init__(self, name, streams, num_classes, freeze_backbones=False):
        super().__init__()
        self.streams = tuple(streams)
        self.network = timm.create_model(name, pretrained=False, num_classes=num_classes)
        self.freeze_backbones = freeze_backbones
        if freeze_backbones:
            for parameter in self.network.parameters():
                parameter.requires_grad_(False)
            for parameter in self.network.get_classifier().parameters():
                parameter.requires_grad_(True)
            self.train(self.training)

    def forward(self, images):
        _check_images(images, len(self.streams))
        return self.network(images[:, 0])

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_backbones", False):
            self.network.eval()
            self.network.get_classifier().train(mode)
        return self

    def state_dict(self, *args, **kwargs):
        return self.network.state_dict(*args, **kwargs)

    def load_state_dict(self, state, strict=True, **kwargs):
        if not strict:
            raise ValueError("Partial weight loading is disabled")
        return self.network.load_state_dict(state, strict=True, **kwargs)


class MultiStreamClassifier(nn.Module):
    def __init__(self, name, streams, num_classes, hidden_dim=2048,
                 freeze_backbones=False):
        super().__init__()
        self.streams = tuple(streams)
        self.backbones = nn.ModuleList([
            timm.create_model(name, pretrained=False, num_classes=0, global_pool="avg")
            for _ in streams
        ])
        features = sum(int(backbone.num_features) for backbone in self.backbones)
        self.fusion = nn.Sequential(
            nn.Linear(features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.freeze_backbones = freeze_backbones
        if freeze_backbones:
            for parameter in self.backbones.parameters():
                parameter.requires_grad_(False)
            self.train(self.training)

    def forward(self, images):
        _check_images(images, len(self.streams))
        features = [backbone(images[:, index])
                    for index, backbone in enumerate(self.backbones)]
        return self.classifier(self.fusion(torch.cat(features, dim=1)))

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_backbones", False):
            self.backbones.eval()
        return self


def _check_images(images, streams):
    if not isinstance(images, torch.Tensor) or images.ndim != 5:
        raise ValueError("Expected images shaped [B, S, 3, H, W]")
    if images.shape[0] < 1 or images.shape[1] != streams or images.shape[2] != 3:
        raise ValueError("Batch, stream count or RGB channels do not match")


def build_model(model_config, streams, num_classes, initialize=False):
    if initialize:
        raise ValueError("Public loading does not perform pretrained initialization")
    if not isinstance(streams, (list, tuple)) or not 1 <= len(streams) <= 8:
        raise ValueError("Expected 1 to 8 explicitly ordered streams")
    if any(not isinstance(stream, str) or not stream for stream in streams):
        raise ValueError("Stream names must be nonempty strings")
    if len(set(streams)) != len(streams):
        raise ValueError("Stream names must be unique")
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("Expected at least two classes")
    name = model_config["name"]
    if not isinstance(name, str) or not timm.is_model(name):
        raise ValueError("Architecture must be registered locally in timm")
    hidden = model_config.get("hidden_dim", 2048)
    if type(hidden) is not int or hidden < 1:
        raise ValueError("hidden_dim must be a positive integer")
    frozen = model_config.get("freeze_backbones", False)
    if type(frozen) is not bool:
        raise ValueError("freeze_backbones must be a boolean")
    if len(streams) == 1:
        return SingleStreamClassifier(name, streams, num_classes, frozen)
    return MultiStreamClassifier(name, streams, num_classes, hidden, frozen)


def load_public_model(directory, device="cpu"):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("public_format_version") != 1:
        raise ValueError("Unsupported public export version")
    classes = config["class_to_idx"]
    if (not isinstance(classes, dict) or len(classes) < 2
            or any(not isinstance(name, str) or type(index) is not int
                   for name, index in classes.items())
            or sorted(classes.values()) != list(range(len(classes)))):
        raise ValueError("Invalid public class mapping")
    model = build_model(config["model_config"], config["streams"], len(classes))
    _strict_load(model, load_file(str(directory / "model.safetensors"), device="cpu"))
    model.public_config = config
    model.class_to_idx = classes
    model.preprocessing = config["preprocessing"]
    return model.to(device).eval()


def preprocess_streams(paths, preprocessing):
    """Return [1, S, 3, H, W] in the supplied path order."""
    from PIL import Image
    from torchvision.transforms import functional as F
    from torchvision.transforms import InterpolationMode
    tensors = []
    backend = preprocessing.get("resize_backend", "pil")
    for path in paths:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if backend == "tensor":
                tensor = F.resize(
                    F.to_tensor(image), preprocessing["resize_size"],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=preprocessing["antialias"],
                )
            elif backend == "pil":
                image = F.resize(image, preprocessing["resize_size"],
                                 interpolation=InterpolationMode.BILINEAR)
                tensor = F.to_tensor(image)
            else:
                raise ValueError("Unsupported resize backend")
        tensor = F.center_crop(tensor, [preprocessing["image_size"]] * 2)
        tensors.append(F.normalize(tensor, preprocessing["mean"], preprocessing["std"]))
    if not tensors:
        raise ValueError("Supply one image path per declared stream")
    return torch.stack(tensors).unsqueeze(0)
'''


def _ordered_streams(value):
    if (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 8
            or any(not isinstance(v, str) or not v for v in value)
            or len(set(value)) != len(value)):
        raise ValueError("Supply 1 to 8 unique, explicitly ordered stream names")
    return list(value)


def _class_mapping(value):
    if (not isinstance(value, dict) or len(value) < 2
            or any(not isinstance(k, str) or not k or type(v) is not int
                   for k, v in value.items())
            or sorted(value.values()) != list(range(len(value)))):
        raise ValueError("class_to_idx must use unique contiguous indices from zero")
    return dict(value)


def _public_preprocessing(value, legacy):
    if not isinstance(value, dict):
        raise ValueError("Explicit preprocessing metadata is required")
    required = {"image_size", "resize_size", "mean", "std"}
    if legacy:
        required.update({"resize_backend", "antialias"})
    if not required.issubset(value):
        raise ValueError(
            "preprocessing requires image_size, resize_size, mean and std; "
            "raw legacy settings also require explicit resize_backend and antialias"
        )
    for field in ("image_size", "resize_size"):
        if type(value[field]) is not int or value[field] <= 0:
            raise ValueError(field + " must be a positive integer")
    vectors = {}
    for field in ("mean", "std"):
        vector = value[field]
        if (not isinstance(vector, (list, tuple)) or len(vector) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in vector)):
            raise ValueError(field + " must contain three finite RGB values")
        vectors[field] = [float(v) for v in vector]
    if any(v <= 0 for v in vectors["std"]):
        raise ValueError("RGB std must be positive")
    backend = value.get("resize_backend", "pil")
    if backend not in ("pil", "tensor"):
        raise ValueError("resize_backend must be pil or tensor")
    interpolation = value.get("interpolation", "bilinear")
    if interpolation != "bilinear":
        raise ValueError("This loader supports explicit bilinear resizing")
    antialias = value.get("antialias", True)
    if type(antialias) is not bool:
        raise ValueError("antialias must be a boolean")
    if backend == "pil":
        antialias = True  # PIL bilinear resize always applies antialiasing.
    return {
        "image_size": value["image_size"], "resize_size": value["resize_size"],
        **vectors, "resize_backend": backend, "interpolation": "bilinear",
        "antialias": antialias, "crop": "center",
    }


def _safe_model_config(value):
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        raise ValueError("Explicit model.name is required")
    hidden = value.get("hidden_dim", 2048)
    frozen = value.get("freeze_backbones", False)
    if type(hidden) is not int or hidden < 1 or type(frozen) is not bool:
        raise ValueError("Invalid hidden_dim or freeze_backbones")
    # Never copy initialization, initialization_context or arbitrary nested fields.
    return {"name": value["name"], "hidden_dim": hidden,
            "freeze_backbones": frozen, "pretrained": False}


def _tensor_state(value):
    import torch
    if (not isinstance(value, Mapping) or not value
            or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                   for k, v in value.items())):
        raise ValueError("Weights must be a nonempty string-to-tensor mapping")
    return value


def _legacy_keys(state, stream_count):
    if stream_count == 1:
        return dict(state)  # Single-stream legacy files contain raw timm keys.
    result = {}
    for key, tensor in state.items():
        branch = re.fullmatch(r"b([1-8])\.(.+)", key)
        if branch:
            index = int(branch.group(1)) - 1
            if index >= stream_count:
                raise ValueError("Legacy state contains an unexpected backbone")
            mapped = "backbones.%d.%s" % (index, branch.group(2))
        elif key.startswith("merger.0.") or key.startswith("merger.2."):
            mapped = "fusion." + key[len("merger."):]
        elif key.startswith("fc."):
            mapped = "classifier." + key[len("fc."):]
        else:
            raise ValueError("Unsupported legacy weight key: " + key)
        if mapped in result:
            raise ValueError("Legacy weight-key mapping collided")
        result[mapped] = tensor
    return result


def _merge_config(base, child):
    """Merge settings while replacing every class_to_idx mapping as a whole."""
    merged = copy.deepcopy(base)
    for key, value in child.items():
        if (key != "class_to_idx" and isinstance(value, dict)
                and isinstance(merged.get(key), dict)):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_legacy_config(path, max_depth=16):
    """Resolve safe YAML extends relative to each declaring file's directory."""
    import yaml
    if type(max_depth) is not int or max_depth < 1:
        raise ValueError("Config extends depth limit must be a positive integer")

    def load(current, active):
        current = Path(current).resolve()
        identity = str(current).casefold()
        if identity in active:
            raise ValueError("Config extends contains a cycle")
        if len(active) >= max_depth:
            raise ValueError("Config extends exceeds the permitted depth")
        value = yaml.safe_load(current.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Each legacy YAML config must contain a mapping")
        parents = value.pop("extends", None)
        if parents is None:
            parents = []
        elif isinstance(parents, str):
            parents = [parents]
        elif not isinstance(parents, list):
            raise ValueError("extends must be a path or a list of paths")
        if any(not isinstance(parent, str) or not parent.strip() for parent in parents):
            raise ValueError("Config extends paths must be nonempty strings")
        result = {}
        for parent in parents:
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = current.parent / parent_path
            result = _merge_config(result, load(parent_path, active + (identity,)))
        return _merge_config(result, value)

    return load(Path(path), ())


def read_public_source(checkpoint_path, legacy_config=None, kind="research"):
    """Read safely and return only tensors plus allowlisted public metadata."""
    import torch
    try:
        source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as error:
        if "weights_only" in str(error):
            raise RuntimeError("Upgrade PyTorch; unsafe pickle fallback is disabled") from error
        raise
    versioned = isinstance(source, dict) and source.get("format_version") == 1
    if versioned:
        required = {"state_dict", "model_config", "streams", "class_to_idx", "preprocessing"}
        if not required.issubset(source):
            raise ValueError("Versioned checkpoint is missing public inference metadata")
        state = _tensor_state(source["state_dict"])
        model_config = _safe_model_config(source["model_config"])
        streams = _ordered_streams(source["streams"])
        classes = _class_mapping(source["class_to_idx"])
        legacy = bool(source.get("legacy_no_split_metadata", False))
        # Versioned sources are interpreted by the new loader, whose default
        # image pipeline is PIL/antialias=True even for converted legacy weights.
        preprocessing = _public_preprocessing(source["preprocessing"], False)
        source_format = "converted_legacy" if legacy else "versioned_checkpoint"
        settings_provenance = "operator_supplied" if legacy else "checkpoint_metadata"
    else:
        if not legacy_config:
            raise ValueError("Raw legacy weights require --legacy-config with explicit stream order/settings")
        config = resolve_legacy_config(legacy_config)
        if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
            raise ValueError("Legacy config must declare data, model and preprocessing")
        model_config = _safe_model_config(config.get("model"))
        streams = _ordered_streams(config["data"].get("streams"))
        classes = _class_mapping(config["data"].get("class_to_idx"))
        preprocessing = _public_preprocessing(config.get("preprocessing"), True)
        raw = source
        if isinstance(source, dict) and "state_dict" in source:
            raw = source["state_dict"]
        elif isinstance(source, dict) and "model_state_dict" in source:
            raw = source["model_state_dict"]
        state = _legacy_keys(_tensor_state(raw), len(streams))
        legacy, source_format, settings_provenance = True, "legacy_state_dict", "operator_supplied"
    if kind not in ("demo", "research"):
        raise ValueError("kind must be demo or research")
    if "smoke_output" in str(checkpoint_path).replace("\\", "/").casefold():
        kind = "demo"
    metadata = {
        "public_format_version": 1, "model_config": model_config,
        "streams": streams, "class_to_idx": classes, "preprocessing": preprocessing,
        "artifact_kind": kind, "source_format": source_format,
        "settings_provenance": settings_provenance,
        "private_training_metadata_omitted": True,
        "legacy_no_split_metadata": legacy,
    }
    return state, metadata


def _reference_loader():
    namespace = {"__name__": "_trusted_public_model_loader"}
    exec(compile(PUBLIC_LOADER, "<trusted public loader template>", "exec"), namespace)
    return namespace


def _validate_state(state, metadata):
    import torch
    loader = _reference_loader()
    model = loader["build_model"](metadata["model_config"], metadata["streams"],
                                 len(metadata["class_to_idx"]), initialize=False)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("Weight keys do not strictly match the supplied architecture/stream count")
    for key, tensor in state.items():
        if tuple(tensor.shape) != tuple(expected[key].shape):
            raise ValueError("Weight shape does not match for " + key)
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("Nonfinite weight tensor: " + key)
    loader["_strict_load"](model, state)
    del model


def _public_code_url(value):
    if value is None:
        return None
    parsed = urlparse(value)
    if (parsed.scheme != "https" or parsed.hostname != "github.com"
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("--code-url must be a GitHub HTTPS URL without credentials/query")
    return value


def _multilingual_card(render_section, tags):
    """Render one English-first README with stable same-file language links."""
    header = "---\nlanguage:\n- en\n- zh\n- ja\ntags:\n"
    header += "".join("- " + tag + "\n" for tag in tags)
    header += ("---\n\n[English](#english) | [繁體中文](#traditional-chinese) | "
               "[日本語](#japanese)\n\n")
    sections = []
    for language, anchor in (("en", "english"), ("zh", "traditional-chinese"), ("ja", "japanese")):
        sections.append('<a id="' + anchor + '"></a>\n\n' + render_section(language).rstrip() + "\n")
    return header + "\n---\n\n".join(sections)


def _model_card_section(metadata, code_url=None, language="en", loading=None):
    def text(english, chinese, japanese):
        return {"en": english, "zh": chinese, "ja": japanese}[language]

    order = json.dumps(metadata["streams"], ensure_ascii=False)
    classes = json.dumps(metadata["class_to_idx"], ensure_ascii=False)
    model = metadata["model_config"]
    kind = metadata["artifact_kind"]
    if kind == "demo":
        scope = text(
            "This is a demo/smoke-test artifact. It is not a formally trained ophthalmology model. Demo metrics are software checks, not clinical results.",
            "這是示範／冒煙測試用產物，並非正式訓練的眼科模型。示範指標僅用於檢查軟體，不代表臨床結果。",
            "これはデモ／スモークテスト用の成果物であり、正式に学習した眼科モデルではありません。デモの指標はソフトウェアの確認結果であり、臨床成績ではありません。")
    else:
        scope = text(
            "This is a research artifact. This export does not establish whether training used real or synthetic data, independent validation, or clinical readiness.",
            "這是研究用產物。本次匯出無法證實訓練採用真實或合成資料、是否經過獨立驗證，或是否已具備臨床應用條件。",
            "これは研究用の成果物です。このエクスポートによって、学習データが実データか合成データか、独立した検証が行われたか、臨床利用の準備が整っているかを確認することはできません。")
    if metadata["legacy_no_split_metadata"]:
        limitations = text(
            "The source is a legacy model. Architecture and tensor keys/shapes were checked strictly, "
            "but the historical stream order, label meaning and preprocessing cannot be proved from "
            "weights. These settings were supplied by the operator. Earlier experiments used "
            "case-level random folds with identified patient-overlap concerns and selected epochs "
            "on validation accuracy. Their scores must not be presented as independent "
            "patient-grouped cross-validation results. Converted format does not correct those experiments.",
            "來源為歷史模型。已嚴格核對架構、張量鍵與形狀，但無法僅憑權重證實歷史輸入流順序、標籤意義及前處理方式；這些設定由操作者提供。早期實驗採用病例層級的隨機 fold，已發現跨 fold 病人重疊的疑慮，並依驗證準確率選取 epoch。不得將其分數稱為以病人分組且彼此獨立的交叉驗證結果。格式轉換不會修正原實驗的這些問題。",
            "変換元は過去のモデルです。モデル構造、テンソルのキーと形状を厳密に確認していますが、過去のストリーム順序、ラベルの意味、前処理は重みだけでは証明できません。これらの設定は操作者が提供したものです。初期の実験では症例単位のランダムな fold 分割を用い、fold 間で患者が重複する懸念が確認されており、検証正解率に基づいて epoch を選択していました。これらのスコアを、患者単位で分割された独立した交差検証の結果と表記しないでください。形式を変換しても、元の実験の問題は修正されません。")
    else:
        limitations = text(
            "Inference settings come from the source checkpoint. Private training partitions and "
            "dataset provenance are deliberately omitted. No training-data or performance claim is "
            "made by this export; substantiate any such claim using the original private experiment records.",
            "推論設定來自來源 checkpoint。公開匯出刻意省略私有訓練分割與資料集來源資訊。本次匯出不對訓練資料或模型效能作出任何主張；如需提出相關主張，須以原始非公開實驗紀錄佐證。",
            "推論設定は元のチェックポイントに由来します。非公開の学習データ分割とデータセットの来歴は、意図的に省略しています。このエクスポートでは、学習データや性能に関する主張は行いません。そのような主張には、元の非公開実験記録による裏付けが必要です。")
    code = ""
    if code_url:
        code = text("Public code: ", "公開程式碼：", "公開コード：") + "[GitHub](" + code_url + ")\n\n"
    if loading is None:
        example = (
            "```python\nfrom pathlib import Path\nimport sys, torch\n"
            'bundle = Path("downloaded_model").resolve()\n'
            "sys.path.insert(0, str(bundle))\n"
            "from modeling_multistream import load_public_model, preprocess_streams\n"
            'model = load_public_model(bundle, device="cpu")\n'
            "# Replace these paths with matching images, preserving stream order.\n"
            'image_paths = [str(Path("my_images") / (stream + ".jpg"))\n'
            "               for stream in model.streams]\n"
            "images = preprocess_streams(image_paths, model.preprocessing)\n"
            "with torch.no_grad():\n"
            "    probabilities = torch.softmax(model(images), dim=1)\n```\n\n")
        loading = text("## Loading\n\n", "## 載入模型\n\n", "## モデルの読み込み\n\n") + text(
            "Install the dependencies in `requirements.txt` and place the exported files in one directory. The public loader uses safetensors and never asks for private patient metadata.\n\n",
            "請安裝 `requirements.txt` 中的相依套件，並將匯出檔案放在同一個目錄。公開載入器使用 safetensors，不需要任何私有病人資料。\n\n",
            "`requirements.txt` の依存パッケージをインストールし、エクスポートされたファイルを同じディレクトリに配置してください。公開ローダーは safetensors を使用し、非公開の患者メタデータを要求しません。\n\n") + example + text(
            "Inputs have shape [batch, streams, 3, height, width]. Generic single-image Transformers pipelines are not a loader for this custom multistream architecture.\n\n",
            "輸入形狀為 [batch, streams, 3, height, width]。一般的單張影像 Transformers pipeline 無法直接載入此自訂多流架構。\n\n",
            "入力の形状は [batch, streams, 3, height, width] です。一般的な単一画像用の Transformers パイプラインでは、この独自のマルチストリーム構造を読み込むことはできません。\n\n")
    return (
        text("# Multistream model artifact", "# 多流模型產物", "# マルチストリームモデルの成果物") + "\n\n" + scope + "\n\n"
        + text(f"Architecture: {model['name']}; hidden dimension: {model['hidden_dim']}.",
               f"模型架構：{model['name']}；隱藏維度：{model['hidden_dim']}。",
               f"モデル構造：{model['name']}。隠れ次元数：{model['hidden_dim']}。") + "\n\n"
        + text(f"Ordered streams: `{order}`\n\nClass mapping: `{classes}`",
               f"輸入流順序：`{order}`\n\n類別對應：`{classes}`",
               f"ストリームの順序：`{order}`\n\nクラスの対応関係：`{classes}`") + "\n\n"
        + text("Class names are labels only; do not infer healthy/diseased semantics without the study definition.",
               "類別名稱僅為標籤；未取得研究定義前，不得將其直接解讀為健康／患病。",
               "クラス名はラベルです。研究上の定義を確認せずに、健康／疾患を意味すると解釈しないでください。") + "\n\n"
        + limitations + "\n\n" + code + loading
        + text("## Preprocessing and verification\n\n", "## 前處理與核對\n\n", "## 前処理と確認\n\n")
        + text("`config.json` records the RGB resize backend, bilinear interpolation, center crop and normalization. For raw legacy exports, verify these operator-supplied settings against the original experiment. Export verification checks every tensor key and shape and rejects nonfinite floating-point weights; it does not run inference or validate performance.",
               "`config.json` 記錄 RGB 影像縮放後端、雙線性插值、中心裁切與正規化方式。若匯出原始歷史權重，須依原始實驗核對操作者提供的設定。匯出核對會檢查每個張量鍵與形狀，並拒絕含非有限浮點數的權重；不會執行推論或驗證效能。",
               "`config.json` に RGB 画像のリサイズ処理、双線形補間、中央クロップ、正規化を記録しています。旧形式の重みをエクスポートする場合は、操作者が提供した設定を元の実験と照合してください。エクスポート時の確認では、すべてのテンソルのキーと形状を検査し、非有限の浮動小数点値を含む重みを拒否します。推論や性能検証は実施しません。") + "\n\n"
        + text("## License\n\n", "## 授權\n\n", "## ライセンス\n\n")
        + text("No model license was inferred by the exporter. Supply the authorized model license and complete the study-specific model card before treating this as a documented release.",
               "匯出器未推定或擅自指定模型授權。將本產物視為具完整文件的發布版本前，須提供經授權的模型授權條款，並完成該研究的模型卡。",
               "エクスポーターはモデルのライセンスを推測に基づいて指定していません。文書を整備した公開版として扱う前に、正当に許諾されたモデルのライセンスを提示し、本研究固有のモデルカードを完成させてください。") + "\n")


def _model_card(metadata, code_url=None):
    return _multilingual_card(
        lambda language: _model_card_section(metadata, code_url, language),
        ("pytorch", "timm", "multistream", "image-classification", metadata["artifact_kind"]))


def _rename_with_retry(source, target):
    """Commit an ordinary directory without overwriting or discarding staged files."""
    source, target = Path(source).absolute(), Path(target).absolute()
    delays = (0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    for attempt in range(len(delays) + 1):
        # Recheck on every attempt: a destination may have appeared during a
        # transient Windows scanner/file-sharing lock. Never replace that path.
        for directory in (source, *source.parents, target.parent, *target.parent.parents):
            details = directory.lstat()
            if (stat.S_ISLNK(details.st_mode)
                    or getattr(details, "st_file_attributes", 0) & 0x400):
                raise ValueError("Linked rename locations are not allowed")
            if not stat.S_ISDIR(details.st_mode):
                raise ValueError("Model rename locations must be ordinary directories")
        try:
            target.lstat()
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Model destination already exists; refusing overwrite")
        try:
            return source.rename(target)
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def export_public_model(checkpoint_path, output, legacy_config=None, kind="research",
                        code_url=None, preview=False):
    """Export tensors and public metadata; never copy the full checkpoint object."""
    from safetensors.torch import save_file
    state, metadata = read_public_source(checkpoint_path, legacy_config, kind)
    code_url = _public_code_url(code_url)
    _validate_state(state, metadata)
    if preview:
        return metadata
    if output is None:
        raise ValueError("--output is required unless --preview is used")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new export directory; existing files are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".public-model-", dir=output.parent)).resolve()
    if stage.parent != output.parent or not stage.name.startswith(".public-model-"):
        raise RuntimeError("Temporary export directory resolved outside its intended parent")
    known_files = {"model.safetensors", "config.json", "modeling_multistream.py",
                   "README.md", "requirements.txt"}
    stage_complete = False
    try:
        # Cloning each tensor prevents shared-storage serialization surprises.
        public_tensors = {key: value.detach().cpu().contiguous().clone()
                          for key, value in state.items()}
        save_file(public_tensors, str(stage / "model.safetensors"))
        (stage / "config.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / "modeling_multistream.py").write_text(PUBLIC_LOADER, encoding="utf-8")
        (stage / "README.md").write_text(_model_card(metadata, code_url), encoding="utf-8")
        dependencies = []
        for distribution in ("torch", "torchvision", "timm", "safetensors"):
            version = importlib.metadata.version(distribution).split("+", 1)[0]
            dependencies.append(distribution + "==" + version)
        (stage / "requirements.txt").write_text("\n".join(dependencies) + "\n", encoding="utf-8")
        if {p.name for p in stage.iterdir()} != known_files:
            raise RuntimeError("Unexpected file in the public export")
        stage_complete = True
        if output.exists():
            raise FileExistsError("Export destination appeared while staging; refusing overwrite")
        _rename_with_retry(stage, output)
    finally:
        # A complete export blocked at commit is useful recovery evidence. Keep
        # its tensors and metadata instead of deleting the expensive output.
        # Only remove known files from the exact directory created above.
        # Never recurse into a computed or operator-owned directory.
        if (not stage_complete and stage.exists() and stage.parent == output.parent
                and stage.name.startswith(".public-model-")):
            for name in known_files:
                file = stage / name
                if file.is_file() and not file.is_symlink():
                    file.unlink()
            if not any(stage.iterdir()):
                stage.rmdir()
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export model tensors with public inference settings only; no remote writes."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--legacy-config", help="Explicit YAML settings for raw legacy weights")
    parser.add_argument("--kind", choices=("demo", "research"), default="research")
    parser.add_argument("--code-url", help="Public GitHub source URL without credentials")
    parser.add_argument("--preview", action="store_true", help="Validate and print public settings without writing")
    args = parser.parse_args(argv)
    try:
        metadata = export_public_model(
            args.checkpoint, args.output, args.legacy_config, args.kind,
            args.code_url, args.preview,
        )
    except Exception as error:
        parser.exit(2, "Export failed: " + str(error) + "\n")
    # This object was constructed using an allowlist. No patient/sample IDs,
    # source paths, nested contexts or dataset/split fingerprints are printed.
    print(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

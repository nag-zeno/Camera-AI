"""
role_classifier.py — Tầng 3: Understanding (Role Classification)

Nhận vào: frame + TrackedObject (persons)
Trả ra  : TrackedObject với role, role_confidence, role_evidence đã được điền

Năm chế độ hoạt động (theo thứ tự ưu tiên):
  1. [ONNX-GPU]  ONNX Runtime DirectML/CUDA — nhanh nhất trên GPU
  2. [ONNX-CPU]  ONNX Runtime CPU — tối ưu đa nhân
  3. [ML V3]     RoleNet v3 (ConvNeXt-Tiny, ~99.96%) — PyTorch fallback
  4. [ML V2]     RoleNet v2 (EfficientNet-B2, 64%) — fallback
  5. [ML V1]     RoleNet v1 (MobileNetV3-Small, 59%) — fallback
  6. [Rule]      HSV color + accessory rules — fallback cuối cùng

Hỗ trợ batch inference (xử lý nhiều người cùng lúc) — hiệu quả hơn nhiều trên GPU.
"""
import cv2
import numpy as np
import logging
import json
from pathlib import Path
from typing import Optional

from config import ROLE_CONFIG, MODELS_DIR
from models import TrackedObject, SocialRole, RoleEvidence, ObjectCategory
from modules.gpu_manager import create_ort_session, get_torch_device

logger = logging.getLogger(__name__)

# Trọng số rule-based (chỉ dùng khi fallback)
W_COLOR     = 0.65
W_ACCESSORY = 0.35

# Tên các file model — ONNX ưu tiên, fallback PyTorch V3 → V2 → V1
MODEL_ONNX_NAME    = "rolenet_v3.onnx"            # ONNX FP32 (export_onnx.py)
MODEL_ONNX_QUANT   = "rolenet_v3_quant.onnx"      # ONNX INT8 quantized (nhanh hơn)
MODEL_V3_PT_NAME   = "rolenet_v3_best.pt"
MODEL_V3_META_NAME = "rolenet_v3_metadata.json"
MODEL_V2_PT_NAME   = "rolenet_v2_best.pt"
MODEL_V2_META_NAME = "rolenet_v2_metadata.json"
MODEL_PT_NAME      = "rolenet_best.pt"        # V1 fallback
MODEL_META_NAME    = "rolenet_metadata.json"  # V1 fallback

# Input size RoleNet: (H, W)
ROLENET_INPUT_H  = 256
ROLENET_INPUT_W  = 128

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _try_load_rolenet_onnx():
    """
    Thử load RoleNet V3 qua ONNX Runtime — tự động chọn DirectML/CUDA/CPU.
    Ưu tiên INT8 quantized → FP32.
    Trả về (session, version_tag) hoặc (None, None).
    """
    for onnx_name, version_tag in [
        (MODEL_ONNX_QUANT, "onnx_int8"),
        (MODEL_ONNX_NAME,  "onnx_fp32"),
    ]:
        onnx_path = MODELS_DIR / onnx_name
        if not onnx_path.exists():
            continue
        sess, provider = create_ort_session(str(onnx_path))
        if sess is not None:
            size_mb = onnx_path.stat().st_size / 1e6
            logger.info(
                f"[RoleClassifier] ONNX loaded: {onnx_name} "
                f"({size_mb:.1f} MB, provider={provider})"
            )
            return sess, version_tag

    return None, None


def _try_load_rolenet_v3():
    """
    Thử load RoleNet V3 (ConvNeXt-Tiny) — độ chính xác cao nhất.
    Trả về (model, preprocess, device, version) hoặc (None, None, None, None).
    """
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms

        model_path = MODELS_DIR / MODEL_V3_PT_NAME
        if not model_path.exists():
            return None, None, None, None

        try:
            import timm
        except ImportError:
            logger.warning("[RoleClassifier] timm chưa cài. Chạy: pip install timm")
            return None, None, None, None

        device = "cuda" if get_torch_device() != "cpu" else "cpu"
        ckpt   = torch.load(str(model_path), map_location=device, weights_only=False)

        # Tái tạo kiến trúc V3 (ConvNeXt-Tiny + LayerNorm head)
        backbone    = timm.create_model("convnext_tiny", pretrained=False,
                                        num_classes=0, drop_rate=0.3)
        in_features = backbone.num_features  # 768 với ConvNeXt-Tiny

        head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 16),
        )

        class _RoleNetV3(nn.Module):
            def __init__(self, bb, cl):
                super().__init__()
                self.backbone   = bb
                self.classifier = cl
            def forward(self, x):
                return self.classifier(self.backbone.forward_features(x))

        net = _RoleNetV3(backbone, head)
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        net.to(device)

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((ROLENET_INPUT_H, ROLENET_INPUT_W)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        val_acc = ckpt.get("val_acc", 0.0)
        logger.info(
            f"[RoleClassifier] ✅ RoleNet V3 (ConvNeXt-Tiny) loaded! "
            f"Device={device}, Val Acc={val_acc*100:.1f}%"
        )
        return net, preprocess, device, "v3"

    except Exception as e:
        logger.warning(f"[RoleClassifier] Không load được RoleNet V3: {e}")
        return None, None, None, None


def _try_load_rolenet_v2():
    """
    Thử load RoleNet V2 (EfficientNet-B2).
    Trả về (model, preprocess, device, version) hoặc (None, None, None, None).
    """
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms

        model_path = MODELS_DIR / MODEL_V2_PT_NAME
        meta_path  = MODELS_DIR / MODEL_V2_META_NAME

        if not model_path.exists():
            return None, None, None, None

        # Cần timm để load EfficientNet-B2
        try:
            import timm
        except ImportError:
            logger.warning("[RoleClassifier] timm chưa cài. Chạy: pip install timm")
            return None, None, None, None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt   = torch.load(str(model_path), map_location=device, weights_only=False)

        # Tái tạo kiến trúc V2 (EfficientNet-B2 + custom head)
        backbone = timm.create_model("efficientnet_b2", pretrained=False,
                                     num_classes=0, drop_rate=0.35)
        in_features = backbone.num_features  # 1408

        head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.175),
            nn.Linear(512, 16),
        )

        class _RoleNetV2(nn.Module):
            def __init__(self, bb, cl):
                super().__init__()
                self.backbone   = bb
                self.classifier = cl
            def forward(self, x):
                return self.classifier(self.backbone.forward_features(x))

        net = _RoleNetV2(backbone, head)
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        net.to(device)

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((ROLENET_INPUT_H, ROLENET_INPUT_W)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        val_acc = ckpt.get("val_acc", 0.0)
        logger.info(
            f"[RoleClassifier] ✅ RoleNet V2 (EfficientNet-B2) loaded! "
            f"Device={device}, Val Acc={val_acc*100:.1f}%"
        )
        return net, preprocess, device, "v2"

    except Exception as e:
        logger.warning(f"[RoleClassifier] Không load được RoleNet V2: {e}")
        return None, None, None, None


def _try_load_rolenet_v1():
    """
    Thử load RoleNet V1 (MobileNetV3-Small) — fallback.
    Trả về (model, preprocess, device, version) hoặc (None, None, None, None).
    """
    try:
        import torch
        import torchvision.transforms as transforms

        model_path = MODELS_DIR / MODEL_PT_NAME
        meta_path  = MODELS_DIR / MODEL_META_NAME

        if not model_path.exists():
            logger.info(f"[RoleClassifier] RoleNet V1 không tìm thấy tại {model_path}.")
            return None, None, None, None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt   = torch.load(str(model_path), map_location=device, weights_only=False)

        from torchvision import models as tv_models
        import torch.nn as nn

        net = tv_models.mobilenet_v3_small(weights=None)
        in_features = net.classifier[0].in_features
        net.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 16),
        )
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        net.to(device)

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((ROLENET_INPUT_H, ROLENET_INPUT_W)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        val_acc = ckpt.get("val_acc", 0.0)
        logger.info(
            f"[RoleClassifier] ✅ RoleNet V1 (MobileNetV3) loaded! "
            f"Device={device}, Val Acc={val_acc*100:.1f}%"
        )
        return net, preprocess, device, "v1"

    except Exception as e:
        logger.warning(f"[RoleClassifier] Không load được RoleNet V1: {e}. Dùng rule-based fallback.")
        return None, None, None, None


def _try_load_rolenet():
    """
    Wrapper: thử V3 → V2 → V1 → trả về (model, preprocess, device).
    Để backward-compatible với code cũ.
    """
    net, prep, dev, ver = _try_load_rolenet_v3()
    if net is not None:
        return net, prep, dev
    net, prep, dev, ver = _try_load_rolenet_v2()
    if net is not None:
        return net, prep, dev
    net, prep, dev, ver = _try_load_rolenet_v1()
    return net, prep, dev


class RoleClassifier:
    """
    Nhận diện vai trò xã hội từ person crop.

    Chế độ ML (khi có model):
      - Crop người → resize 128×256 → ConvNeXt-Tiny/EfficientNet-B2 → softmax → SocialRole
      - Confidence là xác suất lớp được chọn
      - TTA (Test-Time Augmentation) bật với V2 và V3

    Chế độ Rule-based (fallback):
      - Phân tích màu HSV vùng torso
      - Co-occurrence phụ kiện (backpack, motorcycle...)
    """

    # Class-level: shared model across instances
    # --- ONNX Runtime path ---
    _onnx_sess    = None        # ort.InferenceSession (nếu có)
    # --- PyTorch path ---
    _net          = None
    _preprocess   = None
    _device       = None
    _model_version = None      # "onnx_fp32", "onnx_int8", "v3", "v2", "v1", hoặc None
    _model_loaded  = False

    # Mapping từ class names của model
    ROLENET_CLASSES = [
        "shipper", "doctor", "police", "military", "security", "student",
        "chef", "janitor", "construction", "nurse", "postman", "technician",
        "worker", "civil_guard", "normal", "unknown",
    ]

    def __init__(self):
        self._cfg        = ROLE_CONFIG
        self._hsv_ranges = {
            name: (np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            for name, (lo, hi) in self._cfg["hsv_ranges"].items()
        }
        self._roles      = self._cfg["roles"]
        self._crop_size  = self._cfg["crop_size"]
        self._global_thr = self._cfg["confidence_threshold"]

        # Load model một lần duy nhất (class-level singleton)
        # Ưu tiên: ONNX → PyTorch V3 → V2 → V1 → rule-based
        if not RoleClassifier._model_loaded:
            # 1. Thử ONNX Runtime trước (nhanh nhất)
            sess, ver = _try_load_rolenet_onnx()
            if sess is not None:
                RoleClassifier._onnx_sess     = sess
                RoleClassifier._model_version = ver
                RoleClassifier._model_loaded  = True
            else:
                # 2. Fallback: PyTorch V3 → V2 → V1
                net, prep, dev, ver = _try_load_rolenet_v3()
                if net is None:
                    net, prep, dev, ver = _try_load_rolenet_v2()
                if net is None:
                    net, prep, dev, ver = _try_load_rolenet_v1()
                RoleClassifier._net           = net
                RoleClassifier._preprocess    = prep
                RoleClassifier._device        = dev
                RoleClassifier._model_version = ver
                RoleClassifier._model_loaded  = True

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    @property
    def using_onnx(self) -> bool:
        """True nếu đang dùng ONNX Runtime (nhanh nhất)."""
        return RoleClassifier._onnx_sess is not None

    @property
    def using_ml_model(self) -> bool:
        """True nếu có bất kỳ ML model nào (ONNX hoặc PyTorch)."""
        return RoleClassifier._onnx_sess is not None or RoleClassifier._net is not None

    @property
    def model_version(self) -> Optional[str]:
        return RoleClassifier._model_version  # "onnx_fp32", "onnx_int8", "v3", "v2", "v1", None

    @property
    def use_tta(self) -> bool:
        """TTA: chỉ dùng với PyTorch V2/V3 — ONNX không cần (đã export với TTA ở eval mode)."""
        return RoleClassifier._model_version in ("v2", "v3")

    def classify_batch(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        nearby_objects: list[TrackedObject] | None = None,
    ) -> list[TrackedObject]:
        """
        Classify nhiều person cùng lúc bằng batch inference GPU — hiệu quả hơn nhiều.
        Fallback về classify() từng cái nếu không hỗ trợ ONNX.
        """
        persons = [o for o in objects if o.category == ObjectCategory.PERSON]
        if not persons:
            return objects

        # Batch ONNX inference (GPU hiệu quả cao khi batch > 1)
        if self.using_onnx and len(persons) > 1:
            try:
                nearby_names = [o.class_name for o in (nearby_objects or [])]
                return self._classify_onnx_batch(persons, frame, nearby_names)
            except Exception as e:
                logger.debug(f"[RoleClassifier] batch inference lỗi, fallback per-person: {e}")

        # Fallback: per-person
        for obj in persons:
            self.classify(frame, obj, nearby_objects)
        return objects

    def _classify_onnx_batch(
        self,
        persons: list[TrackedObject],
        frame: np.ndarray,
        nearby_names: list[str],
    ) -> list[TrackedObject]:
        """
        Chạy ONNX batch inference cho nhiều person crops cùng lúc.
        Tất cả crops được preprocess và stack thành 1 batch tensor (N,C,H,W).
        """
        crops_norm = []
        valid_idx  = []
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)

        for i, obj in enumerate(persons):
            crop = self._extract_crop(frame, obj)
            if crop is None or crop.size == 0:
                continue
            rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (ROLENET_INPUT_W, ROLENET_INPUT_H))
            arr    = resized.astype(np.float32) / 255.0
            arr    = (arr - mean) / std          # H×W×C
            arr    = arr.transpose(2, 0, 1)      # C×H×W
            crops_norm.append(arr)
            valid_idx.append(i)

        if not crops_norm:
            return persons

        # Stack thành batch (N,C,H,W)
        batch = np.stack(crops_norm, axis=0)  # (N,3,256,128)

        # Single GPU call cho cả batch
        logits_batch = RoleClassifier._onnx_sess.run(
            ["logits"], {"input": batch}
        )[0]  # shape: (N, 16)

        # Parse kết quả cho từng person
        for batch_i, obj_i in enumerate(valid_idx):
            obj    = persons[obj_i]
            logits = logits_batch[batch_i]
            e      = np.exp(logits - logits.max())
            probs  = e / e.sum()

            top_idx  = int(probs.argmax())
            top_conf = float(probs[top_idx])
            role_str = self.ROLENET_CLASSES[top_idx]

            thr = 0.28
            if top_conf < thr:
                role_str, top_conf = "normal", max(0.5, top_conf)

            role_def   = self._roles.get(role_str, {})
            acc_bonus  = self._score_accessories(nearby_names, role_def.get("accessories", []))
            final_conf = min(1.0, top_conf + 0.05 * acc_bonus)

            top3_idx  = probs.argsort()[::-1][:3]
            ver_tag   = f"ONNX-Batch({'INT8' if self.model_version == 'onnx_int8' else 'FP32'})"
            top3_info = [
                f"{self.ROLENET_CLASSES[i]}({probs[i]*100:.0f}%)"
                for i in top3_idx
            ]

            obj.role            = SocialRole(role_str)
            obj.role_confidence = round(final_conf, 3)
            obj.role_evidence   = RoleEvidence(
                color_match       = f"{ver_tag}: {' | '.join(top3_info)}",
                color_ratio       = round(top_conf, 3),
                region            = "full_body",
                accessories_found = [n for n in nearby_names
                                     if n in role_def.get("accessories", [])],
            )

        return persons

    def classify(
        self,
        frame: np.ndarray,
        obj: TrackedObject,
        nearby_objects: list[TrackedObject] | None = None,
    ) -> TrackedObject:
        """
        Phân loại vai trò cho 1 TrackedObject (phải là PERSON).

        Args:
            frame         : BGR frame gốc
            obj           : TrackedObject cần classify
            nearby_objects: Các object khác trong frame (để check accessories)

        Returns:
            obj đã được cập nhật role, role_confidence, role_evidence
        """
        if obj.category != ObjectCategory.PERSON:
            return obj

        crop = self._extract_crop(frame, obj)
        if crop is None or crop.size == 0:
            obj.role            = SocialRole.NORMAL
            obj.role_confidence = 0.5
            return obj

        nearby_names = [o.class_name for o in (nearby_objects or [])]

        if self.using_onnx:
            return self._classify_onnx(obj, crop, nearby_names)
        elif self.using_ml_model:
            return self._classify_ml(obj, crop, nearby_names)
        else:
            return self._classify_rules(obj, crop, nearby_names)

    # ----------------------------------------------------------
    # ONNX Runtime Classification (Ưu tiên cao nhất)
    # ----------------------------------------------------------

    def _classify_onnx(
        self,
        obj: TrackedObject,
        crop_bgr: np.ndarray,
        nearby_names: list[str],
    ) -> TrackedObject:
        """Inference nhanh qua ONNX Runtime. Fallback về PyTorch nếu lỗi."""
        try:
            import numpy as np

            # Tiền xử lý: resize → normalize → NCHW
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            resized  = cv2.resize(crop_rgb, (ROLENET_INPUT_W, ROLENET_INPUT_H))
            arr      = resized.astype(np.float32) / 255.0

            mean = np.array(IMAGENET_MEAN, dtype=np.float32)
            std  = np.array(IMAGENET_STD,  dtype=np.float32)
            arr  = (arr - mean) / std                    # H×W×C
            arr  = arr.transpose(2, 0, 1)[np.newaxis]    # 1×C×H×W

            # Inference
            logits = RoleClassifier._onnx_sess.run(
                ["logits"], {"input": arr}
            )[0][0]  # shape: (16,)

            # Softmax thủ công (ONNX trả logits)
            e     = np.exp(logits - logits.max())
            probs = e / e.sum()

            top_idx  = int(probs.argmax())
            top_conf = float(probs[top_idx])
            role_str = self.ROLENET_CLASSES[top_idx]

            # Confidence thấp → NORMAL
            thr = 0.28
            if top_conf < thr:
                role_str, top_conf = "normal", max(0.5, top_conf)

            # Bonus accessories
            role_def  = self._roles.get(role_str, {})
            acc_bonus = self._score_accessories(nearby_names, role_def.get("accessories", []))
            final_conf = min(1.0, top_conf + 0.05 * acc_bonus)

            # Top-3 cho diagnostics
            top3_idx  = probs.argsort()[::-1][:3]
            ver_tag   = f"ONNX({'INT8' if self.model_version == 'onnx_int8' else 'FP32'})"
            top3_info = [
                f"{self.ROLENET_CLASSES[i]}({probs[i]*100:.0f}%)"
                for i in top3_idx
            ]

            obj.role            = SocialRole(role_str)
            obj.role_confidence = round(final_conf, 3)
            obj.role_evidence   = RoleEvidence(
                color_match       = f"{ver_tag}: {' | '.join(top3_info)}",
                color_ratio       = round(top_conf, 3),
                region            = "full_body",
                accessories_found = [n for n in nearby_names
                                     if n in role_def.get("accessories", [])],
            )
            return obj

        except Exception as exc:
            logger.warning(
                f"[RoleClassifier] ONNX inference lỗi: {exc}. "
                "Fallback sang PyTorch."
            )
            # Thử PyTorch nếu có
            if RoleClassifier._net is not None:
                return self._classify_ml(obj, crop_bgr, nearby_names)
            return self._classify_rules(obj, crop_bgr, nearby_names)

    # ----------------------------------------------------------
    # ML Classification (RoleNet PyTorch)
    # ----------------------------------------------------------

    def _classify_ml(
        self,
        obj: TrackedObject,
        crop_bgr: np.ndarray,
        nearby_names: list[str],
    ) -> TrackedObject:
        """Dùng RoleNet model (V1 hoặc V2) để phân loại. V2 có TTA."""
        try:
            import torch

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            x = RoleClassifier._preprocess(crop_rgb).unsqueeze(0).to(RoleClassifier._device)

            with torch.no_grad():
                if self.use_tta:
                    # TTA: gốc + flip ngang → lấy trung bình
                    logits_orig = RoleClassifier._net(x)
                    logits_flip = RoleClassifier._net(x.flip(-1))
                    probs = (
                        torch.softmax(logits_orig, dim=1) +
                        torch.softmax(logits_flip, dim=1)
                    )[0].cpu().numpy() / 2.0
                else:
                    logits = RoleClassifier._net(x)
                    probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            top_idx  = int(probs.argmax())
            top_conf = float(probs[top_idx])
            role_str = self.ROLENET_CLASSES[top_idx]

            # Confidence thấp → NORMAL
            thr = 0.28 if self.model_version == "v3" else (0.30 if self.use_tta else 0.35)
            if top_conf < thr:
                role_str, top_conf = "normal", max(0.5, top_conf)

            # Bonus accessories
            role_def  = self._roles.get(role_str, {})
            acc_bonus = self._score_accessories(nearby_names, role_def.get("accessories", []))
            final_conf = min(1.0, top_conf + 0.05 * acc_bonus)

            # Top-3
            top3_idx  = probs.argsort()[::-1][:3]
            ver_tag   = f"RoleNet{self.model_version.upper() if self.model_version else ''}"
            top3_info = [
                f"{self.ROLENET_CLASSES[i]}({probs[i]*100:.0f}%)"
                for i in top3_idx
            ]

            obj.role            = SocialRole(role_str)
            obj.role_confidence = round(final_conf, 3)
            obj.role_evidence   = RoleEvidence(
                color_match       = f"{ver_tag}: {' | '.join(top3_info)}",
                color_ratio       = round(top_conf, 3),
                region            = "full_body",
                accessories_found = [n for n in nearby_names
                                     if n in role_def.get("accessories", [])],
            )
            return obj

        except Exception as e:
            logger.error(f"[RoleClassifier] ML inference lỗi: {e}. Fallback sang rules.")
            return self._classify_rules(obj, crop_bgr, nearby_names)

    # ----------------------------------------------------------
    # Rule-based Classification (Fallback HSV)
    # ----------------------------------------------------------

    def _classify_rules(
        self,
        obj: TrackedObject,
        crop_bgr: np.ndarray,
        nearby_names: list[str],
    ) -> TrackedObject:
        """Rule-based HSV + accessory scoring (fallback)."""
        best_role     = SocialRole.NORMAL
        best_score    = 0.0
        best_evidence: Optional[RoleEvidence] = None

        for role_name, role_def in self._roles.items():
            if role_name in ("normal", "unknown"):
                continue

            color_score, color_ev = self._score_color(crop_bgr, role_def["color_rules"])
            acc_score             = self._score_accessories(nearby_names, role_def["accessories"])

            total  = W_COLOR * color_score + W_ACCESSORY * acc_score
            thresh = role_def.get("threshold", self._global_thr)

            if total >= thresh and total > best_score:
                best_score = total
                best_role  = SocialRole(role_name)
                best_evidence = RoleEvidence(
                    color_match       = color_ev.get("color_name", ""),
                    color_ratio       = color_ev.get("ratio", 0.0),
                    region            = "torso",
                    accessories_found = [n for n in nearby_names
                                         if n in role_def["accessories"]],
                )

        if best_score < self._global_thr:
            best_role  = SocialRole.NORMAL
            best_score = max(0.5, best_score)

        obj.role            = best_role
        obj.role_confidence = min(1.0, best_score)
        obj.role_evidence   = best_evidence
        return obj

    # ----------------------------------------------------------
    # Internal Helpers
    # ----------------------------------------------------------

    def _extract_crop(self, frame: np.ndarray, obj: TrackedObject) -> Optional[np.ndarray]:
        """Cắt vùng person từ frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (
            max(0, int(obj.bbox.x1)),
            max(0, int(obj.bbox.y1)),
            min(w, int(obj.bbox.x2)),
            min(h, int(obj.bbox.y2)),
        )
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if self.using_ml_model:
            # RoleNet xử lý resize nội bộ; trả về crop gốc
            return crop
        else:
            # Rule-based cần resize về crop_size (w, h)
            return cv2.resize(crop, self._crop_size)

    def _score_color(
        self,
        crop: np.ndarray,
        color_rules: list[dict],
    ) -> tuple[float, dict]:
        """Tính điểm màu sắc từ HSV torso region."""
        if not color_rules:
            return 0.0, {}

        h, w    = crop.shape[:2]
        y_start = int(h * 0.20)
        y_end   = int(h * 0.75)
        torso   = crop[y_start:y_end, :]

        if torso.size == 0:
            return 0.0, {}

        hsv      = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        total_px = torso.shape[0] * torso.shape[1]

        best_ratio = 0.0
        best_name  = ""

        for rule in color_rules:
            color_name = rule["color"]
            lo, hi     = self._hsv_ranges.get(color_name, (None, None))
            if lo is None:
                continue

            mask  = cv2.inRange(hsv, lo, hi)
            ratio = mask.sum() / 255 / total_px

            if ratio > best_ratio:
                best_ratio = ratio
                best_name  = color_name

        score = 0.0
        if best_ratio > 0:
            ref_ratio = color_rules[0].get("min_ratio", 0.20)
            score     = min(1.0, best_ratio / (ref_ratio * 1.5))

        return score, {"color_name": best_name, "ratio": best_ratio}

    @staticmethod
    def _score_accessories(nearby_names: list[str], required: list[str]) -> float:
        """Điểm phụ kiện: tỷ lệ required accessories thực sự xuất hiện gần đây."""
        if not required:
            return 0.0
        found = sum(1 for acc in required if acc in nearby_names)
        return found / len(required)

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------

    def get_status(self) -> dict:
        """Trả về trạng thái hiện tại của classifier."""
        if self.using_onnx:
            mode = self.model_version or "onnx_fp32"
        elif self.using_ml_model:
            mode = f"ml_rolenet_{self.model_version}"
        else:
            mode = "rule_based"

        if self.model_version == "v3":
            pt_name = MODEL_V3_PT_NAME
        elif self.model_version == "v2":
            pt_name = MODEL_V2_PT_NAME
        else:
            pt_name = MODEL_PT_NAME

        return {
            "mode"            : mode,
            "model_version"   : self.model_version or "none",
            "using_onnx"      : self.using_onnx,
            "onnx_fp32_exists": (MODELS_DIR / MODEL_ONNX_NAME).exists(),
            "onnx_int8_exists": (MODELS_DIR / MODEL_ONNX_QUANT).exists(),
            "model_path"      : str(MODELS_DIR / pt_name),
            "model_exists"    : (MODELS_DIR / pt_name).exists(),
            "device"          : RoleClassifier._device or ("onnx" if self.using_onnx else "N/A"),
            "tta_enabled"     : self.use_tta,
            "classes"         : self.ROLENET_CLASSES if self.using_ml_model else [],
            "v3_available"    : (MODELS_DIR / MODEL_V3_PT_NAME).exists(),
            "v2_available"    : (MODELS_DIR / MODEL_V2_PT_NAME).exists(),
            "v1_available"    : (MODELS_DIR / MODEL_PT_NAME).exists(),
        }

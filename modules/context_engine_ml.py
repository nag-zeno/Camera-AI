"""
context_engine_ml.py — ML-enhanced Context Reasoning Engine

Cải tiến so với rule engine:
  - Dùng XGBoost model thay vì if-else cứng
  - Học được pattern phức tạp từ data
  - Tự động fallback về RuleEngine nếu model chưa sẵn sàng
  - Giải thích được quyết định qua feature importance

Tích hợp:
  Thay 'from modules.context_engine import ContextEngine' bằng
       'from modules.context_engine_ml import ContextEngineML as ContextEngine'
  trong pipeline.py
"""
import pickle
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from config import REASONING_CONFIG
from models import (
    TrackedObject, AlertLevel, AlertEvent,
    SocialRole, ZoneType, ZoneStatus, IdentityStatus, ObjectCategory, ActionLabel,
)
from modules.context_engine import ContextEngine as RuleEngine, generate_vietnamese_reason

logger = logging.getLogger(__name__)

# ============================================================
# Mapping enum → int (phải khớp với generate_context_data.py)
# ============================================================
ROLE_IDS      = {r: i for i, r in enumerate(SocialRole)}
IDENTITY_IDS  = {s: i for i, s in enumerate(IdentityStatus)}
ZONE_TYPE_IDS = {None: 0, ZoneType.ALLOWED: 1, ZoneType.RESTRICTED: 2}
ZONE_STATUS_IDS = {s: i for i, s in enumerate(ZoneStatus)}
DIRECTION_IDS = {
    "stationary": 0, "moving_in": 1, "moving_out": 2, "moving": 3
}
CATEGORY_IDS  = {c: i for i, c in enumerate(ObjectCategory)}
ACTION_IDS    = {a: i for i, a in enumerate(ActionLabel)}

FEATURE_NAMES = [
    "role_id", "identity_id", "zone_type_id", "zone_status_id",
    "loitering", "time_in_zone", "visit_count", "direction_id",
    "hour", "role_confidence", "category_id", "frames_tracked",
    "is_night", "is_business_hour",
    "action_id", "action_confidence",  # <- NEW: ActionNet features
]

ALERT_EMOJI = {
    "ignore": "⬜", "normal": "✅", "watch": "👁",
    "warning": "⚠️",  "alert": "🔴", "critical": "🚨",
}


def _obj_to_features(obj: TrackedObject, hour: int) -> np.ndarray:
    """Chuyển TrackedObject → feature vector 16 chiều (bao gồm action)."""
    is_night    = int(hour < 6 or hour >= 22)
    is_business = int(9 <= hour < 18)

    return np.array([
        ROLE_IDS.get(obj.role, 0),
        IDENTITY_IDS.get(obj.identity, 1),
        ZONE_TYPE_IDS.get(obj.zone_type, 0),
        ZONE_STATUS_IDS.get(obj.zone_status, 0),
        int(obj.loitering),
        obj.time_in_zone,
        obj.visit_count,
        DIRECTION_IDS.get(obj.direction, 0),
        hour,
        obj.role_confidence,
        CATEGORY_IDS.get(obj.category, 0),
        obj.frames_tracked,
        is_night,
        is_business,
        ACTION_IDS.get(obj.action, 0),         # <- NEW: action class index
        obj.action_confidence,                  # <- NEW: action confidence
    ], dtype=np.float32)


# ============================================================
# ML Context Engine
# ============================================================

class ContextEngineML:
    """
    ML-based Context Reasoning Engine.

    Hành vi:
      1. Tìm model tại models/context_net.pkl
      2. Nếu có → dùng XGBoost để predict alert level
      3. Nếu không có → tự động dùng RuleEngine (rule-based)
      4. Log rõ ràng để người dùng biết đang dùng chế độ nào
    """

    DEFAULT_MODEL_PATH = Path("models/context_net.pkl")

    def __init__(self, model_path: Path | None = None):
        self._cfg      = REASONING_CONFIG
        self._cooldown = self._cfg.get("alert_cooldown", 30)
        self._alert_times: dict[str, float] = {}

        # Fallback engine (luôn có sẵn)
        self._rule_engine = RuleEngine()

        # ML state
        self._model        = None
        self._label_encoder = None
        self._feature_names = FEATURE_NAMES
        self._ml_ready     = False

        # SHAP explainer — lazy-init khi lần đầu gọi get_shap_explanation()
        self._shap_explainer = None

        # Load model
        path = model_path or self.DEFAULT_MODEL_PATH
        self._try_load_model(path)

    # ----------------------------------------------------------
    # Public API — giống giao diện ContextEngine cũ
    # ----------------------------------------------------------

    def evaluate(
        self, obj: TrackedObject
    ) -> tuple[TrackedObject, Optional[AlertEvent]]:
        """
        Đánh giá 1 TrackedObject.
        Dùng ML model nếu có, RuleEngine nếu không.
        """
        if self._ml_ready:
            return self._evaluate_ml(obj)
        return self._rule_engine.evaluate(obj)

    def add_rule(self, rule):
        """Forward tới rule engine (tương thích ngược)."""
        self._rule_engine.add_rule(rule)

    def update_zones(self, *args, **kwargs):
        """Forward (nếu có)."""
        if hasattr(self._rule_engine, "update_zones"):
            self._rule_engine.update_zones(*args, **kwargs)

    @property
    def using_ml(self) -> bool:
        """True nếu đang dùng ML model, False nếu dùng rule engine."""
        return self._ml_ready

    def status(self) -> dict:
        """Trả về trạng thái engine."""
        return {
            "mode"    : "ML (XGBoost)" if self._ml_ready else "Rule Engine",
            "classes" : list(self._label_encoder.classes_) if self._ml_ready else [],
            "features": self._feature_names,
        }

    def get_shap_explanation(self, obj: TrackedObject) -> Optional[dict]:
        """
        Tính SHAP values cho 1 TrackedObject.
        Trả về dict chứa SHAP values, feature values, predicted class và probabilities.
        Trả về None nếu ML chưa sẵn sàng hoặc xảy ra lỗi.

        Lazy-init TreeExplainer khi lần đầu gọi để tránh overhead khi startup.
        """
        if not self._ml_ready:
            return None

        try:
            # Lazy init SHAP explainer
            if self._shap_explainer is None:
                import shap
                self._shap_explainer = shap.TreeExplainer(self._model)
                logger.info("[ContextEngineML] SHAP TreeExplainer khởi tạo thành công.")

            hour     = datetime.now().hour
            features = _obj_to_features(obj, hour).reshape(1, -1)

            # Backward-compat: trim features nếu model cũ hơn
            model_nf = getattr(self, "_model_n_features", features.shape[1])
            if features.shape[1] > model_nf:
                features = features[:, :model_nf]

            # Predict
            pred_idx   = int(self._model.predict(features)[0])
            probas     = self._model.predict_proba(features)[0]
            pred_class = self._label_encoder.classes_[pred_idx]

            # SHAP values — shap_values có shape (n_classes, n_samples, n_features)
            shap_values = self._shap_explainer.shap_values(features)

            # Lấy SHAP cho predicted class
            if isinstance(shap_values, list):
                # Dạng cũ: list[n_classes] of (n_samples, n_features)
                sv = shap_values[pred_idx][0].tolist()
                base_val = float(self._shap_explainer.expected_value[pred_idx])
            else:
                # Dạng mới: (n_samples, n_features, n_classes)
                sv = shap_values[0, :, pred_idx].tolist()
                ev = self._shap_explainer.expected_value
                base_val = float(ev[pred_idx] if hasattr(ev, '__len__') else ev)

            fnames  = list(self._feature_names[:model_nf])
            fvalues = features[0].tolist()

            return {
                "ml_ready"       : True,
                "predicted_class": pred_class,
                "probabilities"  : {
                    c: round(float(p), 4)
                    for c, p in zip(self._label_encoder.classes_, probas)
                },
                "base_value"     : round(base_val, 4),
                "shap_values"    : [round(v, 4) for v in sv],
                "feature_names"  : fnames,
                "feature_values" : [round(v, 4) for v in fvalues],
            }

        except Exception as exc:
            logger.warning(f"[ContextEngineML] SHAP explanation lỗi: {exc}")
            return None

    def get_feature_importance(self) -> Optional[dict]:
        """
        Trả về global feature importance từ XGBoost model.feature_importances_.
        Dùng để vẽ biểu đồ global importance khi chưa chọn object cụ thể.
        Trả về None nếu ML chưa sẵn sàng.
        """
        if not self._ml_ready:
            return None

        try:
            importances = self._model.feature_importances_.tolist()
            model_nf    = getattr(self, "_model_n_features", len(importances))
            fnames      = list(self._feature_names[:model_nf])

            # Sort by importance descending
            pairs = sorted(
                zip(fnames, importances),
                key=lambda x: x[1],
                reverse=True,
            )

            return {
                "ml_ready"       : True,
                "feature_names"  : [p[0] for p in pairs],
                "importances"    : [round(p[1], 4) for p in pairs],
                "classes"        : list(self._label_encoder.classes_),
            }

        except Exception as exc:
            logger.warning(f"[ContextEngineML] Feature importance lỗi: {exc}")
            return None

    # ----------------------------------------------------------
    # Internal — ML evaluation path
    # ----------------------------------------------------------

    def _evaluate_ml(
        self, obj: TrackedObject
    ) -> tuple[TrackedObject, Optional[AlertEvent]]:
        """Dánh giá bằng ML model (hỗ trợ cả model cũ  14-feat và mới 16-feat)."""
        hour     = datetime.now().hour
        features = _obj_to_features(obj, hour).reshape(1, -1)

        # Backward-compat: trim features nếu model cũ hơn
        model_nf = getattr(self, "_model_n_features", features.shape[1])
        if features.shape[1] > model_nf:
            features = features[:, :model_nf]

        try:
            pred_idx    = int(self._model.predict(features)[0])
            probas      = self._model.predict_proba(features)[0]
            label_str   = self._label_encoder.classes_[pred_idx]
            confidence  = float(probas[pred_idx])

            obj.alert_level  = AlertLevel(label_str)
            obj.alert_reason = self._format_reason(obj, label_str, confidence)
            obj.rule_name    = "context_net_ml"

        except Exception as exc:
            logger.warning(f"ML predict lỗi ({exc}) — dùng rule engine cho object #{obj.track_id}")
            return self._rule_engine.evaluate(obj)

        # Phát AlertEvent (có cooldown)
        alert_event = None
        if obj.alert_level.value in ("warning", "alert", "critical"):
            alert_event = self._maybe_emit(obj)

        return obj, alert_event

    @staticmethod
    def _format_reason(obj: TrackedObject, level: str, conf: float) -> str:
        """
        Tạo chuỗi lý do bằng ngôn ngữ tự nhiên tiếng Việt.

        Thứ tự ưu tiên:
          1. Gemini NLG Engine (sinh câu tự nhiên, thân thiện)
          2. generate_vietnamese_reason() (template tĩnh — fallback)
        """
        # Chuẩn bị dữ liệu ngữ cảnh cho NLG
        ctx_data = {
            "role"        : obj.role.value if obj.role else "unknown",
            "action"      : obj.action.value if obj.action else "unknown",
            "zone"        : obj.zone_name,
            "zone_type"   : obj.zone_type.value if obj.zone_type else "allowed",
            "alert_level" : level,
            "time_in_zone": obj.time_in_zone,
            "loitering"   : obj.loitering,
        }

        try:
            from modules.nlg_engine import get_nlg_engine
            nlg = get_nlg_engine()
            if nlg.is_available:
                result = nlg.generate(
                    ctx_data,
                    fallback_fn=lambda: generate_vietnamese_reason(obj)
                )
                return result
        except Exception as exc:
            logger.debug(f"[ContextEngineML] NLG fallback ({exc})")

        # Fallback: template tiếng Việt nội bộ
        return generate_vietnamese_reason(obj)

    def _maybe_emit(self, obj: TrackedObject) -> Optional[AlertEvent]:
        """Phát AlertEvent nếu chưa hết cooldown."""
        key = f"ml:{obj.rule_name}:{obj.track_id}"
        now = time.time()
        if now - self._alert_times.get(key, 0.0) < self._cooldown:
            return None
        self._alert_times[key] = now
        event = AlertEvent.create(obj)
        logger.warning(
            f"[{event.level.value.upper()}-ML] {obj.alert_reason} "
            f"(id={event.event_id})"
        )
        return event

    # ----------------------------------------------------------
    # Load model
    # ----------------------------------------------------------

    def _try_load_model(self, path: Path):
        """Load XGBoost model. Không raise exception — chỉ log và fallback."""
        if not path.exists():
            logger.info(
                f"ContextEngineML: Chưa tìm thấy model tại '{path}'. "
                "Đang dùng Rule Engine.\n"
                "  → Để train: python scripts/generate_context_data.py && "
                "python scripts/train_context_model.py"
            )
            return

        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)

            self._model         = bundle["model"]
            self._label_encoder = bundle["label_encoder"]
            if "feature_names" in bundle:
                self._feature_names = bundle["feature_names"]

            # Kiểm tra số features của model đã train
            self._model_n_features = (
                self._model.n_features_in_
                if hasattr(self._model, "n_features_in_")
                else len(self._feature_names)
            )

            self._ml_ready = True
            acc = bundle.get("accuracy", "?")

            new_features = len(FEATURE_NAMES)
            if self._model_n_features < new_features:
                logger.warning(
                    f"ContextEngineML: Model train với {self._model_n_features} features "
                    f"(hiện tại có {new_features}). "
                    f"Thiếu action features — một số tính năng mới bị bỏ qua.\n"
                    f"  → Nên retrain: python scripts/generate_context_data.py && python scripts/train_context_model.py"
                )
            logger.info(
                f"✅ ContextEngineML: XGBoost model tải thành công từ '{path}' "
                f"| features={self._model_n_features} | classes={list(self._label_encoder.classes_)} "
                f"| accuracy={acc if acc == '?' else f'{acc:.1%}'}"
            )

        except Exception as exc:
            logger.error(
                f"Lỗi tải ContextNet model ({exc}). "
                "Đang dùng Rule Engine làm fallback."
            )
            self._ml_ready = False

"""
Smart AI Security Camera System
================================
Context-aware camera security system với 16 vai trò xã hội.

Cấu trúc module:
  modules/
    video_processor.py    — Tầng 1: Video Input
    object_detector.py    — Tầng 2: Perception
    object_tracker.py     — Tầng 2: Perception
    role_classifier.py    — Tầng 3: Understanding
    identity_manager.py   — Tầng 3: Understanding
    zone_detector.py      — Tầng 3: Understanding
    behavior_analyzer.py  — Tầng 3: Understanding
    action_recognizer.py  — Tầng 3: Understanding (Action)
    context_engine.py     — Tầng 4: Reasoning (Rule-based)
    context_engine_ml.py  — Tầng 4: Reasoning (ML/XGBoost)
    nlg_engine.py         — Tầng 5: Output (NLG - Gemini AI)
    visualizer.py         — Tầng 5: Output
    event_logger.py       — Tầng 5: Output
    telegram_notifier.py  — Tầng 5: Output (Push Notification)
    alert_recorder.py     — Tầng 5: Output (Video Recording)
"""

from .video_processor    import VideoProcessor
from .object_detector    import ObjectDetector
from .object_tracker     import ObjectTracker
from .role_classifier    import RoleClassifier
from .identity_manager   import IdentityManager
from .zone_detector      import ZoneDetector
from .behavior_analyzer  import BehaviorAnalyzer
from .action_recognizer  import ActionRecognizer
from .context_engine     import ContextEngine, ContextRule
from .nlg_engine         import NLGEngine, get_nlg_engine
from .visualizer         import Visualizer
from .event_logger       import EventLogger

__all__ = [
    "VideoProcessor",
    "ObjectDetector",
    "ObjectTracker",
    "RoleClassifier",
    "IdentityManager",
    "ZoneDetector",
    "BehaviorAnalyzer",
    "ActionRecognizer",
    "ContextEngine",
    "ContextRule",
    "NLGEngine",
    "get_nlg_engine",
    "Visualizer",
    "EventLogger",
]

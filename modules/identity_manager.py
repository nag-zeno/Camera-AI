"""
identity_manager.py — Tầng 3: Understanding (Identity Awareness)

Nhận vào: frame + TrackedObject (persons)
Trả ra  : TrackedObject với identity = KNOWN / UNKNOWN

Cách hoạt động:
  - Tính color histogram (32 bins * 3 channels) làm "chữ ký" ngoại hình
  - So sánh với danh sách người quen đã đăng ký
  - Không lưu ảnh/tên thật → bảo vệ quyền riêng tư

NOTE: Đây là simplified appearance-based matching, không phải face recognition.
"""
import cv2
import numpy as np
import logging
import time
from pathlib import Path

from config import IDENTITY_CONFIG
from models import TrackedObject, IdentityStatus, ObjectCategory

logger = logging.getLogger(__name__)


class IdentityManager:
    """
    Nhận diện "người quen" bằng color histogram similarity.
    Không yêu cầu face detection; chỉ so sánh ngoại hình tổng thể.
    """

    def __init__(self, known_faces_dir: str | Path | None = None):
        self._cfg         = IDENTITY_CONFIG
        self._threshold   = self._cfg["similarity_threshold"]
        self._recheck_n   = self._cfg["recheck_interval"]
        self._max_known   = self._cfg["max_known_persons"]

        # person_id (str) → list of histograms (np.ndarray)
        self._known: dict[str, list[np.ndarray]] = {}

        # track_id → (IdentityStatus, last_check_frame)
        self._cache: dict[int, tuple[IdentityStatus, int]] = {}

        self._frame_counter = 0

        if known_faces_dir:
            self._load_known_from_dir(known_faces_dir)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def identify(
        self,
        frame: np.ndarray,
        obj: TrackedObject,
    ) -> TrackedObject:
        """
        Xác định identity của 1 person.

        Args:
            frame: BGR frame gốc
            obj  : TrackedObject (phải là PERSON)

        Returns:
            obj với identity đã cập nhật
        """
        self._frame_counter += 1

        if obj.category != ObjectCategory.PERSON:
            return obj

        # Dùng cache nếu chưa đến lúc re-check
        cached = self._cache.get(obj.track_id)
        if cached:
            status, last_check = cached
            if (self._frame_counter - last_check) < self._recheck_n:
                obj.identity = status
                return obj

        # Tính histogram của person
        hist = self._compute_histogram(frame, obj)
        if hist is None:
            obj.identity = IdentityStatus.UNKNOWN
            return obj

        # So sánh với danh sách người quen
        status = IdentityStatus.UNKNOWN
        if self._known:
            max_sim = max(
                max(self._similarity(hist, ref) for ref in refs)
                for refs in self._known.values()
            )
            if max_sim >= self._threshold:
                status = IdentityStatus.KNOWN

        obj.identity = status
        self._cache[obj.track_id] = (status, self._frame_counter)
        return obj

    def register_person(self, person_id: str, frame: np.ndarray, obj: TrackedObject):
        """
        Đăng ký người quen từ frame hiện tại.

        Args:
            person_id: ID hoặc nhãn (không phải tên thật nếu cần ẩn danh)
            frame    : BGR frame
            obj      : TrackedObject của người cần đăng ký
        """
        if len(self._known) >= self._max_known:
            logger.warning("Max known persons reached, cannot register more.")
            return

        hist = self._compute_histogram(frame, obj)
        if hist is None:
            return

        if person_id not in self._known:
            self._known[person_id] = []
        self._known[person_id].append(hist)
        logger.info(f"Registered person '{person_id}' ({len(self._known[person_id])} samples)")

    def forget_track(self, track_id: int):
        """Xóa cache khi track mất."""
        self._cache.pop(track_id, None)

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    def _compute_histogram(
        self,
        frame: np.ndarray,
        obj: TrackedObject,
    ) -> np.ndarray | None:
        """Tính normalized color histogram 32*3 bins từ crop person."""
        h, w = frame.shape[:2]
        x1 = max(0, int(obj.bbox.x1))
        y1 = max(0, int(obj.bbox.y1))
        x2 = min(w, int(obj.bbox.x2))
        y2 = min(h, int(obj.bbox.y2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Resize để chuẩn hóa kích thước
        crop = cv2.resize(crop, (64, 128))
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        hist_parts = []
        for ch in range(3):
            h_i = cv2.calcHist([hsv], [ch], None, [32], [0, 256])
            cv2.normalize(h_i, h_i, 0, 1, cv2.NORM_MINMAX)
            hist_parts.append(h_i.flatten())

        return np.concatenate(hist_parts)

    @staticmethod
    def _similarity(h1: np.ndarray, h2: np.ndarray) -> float:
        """Cosine similarity giữa 2 histogram."""
        dot   = np.dot(h1, h2)
        norms = np.linalg.norm(h1) * np.linalg.norm(h2)
        if norms == 0:
            return 0.0
        return float(dot / norms)

    def _load_known_from_dir(self, directory: str | Path):
        """
        Tùy chọn: load ảnh từ thư mục để tạo histograms ban đầu.
        Cấu trúc thư mục: known_faces/<person_id>/<img1.jpg> ...
        """
        directory = Path(directory)
        if not directory.exists():
            return

        count = 0
        for person_dir in directory.iterdir():
            if not person_dir.is_dir():
                continue
            person_id = person_dir.name
            for img_path in person_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                # Tạo TrackedObject giả để tính histogram
                from models import BoundingBox
                h_im, w_im = img.shape[:2]
                fake_obj = TrackedObject(
                    track_id  = -1,
                    bbox      = BoundingBox(0, 0, w_im, h_im),
                    class_name= "person",
                    category  = ObjectCategory.PERSON,
                    confidence= 1.0,
                )
                hist = self._compute_histogram(img, fake_obj)
                if hist is not None:
                    if person_id not in self._known:
                        self._known[person_id] = []
                    self._known[person_id].append(hist)
                    count += 1

        logger.info(f"Loaded {count} known person histograms from '{directory}'")

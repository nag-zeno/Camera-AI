"""
object_tracker.py — Tầng 2: Perception (Tracking)

Nhận vào: List[Detection] + frame (BGR numpy array)
Trả ra  : List[TrackedObject] với track_id ổn định

Sử dụng IoU tracker nâng cấp v2 với:
  - Velocity EMA: làm mượt velocity qua nhiều frame, loại bỏ box jitter
  - Appearance embedding: color histogram HSV 96-dim để nhận diện lại người
  - Two-stage matching: vòng 1 IoU cao, vòng 2 IoU thấp + appearance
  - Re-ID buffer 30s: khôi phục track đã mất khi người quay lại frame
  - Hungarian algorithm: tối ưu global assignment (scipy)
  - Min-hits guard: track phải confirmed đủ N frame (tránh ghost track)

So sánh v1 → v2:
  v1: IoU-only matching → ID switch khi đứng yên rồi di chuyển
  v2: IoU + appearance → giữ ID ngay cả khi IoU thấp (che khuất, đứng yên)
"""
import time
import logging
from typing import Optional

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from config import TRACKING_CONFIG
from models import TrackedObject, Detection, BoundingBox, ObjectCategory

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================
_VELOCITY_EMA_ALPHA = 0.4    # EMA factor: 0=không update, 1=update tức thì
_APPEAR_BINS        = 32     # Số bins histogram mỗi channel (32 × 3ch = 96-dim)
_APPEAR_ALPHA       = 0.7    # Weight IoU trong combined cost (1-alpha → appearance)
_REID_TTL           = 30.0   # Giữ track trong Re-ID buffer 30 giây
_REID_SIM_THRESH    = 0.65   # Bhattacharyya similarity để coi là "cùng người"
_STAGE1_IOU_THRESH  = 0.25   # Ngưỡng IoU stage 1 (match chắc chắn)
_STAGE2_IOU_THRESH  = 0.08   # Ngưỡng IoU stage 2 (bị che khuất, chỉ dùng appearance)
_STAGE2_APP_THRESH  = 0.70   # Ngưỡng similarity appearance để chấp nhận ở stage 2


# ============================================================
# Appearance helpers
# ============================================================

def _compute_appearance(frame: np.ndarray, bbox: "BoundingBox") -> Optional[np.ndarray]:
    """
    Tính color histogram HSV 32 bins × 3 channels = 96-dim.

    Cắt phần torso (giữa: 20%-80% chiều cao) để giảm ảnh hưởng của nền.
    Trả về vector float32 đã chuẩn hóa L1, hoặc None nếu crop trống.
    """
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox.to_xyxy()]
        h_frame, w_frame = frame.shape[:2]

        # Clamp về frame
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_frame, x2); y2 = min(h_frame, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Lấy phần torso (20%-80% chiều cao)
        h = y2 - y1
        ty1 = y1 + int(h * 0.20)
        ty2 = y1 + int(h * 0.80)
        if ty2 <= ty1:
            ty1 = y1; ty2 = y2

        crop = frame[ty1:ty2, x1:x2]
        if crop.size == 0:
            return None

        # Resize nhỏ để tăng tốc (64×64 là đủ cho histogram)
        crop_small = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(crop_small, cv2.COLOR_BGR2HSV)

        # Histogram 32 bins mỗi channel
        hist_h = cv2.calcHist([hsv], [0], None, [_APPEAR_BINS], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [_APPEAR_BINS], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [_APPEAR_BINS], [0, 256])

        hist = np.concatenate([hist_h, hist_s, hist_v]).flatten().astype(np.float32)

        # Chuẩn hóa L1
        total = hist.sum()
        if total > 0:
            hist /= total

        return hist

    except Exception:
        return None


def _appearance_dist(h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> float:
    """
    Bhattacharyya distance giữa 2 histogram → [0, 1].
    0 = giống hệt nhau, 1 = hoàn toàn khác.
    Trả về 1.0 nếu một trong hai là None.
    """
    if h1 is None or h2 is None:
        return 1.0
    try:
        bc = float(np.sqrt(h1 * h2).sum())   # Bhattacharyya coefficient
        bc = min(bc, 1.0)
        return 1.0 - bc   # distance: 0=giống, 1=khác
    except Exception:
        return 1.0


# ============================================================
# IoU helper
# ============================================================

def _iou(a: list[float], b: list[float]) -> float:
    """Tính IoU giữa 2 box [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _predicted_xyxy(trk: "_Track", use_velocity: bool) -> list[float]:
    """
    Dự đoán bbox ở frame kế tiếp dựa trên velocity EMA.
    Nếu use_velocity=False, trả về bbox hiện tại.
    """
    x1, y1, x2, y2 = trk.bbox.to_xyxy()
    if use_velocity and trk.frames_tracked >= 2:
        vx, vy = trk.velocity
        return [x1 + vx, y1 + vy, x2 + vx, y2 + vy]
    return [x1, y1, x2, y2]


# ============================================================
# Internal Track
# ============================================================

class _Track:
    """Nội bộ: một track đang theo dõi — v2 với appearance + EMA velocity."""
    _id_counter = 0

    def __init__(self, det: Detection, frame: Optional[np.ndarray] = None):
        _Track._id_counter += 1
        self.track_id       = _Track._id_counter
        self.bbox           = det.bbox
        self.class_name     = det.class_name
        self.category       = det.category
        self.confidence     = det.confidence
        self.frames_tracked = 1
        self.missed         = 0
        self.hits           = 1          # Số frame liên tiếp có detection
        self.confirmed      = False      # True khi hits >= min_hits
        self.velocity       = (0.0, 0.0)
        self.first_seen     = time.time()
        self.last_seen      = time.time()

        # Appearance embedding (color histogram 96-dim)
        self.appearance: Optional[np.ndarray] = (
            _compute_appearance(frame, det.bbox) if frame is not None else None
        )

    def update(self, det: Detection, frame: Optional[np.ndarray] = None):
        cx, cy = det.bbox.center
        px, py = self.bbox.center

        # Velocity EMA: làm mượt, tránh box jitter
        raw_vx = cx - px
        raw_vy = cy - py
        old_vx, old_vy = self.velocity
        self.velocity = (
            _VELOCITY_EMA_ALPHA * raw_vx + (1 - _VELOCITY_EMA_ALPHA) * old_vx,
            _VELOCITY_EMA_ALPHA * raw_vy + (1 - _VELOCITY_EMA_ALPHA) * old_vy,
        )

        self.bbox       = det.bbox
        self.confidence = det.confidence
        self.frames_tracked += 1
        self.hits      += 1
        self.missed     = 0
        self.last_seen  = time.time()

        # Cập nhật appearance: EMA của histogram (80% cũ + 20% mới)
        new_app = _compute_appearance(frame, det.bbox) if frame is not None else None
        if new_app is not None:
            if self.appearance is None:
                self.appearance = new_app
            else:
                # Làm mượt histogram theo thời gian
                self.appearance = 0.8 * self.appearance + 0.2 * new_app
                total = self.appearance.sum()
                if total > 0:
                    self.appearance /= total

    def mark_missed(self):
        self.missed += 1
        self.hits    = 0   # Reset hits khi miss

    def appearance_similarity(self, other_hist: Optional[np.ndarray]) -> float:
        """Trả về similarity [0..1]: 1 = giống hệt, 0 = khác hoàn toàn."""
        return 1.0 - _appearance_dist(self.appearance, other_hist)


# ============================================================
# Public Tracker Class
# ============================================================

class ObjectTracker:
    """
    IoU tracker nâng cấp v2:
      - Velocity EMA: loại bỏ box jitter
      - Appearance embedding: color histogram để nhận diện lại
      - Two-stage matching: ưu tiên IoU cao trước, appearance cho missed tracks
      - Re-ID buffer 30s: khôi phục track đã mất khi người quay lại
      - Hungarian / greedy matching
      - Min-hits guard
    """

    def __init__(self):
        self._cfg          = TRACKING_CONFIG
        self._tracks       : dict[int, _Track] = {}   # track_id → Track
        self._min_area     = self._cfg["min_box_area"]
        self._match_thresh = self._cfg["match_thresh"]
        self._buffer       = self._cfg["track_buffer"]
        self._min_hits     = self._cfg.get("min_hits", 2)
        self._use_velocity = self._cfg.get("use_velocity_prediction", True)
        self._category_lock= self._cfg.get("category_lock", True)

        # Re-ID buffer: giữ track đã lost để tìm lại khi người quay lại
        # format: track_id → (_Track, lost_timestamp)
        self._reid_buffer  : dict[int, tuple[_Track, float]] = {}

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def update(self, detections: list[Detection], frame: Optional[np.ndarray] = None) -> list[TrackedObject]:
        """
        Cập nhật tracker với detections của frame hiện tại.

        Args:
            detections: List[Detection] từ ObjectDetector
            frame: BGR numpy array (dùng để tính appearance). Có thể None.

        Returns:
            List[TrackedObject] — chỉ trả về track đã được confirmed
        """
        # 1. Filter detections đủ lớn
        valid_dets = [d for d in detections if d.bbox.area >= self._min_area]

        track_list = list(self._tracks.values())

        matched_det_ids : set[int]   = set()
        matched_trk_ids : set[int]   = set()

        # 2. Xây dựng ma trận IoU và Appearance
        if track_list and valid_dets:
            n_trk = len(track_list)
            n_det = len(valid_dets)

            iou_matrix = np.zeros((n_trk, n_det), dtype=np.float32)
            app_matrix = np.ones((n_trk, n_det), dtype=np.float32)   # dist [0..1]

            for ti, trk in enumerate(track_list):
                pred_box = _predicted_xyxy(trk, self._use_velocity)
                for di, det in enumerate(valid_dets):
                    iou_val = _iou(pred_box, det.bbox.to_xyxy())
                    if self._category_lock and trk.category != det.category:
                        iou_val = 0.0
                    iou_matrix[ti, di] = iou_val

                    # Appearance distance
                    det_app = _compute_appearance(frame, det.bbox) if frame is not None else None
                    app_matrix[ti, di] = _appearance_dist(trk.appearance, det_app)

            # Stage 1: match IoU cao (chắc chắn)
            matched_trk_ids, matched_det_ids = self._two_stage_match(
                track_list, valid_dets, iou_matrix, app_matrix, frame
            )

        # 3. Unmatched detections → thử Re-ID trước khi tạo mới
        now = time.time()
        for di, det in enumerate(valid_dets):
            if di in matched_det_ids:
                continue

            # Tính appearance của detection này
            det_app = _compute_appearance(frame, det.bbox) if frame is not None else None

            # Thử Re-ID: tìm track cũ trong buffer có appearance giống
            revived_trk = self._try_reid(det, det_app, frame)

            if revived_trk is not None:
                # Khôi phục track cũ
                revived_trk.update(det, frame)
                self._tracks[revived_trk.track_id] = revived_trk
                logger.debug(
                    f"[Tracker] Re-ID: track #{revived_trk.track_id} recovered "
                    f"(missing {now - revived_trk.last_seen:.1f}s)"
                )
            else:
                # Tạo track mới
                new_trk = _Track(det, frame)
                self._tracks[new_trk.track_id] = new_trk
                logger.debug(f"[Tracker] New track #{new_trk.track_id} created")

        # 4. Unmatched tracks → increment missed
        for trk in track_list:
            if trk.track_id not in matched_trk_ids:
                trk.mark_missed()

        # 5. Remove lost tracks (missed > buffer) → chuyển vào Re-ID buffer
        lost_ids = [tid for tid, trk in self._tracks.items()
                    if trk.missed > self._buffer]
        for tid in lost_ids:
            trk = self._tracks.pop(tid)
            # Chỉ lưu vào Re-ID buffer nếu track đã confirmed (tránh ghost track)
            if trk.confirmed and trk.appearance is not None:
                self._reid_buffer[tid] = (trk, now)
                logger.debug(f"[Tracker] Track #{tid} lost → Re-ID buffer")
            else:
                logger.debug(f"[Tracker] Track #{tid} lost (not confirmed)")

        # 6. Dọn Re-ID buffer: xóa các track quá cũ
        expired = [tid for tid, (trk, ts) in self._reid_buffer.items()
                   if now - ts > _REID_TTL]
        for tid in expired:
            del self._reid_buffer[tid]
            logger.debug(f"[Tracker] Re-ID buffer: track #{tid} expired")

        # 7. Confirm tracks đủ hits
        for trk in self._tracks.values():
            if not trk.confirmed and trk.hits >= self._min_hits:
                trk.confirmed = True
                logger.debug(f"[Tracker] Track #{trk.track_id} confirmed after {trk.hits} hits")

        # 8. Trả về TrackedObject — chỉ confirmed tracks
        result = [
            self._to_tracked_obj(trk)
            for trk in self._tracks.values()
            if trk.confirmed
        ]

        logger.debug(
            f"[Tracker] active={len(self._tracks)}, confirmed={len(result)}, "
            f"dets={len(valid_dets)}, reid_buf={len(self._reid_buffer)}"
        )
        return result

    def reset(self):
        """Xóa toàn bộ tracks (dùng khi đổi nguồn video)."""
        self._tracks.clear()
        self._reid_buffer.clear()
        logger.info("Tracker reset.")

    # ----------------------------------------------------------
    # Two-stage Matching
    # ----------------------------------------------------------

    def _two_stage_match(
        self,
        track_list: list["_Track"],
        valid_dets : list[Detection],
        iou_matrix : np.ndarray,
        app_matrix : np.ndarray,
        frame      : Optional[np.ndarray],
    ) -> tuple[set, set]:
        """
        Matching 2 vòng:

        Stage 1 (IoU cao): cost = α×(1-IoU) + (1-α)×app_dist
                           Ngưỡng IoU ≥ _STAGE1_IOU_THRESH (0.25)

        Stage 2 (IoU thấp + appearance): chỉ dùng app_dist cho tracks đã missed ≥ 1
                           Ngưỡng IoU ≥ _STAGE2_IOU_THRESH (0.08)
                           Ngưỡng appearance similarity ≥ _STAGE2_APP_THRESH (0.70)
        """
        matched_trk_ids: set[int]   = set()
        matched_det_ids: set[int]   = set()

        n_trk = len(track_list)
        n_det = len(valid_dets)

        # ── Stage 1: Combined IoU + appearance ──────────────────
        combined_cost = (
            _APPEAR_ALPHA * (1.0 - iou_matrix)
            + (1.0 - _APPEAR_ALPHA) * app_matrix
        )
        # Vô hiệu hoá cặp có IoU quá thấp (dưới stage 1 threshold)
        combined_cost[iou_matrix < _STAGE1_IOU_THRESH] = 2.0   # penalty lớn

        s1_trk_ids, s1_det_ids = self._run_assignment(
            track_list, valid_dets, combined_cost, iou_matrix,
            iou_thresh=_STAGE1_IOU_THRESH, frame=frame
        )
        matched_trk_ids.update(s1_trk_ids)
        matched_det_ids.update(s1_det_ids)

        # ── Stage 2: Appearance-only cho tracks đã missed ──────
        # Chỉ thực hiện nếu còn unmatched tracks AND unmatched dets
        unmatched_trk_indices = [
            ti for ti, trk in enumerate(track_list)
            if trk.track_id not in matched_trk_ids and trk.missed >= 1
        ]
        unmatched_det_indices = [
            di for di in range(n_det)
            if di not in matched_det_ids
        ]

        if unmatched_trk_indices and unmatched_det_indices:
            sub_trk   = [track_list[ti] for ti in unmatched_trk_indices]
            sub_det   = [valid_dets[di] for di in unmatched_det_indices]

            sub_iou = iou_matrix[np.ix_(unmatched_trk_indices, unmatched_det_indices)]
            sub_app = app_matrix[np.ix_(unmatched_trk_indices, unmatched_det_indices)]

            # Stage 2: appearance-primary cost, loại bỏ cặp IoU quá thấp
            s2_cost = sub_app.copy()
            s2_cost[sub_iou < _STAGE2_IOU_THRESH] = 2.0

            for sti, (trk, det_indices_in_sub) in enumerate(
                self._greedy_appearance_match(sub_trk, sub_det, s2_cost, sub_app)
            ):
                orig_det_idx = unmatched_det_indices[det_indices_in_sub]
                if trk.track_id not in matched_trk_ids and orig_det_idx not in matched_det_ids:
                    trk.update(valid_dets[orig_det_idx], frame)
                    matched_trk_ids.add(trk.track_id)
                    matched_det_ids.add(orig_det_idx)
                    logger.debug(
                        f"[Tracker] Stage2 match: track #{trk.track_id} "
                        f"← det #{orig_det_idx}"
                    )

        return matched_trk_ids, matched_det_ids

    def _run_assignment(
        self,
        track_list : list["_Track"],
        valid_dets : list[Detection],
        cost_matrix: np.ndarray,
        iou_matrix : np.ndarray,
        iou_thresh : float,
        frame      : Optional[np.ndarray],
    ) -> tuple[set, set]:
        """Chạy Hungarian (hoặc greedy) và cập nhật tracks đã match."""
        matched_trk_ids: set[int] = set()
        matched_det_ids: set[int] = set()

        if _SCIPY_AVAILABLE:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            pairs = list(zip(row_ind, col_ind))
        else:
            pairs = self._greedy_pairs(cost_matrix)

        for ti, di in pairs:
            if iou_matrix[ti, di] >= iou_thresh:
                trk = track_list[ti]
                trk.update(valid_dets[di], frame)
                matched_trk_ids.add(trk.track_id)
                matched_det_ids.add(di)

        return matched_trk_ids, matched_det_ids

    @staticmethod
    def _greedy_pairs(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
        """Greedy matching: lấy min cost → assign → zeroed row/col → lặp."""
        pairs = []
        mat = cost_matrix.copy()
        while mat.min() < 1.5:   # < penalty (2.0)
            ti, di = np.unravel_index(mat.argmin(), mat.shape)
            pairs.append((int(ti), int(di)))
            mat[ti, :] = 2.0
            mat[:, di] = 2.0
        return pairs

    @staticmethod
    def _greedy_appearance_match(
        sub_trk : list["_Track"],
        sub_det : list[Detection],
        cost_mat: np.ndarray,
        app_mat : np.ndarray,
    ):
        """Greedy appearance match: yield (track, det_idx_in_sub) pairs."""
        mat = cost_mat.copy()
        while mat.min() < 1.5:
            ti, di = np.unravel_index(mat.argmin(), mat.shape)
            sim = 1.0 - float(app_mat[ti, di])
            if sim >= _STAGE2_APP_THRESH:
                yield sub_trk[ti], di
            mat[ti, :] = 2.0
            mat[:, di] = 2.0

    # ----------------------------------------------------------
    # Re-ID
    # ----------------------------------------------------------

    def _try_reid(
        self,
        det     : Detection,
        det_app : Optional[np.ndarray],
        frame   : Optional[np.ndarray],
    ) -> Optional["_Track"]:
        """
        Tìm track cũ trong Re-ID buffer khớp với detection mới.

        Điều kiện ghép:
          - Cùng category
          - Appearance similarity >= _REID_SIM_THRESH (0.65)
          - Lấy track có similarity cao nhất
        """
        if not self._reid_buffer or det_app is None:
            return None

        best_sim   = _REID_SIM_THRESH - 0.01   # Cần vượt threshold
        best_track = None

        for tid, (trk, lost_ts) in self._reid_buffer.items():
            if self._category_lock and trk.category != det.category:
                continue
            sim = 1.0 - _appearance_dist(trk.appearance, det_app)
            if sim > best_sim:
                best_sim   = sim
                best_track = trk

        if best_track is not None:
            # Xóa khỏi Re-ID buffer
            del self._reid_buffer[best_track.track_id]
            # Reset missed counter để track hoạt động lại bình thường
            best_track.missed     = 0
            best_track.hits       = self._min_hits   # Vẫn confirmed
            best_track.confirmed  = True
            return best_track

        return None

    # ----------------------------------------------------------
    # Legacy matching (greedy fallback khi không có scipy, dùng cho stage 1)
    # ----------------------------------------------------------

    def _greedy_match(
        self,
        track_list : list["_Track"],
        valid_dets : list[Detection],
        iou_matrix : np.ndarray,
        frame      : Optional[np.ndarray],
    ) -> tuple[set, set]:
        """Greedy matching fallback (dùng IoU thuần)."""
        matched_trk_ids: set[int] = set()
        matched_det_ids: set[int] = set()
        iou_mat = iou_matrix.copy()

        while True:
            max_val = iou_mat.max()
            if max_val < self._match_thresh:
                break
            ti, di = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            track_list[ti].update(valid_dets[di], frame)
            matched_trk_ids.add(track_list[ti].track_id)
            matched_det_ids.add(di)
            iou_mat[ti, :] = 0
            iou_mat[:, di] = 0

        return matched_trk_ids, matched_det_ids

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    @staticmethod
    def _to_tracked_obj(trk: "_Track") -> TrackedObject:
        return TrackedObject(
            track_id       = trk.track_id,
            bbox           = trk.bbox,
            class_name     = trk.class_name,
            category       = trk.category,
            confidence     = trk.confidence,
            frames_tracked = trk.frames_tracked,
            velocity       = trk.velocity,
            first_seen     = trk.first_seen,
            last_seen      = trk.last_seen,
        )

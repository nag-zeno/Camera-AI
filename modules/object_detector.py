"""
object_detector.py — Tầng 2: Perception (Detection)

Nhận vào: BGR frame (numpy)
Trả ra  : List[Detection] — kết quả raw từ YOLOv8 (ONNX DirectML / CUDA / CPU)

Backend ưu tiên:
  1. ONNX DirectML  — AMD/Intel/NVIDIA qua Windows DirectX 12
  2. ONNX CUDA      — NVIDIA CUDA
  3. PyTorch CUDA   — NVIDIA (fallback)
  4. ONNX CPU       — CPU đa nhân
"""
import logging
import cv2
import numpy as np
from pathlib import Path

from config import DETECTION_CONFIG, MODELS_DIR
from models import Detection, BoundingBox, ObjectCategory
from modules.gpu_manager import create_ort_session, get_torch_device, log_gpu_summary

logger = logging.getLogger(__name__)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize và pad ảnh giữ nguyên tỉ lệ aspect ratio cho YOLOv8."""
    shape = img.shape[:2]  # shape hiện tại [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Tỉ lệ scale (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Kích thước mới sau scale (chưa pad)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2.0  # chia đều sang 2 bên
    dh /= 2.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


class ObjectDetector:
    """
    Bao bọc YOLOv8. Hỗ trợ chạy:
      1. ONNX Runtime (DirectML / GPU AMD) — nhanh nhất và mượt nhất.
      2. PyTorch (CUDA / CPU) — fallback.
    """

    def __init__(self):
        self._model = None
        self._ort_session = None
        self._using_onnx = False
        self._device = None
        self._cfg = DETECTION_CONFIG
        self._class_map = self._cfg["target_classes"]       # {id: name}
        self._cat_map = self._cfg["class_categories"]     # {name: category}

    def load(self):
        """Tải model YOLOv8 (ưu tiên ONNX DirectML/CUDA → fallback PyTorch)."""
        # In tóm tắt GPU một lần khi khởi động
        log_gpu_summary()

        device_cfg      = self._cfg.get("device", "auto")
        onnx_model_path = MODELS_DIR / "yolov8n.onnx"

        # 1. Thử tải qua ONNX Runtime (hỗ trợ DirectML/CUDA tự động)
        if onnx_model_path.exists():
            sess, provider = create_ort_session(str(onnx_model_path), prefer=device_cfg)
            if sess is not None:
                self._ort_session  = sess
                self._input_name   = sess.get_inputs()[0].name
                self._output_name  = sess.get_outputs()[0].name
                self._device       = provider
                self._using_onnx   = True
                logger.info(
                    f"[ObjectDetector] YOLOv8 ONNX loaded on [{provider}] "
                    f"({onnx_model_path.stat().st_size/1024**2:.1f} MB)"
                )
                return

        # 2. Fallback về PyTorch
        logger.warning(
            "[ObjectDetector] ONNX model không có hoặc ORT chưa cài. "
            "Fallback về PyTorch. Chạy: pip install onnxruntime-directml"
        )
        self._load_pytorch()

    def _load_pytorch(self):
        """Load YOLO qua PyTorch Ultralytics (CUDA/CPU)."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

        model_name = self._cfg["model_name"]
        self._device = get_torch_device()

        logger.info(f"[ObjectDetector] Loading PyTorch model '{model_name}' on device '{self._device}'...")
        self._model = YOLO(model_name)
        self._model.to(self._device)

        # Bật half precision nếu GPU CUDA
        if "cuda" in self._device:
            try:
                self._model.model.half()  # FP16 inference
                logger.info("[ObjectDetector] FP16 half-precision enabled on CUDA")
            except Exception:
                pass

        self._using_onnx = False
        logger.info(f"[ObjectDetector] PyTorch model loaded on [{self._device}].")

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> list[Detection]:
        """
        Chạy inference trên 1 frame.
        """
        if self._using_onnx:
            return self._detect_onnx(frame, frame_id)
        else:
            return self._detect_pytorch(frame, frame_id)

    def _detect_onnx(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        """Inference bằng ONNX Runtime DirectML."""
        h_orig, w_orig = frame.shape[:2]

        # 1. Preprocess: letterbox (640x640)
        input_img, r, (dw, dh) = letterbox(frame, (640, 640))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)[np.newaxis, ...]  # shape: (1, 3, 640, 640)

        # 2. Run inference
        outputs = self._ort_session.run([self._output_name], {self._input_name: input_img})
        
        # Output shape: (1, 84, 8400)
        output = outputs[0][0]
        output = output.T  # transpose sang (8400, 84)

        boxes = []
        confidences = []
        class_ids = []

        target_ids = list(self._class_map.keys())
        conf_threshold = self._cfg["confidence_threshold"]

        # 3. Parse outputs
        for row in output:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]

            if confidence >= conf_threshold and class_id in target_ids:
                xc, yc, w, h = row[0], row[1], row[2], row[3]
                # Top-left corner
                x1 = xc - w / 2.0
                y1 = yc - h / 2.0
                boxes.append([int(x1), int(y1), int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

        # 4. NMS (Non-Maximum Suppression) bằng OpenCV
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            conf_threshold,
            self._cfg["iou_threshold"]
        )

        detections: list[Detection] = []

        if len(indices) > 0:
            # OpenCV NMSBoxes có thể trả về array 1D hoặc 2D tùy phiên bản
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box[0], box[1], box[2], box[3]

                # Map ngược tọa độ về ảnh gốc (loại bỏ padding và chia tỉ lệ scale)
                x1 = (x - dw) / r
                y1 = (y - dh) / r
                x2 = (x + w - dw) / r
                y2 = (y + h - dh) / r

                # Clip tọa độ nằm trong ảnh gốc
                x1 = max(0, min(w_orig, x1))
                y1 = max(0, min(h_orig, y1))
                x2 = max(0, min(w_orig, x2))
                y2 = max(0, min(h_orig, y2))

                # Lọc box quá nhỏ
                area = (x2 - x1) * (y2 - y1)
                if area < self._cfg.get("min_box_area", 400):
                    continue

                cls_id = class_ids[i]
                cls_name = self._class_map.get(cls_id)
                category_str = self._cat_map.get(cls_name, "unknown")
                
                try:
                    category = ObjectCategory(category_str)
                except ValueError:
                    category = ObjectCategory.UNKNOWN

                detections.append(Detection(
                    bbox=BoundingBox(x1, y1, x2, y2),
                    class_name=cls_name,
                    category=category,
                    confidence=confidences[i],
                    frame_id=frame_id,
                ))

        return detections

    def _detect_pytorch(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        """Inference bằng PyTorch YOLOv8 (nguyên bản)."""
        if self._model is None:
            raise RuntimeError("Call load() before detect()")

        results = self._model.predict(
            source=frame,
            conf=self._cfg["confidence_threshold"],
            iou=self._cfg["iou_threshold"],
            classes=list(self._class_map.keys()),
            verbose=False,
            device=self._device,
        )

        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self._class_map.get(cls_id)
                if cls_name is None:
                    continue

                category_str = self._cat_map.get(cls_name, "unknown")
                try:
                    category = ObjectCategory(category_str)
                except ValueError:
                    category = ObjectCategory.UNKNOWN

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())

                # Lọc box quá nhỏ
                area = (x2 - x1) * (y2 - y1)
                if area < self._cfg.get("min_box_area", 400):
                    continue

                detections.append(Detection(
                    bbox=BoundingBox(x1, y1, x2, y2),
                    class_name=cls_name,
                    category=category,
                    confidence=conf,
                    frame_id=frame_id,
                ))

        return detections

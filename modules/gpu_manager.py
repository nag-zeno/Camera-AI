"""
gpu_manager.py — Quản lý GPU/Inference Backend Tập trung

Tự động phát hiện và chọn backend tốt nhất:
  1. ONNX DirectML  — AMD/Intel/NVIDIA qua DirectML (Windows DX12)
  2. ONNX CUDA      — NVIDIA GPU qua CUDA  
  3. PyTorch CUDA   — NVIDIA GPU (fallback)
  4. ONNX CPU       — CPU đa nhân (luôn có)

Sử dụng:
    from modules.gpu_manager import get_ort_providers, get_torch_device, gpu_info
"""
import os
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Đảm bảo FFmpeg dùng TCP cho RTSP (set trước khi import OpenCV)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|threads;1")


# ============================================================
# ONNX Runtime Provider Detection
# ============================================================

@lru_cache(maxsize=1)
def get_ort_providers(prefer: str = "auto") -> list[str]:
    """
    Trả về danh sách ONNX Runtime providers theo thứ tự ưu tiên.

    Args:
        prefer: "auto" | "dml" | "cuda" | "cpu"

    Returns:
        Danh sách providers, ví dụ: ['DmlExecutionProvider', 'CPUExecutionProvider']
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        logger.warning("[GPU] onnxruntime chưa cài. Chạy: pip install onnxruntime-directml")
        return ["CPUExecutionProvider"]

    providers: list[str] = []

    if prefer in ("auto", "dml"):
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
            logger.debug("[GPU] DmlExecutionProvider: AVAILABLE (AMD/Intel/NVIDIA via DirectML)")

    if prefer in ("auto", "cuda"):
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
            logger.debug("[GPU] CUDAExecutionProvider: AVAILABLE (NVIDIA CUDA)")

    if "TensorrtExecutionProvider" in available and prefer in ("auto", "trt"):
        providers.append("TensorrtExecutionProvider")

    # CPU luôn là fallback cuối cùng
    providers.append("CPUExecutionProvider")
    return providers


@lru_cache(maxsize=1)
def get_ort_session_options():
    """Trả về ORT SessionOptions đã được tối ưu."""
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Tự động dùng tất cả cores cho CPU path
        n_cpu = os.cpu_count() or 4
        opts.intra_op_num_threads = max(1, n_cpu)
        opts.inter_op_num_threads = 1
        # Tắt verbose logging của ORT
        opts.log_severity_level = 3  # ERROR only
        return opts
    except ImportError:
        return None


def create_ort_session(model_path: str, prefer: str = "auto"):
    """
    Tạo ONNX Runtime InferenceSession với GPU acceleration tốt nhất.

    Returns:
        (session, provider_used) hoặc (None, None) nếu lỗi
    """
    try:
        import onnxruntime as ort
        from pathlib import Path
        if not Path(model_path).exists():
            logger.warning(f"[GPU] Model không tồn tại: {model_path}")
            return None, None

        opts      = get_ort_session_options()
        providers = get_ort_providers(prefer)

        sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers,
        )
        provider_used = sess.get_providers()[0].replace("ExecutionProvider", "")
        return sess, provider_used

    except Exception as e:
        logger.error(f"[GPU] Không thể tạo ORT session cho {model_path}: {e}")
        return None, None


# ============================================================
# PyTorch Device Detection
# ============================================================

@lru_cache(maxsize=1)
def get_torch_device(prefer: str = "auto") -> str:
    """
    Trả về device string tốt nhất cho PyTorch.

    Args:
        prefer: "auto" | "cuda" | "cpu"

    Returns:
        "cuda", "cuda:0", hoặc "cpu"
    """
    try:
        import torch
        if prefer in ("auto", "cuda") and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            logger.debug(f"[GPU] PyTorch device: {device} ({torch.cuda.get_device_name(0)})")
            return device
    except ImportError:
        pass

    return "cpu"


# ============================================================
# GPU Information
# ============================================================

@lru_cache(maxsize=1)
def gpu_info() -> dict:
    """
    Thu thập thông tin GPU từ tất cả các nguồn.

    Returns:
        dict với các key: ort_version, ort_providers, torch_device, gpu_name, vram_gb
    """
    info: dict = {
        "ort_version"    : None,
        "ort_providers"  : [],
        "active_provider": "CPU",
        "torch_device"   : "cpu",
        "torch_version"  : None,
        "gpu_name"       : "Unknown",
        "vram_gb"        : None,
        "dml_available"  : False,
        "cuda_available" : False,
    }

    # ONNX Runtime info
    try:
        import onnxruntime as ort
        info["ort_version"]   = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
        info["dml_available"] = "DmlExecutionProvider" in info["ort_providers"]
        info["cuda_available"] = "CUDAExecutionProvider" in info["ort_providers"]

        providers = get_ort_providers()
        info["active_provider"] = providers[0].replace("ExecutionProvider", "") if providers else "CPU"
    except ImportError:
        pass

    # PyTorch info
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["torch_device"] = f"cuda:{torch.cuda.current_device()}"
            info["gpu_name"]     = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"]      = round(vram / 1024**3, 1)
            if not info["dml_available"]:
                info["cuda_available"] = True
    except ImportError:
        pass

    # Windows GPU name (qua subprocess nếu chưa có)
    if info["gpu_name"] == "Unknown":
        try:
            import subprocess
            out = subprocess.check_output(
                ["powershell", "-Command",
                 "(Get-WmiObject Win32_VideoController | Select-Object -First 1).Name"],
                text=True, timeout=5, stderr=subprocess.DEVNULL
            ).strip()
            if out:
                info["gpu_name"] = out
        except Exception:
            pass

    return info


def log_gpu_summary():
    """In ra tóm tắt GPU khi khởi động."""
    info = gpu_info()
    lines = [
        f"[GPU] ONNX Runtime: {info['ort_version'] or 'not installed'}",
        f"[GPU] Active provider: {info['active_provider']}",
        f"[GPU] GPU: {info['gpu_name']}",
        f"[GPU] DirectML: {'YES' if info['dml_available'] else 'NO'}  |  "
        f"CUDA: {'YES' if info['cuda_available'] else 'NO'}",
        f"[GPU] PyTorch device: {info['torch_device']} ({info['torch_version'] or 'N/A'})",
    ]
    if info["vram_gb"]:
        lines.append(f"[GPU] VRAM: {info['vram_gb']} GB")
    for l in lines:
        logger.info(l)
    return info


def run_gpu_benchmark(quick: bool = True) -> dict:
    """
    Chạy benchmark nhanh để so sánh DML vs CPU.
    Chỉ chạy nếu có ONNX model sẵn.

    Returns:
        dict: {model_name: {dml_ms, cpu_ms, speedup}}
    """
    import time
    import numpy as np
    from pathlib import Path

    results = {}
    test_cases = [
        ("yolov8n",      "models/yolov8n.onnx",           (1, 3, 640, 640), "images"),
        ("rolenet_int8", "models/rolenet_v3_quant.onnx",   (1, 3, 256, 128), "input"),
    ]
    n_runs = 5 if quick else 30

    try:
        import onnxruntime as ort
    except ImportError:
        return results

    for name, path, shape, inp in test_cases:
        if not Path(path).exists():
            continue

        dummy = np.random.rand(*shape).astype(np.float32)
        row   = {}

        for prov_list, key in [
            (["DmlExecutionProvider", "CPUExecutionProvider"], "dml_ms"),
            (["CPUExecutionProvider"],                         "cpu_ms"),
        ]:
            avail = ort.get_available_providers()
            actual_prov = [p for p in prov_list if p in avail]
            if not actual_prov:
                continue
            try:
                sess     = ort.InferenceSession(path, providers=actual_prov)
                out_name = sess.get_outputs()[0].name
                # Warmup
                for _ in range(2):
                    sess.run([out_name], {inp: dummy})
                # Measure
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    sess.run([out_name], {inp: dummy})
                row[key] = round((time.perf_counter() - t0) / n_runs * 1000, 1)
            except Exception as e:
                logger.debug(f"[GPU] Benchmark {name}/{key} lỗi: {e}")

        if "dml_ms" in row and "cpu_ms" in row and row["dml_ms"] > 0:
            row["speedup"] = round(row["cpu_ms"] / row["dml_ms"], 1)
        results[name] = row

    return results

"""
export_onnx.py — Export RoleNet V3 (ConvNeXt-Tiny) sang ONNX

Lợi ích:
  - Inference nhanh hơn 2–4x trên CPU (không cần PyTorch runtime)
  - Dễ deploy (chỉ cần onnxruntime, không cần torch/timm)
  - Hỗ trợ INT8 quantization (tùy chọn) → thêm 1.5x nhanh hơn

Cách chạy:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --no-benchmark
    python scripts/export_onnx.py --quantize          # Thêm INT8 quantized version
    python scripts/export_onnx.py --opset 17          # Opset ONNX cụ thể

Output:
    models/rolenet_v3.onnx           (FP32, ~107MB)
    models/rolenet_v3_quant.onnx     (INT8, ~28MB) — nếu --quantize
"""

import sys
import os
import time
import argparse
import json
import logging
from pathlib import Path

# Thêm project root vào sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODELS_DIR   = Path(__file__).parent.parent / "models"
ONNX_OUTPUT  = MODELS_DIR / "rolenet_v3.onnx"
QUANT_OUTPUT = MODELS_DIR / "rolenet_v3_quant.onnx"

# Kiến trúc V3 (phải match với role_classifier.py)
ROLENET_CLASSES = [
    "shipper", "doctor", "police", "military", "security", "student",
    "chef", "janitor", "construction", "nurse", "postman", "technician",
    "worker", "civil_guard", "normal", "unknown",
]
INPUT_H = 256
INPUT_W = 128
NUM_CLASSES = 16


# ============================================================
# Tái tạo model V3
# ============================================================

def build_rolenet_v3(ckpt_path: Path, device: str):
    """Load checkpoint và tái tạo kiến trúc ConvNeXt-Tiny."""
    import torch
    import torch.nn as nn

    try:
        import timm
    except ImportError:
        raise RuntimeError("timm chưa cài. Chạy: pip install timm")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    backbone    = timm.create_model("convnext_tiny", pretrained=False,
                                    num_classes=0, drop_rate=0.3)
    in_features = backbone.num_features  # 768

    head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LayerNorm(in_features),
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(p=0.15),
        nn.Linear(256, NUM_CLASSES),
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

    val_acc = ckpt.get("val_acc", 0.0)
    log.info(f"  ✅ Đã load RoleNet V3 — val_acc={val_acc*100:.2f}%")
    return net


# ============================================================
# Export FP32 ONNX
# ============================================================

def export_fp32(ckpt_path: Path, output_path: Path, opset: int, device: str):
    """Export model sang ONNX FP32."""
    import torch

    log.info(f"\n📦 Bắt đầu export ONNX FP32 (opset={opset})...")
    log.info(f"   Input:  {ckpt_path.name}")
    log.info(f"   Output: {output_path.name}")

    net = build_rolenet_v3(ckpt_path, device)

    # Dummy input: batch=1, C=3, H=256, W=128
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, device=device)

    # Export dùng legacy exporter (không cần onnxscript)
    t0 = time.time()
    try:
        # PyTorch >= 2.3: dùng dynamo=False để dùng legacy exporter
        torch.onnx.export(
            net,
            dummy,
            str(output_path),
            opset_version       = opset,
            input_names         = ["input"],
            output_names        = ["logits"],
            dynamic_axes        = {
                "input" : {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            do_constant_folding = True,
            export_params       = True,
            verbose             = False,
            dynamo              = False,   # dùng legacy TorchScript-based exporter
        )
    except TypeError:
        # Phiên bản PyTorch cũ không có tham số dynamo
        torch.onnx.export(
            net,
            dummy,
            str(output_path),
            opset_version       = opset,
            input_names         = ["input"],
            output_names        = ["logits"],
            dynamic_axes        = {
                "input" : {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            do_constant_folding = True,
            export_params       = True,
            verbose             = False,
        )
    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / 1e6

    log.info(f"  ✅ Export xong trong {elapsed:.1f}s — kích thước: {size_mb:.1f} MB")
    return output_path


# ============================================================
# Verify & Validate
# ============================================================

def verify_onnx(pt_path: Path, onnx_path: Path, device: str):
    """So sánh output PyTorch vs ONNX để đảm bảo chính xác."""
    import torch
    import numpy as np

    log.info("\n🔍 Kiểm tra tính đúng đắn (PyTorch vs ONNX)...")

    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("  ⚠️  onnxruntime chưa cài. Bỏ qua validation.")
        log.warning("       Cài: pip install onnxruntime")
        return False

    # Tạo input ngẫu nhiên
    rng   = np.random.RandomState(42)
    arr   = rng.randn(1, 3, INPUT_H, INPUT_W).astype(np.float32)
    t_inp = torch.tensor(arr, device=device)

    # PyTorch output
    net   = build_rolenet_v3(pt_path, device)
    with torch.no_grad():
        pt_out = torch.softmax(net(t_inp), dim=1).cpu().numpy()

    # ONNX output
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if device == "cuda" else ["CPUExecutionProvider"]
    sess      = ort.InferenceSession(str(onnx_path), providers=providers)
    onnx_out  = sess.run(["logits"], {"input": arr})[0]
    onnx_out  = _softmax(onnx_out)

    # So sánh
    max_diff  = float(np.abs(pt_out - onnx_out).max())
    log.info(f"  Max diff PyTorch vs ONNX: {max_diff:.2e}")

    if max_diff < 1e-4:
        log.info("  ✅ Kết quả khớp hoàn toàn (diff < 1e-4)")
        return True
    else:
        log.warning(f"  ⚠️  Sai lệch lớn: {max_diff:.4f} — kiểm tra lại quá trình export")
        return False


def _softmax(x):
    import numpy as np
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ============================================================
# INT8 Quantization
# ============================================================

def quantize_onnx(fp32_path: Path, quant_path: Path):
    """Quantize ONNX FP32 → INT8 để tăng tốc thêm trên CPU."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        log.warning("\n⚠️  onnxruntime[quantization] chưa cài.")
        log.warning("     Cài: pip install onnxruntime")
        return None

    log.info(f"\n⚡ Quantize INT8...")
    log.info(f"   Input:  {fp32_path.name}")
    log.info(f"   Output: {quant_path.name}")

    t0 = time.time()
    quantize_dynamic(
        model_input   = str(fp32_path),
        model_output  = str(quant_path),
        weight_type   = QuantType.QUInt8,
        per_channel   = False,
        reduce_range  = False,
    )
    elapsed = time.time() - t0
    size_mb = quant_path.stat().st_size / 1e6

    log.info(f"  ✅ Quantize xong trong {elapsed:.1f}s — kích thước: {size_mb:.1f} MB")
    return quant_path


# ============================================================
# Benchmark
# ============================================================

def benchmark(pt_path: Path, onnx_path: Path, quant_path: Path | None, device: str,
              n_warmup: int = 10, n_runs: int = 100):
    """So sánh tốc độ inference: PyTorch vs ONNX FP32 vs ONNX INT8."""
    import torch
    import numpy as np

    log.info(f"\n📊 Benchmark ({n_warmup} warmup + {n_runs} runs)...")

    rng  = np.random.RandomState(0)
    arr  = rng.randn(1, 3, INPUT_H, INPUT_W).astype(np.float32)
    t_in = torch.tensor(arr, device=device)

    results = {}

    # --- PyTorch ---
    net = build_rolenet_v3(pt_path, device)
    for _ in range(n_warmup):
        with torch.no_grad():
            net(t_in)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            net(t_in)
    pt_ms = (time.perf_counter() - t0) / n_runs * 1000
    results["PyTorch (FP32)"] = pt_ms
    log.info(f"  🔵 PyTorch FP32    : {pt_ms:6.2f} ms/frame")

    # --- ONNX FP32 ---
    try:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if device == "cuda" else ["CPUExecutionProvider"]
        sess_fp32 = ort.InferenceSession(str(onnx_path), providers=providers)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4

        for _ in range(n_warmup):
            sess_fp32.run(["logits"], {"input": arr})

        t0 = time.perf_counter()
        for _ in range(n_runs):
            sess_fp32.run(["logits"], {"input": arr})
        onnx_ms = (time.perf_counter() - t0) / n_runs * 1000
        results["ONNX FP32"] = onnx_ms
        speedup = pt_ms / onnx_ms
        log.info(f"  🟢 ONNX FP32       : {onnx_ms:6.2f} ms/frame  ({speedup:.1f}x speedup)")

        # --- ONNX INT8 ---
        if quant_path and quant_path.exists():
            sess_q = ort.InferenceSession(str(quant_path), providers=providers)
            for _ in range(n_warmup):
                sess_q.run(["logits"], {"input": arr})

            t0 = time.perf_counter()
            for _ in range(n_runs):
                sess_q.run(["logits"], {"input": arr})
            q_ms   = (time.perf_counter() - t0) / n_runs * 1000
            speedup_q = pt_ms / q_ms
            results["ONNX INT8"] = q_ms
            log.info(f"  🟡 ONNX INT8 (quant): {q_ms:6.2f} ms/frame  ({speedup_q:.1f}x speedup)")

    except ImportError:
        log.warning("  ⚠️  onnxruntime chưa cài — bỏ qua ONNX benchmark.")
        log.warning("       Cài: pip install onnxruntime")

    return results


# ============================================================
# Lưu metadata ONNX
# ============================================================

def save_onnx_meta(onnx_path: Path, quant_path: Path | None, benchmark_results: dict):
    """Lưu thông tin ONNX để role_classifier.py có thể đọc."""
    meta = {
        "onnx_model"  : onnx_path.name,
        "quant_model" : quant_path.name if quant_path and quant_path.exists() else None,
        "input_shape" : [1, 3, INPUT_H, INPUT_W],
        "input_name"  : "input",
        "output_name" : "logits",
        "classes"     : ROLENET_CLASSES,
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std" : [0.229, 0.224, 0.225],
        "benchmark_ms" : benchmark_results,
        "num_classes"  : NUM_CLASSES,
    }
    meta_path = onnx_path.parent / "rolenet_v3_onnx_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"\n  💾 Metadata ONNX lưu tại: {meta_path.name}")
    return meta_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export RoleNet V3 ConvNeXt-Tiny → ONNX"
    )
    parser.add_argument("--ckpt",
                        default=str(MODELS_DIR / "rolenet_v3_best.pt"),
                        help="Đường dẫn file .pt checkpoint")
    parser.add_argument("--output",
                        default=str(ONNX_OUTPUT),
                        help="Đường dẫn file ONNX output")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (mặc định: 17)")
    parser.add_argument("--quantize", action="store_true",
                        help="Thêm bước INT8 dynamic quantization")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Bỏ qua benchmark tốc độ")
    parser.add_argument("--no-verify", action="store_true",
                        help="Bỏ qua bước kiểm tra tính đúng đắn")
    args = parser.parse_args()

    ckpt_path   = Path(args.ckpt)
    onnx_path   = Path(args.output)
    quant_path  = QUANT_OUTPUT if args.quantize else None

    # Kiểm tra prerequisites
    log.info("=" * 60)
    log.info("  🔧 RoleNet V3 → ONNX Exporter")
    log.info("=" * 60)

    if not ckpt_path.exists():
        log.error(f"\n❌ Không tìm thấy checkpoint: {ckpt_path}")
        log.error("   Hãy chắc chắn file rolenet_v3_best.pt tồn tại trong models/")
        sys.exit(1)

    # Kiểm tra torch
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"\n  Device: {device}")
        log.info(f"  Checkpoint: {ckpt_path} ({ckpt_path.stat().st_size/1e6:.1f} MB)")
    except ImportError:
        log.error("❌ PyTorch chưa cài. Cần PyTorch để export.")
        sys.exit(1)

    # 1. Export FP32
    export_fp32(ckpt_path, onnx_path, args.opset, device)

    # 2. Verify
    if not args.no_verify:
        verify_onnx(ckpt_path, onnx_path, device)

    # 3. Quantize (optional)
    if args.quantize:
        quantize_onnx(onnx_path, QUANT_OUTPUT)

    # 4. Benchmark
    bench_results = {}
    if not args.no_benchmark:
        bench_results = benchmark(ckpt_path, onnx_path, quant_path, device)

    # 5. Lưu metadata
    save_onnx_meta(onnx_path, quant_path, bench_results)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("  ✅ HOÀN THÀNH!")
    log.info("=" * 60)
    log.info(f"\n  FP32 ONNX  : {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)")
    if quant_path and quant_path.exists():
        log.info(f"  INT8 ONNX  : {quant_path}  ({quant_path.stat().st_size/1e6:.1f} MB)")
    log.info("\n  Bước tiếp theo:")
    log.info("    pip install onnxruntime   # hoặc onnxruntime-gpu")
    log.info("    → role_classifier.py sẽ tự động dùng ONNX khi khởi động")


if __name__ == "__main__":
    main()

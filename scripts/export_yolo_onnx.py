import sys
import shutil
from pathlib import Path

# Thêm thư mục gốc vào path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics is not installed. Please install it first.")
    sys.exit(1)

def main():
    model_path = project_root / "yolov8n.pt"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  📦 YOLOv8 → ONNX Exporter")
    print("=" * 60)

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure yolov8n.pt exists in the root directory.")
        sys.exit(1)

    print(f"Loading PyTorch YOLOv8 model from {model_path}...")
    model = YOLO(str(model_path))

    print("\nExporting model to ONNX format (dynamic=True, imgsz=640)...")
    # Export model. dynamic=True cho phép xử lý batch size động
    onnx_path_str = model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    
    onnx_path = Path(onnx_path_str)
    dest_path = models_dir / "yolov8n.onnx"

    if onnx_path.exists():
        print(f"\nExport successful! Saved at: {onnx_path}")
        # Di chuyển vào thư mục models/
        if onnx_path.absolute() != dest_path.absolute():
            if dest_path.exists():
                dest_path.unlink()
            shutil.move(str(onnx_path), str(dest_path))
            print(f"Moved ONNX model to: {dest_path}")
    else:
        print("\nError: Export failed, ONNX file was not created.")
        sys.exit(1)

    print("=" * 60)
    print("  ✅ EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()

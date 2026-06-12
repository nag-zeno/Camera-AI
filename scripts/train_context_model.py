"""
train_context_model.py — Huấn luyện ContextNet (XGBoost Classifier)

Quy trình:
    1. Đọc CSV từ generate_context_data.py
    2. Encode nhãn theo thứ tự severity
    3. Train XGBoost với class-weight cân bằng
    4. Đánh giá + in report
    5. Lưu model → models/context_net.pkl

Cách chạy:
    # Bước 1: sinh data (nếu chưa có)
    python scripts/generate_context_data.py

    # Bước 2: train
    python scripts/train_context_model.py

    # Bước 3: hệ thống tự tải model khi khởi động
"""
import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  xgboost chưa cài. Chạy: pip install xgboost pandas scikit-learn")
    sys.exit(1)

# Thứ tự severity của nhãn (QUAN TRỌNG — XGBoost cần thứ tự nhất quán)
ALERT_ORDER = ["ignore", "normal", "watch", "warning", "alert", "critical"]

FEATURE_NAMES = [
    "role_id", "identity_id", "zone_type_id", "zone_status_id",
    "loitering", "time_in_zone", "visit_count", "direction_id",
    "hour", "role_confidence", "category_id", "frames_tracked",
    "is_night", "is_business_hour",
    "action_id", "action_confidence",   # <- ActionNet features (v2)
]


def load_data(csv_path: str):
    print(f"📂 Đang tải data từ: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Tổng: {len(df):,} samples | {df['label'].nunique()} classes")
    print(f"   Phân bố:\n{df['label'].value_counts().to_string()}\n")

    # Lọc chỉ lấy features và label
    available = [f for f in FEATURE_NAMES if f in df.columns]
    X = df[available].values
    y_raw = df["label"].values
    return X, y_raw, available


def encode_labels(y_raw: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """Encode nhãn theo thứ tự severity."""
    present_classes = [c for c in ALERT_ORDER if c in set(y_raw)]
    le = LabelEncoder()
    le.classes_ = np.array(present_classes)
    y = le.transform(y_raw)
    return y, le


def train(X_train, y_train):
    """Train XGBoost với sample weights để xử lý mất cân bằng."""
    # Tính class weight (class ít xuất hiện → weight cao hơn)
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * (class_counts + 1e-9))
    sample_weights = class_weights[y_train]

    model = xgb.XGBClassifier(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.1,
        subsample        = 0.85,
        colsample_bytree = 0.85,
        min_child_weight = 3,
        eval_metric      = "mlogloss",
        use_label_encoder= False,
        random_state     = 42,
        n_jobs           = -1,  # Dùng toàn bộ CPU cores
    )

    model.fit(
        X_train, y_train,
        sample_weight = sample_weights,
        verbose       = False,
    )
    return model


def evaluate(model, X_test, y_test, le: LabelEncoder, feature_names: list):
    """In báo cáo đánh giá chi tiết."""
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    report_str = classification_report(
        y_test, y_pred,
        target_names = le.classes_,
        digits       = 3,
        zero_division= 0,
    )

    print("📊 Kết quả đánh giá trên test set:")
    print(f"   Accuracy: {acc:.1%}\n")
    print(report_str)

    # Feature importance
    print("🔍 Feature Importance (ảnh hưởng đến quyết định):")
    importances = model.feature_importances_
    sorted_pairs = sorted(
        zip(feature_names, importances), key=lambda x: -x[1]
    )
    for name, imp in sorted_pairs:
        bar = "█" * int(imp * 50)
        print(f"  {name:22s} {bar:<30s} {imp:.4f}")

    return y_pred, acc, report_str


def save_artifacts(model, le, feature_names, acc, n_samples, output_path):
    """Lưu model + metadata."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # Model bundle
    bundle = {
        "model"         : model,
        "label_encoder" : le,
        "feature_names" : feature_names,
        "accuracy"      : acc,
    }
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    # Metadata JSON (readable)
    meta = {
        "feature_names" : feature_names,
        "classes"       : le.classes_.tolist(),
        "n_samples"     : n_samples,
        "test_accuracy" : round(acc, 4),
        "n_estimators"  : model.n_estimators,
        "max_depth"     : model.max_depth,
    }
    meta_path = output_path.replace(".pkl", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Model lưu tại    : {output_path}")
    print(f"✅ Metadata lưu tại : {meta_path}")


def optional_shap(model, X_test, feature_names):
    """Tạo SHAP plot nếu có thư viện."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print("\n🔬 Đang tính SHAP values (explainability)...")
        explainer = shap.TreeExplainer(model)
        sample = X_test[:min(500, len(X_test))]
        shap_values = explainer.shap_values(sample)

        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            # Multi-class: lấy class nguy hiểm nhất (critical)
            shap.summary_plot(
                shap_values[-1], sample,
                feature_names=feature_names,
                show=False, plot_size=(10, 6)
            )
        else:
            shap.summary_plot(
                shap_values, sample,
                feature_names=feature_names,
                show=False, plot_size=(10, 6)
            )

        out = "reports/context_net_shap.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"✅ SHAP plot lưu tại: {out}")

    except ImportError:
        print("⚠️  (Bỏ qua SHAP — cài 'pip install shap matplotlib' nếu cần)")
    except Exception as e:
        print(f"⚠️  SHAP lỗi: {e}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train ContextNet (XGBoost)")
    parser.add_argument("--data",    default="data/context_training_data.csv",
                        help="CSV từ generate_context_data.py")
    parser.add_argument("--output",  default="models/context_net.pkl",
                        help="Đường dẫn lưu model")
    parser.add_argument("--no-shap", action="store_true",
                        help="Tắt SHAP plot")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"❌ Không tìm thấy {args.data}")
        print("   Chạy trước: python scripts/generate_context_data.py")
        sys.exit(1)

    # 1. Load data
    X, y_raw, feature_names = load_data(args.data)

    # 2. Encode labels
    y, le = encode_labels(y_raw)
    print(f"Classes (theo severity): {list(le.classes_)}\n")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}\n")

    # 4. Train
    print("🚀 Đang train XGBoost...")
    model = train(X_train, y_train)
    print("   Train xong!\n")

    # 5. Evaluate
    y_pred, acc, report_str = evaluate(model, X_test, y_test, le, feature_names)

    # 6. Save
    save_artifacts(model, le, feature_names, acc, len(X), args.output)

    # 7. SHAP
    if not args.no_shap:
        optional_shap(model, X_test, feature_names)

    # Done
    print(f"\n{'='*50}")
    print(f"✅ ContextNet training HOÀN THÀNH!")
    print(f"   Test accuracy: {acc:.1%}")
    print(f"   Pipeline sẽ tự tải model khi khởi động lại.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

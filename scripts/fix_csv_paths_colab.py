"""
fix_csv_paths.py
Chạy cell này trên Colab TRƯỚC Bước 4 để fix Windows backslash trong CSV.

Copy và chạy đoạn code này trong 1 cell mới trên Colab:
"""

# ── CELL FIX: Chạy trước Bước 4 ──────────────────────────────
# Vấn đề: CSV được tạo trên Windows nên path dùng backslash (shipper\000001.jpg)
#          Trên Linux/Colab cần forward slash (shipper/000001.jpg)

import pandas as pd
from pathlib import Path

splits_dir = Path('/content/rolenet_dataset/splits')

fixed_count = 0
for csv_name in ['train.csv', 'val.csv', 'test.csv']:
    csv_path = splits_dir / csv_name
    if not csv_path.exists():
        print(f'❌ {csv_name} not found!')
        continue

    df = pd.read_csv(csv_path)

    # Kiểm tra xem có backslash không
    sample_path = str(df['path'].iloc[0])
    if '\\' in sample_path:
        # Fix: thay backslash -> forward slash
        df['path'] = df['path'].str.replace('\\\\', '/', regex=False)
        df.to_csv(csv_path, index=False)
        fixed_count += 1
        print(f'  ✅ Fixed {csv_name}: {sample_path!r} → {df["path"].iloc[0]!r}')
    else:
        print(f'  ✅ {csv_name}: paths OK (no backslash)')

    # Verify file tồn tại
    aug_root = Path('/content/rolenet_dataset/augmented')
    test_path = aug_root / df['path'].iloc[0]
    exists = '✅ exists' if test_path.exists() else '❌ NOT found'
    print(f'     Sample check: {test_path} → {exists}')

print(f'\\n📋 Fixed {fixed_count}/3 CSV files.')
print('👉 Bây giờ chạy Bước 4 (Dataset & DataLoader)")

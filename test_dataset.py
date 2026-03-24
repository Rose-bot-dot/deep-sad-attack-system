import os
import sys

# 把 src 目录加入 Python 搜索路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from datasets.main import load_dataset

dataset = load_dataset(
    dataset_name='attack_csv',
    data_path='data/attack_data',
    normal_class=0,
    known_outlier_class=1,
    n_known_outlier_classes=1,
    ratio_known_normal=0.0,
    ratio_known_outlier=0.05,
    ratio_pollution=0.0,
    random_state=42
)

train_loader, test_loader = dataset.loaders(batch_size=4)

print("输入维度:", dataset.input_dim)

count = 0
for batch in train_loader:
    x, y, semi_y, idx = batch
    print("x shape:", x.shape)
    print("y:", y.tolist())
    print("semi_y:", semi_y.tolist())
    print("idx:", idx.tolist())
    print("-" * 50)

    count += 1
    if count >= 3:
        break
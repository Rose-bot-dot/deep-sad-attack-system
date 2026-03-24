import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


class AttackCSVDataset:
    def __init__(
        self,
        root,
        normal_class=0,
        known_outlier_class=1,
        ratio_known_normal=0.0,
        ratio_known_outlier=0.0,
        ratio_pollution=0.0
    ):
        self.root = root

        train_path = os.path.join(root, "train.csv")
        test_path = os.path.join(root, "test.csv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到训练文件: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"找不到测试文件: {test_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if "label" not in train_df.columns:
            raise ValueError("train.csv 缺少 label 列")
        if "label" not in test_df.columns:
            raise ValueError("test.csv 缺少 label 列")

        self.X_train = train_df.drop(columns=["label"]).values.astype(np.float32)
        self.y_train = train_df["label"].values.astype(np.int64)

        self.X_test = test_df.drop(columns=["label"]).values.astype(np.float32)
        self.y_test = test_df["label"].values.astype(np.int64)

        self.input_dim = self.X_train.shape[1]

        # 关键修正：
        # 普通标签：0=正常，1=异常
        # Deep SAD 的半监督标签建议用：1=已知正常，-1=已知异常
        self.semi_y_train = np.where(self.y_train == 0, 1, -1).astype(np.int64)
        self.semi_y_test = np.where(self.y_test == 0, 1, -1).astype(np.int64)

        self.train_set = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.int64),
            torch.tensor(self.semi_y_train, dtype=torch.int64),
            torch.arange(len(self.X_train))
        )

        self.test_set = TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.int64),
            torch.tensor(self.semi_y_test, dtype=torch.int64),
            torch.arange(len(self.X_test))
        )

    def loaders(self, batch_size=128, shuffle_train=True, shuffle_test=False, num_workers=0):
        train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            self.test_set,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=num_workers
        )
        return train_loader, test_loader
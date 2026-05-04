import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


REALTIME_FEATURE_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Max",
    "Packet Length Min",
    "Packet Length Mean",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
]


class AttackCSVDataset:
    """
    用于 Deep SAD 攻击检测的 CSV 数据集。

    这个版本强制训练阶段只使用实时监控模块能够提取的 23 个特征。
    并且修复 SimpleImputer 遇到全空列时自动丢列，导致 23 维变 22 维的问题。
    """

    def __init__(
        self,
        root,
        normal_class=0,
        known_outlier_class=1,
        ratio_known_normal=0.0,
        ratio_known_outlier=0.0,
        ratio_pollution=0.0,
    ):
        self.root = root
        self.normal_class = normal_class
        self.known_outlier_class = known_outlier_class

        train_path = os.path.join(root, "train.csv")
        test_path = os.path.join(root, "test.csv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到训练文件: {train_path}")

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"找不到测试文件: {test_path}")

        train_df = pd.read_csv(train_path, low_memory=False)
        test_df = pd.read_csv(test_path, low_memory=False)

        train_df = self._normalize_columns(train_df)
        test_df = self._normalize_columns(test_df)

        if "label" not in train_df.columns:
            raise ValueError("train.csv 缺少 label 或 Label 列")

        if "label" not in test_df.columns:
            raise ValueError("test.csv 缺少 label 或 Label 列")

        self.y_train = self._build_binary_label(train_df["label"])
        self.y_test = self._build_binary_label(test_df["label"])

        train_feature_df = self._build_realtime_feature_df(train_df)
        test_feature_df = self._build_realtime_feature_df(test_df)

        self.feature_columns = REALTIME_FEATURE_COLUMNS.copy()

        # 兼容 train_service.py
        self.data_df = train_feature_df.copy()

        # 缺失值填补
        self.imputer = SimpleImputer(strategy="median")
        X_train_imputed = self.imputer.fit_transform(train_feature_df)
        X_test_imputed = self.imputer.transform(test_feature_df)

        # 关键保险：如果某些 sklearn 版本仍然导致列数异常，直接报出明确错误
        if X_train_imputed.shape[1] != len(self.feature_columns):
            raise RuntimeError(
                f"训练特征维度异常：期望 {len(self.feature_columns)} 维，"
                f"实际 {X_train_imputed.shape[1]} 维。"
                f"请检查是否存在全空列或列名不一致。"
            )

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train_imputed).astype(np.float32)
        self.X_test = self.scaler.transform(X_test_imputed).astype(np.float32)

        self.input_dim = self.X_train.shape[1]

        self.semi_y_train = np.where(self.y_train == 0, 1, -1).astype(np.int64)
        self.semi_y_test = np.where(self.y_test == 0, 1, -1).astype(np.int64)

        self.train_set = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.int64),
            torch.tensor(self.semi_y_train, dtype=torch.int64),
            torch.arange(len(self.X_train)),
        )

        self.test_set = TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.int64),
            torch.tensor(self.semi_y_test, dtype=torch.int64),
            torch.arange(len(self.X_test)),
        )

        print("=" * 60)
        print("[AttackCSVDataset] 数据集加载完成")
        print(f"[AttackCSVDataset] 训练样本数: {len(self.X_train)}")
        print(f"[AttackCSVDataset] 测试样本数: {len(self.X_test)}")
        print(f"[AttackCSVDataset] 输入特征维度: {self.input_dim}")
        print(f"[AttackCSVDataset] 使用特征列: {self.feature_columns}")
        print("=" * 60)

    def _normalize_columns(self, df):
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]

        if "label" not in df.columns and "Label" in df.columns:
            df = df.rename(columns={"Label": "label"})

        return df

    def _build_binary_label(self, label_series):
        """
        标签统一转成：
        0 = 正常
        1 = 异常
        """
        numeric_label = pd.to_numeric(label_series, errors="coerce")

        if numeric_label.notna().all():
            return np.where(
                numeric_label.astype(int).values == int(self.normal_class),
                0,
                1,
            ).astype(np.int64)

        text_label = label_series.astype(str).str.strip().str.lower()

        normal_keywords = {
            "0",
            "normal",
            "benign",
            "benign traffic",
            "normal traffic",
        }

        return np.where(text_label.isin(normal_keywords), 0, 1).astype(np.int64)

    def _build_realtime_feature_df(self, df):
        """
        只保留实时监控能够提取出来的 23 个特征。

        关键修复点：
        1. CSV 缺少的列补 0；
        2. inf / -inf 转 NaN；
        3. 全 NaN 的列强制补 0，防止 SimpleImputer 自动删列；
        4. 最终强制列顺序等于 REALTIME_FEATURE_COLUMNS。
        """
        feature_df = pd.DataFrame(index=df.index)

        for col in REALTIME_FEATURE_COLUMNS:
            if col in df.columns:
                feature_df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                feature_df[col] = 0.0

        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        # 关键修复：如果某列全是 NaN，先补 0，否则 SimpleImputer 可能会直接丢掉这一列
        for col in REALTIME_FEATURE_COLUMNS:
            if feature_df[col].isna().all():
                feature_df[col] = 0.0

        # 剩余局部 NaN 交给 SimpleImputer 处理
        feature_df = feature_df[REALTIME_FEATURE_COLUMNS]

        return feature_df

    def get_preprocessor(self):
        return {
            "imputer": self.imputer,
            "scaler": self.scaler,
        }

    def loaders(self, batch_size=128, shuffle_train=True, shuffle_test=False, num_workers=0):
        train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
        )

        test_loader = DataLoader(
            self.test_set,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=num_workers,
        )

        return train_loader, test_loader
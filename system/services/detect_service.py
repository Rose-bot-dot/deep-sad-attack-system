# system/services/detect_service.py

import os
import sys
import json
import joblib


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from DeepSAD import DeepSAD


class AttackDetector:
    """
    Deep SAD 攻击检测器。

    修复点：
    1. 兼容 CICIDS2017 常见列名；
    2. 不再错误删除 Destination Port；
    3. 缺失特征自动补 0；
    4. 标准化后做 clip，防止实时检测分数爆炸；
    5. 输出 score 保持为模型原始距离分数，但已经通过输入裁剪避免几千万、几亿的问题。
    """

    def __init__(self, model_path="saved_models/attack_model.tar"):
        import torch
        import pandas as pd
        import numpy as np

        self.torch = torch
        self.pd = pd
        self.np = np

        self.model_path = (
            os.path.join(PROJECT_ROOT, model_path)
            if not os.path.isabs(model_path)
            else model_path
        )

        self.preprocessor_path = os.path.join(
            PROJECT_ROOT,
            "saved_models",
            "preprocessor.joblib",
        )

        self.feature_cols_path = os.path.join(
            PROJECT_ROOT,
            "saved_models",
            "feature_columns.json",
        )

        self._load_model()
        self._load_preprocessor_and_columns()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")

        self.model = DeepSAD(eta=1.0)
        self.model.set_network("attack_mlp")
        self.model.load_model(
            model_path=self.model_path,
            load_ae=True,
            map_location="cpu",
        )

        if isinstance(self.model.c, list):
            self.model.c = self.torch.tensor(self.model.c, dtype=self.torch.float32)
        elif not isinstance(self.model.c, self.torch.Tensor):
            self.model.c = self.torch.tensor(self.model.c, dtype=self.torch.float32)

        self.model.net.eval()

    def _load_preprocessor_and_columns(self):
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"预处理器文件不存在：{self.preprocessor_path}")

        if not os.path.exists(self.feature_cols_path):
            raise FileNotFoundError(f"特征列文件不存在：{self.feature_cols_path}")

        self.preprocessor = joblib.load(self.preprocessor_path)

        with open(self.feature_cols_path, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

        if not isinstance(self.feature_columns, list) or len(self.feature_columns) == 0:
            raise ValueError("feature_columns.json 内容无效，未读取到有效特征列")

        print("=" * 60)
        print("[AttackDetector] 模型加载完成")
        print(f"[AttackDetector] 模型路径: {self.model_path}")
        print(f"[AttackDetector] 特征维度: {len(self.feature_columns)}")
        print(f"[AttackDetector] 特征列: {self.feature_columns}")
        print("=" * 60)

    def _normalize_columns(self, df):
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        return df

    def _apply_column_aliases(self, df):
        """
        兼容 CICIDS2017 / CICIDS2018 / 实时监控中可能出现的不同列名。
        """
        df = df.copy()

        alias_map = {
            "Dst Port": "Destination Port",
            "DstPort": "Destination Port",
            "Destination_Port": "Destination Port",

            "Tot Fwd Pkts": "Total Fwd Packets",
            "Total Fwd Packet": "Total Fwd Packets",

            "Tot Bwd Pkts": "Total Backward Packets",
            "Total Bwd Packets": "Total Backward Packets",

            "TotLen Fwd Pkts": "Total Length of Fwd Packets",
            "Total Length of Fwd Packet": "Total Length of Fwd Packets",

            "TotLen Bwd Pkts": "Total Length of Bwd Packets",
            "Total Length of Bwd Packet": "Total Length of Bwd Packets",

            "Fwd Pkt Len Max": "Fwd Packet Length Max",
            "Fwd Pkt Len Min": "Fwd Packet Length Min",
            "Fwd Pkt Len Mean": "Fwd Packet Length Mean",

            "Bwd Pkt Len Max": "Bwd Packet Length Max",
            "Bwd Pkt Len Min": "Bwd Packet Length Min",
            "Bwd Pkt Len Mean": "Bwd Packet Length Mean",

            "Flow Byts/s": "Flow Bytes/s",
            "Flow Pkts/s": "Flow Packets/s",

            "Max Packet Length": "Packet Length Max",
            "Min Packet Length": "Packet Length Min",
            "Packet Length Std": "Packet Length Std",
            "Packet Length Variance": "Packet Length Variance",
            "Average Packet Size": "Packet Length Mean",

            "FIN Flag Cnt": "FIN Flag Count",
            "SYN Flag Cnt": "SYN Flag Count",
            "RST Flag Cnt": "RST Flag Count",
            "PSH Flag Cnt": "PSH Flag Count",
            "ACK Flag Cnt": "ACK Flag Count",
            "URG Flag Cnt": "URG Flag Count",
        }

        for old_name, new_name in alias_map.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]

        return df

    def _derive_missing_packet_length_columns(self, df):
        df = df.copy()

        if "Packet Length Max" not in df.columns:
            candidate_cols = [
                col for col in [
                    "Fwd Packet Length Max",
                    "Bwd Packet Length Max",
                ]
                if col in df.columns
            ]

            if candidate_cols:
                numeric_df = df[candidate_cols].apply(self.pd.to_numeric, errors="coerce")
                df["Packet Length Max"] = numeric_df.max(axis=1)

        if "Packet Length Min" not in df.columns:
            candidate_cols = [
                col for col in [
                    "Fwd Packet Length Min",
                    "Bwd Packet Length Min",
                ]
                if col in df.columns
            ]

            if candidate_cols:
                numeric_df = df[candidate_cols].apply(self.pd.to_numeric, errors="coerce")
                df["Packet Length Min"] = numeric_df.min(axis=1)

        if "Packet Length Mean" not in df.columns:
            candidate_cols = [
                col for col in [
                    "Fwd Packet Length Mean",
                    "Bwd Packet Length Mean",
                ]
                if col in df.columns
            ]

            if candidate_cols:
                numeric_df = df[candidate_cols].apply(self.pd.to_numeric, errors="coerce")
                df["Packet Length Mean"] = numeric_df.mean(axis=1)

        return df

    def _sanitize_dataframe(self, df):
        df = self._normalize_columns(df)
        df = self._apply_column_aliases(df)
        df = self._derive_missing_packet_length_columns(df)

        df = df.drop(columns=["Label", "label"], errors="ignore")

        drop_cols = [
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Timestamp",
        ]

        df = df.drop(columns=drop_cols, errors="ignore")

        for col in df.columns:
            df[col] = self.pd.to_numeric(df[col], errors="coerce")

        df = df.replace([self.np.inf, -self.np.inf], self.np.nan)

        return df

    def _transform_dataframe(self, df, strict=False):
        """
        将输入数据转换为训练时一致的特征格式。

        strict=False：
        缺失列自动补 0，更适合演示系统，避免因为 CSV 列名差异直接崩溃。
        """
        df = df.copy()

        missing_cols = [
            col for col in self.feature_columns
            if col not in df.columns
        ]

        if missing_cols:
            print(f"[AttackDetector] 输入数据缺少特征列，已自动补 0: {missing_cols}")

            if strict:
                raise ValueError(f"输入数据缺少必要特征列: {missing_cols}")

            for col in missing_cols:
                df[col] = 0.0

        df = df[self.feature_columns].copy()

        for col in df.columns:
            df[col] = self.pd.to_numeric(df[col], errors="coerce")

        df = df.replace([self.np.inf, -self.np.inf], self.np.nan)

        X_imputed = self.preprocessor["imputer"].transform(df)
        X_scaled = self.preprocessor["scaler"].transform(X_imputed)

        X_scaled = self.np.nan_to_num(
            X_scaled,
            nan=0.0,
            posinf=5.0,
            neginf=-5.0,
        )

        # 核心修复：防止实时监控特征偏离训练集太远导致 Deep SAD 距离分数爆炸
        X_scaled = self.np.clip(X_scaled, -3.0, 3.0)

        return X_scaled

    def _prepare_input(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"待检测文件不存在：{csv_path}")

        df = self.pd.read_csv(csv_path, low_memory=False)
        df = self._sanitize_dataframe(df)

        return self._transform_dataframe(df, strict=False)

    def _build_dataframe_from_feature_dict(self, feature_dict):
        if not isinstance(feature_dict, dict):
            raise TypeError("feature_dict 必须是字典类型")

        row = {}

        for col in self.feature_columns:
            value = feature_dict.get(col, 0.0)

            try:
                row[col] = float(value)
            except (TypeError, ValueError):
                row[col] = self.np.nan

        df = self.pd.DataFrame([row])
        df = df.replace([self.np.inf, -self.np.inf], self.np.nan)

        return df

    def _predict_array(self, X, threshold=0.03):
        results = []

        for row in X:
            x = self.torch.tensor(row, dtype=self.torch.float32).unsqueeze(0)

            with self.torch.no_grad():
                output = self.model.net(x)
                dist = self.torch.sum((output - self.model.c) ** 2, dim=1)

                score = float(dist.item())

                if score != score:
                    score = 0.0

                if score == float("inf"):
                    score = 999.999

                # 只做显示层保护，避免前端出现几千万、几亿这种难看的数
                display_score = min(score, 999.999)

                pred_label = 1 if score > threshold else 0

                results.append({
                    "score": round(float(display_score), 6),
                    "raw_score": round(float(score), 6),
                    "label": pred_label,
                })

        return results

    def predict_csv(self, csv_path, threshold=0.03):
        X = self._prepare_input(csv_path)
        results = self._predict_array(X, threshold=threshold)

        scores_only = [item["score"] for item in results]

        if scores_only:
            print("==== 检测分数统计 ====")
            print("min:", min(scores_only))
            print("max:", max(scores_only))
            print("mean:", sum(scores_only) / len(scores_only))
            print("前10个分数:", scores_only[:10])
            print("当前阈值:", threshold)

        return results

    def predict_feature_dict(self, feature_dict, threshold=0.03):
        df = self._build_dataframe_from_feature_dict(feature_dict)
        X = self._transform_dataframe(df, strict=False)
        result = self._predict_array(X, threshold=threshold)[0]
        return result

    def predict_feature_list(self, feature_list, threshold=0.03):
        if not isinstance(feature_list, list):
            raise TypeError("feature_list 必须是列表类型")

        rows = []

        for item in feature_list:
            if not isinstance(item, dict):
                raise TypeError("feature_list 中的每个元素都必须是字典")

            row = {}

            for col in self.feature_columns:
                value = item.get(col, 0.0)

                try:
                    row[col] = float(value)
                except (TypeError, ValueError):
                    row[col] = self.np.nan

            rows.append(row)

        df = self.pd.DataFrame(rows)
        X = self._transform_dataframe(df, strict=False)

        return self._predict_array(X, threshold=threshold)
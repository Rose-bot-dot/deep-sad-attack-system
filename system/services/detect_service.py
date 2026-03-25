import os
import sys
import json
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from DeepSAD import DeepSAD


class AttackDetector:
    def __init__(self, model_path='saved_models/attack_model.tar'):
        import torch
        import pandas as pd
        import numpy as np

        self.torch = torch
        self.pd = pd
        self.np = np

        self.model_path = os.path.join(PROJECT_ROOT, model_path) if not os.path.isabs(model_path) else model_path
        self.preprocessor_path = os.path.join(PROJECT_ROOT, 'saved_models', 'preprocessor.joblib')
        self.feature_cols_path = os.path.join(PROJECT_ROOT, 'saved_models', 'feature_columns.json')

        self._load_model()
        self._load_preprocessor_and_columns()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'模型文件不存在：{self.model_path}')

        self.model = DeepSAD(eta=1.0)
        self.model.set_network('attack_mlp')
        self.model.load_model(model_path=self.model_path, load_ae=True, map_location='cpu')

        # 统一 c 的类型，避免后续 dist 计算报错
        if isinstance(self.model.c, list):
            self.model.c = self.torch.tensor(self.model.c, dtype=self.torch.float32)
        elif not isinstance(self.model.c, self.torch.Tensor):
            self.model.c = self.torch.tensor(self.model.c, dtype=self.torch.float32)

    def _load_preprocessor_and_columns(self):
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f'预处理器文件不存在：{self.preprocessor_path}')
        if not os.path.exists(self.feature_cols_path):
            raise FileNotFoundError(f'特征列文件不存在：{self.feature_cols_path}')

        self.preprocessor = joblib.load(self.preprocessor_path)

        with open(self.feature_cols_path, 'r', encoding='utf-8') as f:
            self.feature_columns = json.load(f)

        if not isinstance(self.feature_columns, list) or len(self.feature_columns) == 0:
            raise ValueError('feature_columns.json 内容无效，未读取到有效特征列')

    def _sanitize_dataframe(self, df):
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]

        # 丢弃标签列
        if 'Label' in df.columns:
            df = df.drop(columns=['Label'])
        if 'label' in df.columns:
            df = df.drop(columns=['label'])

        # 丢弃训练阶段通常不参与建模的非通用字段
        drop_cols = [
            'Flow ID',
            'Source IP',
            'Source Port',
            'Destination IP',
            'Destination Port',
            'Timestamp'
        ]
        df = df.drop(columns=drop_cols, errors='ignore')

        # 将所有字段尽量转为数值型
        for col in df.columns:
            df[col] = self.pd.to_numeric(df[col], errors='coerce')

        # 替换 inf
        df = df.replace([self.np.inf, -self.np.inf], self.np.nan)
        return df

    def _transform_dataframe(self, df, strict=True):
        """
        将 dataframe 转换为训练时同口径特征，再走缺失值填补与标准化。
        strict=True  : 缺少训练特征列时直接报错（适用于上传 CSV）
        strict=False : 缺失列自动补 0（适用于实时流特征字典）
        """
        df = df.copy()

        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            if strict:
                raise ValueError(f'输入数据缺少必要特征列: {missing_cols[:10]}')
            for col in missing_cols:
                df[col] = 0.0

        # 只保留训练阶段使用的列，顺序严格一致
        df = df[self.feature_columns].copy()

        # 再次数值化，兜底处理
        for col in df.columns:
            df[col] = self.pd.to_numeric(df[col], errors='coerce')

        df = df.replace([self.np.inf, -self.np.inf], self.np.nan)

        X_imputed = self.preprocessor['imputer'].transform(df)
        X_scaled = self.preprocessor['scaler'].transform(X_imputed)
        return X_scaled

    def _prepare_input(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'待检测文件不存在：{csv_path}')

        df = self.pd.read_csv(csv_path, low_memory=False)
        df = self._sanitize_dataframe(df)
        return self._transform_dataframe(df, strict=True)

    def _build_dataframe_from_feature_dict(self, feature_dict):
        """
        将单条流特征字典转换为 DataFrame。
        未提供的训练列统一补 0，以适配实时监控场景。
        """
        if not isinstance(feature_dict, dict):
            raise TypeError('feature_dict 必须是字典类型')

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
                score = dist.item()

            pred_label = 1 if score > threshold else 0
            results.append({
                'score': score,
                'label': pred_label
            })

        return results

    def predict_csv(self, csv_path, threshold=0.03):
        X = self._prepare_input(csv_path)
        results = self._predict_array(X, threshold=threshold)

        scores_only = [item['score'] for item in results]
        if scores_only:
            print('==== 检测分数统计 ====')
            print('min:', min(scores_only))
            print('max:', max(scores_only))
            print('mean:', sum(scores_only) / len(scores_only))
            print('前10个分数:', scores_only[:10])
            print('当前阈值:', threshold)

        return results

    def predict_feature_dict(self, feature_dict, threshold=0.03):
        """
        供实时监控模块调用：
        输入单条流特征字典，输出单条检测结果。
        """
        df = self._build_dataframe_from_feature_dict(feature_dict)
        X = self._transform_dataframe(df, strict=False)
        result = self._predict_array(X, threshold=threshold)[0]
        return result

    def predict_feature_list(self, feature_list, threshold=0.03):
        """
        可选扩展：
        输入多条流特征字典，批量返回检测结果。
        """
        if not isinstance(feature_list, list):
            raise TypeError('feature_list 必须是列表类型')

        rows = []
        for item in feature_list:
            if not isinstance(item, dict):
                raise TypeError('feature_list 中的每个元素都必须是字典')
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
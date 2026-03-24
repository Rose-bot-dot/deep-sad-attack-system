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

        self.model = DeepSAD(eta=1.0)
        self.model.set_network('attack_mlp')
        self.model.load_model(model_path=model_path, load_ae=True, map_location='cpu')

        if isinstance(self.model.c, list):
            self.model.c = self.torch.tensor(self.model.c, dtype=self.torch.float32)
        elif not isinstance(self.model.c, self.torch.Tensor):
            self.model.c = self.torch.tensor(self.model.c, dtype=self.torch.float32)

        # 加载训练时的预处理器和特征列
        preprocessor_path = os.path.join(PROJECT_ROOT, 'saved_models', 'preprocessor.joblib')
        feature_cols_path = os.path.join(PROJECT_ROOT, 'saved_models', 'feature_columns.json')

        self.preprocessor = joblib.load(preprocessor_path)
        with open(feature_cols_path, 'r', encoding='utf-8') as f:
            self.feature_columns = json.load(f)

    def _prepare_input(self, csv_path):
        df = self.pd.read_csv(csv_path, low_memory=False)
        df.columns = [col.strip() for col in df.columns]

        # 如果是 CICIDS2017 原始文件，有原始标签列，先丢掉
        if 'Label' in df.columns:
            df = df.drop(columns=['Label'])

        if 'label' in df.columns:
            df = df.drop(columns=['label'])

        # 删除训练阶段排除过的非通用字段
        drop_cols = [
            'Flow ID',
            'Source IP',
            'Source Port',
            'Destination IP',
            'Destination Port',
            'Timestamp'
        ]
        df = df.drop(columns=drop_cols, errors='ignore')

        # 只保留训练时使用的列
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f'上传文件缺少必要特征列: {missing_cols[:10]}')

        df = df[self.feature_columns].copy()

        # 替换 inf
        df = df.replace([self.np.inf, -self.np.inf], self.np.nan)

        # 同样的缺失值填补与标准化
        X_imputed = self.preprocessor['imputer'].transform(df)
        X_scaled = self.preprocessor['scaler'].transform(X_imputed)

        return X_scaled

    def predict_csv(self, csv_path, threshold=0.03):
        X = self._prepare_input(csv_path)

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

            scores_only = [item['score'] for item in results]
            if scores_only:
                print("==== 检测分数统计 ====")
                print("min:", min(scores_only))
                print("max:", max(scores_only))
                print("mean:", sum(scores_only) / len(scores_only))
                print("前10个分数:", scores_only[:10])
                print("当前阈值:", threshold)


        return results
import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from DeepSAD import DeepSAD


class ThresholdRecommender:
    def __init__(self, model_path='saved_models/attack_model.tar'):
        # 延迟导入，避免模块一加载就触发 torch DLL 初始化
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.metrics import precision_recall_curve

        self.np = np
        self.pd = pd
        self.torch = torch
        self.precision_recall_curve = precision_recall_curve

        self.model = DeepSAD(eta=1.0)
        self.model.set_network('attack_mlp')
        self.model.load_model(model_path=model_path, load_ae=True, map_location='cpu')

        # 修复 c 的类型
        if isinstance(self.model.c, list):
            self.model.c = self.torch.tensor(self.model.c, dtype=self.torch.float32)
        elif not isinstance(self.model.c, self.torch.Tensor):
            self.model.c = self.torch.tensor(self.model.c, dtype=self.torch.float32)

    def _load_csv(self, csv_path):
        df = self.pd.read_csv(csv_path)

        if 'label' not in df.columns:
            raise ValueError("用于推荐阈值的数据必须包含 label 列")

        X = df.drop(columns=['label']).values.astype(self.np.float32)
        y = df['label'].values.astype(self.np.int64)
        return X, y

    def _predict_scores(self, X):
        scores = []

        for row in X:
            x = self.torch.tensor(row, dtype=self.torch.float32).unsqueeze(0)

            with self.torch.no_grad():
                output = self.model.net(x)
                dist = self.torch.sum((output - self.model.c) ** 2, dim=1)
                score = dist.item()

            scores.append(score)

        return self.np.array(scores, dtype=self.np.float32)

    def recommend_by_pr_f1(self, csv_path):
        """
        基于 PR 曲线和 F1 最大值推荐阈值
        """
        X, y_true = self._load_csv(csv_path)
        scores = self._predict_scores(X)

        precision, recall, thresholds = self.precision_recall_curve(y_true, scores)

        # precision 和 recall 的长度比 thresholds 长 1，需要对齐
        precision_valid = precision[:-1]
        recall_valid = recall[:-1]

        f1_scores = []
        for p, r in zip(precision_valid, recall_valid):
            if p + r == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * p * r / (p + r))

        f1_scores = self.np.array(f1_scores)

        best_idx = self.np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        return {
            'method': 'pr_f1',
            'recommended_threshold': float(best_threshold),
            'best_precision': float(precision_valid[best_idx]),
            'best_recall': float(recall_valid[best_idx]),
            'best_f1': float(f1_scores[best_idx]),
            'all_scores': scores.tolist()
        }

    def save_recommended_threshold(self, csv_path, save_path='saved_models/threshold.json'):
        result = self.recommend_by_pr_f1(csv_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        return result
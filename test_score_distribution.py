import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from DeepSAD import DeepSAD


def load_model():
    import torch

    model = DeepSAD(eta=1.0)
    model.set_network('attack_mlp')
    model.load_model(
        model_path='saved_models/attack_model.tar',
        load_ae=True,
        map_location='cpu'
    )

    if isinstance(model.c, list):
        model.c = torch.tensor(model.c, dtype=torch.float32)
    elif not isinstance(model.c, torch.Tensor):
        model.c = torch.tensor(model.c, dtype=torch.float32)

    return model, torch


def main():
    import pandas as pd
    import numpy as np

    df = pd.read_csv('data/attack_data/test.csv')

    if 'label' not in df.columns:
        raise ValueError('test.csv 中缺少 label 列')

    X = df.drop(columns=['label']).values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    model, torch = load_model()

    scores = []
    for row in X:
        x = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model.net(x)
            dist = torch.sum((output - model.c) ** 2, dim=1)
            score = dist.item()
        scores.append(score)

    scores = np.array(scores)

    normal_scores = scores[y == 0]
    attack_scores = scores[y == 1]

    print("正常样本数量:", len(normal_scores))
    print("异常样本数量:", len(attack_scores))

    if len(normal_scores) > 0:
        print("正常样本平均分数:", float(normal_scores.mean()))
        print("正常样本最小分数:", float(normal_scores.min()))
        print("正常样本最大分数:", float(normal_scores.max()))

    if len(attack_scores) > 0:
        print("异常样本平均分数:", float(attack_scores.mean()))
        print("异常样本最小分数:", float(attack_scores.min()))
        print("异常样本最大分数:", float(attack_scores.max()))

    if len(normal_scores) > 0 and len(attack_scores) > 0:
        print("正常样本95分位数:", float(np.percentile(normal_scores, 95)))
        print("异常样本5分位数:", float(np.percentile(attack_scores, 5)))


if __name__ == '__main__':
    main()
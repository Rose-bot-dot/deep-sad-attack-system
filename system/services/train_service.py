# system/services/train_service.py

import os
import sys
import json
import joblib
import numpy as np

from sklearn.metrics import precision_recall_curve


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from datasets.main import load_dataset
from DeepSAD import DeepSAD


def _abs_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _save_feature_columns_for_network(feature_columns, saved_models_dir):
    """
    networks/main.py 中的 attack_mlp 会从 saved_models/feature_columns.json 读取输入维度。
    所以必须在 deep_sad.set_network('attack_mlp') 之前先保存 feature_columns.json。
    """
    os.makedirs(saved_models_dir, exist_ok=True)

    project_feature_file = os.path.join(saved_models_dir, "feature_columns.json")

    with open(project_feature_file, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=4)

    # 兼容从不同工作目录启动 Flask 的情况
    cwd_saved_models_dir = os.path.join(os.getcwd(), "saved_models")
    cwd_feature_file = os.path.join(cwd_saved_models_dir, "feature_columns.json")

    if os.path.abspath(cwd_feature_file) != os.path.abspath(project_feature_file):
        os.makedirs(cwd_saved_models_dir, exist_ok=True)
        with open(cwd_feature_file, "w", encoding="utf-8") as f:
            json.dump(feature_columns, f, ensure_ascii=False, indent=4)

    return project_feature_file


def _recommend_threshold_from_test_scores(deep_sad, dataset):
    """
    用 test.csv 的测试分数推荐阈值。
    如果 test.csv 同时包含正常和异常标签，则使用 PR-F1 最优阈值。
    如果只有单一类别，则使用 95 分位数兜底。
    """
    deep_sad.test(
        dataset=dataset,
        device="cpu",
        n_jobs_dataloader=0,
    )

    test_scores = deep_sad.results.get("test_scores", [])

    if not test_scores:
        return {
            "method": "fallback_empty_scores",
            "recommended_threshold": 0.03,
            "best_precision": 0.0,
            "best_recall": 0.0,
            "best_f1": 0.0,
            "all_scores": [],
        }

    labels = np.array([item[1] for item in test_scores], dtype=np.int64)
    scores = np.array([item[2] for item in test_scores], dtype=np.float32)

    unique_labels = np.unique(labels)

    # 如果只有正常或只有异常，无法计算 PR 曲线，用 95 分位数兜底
    if len(unique_labels) < 2:
        threshold = float(np.percentile(scores, 95))
        return {
            "method": "percentile_95_single_class",
            "recommended_threshold": threshold,
            "best_precision": 0.0,
            "best_recall": 0.0,
            "best_f1": 0.0,
            "all_scores": scores.tolist(),
        }

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    if len(thresholds) == 0:
        threshold = float(np.percentile(scores, 95))
        return {
            "method": "percentile_95_no_thresholds",
            "recommended_threshold": threshold,
            "best_precision": 0.0,
            "best_recall": 0.0,
            "best_f1": 0.0,
            "all_scores": scores.tolist(),
        }

    precision_valid = precision[:-1]
    recall_valid = recall[:-1]

    f1_scores = []
    for p, r in zip(precision_valid, recall_valid):
        if p + r == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * p * r / (p + r))

    f1_scores = np.array(f1_scores, dtype=np.float32)

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])

    return {
        "method": "pr_f1",
        "recommended_threshold": best_threshold,
        "best_precision": float(precision_valid[best_idx]),
        "best_recall": float(recall_valid[best_idx]),
        "best_f1": float(f1_scores[best_idx]),
        "all_scores": scores.tolist(),
    }


def train_attack_model(
    pretrain_epochs=10,
    train_epochs=10,
    lr=0.001,
    data_path="data/attack_data",
):
    data_path_abs = _abs_path(data_path)

    saved_models_dir = os.path.join(PROJECT_ROOT, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)

    model_path = os.path.join(saved_models_dir, "attack_model.tar")
    feature_columns_file = os.path.join(saved_models_dir, "feature_columns.json")
    preprocessor_file = os.path.join(saved_models_dir, "preprocessor.joblib")
    threshold_file = os.path.join(saved_models_dir, "threshold.json")

    # 1. 加载数据集
    dataset = load_dataset(
        dataset_name="attack_csv",
        data_path=data_path_abs,
        normal_class=0,
        known_outlier_class=1,
        n_known_outlier_classes=1,
        ratio_known_normal=0.0,
        ratio_known_outlier=0.05,
        ratio_pollution=0.0,
        random_state=42,
    )

    # 2. 保存训练特征列
    if hasattr(dataset, "feature_columns"):
        feature_columns = dataset.feature_columns
    elif hasattr(dataset, "data_df"):
        feature_columns = dataset.data_df.columns.tolist()
    else:
        raise AttributeError("dataset 缺少 feature_columns 或 data_df，无法保存训练特征列")

    _save_feature_columns_for_network(feature_columns, saved_models_dir)

    # 3. 初始化 Deep SAD
    deep_sad = DeepSAD(eta=1.0)
    deep_sad.set_network("attack_mlp")

    # 4. 预训练
    deep_sad.pretrain(
        dataset=dataset,
        optimizer_name="adam",
        lr=lr,
        n_epochs=pretrain_epochs,
        lr_milestones=(),
        batch_size=128,
        weight_decay=1e-6,
        device="cpu",
        n_jobs_dataloader=0,
    )

    # 5. 正式训练
    deep_sad.train(
        dataset=dataset,
        optimizer_name="adam",
        lr=lr,
        n_epochs=train_epochs,
        lr_milestones=(),
        batch_size=128,
        weight_decay=1e-6,
        device="cpu",
        n_jobs_dataloader=0,
    )

    # 6. 保存模型
    deep_sad.save_model(model_path, save_ae=True)

    # 7. 保存特征列
    with open(feature_columns_file, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=4)

    # 8. 保存预处理器
    if hasattr(dataset, "get_preprocessor"):
        preprocessor = dataset.get_preprocessor()
    elif hasattr(dataset, "imputer") and hasattr(dataset, "scaler"):
        preprocessor = {
            "imputer": dataset.imputer,
            "scaler": dataset.scaler,
        }
    else:
        raise AttributeError("dataset 缺少预处理器，无法保存 preprocessor.joblib")

    joblib.dump(preprocessor, preprocessor_file)

    # 9. 推荐阈值并保存
    threshold_result = _recommend_threshold_from_test_scores(deep_sad, dataset)

    with open(threshold_file, "w", encoding="utf-8") as f:
        json.dump(threshold_result, f, ensure_ascii=False, indent=4)

    return {
        "model_path": model_path,
        "recommended_threshold": threshold_result["recommended_threshold"],
        "pretrain_epochs": pretrain_epochs,
        "train_epochs": train_epochs,
        "lr": lr,
        "data_path": data_path,
    }
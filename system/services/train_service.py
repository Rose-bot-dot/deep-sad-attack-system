# system/services/train_service.py

import os
import sys
import json
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from datasets.main import load_dataset
from DeepSAD import DeepSAD
from system.services.threshold_service import ThresholdRecommender

def train_attack_model(pretrain_epochs=10, train_epochs=10, lr=0.001, data_path='data/attack_data'):
    # 加载数据集
    dataset = load_dataset(
        dataset_name='attack_csv',
        data_path=data_path,
        normal_class=0,
        known_outlier_class=1,
        n_known_outlier_classes=1,
        ratio_known_normal=0.0,
        ratio_known_outlier=0.05,
        ratio_pollution=0.0,
        random_state=42
    )

    # 初始化 DeepSAD
    deep_sad = DeepSAD(eta=1.0)
    deep_sad.set_network('attack_mlp')

    # 1) 预训练
    deep_sad.pretrain(
        dataset=dataset,
        optimizer_name='adam',
        lr=lr,
        n_epochs=pretrain_epochs,
        lr_milestones=(),
        batch_size=128,
        weight_decay=1e-6,
        device='cpu',
        n_jobs_dataloader=0
    )

    # 2) 正式训练
    deep_sad.train(
        dataset=dataset,
        optimizer_name='adam',
        lr=lr,
        n_epochs=train_epochs,
        lr_milestones=(),
        batch_size=128,
        weight_decay=1e-6,
        device='cpu',
        n_jobs_dataloader=0
    )

    # ========================= #
    # 保存模型
    # ========================= #
    os.makedirs('saved_models', exist_ok=True)
    model_path = 'saved_models/attack_model.tar'
    deep_sad.save_model(model_path, save_ae=True)

    # ========================= #
    # 保存预处理器 & 特征列
    # ========================= #
    # dataset 包含 DataLoader，我们取 raw_df 中的 columns 作为训练列
    feature_columns = dataset.data_df.columns.tolist()  # ⚠ 如果你的 dataset 名称不同，请调整

    # 保存特征列
    feature_columns_file = 'saved_models/feature_columns.json'
    with open(feature_columns_file, 'w', encoding='utf-8') as f:
        json.dump(feature_columns, f, ensure_ascii=False)

    # 保存预处理器
    preprocessor = deep_sad.get_preprocessor()  # 假设 DeepSAD 支持获得 imputer + scaler
    preprocessor_file = 'saved_models/preprocessor.joblib'
    joblib.dump(preprocessor, preprocessor_file)

    # ========================= #
    # 推荐阈值
    # ========================= #
    recommender = ThresholdRecommender(model_path=model_path)
    threshold_result = recommender.save_recommended_threshold(
        csv_path=os.path.join(data_path, 'val.csv'),
        save_path='saved_models/threshold.json'
    )

    return {
        'model_path': model_path,
        'recommended_threshold': threshold_result['recommended_threshold'],
        'pretrain_epochs': pretrain_epochs,
        'train_epochs': train_epochs,
        'lr': lr,
        'data_path': data_path
    }
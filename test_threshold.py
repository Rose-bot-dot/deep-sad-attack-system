import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from system.services.threshold_service import ThresholdRecommender

recommender = ThresholdRecommender(model_path='saved_models/attack_model.tar')

result = recommender.save_recommended_threshold(
    csv_path='data/attack_data/val.csv',
    save_path='saved_models/threshold.json'
)

print("====== PR + F1 推荐阈值结果 ======")
print("推荐阈值:", result['recommended_threshold'])
print("最佳 Precision:", result['best_precision'])
print("最佳 Recall:", result['best_recall'])
print("最佳 F1:", result['best_f1'])
print("已保存到 saved_models/threshold.json")
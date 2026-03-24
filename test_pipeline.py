from system.services.train_service import train_attack_model
from system.services.detect_service import AttackDetector

# 1. 训练模型
model_path = train_attack_model()

# 2. 加载模型
detector = AttackDetector(model_path=model_path)

# 3. 检测测试数据
results = detector.predict_csv('data/attack_data/test.csv', threshold=1)

# 4. 输出汇总结果
total = len(results)
anomaly_count = sum(item['label'] for item in results)
normal_count = total - anomaly_count
scores = [item['score'] for item in results]

print("总样本数:", total)
print("正常样本数:", normal_count)
print("异常样本数:", anomaly_count)
print("平均异常分数:", sum(scores) / total if total > 0 else 0)
print("最大异常分数:", max(scores) if scores else 0)
print("最小异常分数:", min(scores) if scores else 0)
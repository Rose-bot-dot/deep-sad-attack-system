import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from system.services.detect_service import AttackDetector

detector = AttackDetector(model_path='saved_models/attack_model.tar')
results = detector.predict_csv('data/attack_data/test.csv', threshold=1.0)

print("前5条结果:")
for item in results[:5]:
    print(item)
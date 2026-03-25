from datetime import datetime
from system.models.db import db


class DetectRecord(db.Model):
    __tablename__ = 'detect_record'

    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)

    total_count = db.Column(db.Integer, default=0)
    normal_count = db.Column(db.Integer, default=0)
    anomaly_count = db.Column(db.Integer, default=0)

    avg_score = db.Column(db.Float, default=0.0)
    max_score = db.Column(db.Float, default=0.0)
    min_score = db.Column(db.Float, default=0.0)

    # 新增：记录归属用户
    user_id = db.Column(db.Integer, nullable=True)
    username = db.Column(db.String(64), nullable=True)

    create_time = db.Column(db.DateTime, default=datetime.now)
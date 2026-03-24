from datetime import datetime
from system.models.db import db


class DetectRecord(db.Model):
    __tablename__ = 'detect_record'

    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    total_count = db.Column(db.Integer, nullable=False)
    normal_count = db.Column(db.Integer, nullable=False)
    anomaly_count = db.Column(db.Integer, nullable=False)
    avg_score = db.Column(db.Float, nullable=False)
    max_score = db.Column(db.Float, nullable=False)
    min_score = db.Column(db.Float, nullable=False)
    create_time = db.Column(db.DateTime, default=datetime.utcnow)
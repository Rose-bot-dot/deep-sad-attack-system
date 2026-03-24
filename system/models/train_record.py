from datetime import datetime
from system.models.db import db


class TrainRecord(db.Model):
    __tablename__ = 'train_record'

    id = db.Column(db.Integer, primary_key=True)
    model_path = db.Column(db.String(255), nullable=False)
    recommended_threshold = db.Column(db.Float, nullable=False)
    train_epochs = db.Column(db.Integer, nullable=False)
    pretrain_epochs = db.Column(db.Integer, nullable=False)
    data_path = db.Column(db.String(255), nullable=False)
    create_time = db.Column(db.DateTime, default=datetime.utcnow)
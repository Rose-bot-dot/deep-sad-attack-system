from system.models.db import db


class DetectDetail(db.Model):
    __tablename__ = 'detect_detail'

    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.Integer, nullable=False)
    sample_index = db.Column(db.Integer, nullable=False)
    score = db.Column(db.Float, nullable=False)
    label = db.Column(db.Integer, nullable=False)
import os
import sys
import json
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from system.services.detect_service import AttackDetector
from system.services.train_service import train_attack_model

from system.models.db import db
from system.models.user import User
from system.models.detect_record import DetectRecord
from system.models.detect_detail import DetectDetail
from system.models.train_record import TrainRecord

app = Flask(__name__, template_folder='system/templates')
app.config['SECRET_KEY'] = 'deep-sad-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attack_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '请先登录后再访问该页面'
login_manager.login_message_category = 'error'

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_recommended_threshold():
    threshold_file = os.path.join(PROJECT_ROOT, 'saved_models', 'threshold.json')

    if not os.path.exists(threshold_file):
        return 0.03

    try:
        with open(threshold_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return float(data.get('recommended_threshold', 0.03))
    except Exception:
        return 0.03


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def admin_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))

        if current_user.role != 'admin':
            return '无权限访问，该功能仅管理员可用'

        return func(*args, **kwargs)
    return wrapper


with app.app_context():
    db.create_all()

    # 初始化默认管理员
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            password=generate_password_hash('123456'),
            role='admin'
        )
        db.session.add(admin)
        db.session.commit()


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if not username or not password or not confirm_password:
            flash('请填写完整信息', 'error')
            return render_template('register.html')

        if len(username) < 3:
            flash('用户名长度不能少于 3 位', 'error')
            return render_template('register.html')

        if len(password) < 6:
            flash('密码长度不能少于 6 位', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('两次输入的密码不一致', 'error')
            return render_template('register.html')

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('用户名已存在，请更换', 'error')
            return render_template('register.html')

        new_user = User(
            username=username,
            password=generate_password_hash(password),
            role='user'
        )
        db.session.add(new_user)
        db.session.commit()

        flash('注册成功，请登录', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('请输入用户名和密码', 'error')
            return render_template('login.html')

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash(f'欢迎你，{user.username}，登录成功', 'success')
            return redirect(url_for('index'))

        flash('用户名或密码错误', 'error')
        return render_template('login.html')

    return render_template('login.html')



@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('你已成功退出登录', 'success')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    recommended_threshold = load_recommended_threshold()
    return render_template(
        'index.html',
        username=current_user.username,
        role=current_user.role,
        recommended_threshold=recommended_threshold
    )


@app.route('/detect', methods=['POST'])
@login_required
def detect():
    if 'file' not in request.files:
        return '没有上传文件'

    file = request.files['file']
    if file.filename == '':
        return '文件名为空'

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    threshold_str = request.form.get('threshold', str(load_recommended_threshold()))
    try:
        threshold = float(threshold_str)
    except ValueError:
        threshold = load_recommended_threshold()

    detector = AttackDetector(model_path='saved_models/attack_model.tar')
    results = detector.predict_csv(file_path, threshold=threshold)

    total = len(results)
    anomaly_count = sum(item['label'] for item in results)
    normal_count = total - anomaly_count
    scores = [item['score'] for item in results]

    avg_score = sum(scores) / total if total > 0 else 0
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0

    record = DetectRecord(
        file_name=filename,
        total_count=total,
        normal_count=normal_count,
        anomaly_count=anomaly_count,
        avg_score=avg_score,
        max_score=max_score,
        min_score=min_score
    )
    db.session.add(record)
    db.session.commit()

    for idx, item in enumerate(results, start=1):
        detail = DetectDetail(
            record_id=record.id,
            sample_index=idx,
            score=item['score'],
            label=item['label']
        )
        db.session.add(detail)

    db.session.commit()

    return render_template(
        'result.html',
        filename=filename,
        threshold=threshold,
        total=total,
        anomaly_count=anomaly_count,
        normal_count=normal_count,
        avg_score=avg_score,
        max_score=max_score,
        min_score=min_score,
        results=results
    )


@app.route('/records')
@login_required
def records():
    all_records = DetectRecord.query.order_by(DetectRecord.create_time.desc()).all()
    return render_template(
        'records.html',
        records=all_records,
        username=current_user.username,
        role=current_user.role
    )


@app.route('/record/<int:record_id>')
@login_required
def record_detail(record_id):
    record = DetectRecord.query.get_or_404(record_id)
    details = DetectDetail.query.filter_by(record_id=record_id).order_by(DetectDetail.sample_index.asc()).all()

    return render_template(
        'record_detail.html',
        record=record,
        details=details,
        username=current_user.username
    )


@app.route('/delete_record/<int:record_id>', methods=['POST'])
@login_required
@admin_required
def delete_record(record_id):
    record = DetectRecord.query.get_or_404(record_id)

    DetectDetail.query.filter_by(record_id=record_id).delete()
    db.session.delete(record)
    db.session.commit()

    return redirect(url_for('records'))


@app.route('/train', methods=['POST'])
@login_required
@admin_required
def train():
    pretrain_epochs_str = request.form.get('pretrain_epochs', '10')
    train_epochs_str = request.form.get('train_epochs', '10')
    lr_str = request.form.get('lr', '0.001')
    data_path = request.form.get('data_path', 'data/attack_data')

    try:
        pretrain_epochs = int(pretrain_epochs_str)
    except ValueError:
        pretrain_epochs = 10

    try:
        train_epochs = int(train_epochs_str)
    except ValueError:
        train_epochs = 10

    try:
        lr = float(lr_str)
    except ValueError:
        lr = 0.001

    train_result = train_attack_model(
        pretrain_epochs=pretrain_epochs,
        train_epochs=train_epochs,
        lr=lr,
        data_path=data_path
    )

    record = TrainRecord(
        model_path=train_result['model_path'],
        recommended_threshold=train_result['recommended_threshold'],
        train_epochs=train_result['train_epochs'],
        pretrain_epochs=train_result['pretrain_epochs'],
        data_path=train_result['data_path']
    )
    db.session.add(record)
    db.session.commit()

    return render_template(
        'train_result.html',
        model_path=train_result['model_path'],
        recommended_threshold=train_result['recommended_threshold'],
        train_epochs=train_result['train_epochs'],
        pretrain_epochs=train_result['pretrain_epochs'],
        lr=train_result['lr'],
        data_path=train_result['data_path']
    )


@app.route('/train_records')
@login_required
@admin_required
def train_records():
    records = TrainRecord.query.order_by(TrainRecord.create_time.desc()).all()
    return render_template(
        'train_records.html',
        records=records,
        username=current_user.username
    )


@app.route('/delete_train_record/<int:record_id>', methods=['POST'])
@login_required
@admin_required
def delete_train_record(record_id):
    record = TrainRecord.query.get_or_404(record_id)
    db.session.delete(record)
    db.session.commit()
    return redirect(url_for('train_records'))


if __name__ == '__main__':
    app.run(debug=True)
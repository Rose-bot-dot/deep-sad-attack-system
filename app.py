import os
import sys
import json
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from system.services.detect_service import AttackDetector
from system.services.train_service import train_attack_model
from system.services.monitor_service import LiveTrafficMonitor
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

SUMMARY_DIR = os.path.join(PROJECT_ROOT, 'runtime', 'summary')
SUMMARY_FILE = os.path.join(SUMMARY_DIR, 'last_detection_summary.json')

# 全局实时监控对象
monitor = None


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


def get_monitor():
    """
    延迟初始化监控对象，避免应用启动时因模型文件不存在导致直接报错。
    """
    global monitor
    if monitor is None:
        monitor = LiveTrafficMonitor(
            model_path='saved_models/attack_model.tar',
            threshold=load_recommended_threshold()
        )
    return monitor


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def calc_risk_level(anomaly_rate):
    """
    展示层风险等级，不改变模型判断。
    anomaly_rate 是 0-1 之间的小数。
    """
    if anomaly_rate >= 0.5:
        return '高风险'
    elif anomaly_rate >= 0.15:
        return '中风险'
    else:
        return '低风险'


def build_detection_summary(
    filename='检测结果',
    threshold=0,
    total=0,
    normal_count=0,
    anomaly_count=0,
    avg_score=0,
    max_score=0,
    min_score=0,
    results=None,
    source='CSV文件检测'
):
    """
    将一次检测结果整理成大屏页面需要的数据。
    results 推荐格式：
    [
        {'score': 0.12, 'label': 0},
        {'score': 1.25, 'label': 1}
    ]
    """
    results = results or []

    clean_records = []
    score_list = []

    threshold = safe_float(threshold, 0)

    for item in results:
        if isinstance(item, dict):
            score = safe_float(item.get('score', 0))
            label_value = item.get('label', None)

            if label_value is None:
                label = 1 if score > threshold else 0
            else:
                label = safe_int(label_value, 0)
        else:
            score = safe_float(getattr(item, 'score', 0))
            label_value = getattr(item, 'label', None)

            if label_value is None:
                label = 1 if score > threshold else 0
            else:
                label = safe_int(label_value, 0)

        score = round(score, 6)
        label = 1 if label == 1 else 0

        score_list.append(score)
        clean_records.append({
            'score': score,
            'label': label
        })

    total = safe_int(total, len(clean_records))
    if total <= 0:
        total = len(clean_records)

    anomaly_count = safe_int(anomaly_count, 0)
    normal_count = safe_int(normal_count, 0)

    if clean_records:
        anomaly_count = sum(1 for item in clean_records if item['label'] == 1)
        normal_count = len(clean_records) - anomaly_count
        total = len(clean_records)

    anomaly_rate = anomaly_count / total if total > 0 else 0
    anomaly_rate_percent = round(anomaly_rate * 100, 2)

    if score_list:
        avg_score = sum(score_list) / len(score_list)
        max_score = max(score_list)
        min_score = min(score_list)

    summary = {
        'filename': filename,
        'source': source,
        'detect_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'threshold': round(safe_float(threshold), 6),
        'total': total,
        'normal_count': normal_count,
        'anomaly_count': anomaly_count,
        'anomaly_rate': round(anomaly_rate, 4),
        'anomaly_rate_percent': anomaly_rate_percent,
        'avg_score': round(safe_float(avg_score), 6),
        'max_score': round(safe_float(max_score), 6),
        'min_score': round(safe_float(min_score), 6),
        'risk_level': calc_risk_level(anomaly_rate),
        'score_list': score_list[:120],
        'recent_records': clean_records[:30]
    }

    return summary


def save_detection_summary(summary):
    """
    保存最近一次检测总结，供总结大屏读取。
    """
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def load_detection_summary():
    """
    读取最近一次检测总结。
    """
    if not os.path.exists(SUMMARY_FILE):
        return None

    try:
        with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def build_summary_from_monitor(live_monitor):
    """
    从实时监控缓存事件中生成总结大屏数据。
    兼容 event 中可能出现的 score / anomaly_score / label / is_anomaly 字段。
    """
    try:
        status = live_monitor.status()
    except Exception:
        status = {}

    threshold = safe_float(status.get('threshold', load_recommended_threshold()))

    try:
        events = live_monitor.get_events(limit=1000)
    except Exception:
        events = []

    clean_results = []

    for event in events:
        if not isinstance(event, dict):
            continue

        score = safe_float(
            event.get(
                'score',
                event.get(
                    'anomaly_score',
                    event.get('distance', 0)
                )
            ),
            0
        )

        if 'label' in event:
            label = safe_int(event.get('label'), 0)
        elif 'is_anomaly' in event:
            label = 1 if event.get('is_anomaly') else 0
        elif 'result' in event:
            result_text = str(event.get('result'))
            label = 1 if ('异常' in result_text or 'anomaly' in result_text.lower()) else 0
        else:
            label = 1 if score > threshold else 0

        clean_results.append({
            'score': score,
            'label': label
        })

    summary = build_detection_summary(
        filename='实时监控检测结果',
        threshold=threshold,
        results=clean_results,
        source='实时监控'
    )

    return summary


def get_current_monitor_summary():
    """
    如果实时监控对象已经存在，则优先生成实时监控总结；
    如果没有实时监控数据，则返回 None。
    """
    global monitor

    if monitor is None:
        return None

    try:
        summary = build_summary_from_monitor(monitor)
        if summary and summary.get('total', 0) > 0:
            return summary
    except Exception:
        return None

    return None


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


@app.route('/summary')
@login_required
def summary():
    """
    检测总结大屏：
    优先展示实时监控缓存结果；
    如果没有实时监控结果，则展示最近一次 CSV 检测结果。
    """
    summary_data = get_current_monitor_summary()

    if summary_data is None:
        summary_data = load_detection_summary()

    return render_template(
        'summary.html',
        summary=summary_data,
        username=current_user.username,
        role=current_user.role
    )


@app.route('/live-dashboard')
@login_required
def live_dashboard():
    """
    保留原来的大屏地址，但页面内容改为“检测总结大屏”。
    这样你原来首页如果有 live-dashboard 入口，也不用改。
    """
    summary_data = get_current_monitor_summary()

    if summary_data is None:
        summary_data = load_detection_summary()

    return render_template(
        'summary.html',
        summary=summary_data,
        username=current_user.username,
        role=current_user.role
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
        min_score=min_score,
        user_id=current_user.id,
        username=current_user.username
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

    # 保存最近一次检测总结，供总结大屏展示
    summary_data = build_detection_summary(
        filename=filename,
        threshold=threshold,
        total=total,
        normal_count=normal_count,
        anomaly_count=anomaly_count,
        avg_score=avg_score,
        max_score=max_score,
        min_score=min_score,
        results=results,
        source='CSV文件检测'
    )
    save_detection_summary(summary_data)

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
    if current_user.role == 'admin':
        all_records = DetectRecord.query.order_by(DetectRecord.create_time.desc()).all()
    else:
        all_records = DetectRecord.query.filter_by(user_id=current_user.id).order_by(
            DetectRecord.create_time.desc()
        ).all()

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

    if current_user.role != 'admin' and record.user_id != current_user.id:
        return '无权限查看该检测记录'

    details = DetectDetail.query.filter_by(record_id=record_id).order_by(
        DetectDetail.sample_index.asc()
    ).all()

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


# =========================
# 实时监控接口
# =========================

@app.route('/monitor/interfaces')
@login_required
def monitor_interfaces():
    """
    获取当前可用网卡列表。
    会调用 monitor_service.py 中的 list_interfaces()，
    由后端短时间采样每个网卡，默认推荐当前流量最高的网卡。
    """
    try:
        live_monitor = get_monitor()

        if not hasattr(live_monitor, 'list_interfaces'):
            return jsonify({
                'success': False,
                'message': '当前 monitor_service.py 缺少 list_interfaces 方法，请先替换新版 monitor_service.py',
                'interfaces': [],
                'default_iface': None
            })

        data = live_monitor.list_interfaces(sample_seconds=0.6)

        return jsonify({
            'success': True,
            'interfaces': data.get('interfaces', []),
            'default_iface': data.get('default_iface')
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取网卡列表失败：{str(e)}',
            'interfaces': [],
            'default_iface': None
        })


@app.route('/monitor/start', methods=['POST'])
@login_required
def monitor_start():
    try:
        iface = request.form.get('iface', '').strip()

        # 处理前端的“自动选择网卡”
        if (
            iface == ''
            or iface.lower() == 'auto'
            or iface in ['自动', '自动选择', '自动选择网卡']
        ):
            iface = None

        threshold_str = request.form.get('threshold', str(load_recommended_threshold()))

        try:
            threshold = float(threshold_str)
        except ValueError:
            threshold = load_recommended_threshold()

        live_monitor = get_monitor()
        live_monitor.start(iface=iface, threshold=threshold)

        return jsonify({
            'success': True,
            'message': f'实时监控已启动，当前网卡：{live_monitor.iface}'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'启动监控失败：{str(e)}'
        })


@app.route('/monitor/stop', methods=['POST'])
@login_required
def monitor_stop():
    try:
        live_monitor = get_monitor()
        live_monitor.stop()

        # 停止监控后，保存一次实时监控总结
        try:
            summary_data = build_summary_from_monitor(live_monitor)
            if summary_data and summary_data.get('total', 0) > 0:
                save_detection_summary(summary_data)
        except Exception:
            pass

        return jsonify({
            'success': True,
            'message': '实时监控已停止，可进入检测总结大屏查看本次结果',
            'summary_url': url_for('summary')
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'停止监控失败：{str(e)}'
        })


@app.route('/monitor/status')
@login_required
def monitor_status():
    try:
        live_monitor = get_monitor()

        return jsonify({
            'success': True,
            'data': live_monitor.status()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取监控状态失败：{str(e)}'
        })


@app.route('/monitor/data')
@login_required
def monitor_data():
    try:
        live_monitor = get_monitor()

        summary_data = None
        try:
            summary_data = build_summary_from_monitor(live_monitor)
        except Exception:
            summary_data = None

        return jsonify({
            'success': True,
            'status': live_monitor.status(),
            'events': live_monitor.get_events(limit=50),
            'summary': summary_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取监控数据失败：{str(e)}',
            'status': {
                'running': False,
                'iface': None,
                'threshold': load_recommended_threshold(),
                'cached_events': 0
            },
            'events': [],
            'summary': None
        })


@app.route('/users')
@login_required
@admin_required
def user_manage():
    users = User.query.order_by(User.id.asc()).all()
    return render_template(
        'user_manage.html',
        users=users,
        username=current_user.username
    )


@app.route('/users/create', methods=['POST'])
@login_required
@admin_required
def create_user():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    role = request.form.get('role', 'user').strip()

    if not username or not password:
        flash('用户名和密码不能为空', 'error')
        return redirect(url_for('user_manage'))

    if len(username) < 3:
        flash('用户名长度不能少于 3 位', 'error')
        return redirect(url_for('user_manage'))

    if len(password) < 6:
        flash('密码长度不能少于 6 位', 'error')
        return redirect(url_for('user_manage'))

    if role not in ['admin', 'user']:
        role = 'user'

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        flash('用户名已存在，请更换', 'error')
        return redirect(url_for('user_manage'))

    new_user = User(
        username=username,
        password=generate_password_hash(password),
        role=role
    )
    db.session.add(new_user)
    db.session.commit()

    flash(f'用户 {username} 创建成功', 'success')
    return redirect(url_for('user_manage'))


@app.route('/users/<int:user_id>/update', methods=['POST'])
@login_required
@admin_required
def update_user(user_id):
    user = User.query.get_or_404(user_id)

    new_role = request.form.get('role', user.role).strip()
    new_password = request.form.get('password', '').strip()

    if new_role not in ['admin', 'user']:
        new_role = user.role

    # 防止把最后一个管理员降权
    if user.role == 'admin' and new_role != 'admin':
        admin_count = User.query.filter_by(role='admin').count()
        if admin_count <= 1:
            flash('系统至少需要保留一个管理员，不能将最后一个管理员降为普通用户', 'error')
            return redirect(url_for('user_manage'))

    user.role = new_role

    if new_password:
        if len(new_password) < 6:
            flash('新密码长度不能少于 6 位', 'error')
            return redirect(url_for('user_manage'))

        user.password = generate_password_hash(new_password)

    db.session.commit()
    flash(f'用户 {user.username} 信息更新成功', 'success')
    return redirect(url_for('user_manage'))


@app.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)

    # 不能删除自己
    if user.id == current_user.id:
        flash('不能删除当前登录的管理员账号', 'error')
        return redirect(url_for('user_manage'))

    # 防止删除最后一个管理员
    if user.role == 'admin':
        admin_count = User.query.filter_by(role='admin').count()
        if admin_count <= 1:
            flash('系统至少需要保留一个管理员，不能删除最后一个管理员', 'error')
            return redirect(url_for('user_manage'))

    db.session.delete(user)
    db.session.commit()

    flash(f'用户 {user.username} 已删除', 'success')
    return redirect(url_for('user_manage'))


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
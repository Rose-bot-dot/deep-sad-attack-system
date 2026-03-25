import time
import threading
from collections import deque

from scapy.all import sniff, IP, TCP, UDP

from system.services.detect_service import AttackDetector


class LiveTrafficMonitor:
    """
    实时网络流量监控服务：
    1. 抓取数据包
    2. 按五元组聚合为流
    3. 提取基础流特征
    4. 调用 Deep SAD 模型进行异常检测
    5. 缓存最近事件供前端展示
    """

    def __init__(
        self,
        model_path='saved_models/attack_model.tar',
        threshold=0.03,
        iface=None,
        idle_timeout=2,
        max_events=200
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.iface = iface
        self.idle_timeout = idle_timeout
        self.max_events = max_events

        self.detector = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # 当前活动流
        self.flows = {}

        # 最近检测事件
        self.events = deque(maxlen=max_events)

    def _ensure_detector(self):
        """
        延迟加载检测器，避免应用启动时模型文件不存在直接报错。
        """
        if self.detector is None:
            self.detector = AttackDetector(model_path=self.model_path)

    def _make_keys(self, pkt):
        """
        根据数据包构造正向/反向流键。
        流键格式：
        (src_ip, dst_ip, src_port, dst_port, protocol)
        """
        if IP not in pkt:
            return None, None, None

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst

        if TCP in pkt:
            proto = 'TCP'
            src_port = int(pkt[TCP].sport)
            dst_port = int(pkt[TCP].dport)
        elif UDP in pkt:
            proto = 'UDP'
            src_port = int(pkt[UDP].sport)
            dst_port = int(pkt[UDP].dport)
        else:
            proto = 'IP'
            src_port = 0
            dst_port = 0

        fwd_key = (src_ip, dst_ip, src_port, dst_port, proto)
        bwd_key = (dst_ip, src_ip, dst_port, src_port, proto)

        meta = {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': proto
        }
        return fwd_key, bwd_key, meta

    def _get_or_create_flow(self, pkt):
        """
        获取已有流，若不存在则新建。
        """
        fwd_key, bwd_key, meta = self._make_keys(pkt)
        if fwd_key is None:
            return None

        now_ts = float(pkt.time)

        with self.lock:
            if fwd_key in self.flows:
                return self.flows[fwd_key], 'fwd'

            if bwd_key in self.flows:
                return self.flows[bwd_key], 'bwd'

            flow = {
                **meta,
                'flow_key': fwd_key,
                'start_time': now_ts,
                'last_seen': now_ts,

                'fwd_packets': 0,
                'bwd_packets': 0,
                'fwd_bytes': 0,
                'bwd_bytes': 0,

                'fwd_lengths': [],
                'bwd_lengths': [],

                'fin_count': 0,
                'syn_count': 0,
                'rst_count': 0,
                'psh_count': 0,
                'ack_count': 0,
                'urg_count': 0
            }

            self.flows[fwd_key] = flow
            return flow, 'fwd'

    def _handle_packet(self, pkt):
        """
        处理单个数据包，将其归入对应流。
        """
        result = self._get_or_create_flow(pkt)
        if result is None:
            return

        flow, direction = result
        pkt_len = len(pkt)
        now_ts = float(pkt.time)

        with self.lock:
            flow['last_seen'] = now_ts

            if direction == 'fwd':
                flow['fwd_packets'] += 1
                flow['fwd_bytes'] += pkt_len
                flow['fwd_lengths'].append(pkt_len)
            else:
                flow['bwd_packets'] += 1
                flow['bwd_bytes'] += pkt_len
                flow['bwd_lengths'].append(pkt_len)

            if TCP in pkt:
                flags = int(pkt[TCP].flags)
                if flags & 0x01:
                    flow['fin_count'] += 1
                if flags & 0x02:
                    flow['syn_count'] += 1
                if flags & 0x04:
                    flow['rst_count'] += 1
                if flags & 0x08:
                    flow['psh_count'] += 1
                if flags & 0x10:
                    flow['ack_count'] += 1
                if flags & 0x20:
                    flow['urg_count'] += 1

    def _stat_mean(self, values):
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _stat_min(self, values):
        if not values:
            return 0.0
        return min(values)

    def _stat_max(self, values):
        if not values:
            return 0.0
        return max(values)

    def _build_feature_dict(self, flow):
        """
        将流转换为模型可用的特征字典。
        这里只构造一部分常见 CIC-IDS 风格流特征。
        其余训练列由 detect_service 中按 feature_columns 自动补 0。
        """
        duration = max(flow['last_seen'] - flow['start_time'], 1e-6)

        total_fwd_packets = flow['fwd_packets']
        total_bwd_packets = flow['bwd_packets']
        total_packets = total_fwd_packets + total_bwd_packets

        total_fwd_bytes = flow['fwd_bytes']
        total_bwd_bytes = flow['bwd_bytes']
        total_bytes = total_fwd_bytes + total_bwd_bytes

        all_lengths = flow['fwd_lengths'] + flow['bwd_lengths']

        feature_dict = {
            'Destination Port': flow['dst_port'],
            'Flow Duration': duration * 1000000,

            'Total Fwd Packets': total_fwd_packets,
            'Total Backward Packets': total_bwd_packets,

            'Total Length of Fwd Packets': total_fwd_bytes,
            'Total Length of Bwd Packets': total_bwd_bytes,

            'Fwd Packet Length Max': self._stat_max(flow['fwd_lengths']),
            'Fwd Packet Length Min': self._stat_min(flow['fwd_lengths']),
            'Fwd Packet Length Mean': self._stat_mean(flow['fwd_lengths']),

            'Bwd Packet Length Max': self._stat_max(flow['bwd_lengths']),
            'Bwd Packet Length Min': self._stat_min(flow['bwd_lengths']),
            'Bwd Packet Length Mean': self._stat_mean(flow['bwd_lengths']),

            'Flow Bytes/s': total_bytes / duration,
            'Flow Packets/s': total_packets / duration,

            'Packet Length Max': self._stat_max(all_lengths),
            'Packet Length Min': self._stat_min(all_lengths),
            'Packet Length Mean': self._stat_mean(all_lengths),

            'FIN Flag Count': flow['fin_count'],
            'SYN Flag Count': flow['syn_count'],
            'RST Flag Count': flow['rst_count'],
            'PSH Flag Count': flow['psh_count'],
            'ACK Flag Count': flow['ack_count'],
            'URG Flag Count': flow['urg_count'],
        }

        return feature_dict

    def _predict_flow(self, flow):
        """
        对单条流进行异常检测。
        """
        self._ensure_detector()

        feature_dict = self._build_feature_dict(flow)

        if not hasattr(self.detector, 'predict_feature_dict'):
            raise AttributeError(
                'AttackDetector 缺少 predict_feature_dict 方法，请先替换 detect_service.py'
            )

        result = self.detector.predict_feature_dict(
            feature_dict=feature_dict,
            threshold=self.threshold
        )

        event = {
            'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(flow['last_seen'])),
            'src_ip': flow['src_ip'],
            'dst_ip': flow['dst_ip'],
            'src_port': flow['src_port'],
            'dst_port': flow['dst_port'],
            'protocol': flow['protocol'],
            'score': round(float(result.get('score', 0.0)), 6),
            'label': '异常' if int(result.get('label', 0)) == 1 else '正常'
        }

        self.events.appendleft(event)

    def _flush_idle_flows(self):
        """
        将超过空闲超时时间的流判定为结束流，并送入模型检测。
        """
        now_ts = time.time()
        expired_flows = []

        with self.lock:
            expired_keys = []
            for key, flow in self.flows.items():
                if now_ts - flow['last_seen'] >= self.idle_timeout:
                    expired_keys.append(key)

            for key in expired_keys:
                expired_flows.append(self.flows.pop(key))

        for flow in expired_flows:
            try:
                self._predict_flow(flow)
            except Exception as e:
                self.events.appendleft({
                    'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'src_ip': flow.get('src_ip', ''),
                    'dst_ip': flow.get('dst_ip', ''),
                    'src_port': flow.get('src_port', ''),
                    'dst_port': flow.get('dst_port', ''),
                    'protocol': flow.get('protocol', ''),
                    'score': 0,
                    'label': f'检测失败：{str(e)}'
                })

    def _sniff_once(self):
        """
        抓取一个短周期的数据包，然后统一清理空闲流。
        """
        sniff(
            iface=self.iface,
            store=False,
            timeout=1,
            prn=self._handle_packet
        )
        self._flush_idle_flows()

    def _run_loop(self):
        """
        后台监控线程主循环。
        """
        while self.running:
            try:
                self._sniff_once()
            except Exception as e:
                self.events.appendleft({
                    'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'src_ip': '',
                    'dst_ip': '',
                    'src_port': '',
                    'dst_port': '',
                    'protocol': '',
                    'score': 0,
                    'label': f'抓包失败：{str(e)}'
                })
                time.sleep(1)

    def start(self, iface=None, threshold=None):
        """
        启动实时监控。
        """
        if self.running:
            return

        if iface is not None:
            self.iface = iface

        if threshold is not None:
            self.threshold = threshold

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """
        停止实时监控。
        """
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        self._flush_idle_flows()

    def status(self):
        """
        返回当前监控状态。
        """
        return {
            'running': self.running,
            'iface': self.iface,
            'threshold': self.threshold,
            'cached_events': len(self.events)
        }

    def get_events(self, limit=50):
        """
        获取最近事件。
        """
        return list(self.events)[:limit]
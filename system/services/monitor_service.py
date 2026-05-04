# system/services/monitor_service.py

import time
import threading
from collections import deque

from scapy.all import sniff, IP, TCP, UDP, get_if_list, conf

from system.services.detect_service import AttackDetector


class LiveTrafficMonitor:
    """
    实时网络流量监控服务：
    1. 抓取数据包；
    2. 按五元组聚合为流；
    3. 提取基础流特征；
    4. 调用 Deep SAD 模型进行异常检测；
    5. 缓存最近事件供前端展示。

    修复点：
    1. 防止极短流导致 Flow Bytes/s、Flow Packets/s 爆炸；
    2. 跳过单包流、极小流，减少误判；
    3. 对实时特征做上限保护；
    4. 前端异常分数不再动辄几千万、几亿。
    """

    def __init__(
        self,
        model_path="saved_models/attack_model.tar",
        threshold=0.03,
        iface=None,
        idle_timeout=2,
        max_events=200,
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
        self.flows = {}
        self.events = deque(maxlen=max_events)

    def _is_bad_iface(self, iface):
        name = str(iface).lower()

        bad_keywords = [
            "loopback",
            "npcap loopback",
            "virtual",
            "vmware",
            "virtualbox",
            "docker",
            "hyper-v",
            "bluetooth",
            "teredo",
            "isatap",
        ]

        return any(keyword in name for keyword in bad_keywords)

    def _score_iface_once(self, iface, timeout=0.6):
        try:
            packets = sniff(
                iface=iface,
                timeout=timeout,
                store=True,
            )
            return len(packets)
        except Exception as e:
            print(f"[LiveTrafficMonitor] 网卡采样失败：{iface}，原因：{e}")
            return -1

    def list_interfaces(self, sample_seconds=0.6):
        interfaces = get_if_list()
        results = []

        print("\n========== 当前 Scapy 检测到的网卡 ==========")

        for i, iface in enumerate(interfaces):
            iface_text = str(iface)
            is_bad = self._is_bad_iface(iface_text)

            if is_bad:
                score = -1
            else:
                score = self._score_iface_once(iface_text, timeout=sample_seconds)

            item = {
                "index": i,
                "name": iface_text,
                "score": score,
                "disabled": is_bad,
                "display": f"{i} - {iface_text} | 当前流量包数：{score if score >= 0 else '不推荐'}",
            }

            results.append(item)
            print(item["display"])

        valid_results = [
            item for item in results
            if not item["disabled"] and item["score"] >= 0
        ]

        default_iface = None

        if valid_results:
            valid_results.sort(key=lambda x: x["score"], reverse=True)
            default_iface = valid_results[0]["name"]
        else:
            try:
                default_iface = str(conf.iface)
            except Exception:
                default_iface = None

        for item in results:
            item["selected"] = item["name"] == default_iface

        print(f"[LiveTrafficMonitor] 默认推荐网卡：{default_iface}")

        return {
            "interfaces": results,
            "default_iface": default_iface,
        }

    def _auto_select_iface(self):
        data = self.list_interfaces(sample_seconds=0.6)
        default_iface = data.get("default_iface")

        if not default_iface:
            raise RuntimeError("没有找到可用网卡，请检查 Npcap 是否安装正常，或手动选择网卡。")

        print(f"[LiveTrafficMonitor] 自动选择流量最高网卡：{default_iface}")

        return default_iface

    def _ensure_detector(self):
        if self.detector is None:
            self.detector = AttackDetector(model_path=self.model_path)

    def _make_keys(self, pkt):
        if IP not in pkt:
            return None, None, None

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst

        if TCP in pkt:
            proto = "TCP"
            src_port = int(pkt[TCP].sport)
            dst_port = int(pkt[TCP].dport)
        elif UDP in pkt:
            proto = "UDP"
            src_port = int(pkt[UDP].sport)
            dst_port = int(pkt[UDP].dport)
        else:
            proto = "IP"
            src_port = 0
            dst_port = 0

        fwd_key = (src_ip, dst_ip, src_port, dst_port, proto)
        bwd_key = (dst_ip, src_ip, dst_port, src_port, proto)

        meta = {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": proto,
        }

        return fwd_key, bwd_key, meta

    def _get_or_create_flow(self, pkt):
        fwd_key, bwd_key, meta = self._make_keys(pkt)

        if fwd_key is None:
            return None

        now_ts = float(pkt.time)

        with self.lock:
            if fwd_key in self.flows:
                return self.flows[fwd_key], "fwd"

            if bwd_key in self.flows:
                return self.flows[bwd_key], "bwd"

            flow = {
                **meta,
                "flow_key": fwd_key,
                "start_time": now_ts,
                "last_seen": now_ts,

                "fwd_packets": 0,
                "bwd_packets": 0,

                "fwd_bytes": 0,
                "bwd_bytes": 0,

                "fwd_lengths": [],
                "bwd_lengths": [],

                "fin_count": 0,
                "syn_count": 0,
                "rst_count": 0,
                "psh_count": 0,
                "ack_count": 0,
                "urg_count": 0,
            }

            self.flows[fwd_key] = flow

            return flow, "fwd"

    def _handle_packet(self, pkt):
        result = self._get_or_create_flow(pkt)

        if result is None:
            return

        flow, direction = result

        pkt_len = len(pkt)
        now_ts = float(pkt.time)

        with self.lock:
            flow["last_seen"] = now_ts

            if direction == "fwd":
                flow["fwd_packets"] += 1
                flow["fwd_bytes"] += pkt_len
                flow["fwd_lengths"].append(pkt_len)
            else:
                flow["bwd_packets"] += 1
                flow["bwd_bytes"] += pkt_len
                flow["bwd_lengths"].append(pkt_len)

            if TCP in pkt:
                flags = int(pkt[TCP].flags)

                if flags & 0x01:
                    flow["fin_count"] += 1
                if flags & 0x02:
                    flow["syn_count"] += 1
                if flags & 0x04:
                    flow["rst_count"] += 1
                if flags & 0x08:
                    flow["psh_count"] += 1
                if flags & 0x10:
                    flow["ack_count"] += 1
                if flags & 0x20:
                    flow["urg_count"] += 1

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

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    def _build_feature_dict(self, flow):
        """
        将实时流转换为模型可用的 23 个特征。
        """

        raw_duration = float(flow["last_seen"] - flow["start_time"])

        # 核心修复：
        # 原项目使用 1e-6，会导致极短流的 Flow Bytes/s 和 Flow Packets/s 被放大上百万倍。
        # 这里最小按 1 秒计算，更适合演示系统。
        duration_s = max(raw_duration, 1.0)
        duration_us = duration_s * 1_000_000

        total_fwd_packets = int(flow["fwd_packets"])
        total_bwd_packets = int(flow["bwd_packets"])
        total_packets = total_fwd_packets + total_bwd_packets

        total_fwd_bytes = int(flow["fwd_bytes"])
        total_bwd_bytes = int(flow["bwd_bytes"])
        total_bytes = total_fwd_bytes + total_bwd_bytes

        fwd_lengths = list(flow["fwd_lengths"])
        bwd_lengths = list(flow["bwd_lengths"])
        all_lengths = fwd_lengths + bwd_lengths

        flow_bytes_s = total_bytes / duration_s
        flow_packets_s = total_packets / duration_s

        # 对实时速率特征做保护，防止远超训练数据分布
        flow_bytes_s = min(flow_bytes_s, 10_000_000.0)
        flow_packets_s = min(flow_packets_s, 100_000.0)

        feature_dict = {
            "Destination Port": float(flow["dst_port"]),
            "Flow Duration": float(duration_us),

            "Total Fwd Packets": float(total_fwd_packets),
            "Total Backward Packets": float(total_bwd_packets),

            "Total Length of Fwd Packets": float(total_fwd_bytes),
            "Total Length of Bwd Packets": float(total_bwd_bytes),

            "Fwd Packet Length Max": float(self._stat_max(fwd_lengths)),
            "Fwd Packet Length Min": float(self._stat_min(fwd_lengths)),
            "Fwd Packet Length Mean": float(self._stat_mean(fwd_lengths)),

            "Bwd Packet Length Max": float(self._stat_max(bwd_lengths)),
            "Bwd Packet Length Min": float(self._stat_min(bwd_lengths)),
            "Bwd Packet Length Mean": float(self._stat_mean(bwd_lengths)),

            "Flow Bytes/s": float(flow_bytes_s),
            "Flow Packets/s": float(flow_packets_s),

            "Packet Length Max": float(self._stat_max(all_lengths)),
            "Packet Length Min": float(self._stat_min(all_lengths)),
            "Packet Length Mean": float(self._stat_mean(all_lengths)),

            "FIN Flag Count": float(flow["fin_count"]),
            "SYN Flag Count": float(flow["syn_count"]),
            "RST Flag Count": float(flow["rst_count"]),
            "PSH Flag Count": float(flow["psh_count"]),
            "ACK Flag Count": float(flow["ack_count"]),
            "URG Flag Count": float(flow["urg_count"]),
        }

        return feature_dict

    def _should_skip_flow(self, flow):
        """
        过滤信息量太少的流，减少误判。
        """

        total_packets = int(flow["fwd_packets"]) + int(flow["bwd_packets"])
        total_bytes = int(flow["fwd_bytes"]) + int(flow["bwd_bytes"])

        if total_packets < 3:
            return True

        if total_bytes < 120:
            return True

        return False

    def _predict_flow(self, flow):
        self._ensure_detector()

        if self._should_skip_flow(flow):
            return

        feature_dict = self._build_feature_dict(flow)

        if not hasattr(self.detector, "predict_feature_dict"):
            raise AttributeError("AttackDetector 缺少 predict_feature_dict 方法，请先替换 detect_service.py")

        result = self.detector.predict_feature_dict(
            feature_dict=feature_dict,
            threshold=self.threshold,
        )

        score = self._safe_float(result.get("score", 0.0), 0.0)
        raw_score = self._safe_float(result.get("raw_score", score), score)
        label_value = int(result.get("label", 0))

        event = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(flow["last_seen"])),
            "src_ip": flow["src_ip"],
            "dst_ip": flow["dst_ip"],
            "src_port": flow["src_port"],
            "dst_port": flow["dst_port"],
            "protocol": flow["protocol"],
            "score": round(float(score), 6),
            "raw_score": round(float(raw_score), 6),
            "label": "异常" if label_value == 1 else "正常",
        }

        self.events.appendleft(event)

    def _flush_idle_flows(self):
        now_ts = time.time()
        expired_flows = []

        with self.lock:
            expired_keys = []

            for key, flow in self.flows.items():
                if now_ts - flow["last_seen"] >= self.idle_timeout:
                    expired_keys.append(key)

            for key in expired_keys:
                expired_flows.append(self.flows.pop(key))

        for flow in expired_flows:
            try:
                self._predict_flow(flow)
            except Exception as e:
                self.events.appendleft({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "src_ip": flow.get("src_ip", ""),
                    "dst_ip": flow.get("dst_ip", ""),
                    "src_port": flow.get("src_port", ""),
                    "dst_port": flow.get("dst_port", ""),
                    "protocol": flow.get("protocol", ""),
                    "score": 0,
                    "label": f"检测失败：{str(e)}",
                })

    def _sniff_once(self):
        if not self.iface:
            raise RuntimeError("当前没有可用网卡，自动选择网卡失败。")

        sniff(
            iface=self.iface,
            store=False,
            timeout=1,
            prn=self._handle_packet,
        )

        self._flush_idle_flows()

    def _run_loop(self):
        while self.running:
            try:
                self._sniff_once()
            except Exception as e:
                error_msg = f"抓包失败：{str(e)}"

                print(f"[LiveTrafficMonitor] {error_msg}")

                self.events.appendleft({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "src_ip": "",
                    "dst_ip": "",
                    "src_port": "",
                    "dst_port": "",
                    "protocol": "",
                    "score": 0,
                    "label": error_msg,
                })

                time.sleep(1)

    def start(self, iface=None, threshold=None):
        if self.running:
            return

        if iface is None:
            self.iface = self._auto_select_iface()
        else:
            iface_text = str(iface).strip()

            if (
                iface_text == ""
                or iface_text.lower() == "auto"
                or iface_text in ["自动", "自动选择", "自动选择网卡"]
            ):
                self.iface = self._auto_select_iface()
            else:
                self.iface = iface_text

        if threshold is not None:
            self.threshold = threshold

        print(f"[LiveTrafficMonitor] 当前使用网卡：{self.iface}")
        print(f"[LiveTrafficMonitor] 当前检测阈值：{self.threshold}")

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        self._flush_idle_flows()

    def status(self):
        return {
            "running": self.running,
            "iface": self.iface,
            "threshold": self.threshold,
            "cached_events": len(self.events),
        }

    def get_events(self, limit=50):
        return list(self.events)[:limit]
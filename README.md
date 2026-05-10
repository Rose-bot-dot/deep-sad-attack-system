# Deep SAD Attack System

## 项目简介

本项目是一个基于 Deep SAD（Deep Semi-Supervised Anomaly Detection）的网络异常行为监测系统，主要用于对网络流量数据进行异常检测与结果展示。

系统采用 Flask 构建 Web 后端，结合 PyTorch 实现 Deep SAD 模型训练与推理，并使用 Scapy 对实时网络流量进行采集和特征提取。系统支持演示数据集训练、离线 CSV 文件检测、实时流量监控、异常结果展示等功能。

本项目主要用于毕业设计、课程设计和网络异常检测系统演示。

---

## 主要功能

- 用户登录与系统首页展示
- 演示数据集模型训练
- Deep SAD 异常检测模型加载
- CSV 离线流量检测
- 实时网络流量监控
- 网络流特征提取
- 异常分数计算与结果展示
- 检测结果可视化展示

---

## 技术栈

- Python 3.11
- Flask
- Flask-Login
- Flask-SQLAlchemy
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Scapy
- SQLite
- HTML / CSS / JavaScript

---



## Npcap 安装说明

本项目的实时流量监控功能依赖 Scapy 进行网络数据包捕获。在 Windows 系统下，Scapy 需要借助 Npcap 提供底层抓包支持。

如果只使用模型训练和 CSV 离线检测功能，可以不安装 Npcap；如果需要使用实时监控功能，则必须安装 Npcap。

安装 Npcap 时建议勾选以下选项：

```text
√ Install Npcap in WinPcap API-compatible Mode
√ Support loopback traffic

---


## 项目结构

```text
deep-sad-attack-system
├── app.py                         # Flask 主程序入口
├── requirements.txt               # 项目依赖文件
├── saved_models                   # 模型和预处理器保存目录
│   ├── attack_model.tar
│   ├── preprocessor.joblib
│   └── feature_columns.json
├── src                            # Deep SAD 模型相关代码
│   ├── DeepSAD.py
│   └── networks
├── system
│   └── services
│       ├── train_service.py       # 模型训练服务
│       ├── detect_service.py      # 离线检测服务
│       └── monitor_service.py     # 实时监控服务
├── templates                      # 前端页面模板
├── static                         # 静态资源文件
└── instance                       # SQLite 数据库文件

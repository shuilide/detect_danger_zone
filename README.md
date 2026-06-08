# 危险区域人员闯入检测与报警系统 - 运行说明文档

---

## 一、项目概述

基于 YOLOv8 深度学习框架的危险区域人员闯入检测与报警系统，支持视频文件和摄像头实时检测，可手动绘制多边形危险区域，当区域内人数达到设定阈值时触发声光报警。

### 技术栈
- **Python 3.10+**
- **YOLOv8** (ultralytics) - 目标检测
- **ByteTrack** - 多目标追踪
- **PyQt5** - 可视化界面
- **OpenCV** - 图像处理
- **supervision** - 计算机视觉工具库

### 核心功能
1. ✅ 仅检测人员（person）类别，过滤其他类别
2. ✅ 支持本地视频文件和电脑摄像头两种输入源
3. ✅ 鼠标手动绘制任意多边形危险区域（至少3个点闭合）
4. ✅ 自动统计区域内人数和画面总人数
5. ✅ 区域内人数 ≥ 阈值时触发报警（红色警告 + 警报声音）
6. ✅ 集成 ByteTrack 多目标追踪，分配稳定 ID
7. ✅ 实时显示：FPS、检测时长、区域人数、总人数
8. ✅ 可开关：检测框、标签、追踪轨迹显示

---

## 二、项目目录结构

```
yolov8-detect/
├── MainProgram.py          # 【主程序】PyQt5 可视化界面入口
├── VideoTest.py            # 视频检测独立脚本（脱离 GUI）
├── CameraTest.py           # 摄像头检测独立脚本（脱离 GUI）
├── imgTest.py              # 图片检测测试脚本
├── train_v8.py             # YOLOv8 模型训练脚本
├── installPackages.py      # 一键环境安装脚本
├── requirements.txt        # 依赖库清单
├── alarm.wav               # 报警音频文件
├── datasets/               # 数据集文件夹
│   └── Data/
│       └── data.yaml       # 数据集配置文件
├── UIProgram/              # 核心模块
│   ├── __init__.py
│   ├── detector.py         # YOLOv8 检测器
│   ├── tracker.py          # ByteTrack 追踪器
│   ├── zone.py             # 多边形区域管理
│   ├── alarm.py            # 报警系统
│   └── utils.py            # 工具函数
├── Font/                   # UI 字体文件（预留）
├── models/                 # 模型权重目录（首次运行自动下载）
├── runs/                   # 训练日志和结果
└── TestFiles/              # 测试图片和视频（预留）
```

---

## 三、环境安装

### 方法一：使用安装脚本（推荐）

```bash
# 运行一键安装脚本
python installPackages.py

# 然后选择安装方式：
# [1] 普通 pip 安装（适用于系统 Python）
# [2] Anaconda 安装（推荐，需先激活 conda 环境）
```

### 方法二：Anaconda 手动安装

```bash
# 1. 创建并激活环境
conda create -n danger_zone python=3.10 -y
conda activate danger_zone

# 2. 添加清华大学镜像源（可选，加速下载）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

# 3. 安装依赖
conda install -c conda-forge opencv numpy pillow -y
pip install ultralytics PyQt5 supervision -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 方法三：直接使用 pip

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 四、运行程序

### 4.1 主程序（推荐）

```bash
# 激活环境（如果使用 conda）
conda activate danger_zone

# 启动主程序
python MainProgram.py
```

**主界面操作说明：**

| 按钮/控件 | 功能说明 |
|-----------|----------|
| 打开视频 | 选择本地视频文件进行检测，再次点击关闭 |
| 打开摄像头 | 打开默认摄像头进行实时检测，再次点击关闭 |
| 绘制区域 | 进入区域绘制模式，鼠标左键点击添加顶点 |
| 绘制完成 | 闭合多边形区域（需至少3个点） |
| 置信度阈值 | 设置检测置信度阈值（0.1-1.0） |
| IoU阈值 | 设置 NMS 的 IoU 阈值（0.1-1.0） |
| 报警阈值 | 设置触发报警的区域人数阈值（默认1） |
| 显示检测框 | 控制是否显示检测边界框 |
| 显示标签 | 控制是否显示追踪 ID 和置信度标签 |
| 显示追踪轨迹 | 控制是否显示目标运动轨迹 |

### 4.2 视频检测脚本

```bash
# 运行视频检测（脱离 GUI）
python VideoTest.py --video path/to/video.mp4

# 命令行参数
python VideoTest.py --video test.mp4 --model models/yolov8n.pt --conf 0.5 --threshold 1

# 不指定视频路径，弹出文件选择对话框
python VideoTest.py
```

### 4.3 摄像头检测脚本

```bash
# 运行摄像头检测（脱离 GUI）
python CameraTest.py --camera 0

# 命令行参数
python CameraTest.py --camera 0 --model models/yolov8n.pt
```

### 4.4 图片检测脚本

```bash
# 检测单张图片
python imgTest.py --image test.jpg --output result.jpg

# 参数说明
# --image: 输入图片路径
# --model: 模型路径（默认 models/yolov8n.pt）
# --output: 输出图片路径（默认 output.jpg）
# --conf: 置信度阈值（默认 0.5）
```

### 4.5 模型训练

```bash
# 训练人员检测模型（需先准备数据集）
python train_v8.py

# 训练参数在脚本中配置：
# epochs: 训练轮数（默认 150）
# batch: 批次大小（默认 4）
# imgsz: 输入图像大小（默认 640）
# device: 训练设备（默认 'cpu'）
```

---

## 五、快捷键说明

### 视频检测脚本 (VideoTest.py)
| 按键 | 功能 |
|------|------|
| q / ESC | 退出程序 |
| r | 重置绘制的区域 |
| t | 切换轨迹显示 |
| b | 切换检测框显示 |
| l | 切换标签显示 |
| 鼠标左键 | 添加区域顶点 |
| 鼠标右键 | 闭合区域（需≥3个点） |

### 摄像头检测脚本 (CameraTest.py)
| 按键 | 功能 |
|------|------|
| q / ESC | 退出程序 |
| r | 重置绘制的区域 |
| t | 切换轨迹显示 |
| b | 切换检测框显示 |
| l | 切换标签显示 |
| 鼠标左键 | 添加区域顶点 |
| 鼠标右键 | 闭合区域（需≥3个点） |

---

## 六、数据集准备（训练用）

如需训练自定义模型，请按以下结构组织数据集：

```
datasets/Data/
├── images/
│   ├── train/          # 训练图片（约80%）
│   └── val/            # 验证图片（约20%）
└── labels/
    ├── train/          # 训练标签（YOLO格式）
    └── val/            # 验证标签（YOLO格式）
```

**data.yaml 配置示例：**

```yaml
train: datasets/Data_/images/train
val: datasets/Data_/images/val
nc: 1
names: ['person']
```

---

## 七、常见问题

### Q1: 第一次运行时模型文件不存在？
**A:** 程序会自动从 ultralytics 服务器下载 yolov8n.pt 模型（约6MB），首次运行可能需要等待几秒。

### Q2: 摄像头无法打开？
**A:** 请确保摄像头未被其他程序占用，尝试更换摄像头索引（--camera 参数）。

### Q3: 报警声音不播放？
**A:** 检查 alarm.wav 文件是否存在于项目根目录，或检查系统音量设置。

### Q4: 检测速度慢？
**A:** 默认使用 CPU 运行，如需加速请使用支持 CUDA 的 GPU，并将 detector.py 中的 device 参数改为 'cuda'。

### Q5: 区域绘制后不显示？
**A:** 需要至少点击3个点才能闭合区域，点击"绘制完成"按钮后区域才会生效。

---

## 八、技术支持

如有问题，请检查：
1. Python 版本是否为 3.10+
2. 所有依赖是否正确安装
3. 模型文件是否成功下载
4. 摄像头权限是否正常

---

**版本**: v1.0  
**日期**: 2024年11月  
**作者**: YOLOv8 危险区域检测开发团队
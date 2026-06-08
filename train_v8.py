# coding: utf-8
"""
YOLOv8 模型训练脚本
支持人员检测、人员+无人机多类别检测的训练 / 微调

数据集准备（详见下方注释）:
    使用前请先准备好 datasets/Data_/ 目录下的数据，结构如下:
    datasets/Data_/
    ├── data.yaml          # 数据集配置文件
    ├── train/
    │   ├── images/        # 训练图片 (.jpg/.png)
    │   └── labels/        # YOLO 格式标签 (.txt)
    └── val/
        ├── images/        # 验证图片
        └── labels/        # 验证标签
"""

import matplotlib
matplotlib.use('TkAgg')  # 避免 GUI 后端冲突

from ultralytics import YOLO
import os


if __name__ == '__main__':
    # ==================== 模型加载 ====================
    # 方式1：从头训练（随机初始化）
    # model = YOLO('yolov8n.yaml')

    # 方式2（推荐）：基于预训练权重微调
    # 加载 yolov8n.pt 预训练权重，在其基础上微调 person + drone
    model = YOLO('yolov8n.pt')

    # ==================== 训练参数配置 ====================
    # 数据集配置文件路径
    data_yaml = 'datasets/Data/data.yaml'

    # 训练轮数：仅无人机数据，轮数不宜过多避免过拟合
    epochs = 5

    # 批次大小：CPU 训练建议使用较小批次，避免内存不足
    batch = 4

    # 输入图像尺寸
    imgsz = 640

    # 训练设备：'cpu' 使用 CPU，如有 GPU 可改为 'cuda' 或 'cuda:0'
    device = 'cpu'

    # 数据加载进程数：Windows 下设为 0
    workers = 0

    # 冻结主干网络层数：冻结前 10 层，保护 COCO 预训练特征不退化
    # 只有检测头学习 drone，避免忘记 person
    freeze = 10

    # 实验名称
    exp_name = 'person_drone_detect'

    # ==================== 训练信息输出 ====================
    print("=" * 60)
    print("  开始训练 YOLOv8 人员+无人机检测模型")
    print(f"  策略: 冻结主干 (freeze={freeze})，仅训练检测头")
    print(f"  基础模型: yolov8n.pt (COCO 预训练)")
    print(f"  数据配置: {data_yaml}")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch}")
    print(f"  输入尺寸: {imgsz}")
    print(f"  训练设备: {device}")
    print(f"  实验名称: {exp_name}")
    print("=" * 60)

    # ==================== 开始训练 ====================
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        name=exp_name,
        exist_ok=True,
        freeze=freeze,
    )

    # ==================== 训练完成 ====================
    print("=" * 60)
    print("  训练完成！")
    print(f"  最佳模型保存在: runs/detect/{exp_name}/weights/best.pt")
    print(f"  使用模型: 将 best.pt 复制到 models/ 目录下")
    print(f"  修改 MainProgram.py 中的 model_path 和 target_classes")
    print("=" * 60)
# coding: utf-8
"""
YOLOv8 模型训练脚本
用于训练危险区域人员检测模型
"""

import matplotlib
matplotlib.use('TkAgg')  # 避免 GUI 后端冲突

from ultralytics import YOLO
import os


if __name__ == '__main__':
    # ==================== 模型加载 ====================
    # 加载 YOLOv8n 预训练权重（n 表示 nano，最小版本，适合快速实验）
    model = YOLO('yolov8n.pt')

    # ==================== 训练参数配置 ====================
    # 数据集配置文件路径
    data_yaml = 'datasets/Data/data.yaml'

    # 训练轮数：可根据数据集规模和收敛情况调整
    epochs = 150

    # 批次大小：CPU 训练建议使用较小批次，避免内存不足
    batch = 4

    # 输入图像尺寸：640x640 是 YOLOv8 的默认训练尺寸
    imgsz = 640

    # 训练设备：'cpu' 使用 CPU，如有 NVIDIA GPU 可改为 'cuda' 或 'cuda:0'
    device = 'cpu'

    # 数据加载进程数：Windows 下设为 0 避免多进程问题
    workers = 0

    # 实验名称：训练结果将保存在 runs/detect/person_detect/ 目录下
    exp_name = 'person_detect'

    # ==================== 训练信息输出 ====================
    print("=" * 60)
    print("  开始训练 YOLOv8 人员检测模型...")
    print(f"  数据配置: {data_yaml}")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch}")
    print(f"  输入尺寸: {imgsz}")
    print(f"  训练设备: {device}")
    print(f"  实验名称: {exp_name}")
    print("=" * 60)

    # ==================== 开始训练 ====================
    results = model.train(
        data=data_yaml,       # 数据集配置 YAML 文件
        epochs=epochs,        # 训练总轮数
        batch=batch,          # 每批次的图像数量
        imgsz=imgsz,          # 模型输入图像尺寸
        device=device,        # 训练设备（CPU 或 CUDA）
        workers=workers,      # 数据加载子进程数
        name=exp_name,        # 实验名称，用于区分不同训练结果
        exist_ok=True         # 允许覆盖已存在的同名实验目录
    )

    # ==================== 训练完成 ====================
    print("=" * 60)
    print("  训练完成！")
    print(f"  最佳模型保存在: runs/detect/{exp_name}/weights/best.pt")
    print(f"  训练日志保存在: runs/detect/{exp_name}/")
    print("=" * 60)
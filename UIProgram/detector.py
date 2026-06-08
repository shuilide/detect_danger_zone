# coding: utf-8
"""
YOLOv8 目标检测器模块
封装 YOLOv8 模型加载与推理，仅检测 person 类别
"""
import cv2
import numpy as np
from ultralytics import YOLO
import os
import shutil


class YOLODetector:
    """YOLOv8 检测器封装类，仅保留 person 类别检测结果"""

    def __init__(self, model_path='models/yolov8n.pt', device='cpu'):
        """
        初始化 YOLOv8 检测器

        参数:
            model_path: 模型文件路径，默认 'models/yolov8n.pt'
            device: 推理设备，默认 'cpu'
        """
        self.device = device
        resolved_path = model_path

        if not os.path.exists(resolved_path):
            # 计算项目根目录路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.normpath(os.path.join(script_dir, '..'))
            root_pt = os.path.join(project_root, 'yolov8n.pt')

            if os.path.exists(root_pt):
                # 根目录已有下载的文件，复制到 models/ 目录
                os.makedirs(os.path.dirname(os.path.abspath(resolved_path)), exist_ok=True)
                shutil.copy2(root_pt, resolved_path)
                print(f"[YOLODetector] 已将模型从根目录复制到 {resolved_path}")
            else:
                # 本地完全没有，触发下载
                os.makedirs(os.path.dirname(os.path.abspath(resolved_path)), exist_ok=True)
                print(f"[YOLODetector] 本地模型文件不存在，将从 ultralytics 自动下载 yolov8n.pt ...")
                # ultralytics 会下载到当前工作目录，下载后移动到 models/
                temp_model = YOLO('yolov8n.pt')
                if os.path.exists(root_pt):
                    shutil.move(root_pt, resolved_path)
                    print(f"[YOLODetector] 模型已下载并保存到 {resolved_path}")

        self.model = YOLO(resolved_path)
        # 预热模型（首次推理较慢，提前加载）
        self.model.to(self.device)
        print(f"[YOLODetector] 模型已加载，设备: {self.device}")

    def detect(self, frame, conf_thres=0.5, iou_thres=0.5):
        """
        对单帧图像进行目标检测，只返回 person 类别结果

        参数:
            frame: BGR 格式的 numpy 数组图像
            conf_thres: 置信度阈值，默认 0.5
            iou_thres: NMS 的 IoU 阈值，默认 0.5

        返回:
            list[dict]: 检测结果列表，每项包含:
                - bbox: (x1, y1, x2, y2) 边界框坐标
                - confidence: 置信度
                - class_id: 类别 ID
                - class_name: 类别名称
        """
        if frame is None or frame.size == 0:
            return []

        # 执行推理
        results = self.model(
            frame,
            conf=conf_thres,
            iou=iou_thres,
            device=self.device,
            verbose=False
        )

        detections_list = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                # 只保留 person 类别 (COCO 数据集中 class_id=0)
                if cls_id == 0:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    detections_list.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': round(conf, 4),
                        'class_id': cls_id,
                        'class_name': 'person'
                    })

        return detections_list
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
    """YOLOv8 检测器封装类，支持多类别检测（person、drone 等）"""

    # COCO 数据集原始类别名称（前 80 类），用于预训练模型的类别映射
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
        34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
        37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
        41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
        55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
        65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
        78: 'hair drier', 79: 'toothbrush',
    }

    def __init__(self, model_path='models/yolov8n.pt', device='cpu',
                 target_classes=None):
        """
        初始化 YOLOv8 检测器

        参数:
            model_path: 模型文件路径，默认 'models/yolov8n.pt'
            device: 推理设备，默认 'cpu'
            target_classes: 要检测的目标类别，dict 格式 {class_id: 'class_name', ...}
                            默认为 {0: 'person'}，即只检测行人。
                            微调后如需检测无人机，传入 {0: 'person', 1: 'drone'}
        """
        self.device = device
        # 默认只检测 person，保持向后兼容
        if target_classes is None:
            target_classes = {0: 'person'}
        self.target_classes = target_classes
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

    def get_class_name(self, cls_id):
        """
        根据类别 ID 获取类别名称

        参数:
            cls_id: 类别 ID 整数

        返回:
            str: 类别名称，未知类别返回 'unknown'
        """
        return self.target_classes.get(cls_id, 'unknown')

    def detect(self, frame, conf_thres=0.5, iou_thres=0.5):
        """
        对单帧图像进行目标检测，返回 target_classes 中配置的类别结果

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
                # 只保留 target_classes 中配置的类别
                if cls_id in self.target_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    detections_list.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': round(conf, 4),
                        'class_id': cls_id,
                        'class_name': self.target_classes[cls_id]
                    })

        return detections_list
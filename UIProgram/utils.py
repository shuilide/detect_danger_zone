# coding: utf-8
"""
工具函数模块
包含 FPS 计算、检测框绘制、轨迹绘制、颜色映射等
"""
import cv2
import time
import numpy as np


class FPSCounter:
    """FPS（帧率）计算器"""

    def __init__(self):
        """初始化 FPS 计数器"""
        self._start_time = time.time()
        self._frame_count = 0
        self.fps = 0.0

    def update(self):
        """
        更新帧计数并计算当前 FPS

        返回:
            float: 当前 FPS 值
        """
        self._frame_count += 1
        elapsed = time.time() - self._start_time
        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._start_time = time.time()
        return self.fps

    def draw_fps(self, frame):
        """
        在画面左上角绘制 FPS 信息

        参数:
            frame: BGR 格式的 numpy 数组图像（原地修改）

        返回:
            numpy.ndarray: 绘制后的图像
        """
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            frame, fps_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        return frame


def get_color(track_id):
    """
    根据 track_id 生成唯一且稳定的颜色

    参数:
        track_id: 追踪 ID 整数

    返回:
        tuple: (B, G, R) 颜色元组
    """
    # 使用 track_id 作为随机种子，保证同一 ID 颜色一致
    rng = np.random.RandomState(track_id * 7 + 13)
    return tuple(rng.randint(64, 255, 3).tolist())


def draw_detection(frame, bbox, track_id, class_name, conf, color, show_label=True):
    """
    在画面上绘制检测框和标签

    参数:
        frame: BGR 格式的 numpy 数组图像（原地修改）
        bbox: (x1, y1, x2, y2) 边界框坐标
        track_id: 追踪 ID
        class_name: 类别名称
        conf: 置信度
        color: (B, G, R) 绘制颜色
        show_label: 是否显示文字标签，默认 True

    返回:
        numpy.ndarray: 绘制后的图像
    """
    x1, y1, x2, y2 = map(int, bbox)

    # 绘制边界框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if show_label:
        label = f"ID:{track_id} {class_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 标签背景
        label_y = max(y1 - th - 4, 0)
        cv2.rectangle(frame, (x1, label_y), (x1 + tw + 2, y1), color, -1)

        # 标签文字（白色）
        cv2.putText(
            frame, label, (x1 + 1, y1 - 3),
            font, font_scale, (255, 255, 255), thickness
        )

    return frame


def draw_trails(frame, trails, show_trails=True):
    """
    在画面上绘制所有追踪目标的运动轨迹

    参数:
        frame: BGR 格式的 numpy 数组图像（原地修改）
        trails: dict，格式 {track_id: [(cx, cy), ...], ...}
        show_trails: 是否绘制轨迹，默认 True

    返回:
        numpy.ndarray: 绘制后的图像
    """
    if not show_trails or not trails:
        return frame

    for track_id, points in trails.items():
        if len(points) < 2:
            continue
        color = get_color(track_id)
        pts_array = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [pts_array], isClosed=False, color=color, thickness=2)

    return frame


def draw_zone_count(frame, count, zone=None):
    """
    在画面上绘制区域内人数统计

    参数:
        frame: BGR 格式的 numpy 数组图像（原地修改）
        count: 区域内人数
        zone: 可选，用于在区域中心显示统计

    返回:
        numpy.ndarray: 绘制后的图像
    """
    h, w = frame.shape[:2]
    text = f"In Zone: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 255, 0) if count == 0 else (0, 165, 255)

    cv2.putText(frame, text, (10, h - 20), font, font_scale, color, thickness)
    return frame
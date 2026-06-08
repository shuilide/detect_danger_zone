# coding: utf-8
"""
多边形区域管理模块
支持鼠标绘制任意多边形，使用 cv2.pointPolygonTest 判断点是否在区域内
"""
import cv2
import numpy as np


class DangerZone:
    """危险区域管理类，支持多边形绘制与点位判断"""

    def __init__(self):
        """初始化多边形区域"""
        self.points = []          # 多边形顶点列表 [(x1, y1), (x2, y2), ...]
        self.is_closed = False    # 区域是否已闭合

    def add_point(self, x, y):
        """
        添加一个多边形的顶点

        参数:
            x: 顶点 x 坐标
            y: 顶点 y 坐标
        """
        self.points.append((int(x), int(y)))

    def clear(self):
        """清空所有顶点，重置区域状态"""
        self.points.clear()
        self.is_closed = False

    def get_points(self):
        """
        获取所有顶点

        返回:
            list[tuple]: 顶点坐标列表
        """
        return self.points.copy()

    def is_ready(self):
        """
        判断多边形是否至少有 3 个顶点（可闭合）

        返回:
            bool: 是否满足闭合条件
        """
        return len(self.points) >= 3

    def close(self):
        """闭合多边形（标记为已闭合，后续可进行点位判断）"""
        if self.is_ready():
            self.is_closed = True

    def is_point_inside(self, x, y):
        """
        使用 cv2.pointPolygonTest 判断指定点是否在多边形内部

        参数:
            x: 待检测点 x 坐标
            y: 待检测点 y 坐标

        返回:
            bool: 点是否在多边形内部（含边界）
        """
        if not self.is_closed or not self.is_ready():
            return False
        pts = np.array(self.points, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

    def draw_zone(self, frame, color=(0, 0, 255), thickness=2):
        """
        在画面上绘制多边形区域（半透明填充 + 边界线 + 顶点圆）

        参数:
            frame: BGR 格式的 numpy 数组图像（原地修改）
            color: 绘制颜色，默认红色 (0, 0, 255)
            thickness: 边界线粗细，默认 2

        返回:
            numpy.ndarray: 绘制后的图像
        """
        if not self.points:
            return frame

        pts = np.array(self.points, dtype=np.int32)

        # 如果区域已闭合，绘制半透明填充
        if self.is_closed:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
        else:
            # 未闭合时逐段画线
            for i in range(len(self.points) - 1):
                cv2.line(frame, self.points[i], self.points[i + 1], color, thickness)

        # 绘制所有顶点圆
        for pt in self.points:
            cv2.circle(frame, pt, 5, color, -1)

        return frame
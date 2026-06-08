# coding: utf-8
"""
ByteTrack 多目标追踪模块
基于 supervision 库实现人员追踪，维护每个 track_id 的运动轨迹
"""
import numpy as np
from collections import defaultdict
import supervision as sv


class ByteTrackTracker:
    """ByteTrack 多目标追踪器，给每个检测目标分配稳定唯一 ID 并记录轨迹"""

    def __init__(self):
        """初始化 ByteTrack 追踪器"""
        self.byte_track = sv.ByteTrack()
        self.trails = defaultdict(list)       # track_id -> [(cx, cy), ...]
        self.max_trail_length = 30            # 每个轨迹最多保留的坐标点数

    def update(self, detections):
        """
        用新的检测结果更新追踪状态

        参数:
            detections: supervision.tools.detections.Detections 对象
                        需要包含 xyxy, confidence, class_id 字段

        返回:
            supervision.tools.detections.Detections: 带 tracker_id 的追踪结果
        """
        # 画面中无人 → 清空所有轨迹，不做追踪
        if detections is None or len(detections) == 0:
            self.trails.clear()
            return detections

        tracked_detections = self.byte_track.update_with_detections(detections)

        # 更新每个 track_id 的轨迹点（使用边界框中心点）
        if tracked_detections.tracker_id is not None:
            for i, track_id in enumerate(tracked_detections.tracker_id):
                tid = int(track_id)
                x1, y1, x2, y2 = tracked_detections.xyxy[i]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                self.trails[tid].append((cx, cy))
                # 限制轨迹长度
                if len(self.trails[tid]) > self.max_trail_length:
                    self.trails[tid] = self.trails[tid][-self.max_trail_length:]

        # 清理已不在画面中的 track_id 轨迹（无延迟立即移除）
        if tracked_detections.tracker_id is not None:
            active_ids = set(int(tid) for tid in tracked_detections.tracker_id)
            stale_ids = [tid for tid in self.trails if tid not in active_ids]
            for tid in stale_ids:
                del self.trails[tid]

        return tracked_detections

    def get_trails(self):
        """
        获取所有追踪目标的轨迹点

        返回:
            dict: {track_id: [(cx, cy), ...], ...}
        """
        return dict(self.trails)

    def reset(self):
        """重置追踪器状态，清除所有轨迹"""
        self.byte_track = sv.ByteTrack()
        self.trails.clear()
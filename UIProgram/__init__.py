# coding: utf-8
"""
UIProgram 核心模块包
提供目标检测、追踪、区域管理和报警功能
"""
from .detector import YOLODetector
from .tracker import ByteTrackTracker
from .zone import DangerZone
from .alarm import AlarmSystem
from .utils import FPSCounter, get_color, draw_detection, draw_trails, draw_zone_count
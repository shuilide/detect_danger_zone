# coding: utf-8
"""
UIProgram 核心模块包
提供目标检测、追踪、区域管理、报警、多路管理、ReID 功能
"""
from .detector import YOLODetector
from .tracker import ByteTrackTracker
from .zone import DangerZone
from .alarm import AlarmSystem
from .utils import FPSCounter, get_color, draw_detection, draw_trails, draw_zone_count
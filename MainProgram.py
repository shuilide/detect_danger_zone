# coding: utf-8
"""
危险区域人员闯入检测与报警系统 - 主程序
基于 PyQt5 + YOLOv8 + ByteTrack + supervision 实现实时检测与报警

运行方式:
    python MainProgram.py
"""

import sys
import os

# ⚠️ 必须在 import cv2 / torch 之前设置，防止 OpenCV 和 PyTorch 的
# OpenMP/MKL 线程池在 QThread 中产生竞争，导致堆栈溢出 (0xC0000409)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import cv2
import numpy as np
import supervision as sv

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QComboBox, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from UIProgram.detector import YOLODetector
from UIProgram.tracker import ByteTrackTracker
from UIProgram.zone import DangerZone
from UIProgram.alarm import AlarmSystem
from UIProgram.utils import FPSCounter, get_color, draw_detection, draw_trails, draw_zone_count, frame_to_qimage
from UIProgram.multi_cam import MultiCamConfig, CameraWorker, MultiCamManager
from UIProgram.alarm_aggregator import AlarmAggregator, AlarmEvent


class VideoLabel(QLabel):
    """自定义视频显示 QLabel，支持鼠标点击事件（用于绘制区域）"""

    # 鼠标点击信号，传递映射回原始帧的坐标 (x, y)
    mouse_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background-color: #1e1e1e; border: 2px solid #555; color: #888;"
        )
        self.setFont(QFont("Microsoft YaHei", 16))
        self.setText("请打开视频或摄像头")

        # 保存最近一次显示的原始 QPixmap，用于窗口缩放时重新缩放
        self._last_pixmap = None
        # 缩放后的实际显示区域: (x, y, width, height)，相对于 QLabel 左上角
        self._display_rect = (0, 0, 0, 0)
        # 缩放比例 (scale_w, scale_h)，从 QLabel 坐标 → 原始帧坐标
        self._scale = (1.0, 1.0)

    def _update_display_rect(self, scaled_pixmap):
        """计算缩放后图像在 QLabel 内的实际显示区域和缩放比例"""
        if scaled_pixmap is None or scaled_pixmap.isNull():
            self._display_rect = (0, 0, 0, 0)
            self._scale = (1.0, 1.0)
            return

        label_w = self.width()
        label_h = self.height()
        img_w = scaled_pixmap.width()
        img_h = scaled_pixmap.height()

        if img_w <= 0 or img_h <= 0:
            self._display_rect = (0, 0, 0, 0)
            self._scale = (1.0, 1.0)
            return

        # 计算居中偏移（Qt.AlignCenter 会做居中）
        offset_x = (label_w - img_w) // 2
        offset_y = (label_h - img_h) // 2

        self._display_rect = (offset_x, offset_y, img_w, img_h)

        # 计算缩放比例：缩放后像素 → 原始帧像素
        if self._last_pixmap is not None and not self._last_pixmap.isNull():
            orig_w = self._last_pixmap.width()
            orig_h = self._last_pixmap.height()
            if orig_w > 0 and orig_h > 0:
                self._scale = (orig_w / img_w, orig_h / img_h)
            else:
                self._scale = (1.0, 1.0)
        else:
            self._scale = (1.0, 1.0)

    def mousePressEvent(self, event):
        """鼠标点击事件：将 QLabel 坐标映射回原始帧坐标后发射信号"""
        if event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            # 坐标转换：QLabel 坐标 → 原始视频帧坐标
            mapped_x, mapped_y = self._map_to_frame(x, y)
            self.mouse_clicked.emit(mapped_x, mapped_y)
        super().mousePressEvent(event)

    def _map_to_frame(self, label_x, label_y):
        """将 QLabel 上的点击坐标映射回原始视频帧坐标"""
        dx, dy, dw, dh = self._display_rect
        scale_w, scale_h = self._scale

        if dw <= 0 or dh <= 0:
            return label_x, label_y

        # 先减去黑边偏移，得到相对于显示区域的坐标
        rel_x = label_x - dx
        rel_y = label_y - dy

        # 边界裁剪
        rel_x = max(0, min(rel_x, dw))
        rel_y = max(0, min(rel_y, dh))

        # 按比例放大回原始帧坐标
        frame_x = int(rel_x * scale_w)
        frame_y = int(rel_y * scale_h)

        return frame_x, frame_y

    def resizeEvent(self, event):
        """窗口大小变化时重新缩放当前显示的图像"""
        super().resizeEvent(event)
        if self._last_pixmap is not None and not self._last_pixmap.isNull():
            scaled = self._last_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._update_display_rect(scaled)
            self.setPixmap(scaled)

    def display_frame(self, qt_image):
        """接收 QImage 并缩放显示在 QLabel 上"""
        pixmap = QPixmap.fromImage(qt_image)
        self._last_pixmap = pixmap
        scaled = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._update_display_rect(scaled)
        self.setPixmap(scaled)


class DetectionThread(QThread):
    """检测线程：视频读取 → 目标检测 → 追踪 → 区域判断 → 报警 → 绘图 → 发送帧"""

    # 信号定义
    update_frame_signal = pyqtSignal(QImage)               # 处理后的帧
    update_stats_signal = pyqtSignal(float, str, int, int) # fps, 时长, 区域内人数, 总人数
    alarm_status_signal = pyqtSignal(bool)                 # 报警状态
    video_finished_signal = pyqtSignal()                    # 视频播放结束

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None           # 视频源：文件路径或摄像头索引
        self.is_camera = False       # 是否为摄像头模式
        self.running = False

        # 共享检测组件（由主线程传入）
        self.detector = None
        self.tracker = None
        self.zone = None
        self.alarm = None
        self.fps_counter = None

        # 检测参数（运行时可由主线程更新）
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.alarm_threshold = 1

        # 显示选项（运行时可由主线程更新）
        self.show_bbox = True
        self.show_label = True
        self.show_trails = True

        # 性能优化选项
        self.detect_every_n_frames = 2   # 每 N 帧检测一次（1=每帧都检测）
        self.detect_width = 640           # 检测前缩放到该宽度（None=不缩放）
        self._cached_detections = []      # 跳帧期间的缓存检测结果
        self._cached_tracked = None       # 跳帧期间的缓存追踪结果
        self._cached_trails = {}          # 跳帧期间的缓存轨迹

    def set_source(self, source, is_camera=False):
        """设置视频源"""
        self.source = source
        self.is_camera = is_camera

    def set_params(self, conf_thres, iou_thres, alarm_threshold):
        """更新检测参数（线程安全）"""
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.alarm_threshold = alarm_threshold

    def set_display_options(self, show_bbox, show_label, show_trails):
        """更新显示选项（线程安全）"""
        self.show_bbox = show_bbox
        self.show_label = show_label
        self.show_trails = show_trails

    def set_shared_objects(self, detector, tracker, zone, alarm, fps_counter):
        """设置共享的检测组件对象引用"""
        self.detector = detector
        self.tracker = tracker
        self.zone = zone
        self.alarm = alarm
        self.fps_counter = fps_counter

    def stop(self):
        """安全停止检测线程"""
        self.running = False

    def run(self):
        """检测线程主循环"""
        self.running = True

        # ⚠️ 二次保障：限制本线程内 OpenCV / PyTorch 并行度
        # 多线程 OpenMP 在 QThread 中会导致堆栈溢出 (0xC0000409)
        cv2.setNumThreads(1)
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass

        # ---- 打开视频源 ----
        if self.is_camera:
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            video_fps = 30          # 摄像头默认帧率
            start_time = time.time()
        else:
            cap = cv2.VideoCapture(self.source)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30

        if not cap.isOpened():
            self.video_finished_signal.emit()
            return

        frame_count = 0
        _last_emit_time = 0.0          # 帧发送限速（防止 UI 信号队列积压）

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # ========== 1. 目标检测（跳帧优化） ==========
            if frame_count % self.detect_every_n_frames == 1 or self.detect_every_n_frames <= 1:
                # 缩放帧以加速推理（小分辨率 → 更快）
                if self.detect_width and frame.shape[1] > self.detect_width:
                    h, w = frame.shape[:2]
                    scale = self.detect_width / w
                    detect_frame = cv2.resize(frame, (self.detect_width, int(h * scale)))
                else:
                    detect_frame = frame
                    scale = 1.0

                detections_list = self.detector.detect(
                    detect_frame,
                    conf_thres=self.conf_thres,
                    iou_thres=self.iou_thres
                )

                # 将检测框坐标从缩放帧映射回原始帧
                if scale != 1.0:
                    for d in detections_list:
                        x1, y1, x2, y2 = d['bbox']
                        d['bbox'] = (int(x1 / scale), int(y1 / scale),
                                     int(x2 / scale), int(y2 / scale))

                # 转为 supervision Detections 格式
                if detections_list:
                    xyxy = np.array([d['bbox'] for d in detections_list], dtype=np.float32)
                    conf = np.array([d['confidence'] for d in detections_list], dtype=np.float32)
                    cls = np.array([d['class_id'] for d in detections_list], dtype=np.int64)
                    sv_detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
                else:
                    sv_detections = sv.Detections.empty()

                # ========== 2. 多目标追踪 ==========
                tracked_detections = self.tracker.update(sv_detections)
                trails = self.tracker.get_trails()

                # 缓存结果供跳帧使用
                self._cached_detections = detections_list
                self._cached_tracked = tracked_detections
                self._cached_trails = trails
            else:
                # 跳过检测：复用上一帧的检测和追踪结果
                tracked_detections = self._cached_tracked
                trails = self._cached_trails

            total_count = len(self._cached_detections) if self._cached_detections else 0

            # ========== 3. 区域判断 ==========
            zone_count = 0
            if (tracked_detections is not None
                    and tracked_detections.tracker_id is not None):
                for i in range(len(tracked_detections)):
                    x1, y1, x2, y2 = tracked_detections.xyxy[i]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    if self.zone.is_point_inside(cx, cy):
                        zone_count += 1

            # ========== 4. 报警检查 ==========
            frame, is_alarming = self.alarm.check_and_alarm(
                zone_count, self.alarm_threshold, frame
            )
            self.alarm_status_signal.emit(is_alarming)

            # ========== 5. 绘制检测框和标签 ==========
            if (tracked_detections is not None
                    and tracked_detections.tracker_id is not None):
                for i in range(len(tracked_detections)):
                    bbox = tracked_detections.xyxy[i]
                    track_id = int(tracked_detections.tracker_id[i])
                    conf_val = (
                        float(tracked_detections.confidence[i])
                        if tracked_detections.confidence is not None
                        else 0.0
                    )
                    color = get_color(track_id)

                    if self.show_bbox:
                        draw_detection(
                            frame, bbox, track_id,
                            self.detector.get_class_name(int(tracked_detections.class_id[i])),
                            conf_val, color,
                            show_label=self.show_label
                        )

            # ========== 6. 绘制轨迹 ==========
            if trails:
                draw_trails(frame, trails, show_trails=self.show_trails)

            # ========== 7. 绘制区域多边形 ==========
            self.zone.draw_zone(frame)

            # ========== 8. 绘制 FPS ==========
            self.fps_counter.draw_fps(frame)

            # ========== 9. 绘制区域内人数 ==========
            draw_zone_count(frame, zone_count)

            # ========== 10. 计算统计信息 ==========
            fps = self.fps_counter.update()

            if self.is_camera:
                elapsed = time.time() - start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                duration_str = f"{minutes:02d}:{seconds:02d}"
            else:
                total_seconds = frame_count / video_fps
                minutes = int(total_seconds // 60)
                seconds = int(total_seconds % 60)
                duration_str = f"{minutes:02d}:{seconds:02d}"

            self.update_stats_signal.emit(fps, duration_str, zone_count, total_count)

            # ========== 11. 帧转为 QImage 发送到主线程（限速，防止队列积压） ==========
            now = time.time()
            if now - _last_emit_time >= 0.033:  # 最多 30 FPS 显示
                qt_image = frame_to_qimage(frame)
                self.update_frame_signal.emit(qt_image)
                _last_emit_time = now

        # 清理
        cap.release()
        self.video_finished_signal.emit()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("危险区域人员闯入检测与报警系统")
        self.resize(1200, 700)

        # ================================================================
        # 检测模式配置 {模式名称: (模型路径, 类别映射)}
        # 类别映射格式: {class_id: 'class_name'}
        # ================================================================
        self.DETECT_MODES = {
            '人员检测': {
                'model': 'models/yolov8n.pt',
                'target_classes': {0: 'person'},
                'warning_text': "WARNING: 危险区域人员闯入!!",
            },
            '无人机检测': {
                'model': 'models/best.pt',
                'target_classes': {0: 'person', 1: 'drone'},
                'warning_text': "WARNING: 危险区域无人机闯入!!",
            },
        }
        self.current_mode = '人员检测'    # 默认检测模式

        # ---- 运行状态 ----
        self.video_opened = False
        self.camera_opened = False
        self.drawing_mode = False

        # ---- 检测组件 ----
        self.detector = None          # YOLO 检测器（延迟初始化）
        self.tracker = None           # ByteTrack 追踪器
        self.zone = DangerZone()      # 危险区域管理
        self.alarm = AlarmSystem(alarm_sound_path='alarm.wav')
        self.fps_counter = FPSCounter()

        # ---- 检测线程 ----
        self.detect_thread = None

        # ---- 初始化界面 ----
        self._init_ui()
        self._apply_style()

    # ======================== 界面初始化 ========================

    def _init_ui(self):
        """构建主界面布局"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ----- 左侧：视频显示区域 -----
        self.video_label = VideoLabel()
        self.video_label.mouse_clicked.connect(self._on_video_label_clicked)
        main_layout.addWidget(self.video_label, stretch=3)

        # ----- 右侧：控制面板 -----
        right_panel = QWidget()
        right_panel.setFixedWidth(280)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)

        # -- 按钮组 --
        btn_group = QGroupBox("操作控制")
        btn_layout = QVBoxLayout(btn_group)

        self.btn_open_video = QPushButton("📂 打开视频")
        self.btn_open_video.clicked.connect(self._on_open_video)
        btn_layout.addWidget(self.btn_open_video)

        self.btn_open_camera = QPushButton("📷 打开摄像头")
        self.btn_open_camera.clicked.connect(self._on_open_camera)
        btn_layout.addWidget(self.btn_open_camera)

        self.btn_draw_zone = QPushButton("✏️ 绘制区域")
        self.btn_draw_zone.clicked.connect(self._on_draw_zone)
        btn_layout.addWidget(self.btn_draw_zone)

        self.btn_finish_zone = QPushButton("✅ 绘制完成")
        self.btn_finish_zone.setEnabled(False)
        self.btn_finish_zone.clicked.connect(self._on_finish_zone)
        btn_layout.addWidget(self.btn_finish_zone)

        right_layout.addWidget(btn_group)

        # -- 检测模式选择 --
        mode_group = QGroupBox("检测模式")
        mode_layout = QVBoxLayout(mode_group)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(list(self.DETECT_MODES.keys()))
        self.cmb_mode.setCurrentText(self.current_mode)
        self.cmb_mode.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.cmb_mode)

        self.lbl_mode_hint = QLabel(
            "切换模式前请先关闭当前视频源，\n否则需要手动重新打开。"
        )
        self.lbl_mode_hint.setWordWrap(True)
        self.lbl_mode_hint.setStyleSheet("color: #999; font: 11px 'Microsoft YaHei';")
        mode_layout.addWidget(self.lbl_mode_hint)

        right_layout.addWidget(mode_group)

        # -- 参数设置组 --
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout(param_group)

        # 置信度阈值
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("置信度阈值:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.5)
        self.conf_spin.valueChanged.connect(self._on_params_changed)
        row1.addWidget(self.conf_spin)
        param_layout.addLayout(row1)

        # IoU 阈值
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("IoU 阈值:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.5)
        self.iou_spin.valueChanged.connect(self._on_params_changed)
        row2.addWidget(self.iou_spin)
        param_layout.addLayout(row2)

        # 报警阈值
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("报警阈值:"))
        self.alarm_spin = QSpinBox()
        self.alarm_spin.setRange(1, 100)
        self.alarm_spin.setValue(1)
        self.alarm_spin.valueChanged.connect(self._on_params_changed)
        row3.addWidget(self.alarm_spin)
        param_layout.addLayout(row3)

        right_layout.addWidget(param_group)

        # -- 显示选项组 --
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout(display_group)

        self.chk_bbox = QCheckBox("显示检测框")
        self.chk_bbox.setChecked(True)
        self.chk_bbox.stateChanged.connect(self._on_display_options_changed)
        display_layout.addWidget(self.chk_bbox)

        self.chk_label = QCheckBox("显示标签")
        self.chk_label.setChecked(True)
        self.chk_label.stateChanged.connect(self._on_display_options_changed)
        display_layout.addWidget(self.chk_label)

        self.chk_trails = QCheckBox("显示追踪轨迹")
        self.chk_trails.setChecked(True)
        self.chk_trails.stateChanged.connect(self._on_display_options_changed)
        display_layout.addWidget(self.chk_trails)

        right_layout.addWidget(display_group)

        # -- 信息显示组 --
        info_group = QGroupBox("检测信息")
        info_layout = QVBoxLayout(info_group)

        self.lbl_fps = QLabel("FPS: --")
        info_layout.addWidget(self.lbl_fps)

        self.lbl_duration = QLabel("检测时长: --")
        info_layout.addWidget(self.lbl_duration)

        self.lbl_zone_count = QLabel("区域内目标: 0")
        info_layout.addWidget(self.lbl_zone_count)

        self.lbl_total_count = QLabel("检测目标总数: 0")
        info_layout.addWidget(self.lbl_total_count)

        right_layout.addWidget(info_group)

        # 底部弹簧
        right_layout.addStretch()
        main_layout.addWidget(right_panel)

    def _apply_style(self):
        """全局深色主题样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #ddd;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 12px;
                color: #ddd;
                font: 13px "Microsoft YaHei";
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
            QLabel {
                color: #ccc;
                font: 13px "Microsoft YaHei";
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 2px 4px;
                color: #ddd;
                font: 12px "Microsoft YaHei";
                min-width: 70px;
            }
            QCheckBox {
                color: #ccc;
                font: 13px "Microsoft YaHei";
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px 8px;
                color: #ddd;
                font: 13px "Microsoft YaHei";
                min-width: 100px;
            }
            QComboBox:hover {
                border-color: #777;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                border: 1px solid #555;
                color: #ddd;
                selection-background-color: #4a4a4a;
            }
        """)

    # ======================== 按钮事件 ========================

    def _on_mode_changed(self, mode_name):
        """检测模式切换：释放旧检测器，更新报警文字"""
        if mode_name == self.current_mode:
            return
        self.current_mode = mode_name
        # 同步报警文字
        self.alarm.warning_text = self.DETECT_MODES[mode_name]['warning_text']
        # 释放旧检测器，下次 _start_detection 会创建新的
        if self.detector is not None:
            del self.detector
            self.detector = None
            print(f"[MainWindow] 检测模式已切换为: {mode_name}，模型将在下次打开视频源时加载")

    def _on_open_video(self):
        """打开 / 关闭视频文件"""
        if self.video_opened:
            self._stop_detection()
            self._reset_ui_state()
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;所有文件 (*.*)"
        )
        if not file_path:
            return

        self.video_label.setText("正在加载视频...")
        self._start_detection(file_path, is_camera=False)
        self.video_opened = True
        self.camera_opened = False
        self.btn_open_video.setText("⏹ 关闭视频")
        self.btn_open_camera.setText("📷 打开摄像头")

    def _on_open_camera(self):
        """打开 / 关闭摄像头"""
        if self.camera_opened:
            self._stop_detection()
            self._reset_ui_state()
            return

        self.video_label.setText("正在打开摄像头...")
        self._start_detection(0, is_camera=True)
        self.camera_opened = True
        self.video_opened = False
        self.btn_open_camera.setText("⏹ 关闭摄像头")
        self.btn_open_video.setText("📂 打开视频")

    def _on_draw_zone(self):
        """进入区域绘制模式"""
        if not self.video_opened and not self.camera_opened:
            QMessageBox.warning(self, "提示", "请先打开视频或摄像头再绘制区域！")
            return

        self.drawing_mode = True
        self.zone.clear()
        self.btn_draw_zone.setEnabled(False)
        self.btn_finish_zone.setEnabled(False)
        self.video_label.setCursor(Qt.CrossCursor)

    def _on_finish_zone(self):
        """完成区域绘制：闭合多边形"""
        if not self.zone.is_ready():
            QMessageBox.warning(self, "提示", "至少需要 3 个顶点才能闭合多边形！")
            return

        self.zone.close()
        self.drawing_mode = False
        self.btn_draw_zone.setEnabled(True)
        self.btn_finish_zone.setEnabled(False)
        self.video_label.setCursor(Qt.ArrowCursor)

    def _on_video_label_clicked(self, x, y):
        """视频显示区域鼠标点击（仅在绘制模式下响应）"""
        if not self.drawing_mode:
            return

        self.zone.add_point(x, y)

        # 达到 3 个点后启用"绘制完成"按钮
        if self.zone.is_ready():
            self.btn_finish_zone.setEnabled(True)

    # ======================== 参数 / 显示选项变更 ========================

    def _on_params_changed(self):
        """检测参数变更时同步到检测线程"""
        if self.detect_thread is not None and self.detect_thread.isRunning():
            self.detect_thread.set_params(
                self.conf_spin.value(),
                self.iou_spin.value(),
                self.alarm_spin.value()
            )

    def _on_display_options_changed(self):
        """显示选项变更时同步到检测线程"""
        if self.detect_thread is not None and self.detect_thread.isRunning():
            self.detect_thread.set_display_options(
                self.chk_bbox.isChecked(),
                self.chk_label.isChecked(),
                self.chk_trails.isChecked()
            )

    # ======================== 检测线程管理 ========================

    def _start_detection(self, source, is_camera=False):
        """创建并启动检测线程"""
        # 先停止旧线程
        self._stop_detection()

        # 清空上次绘制的危险区域
        self.zone.clear()

        # 同步报警文字为当前模式的配置
        self.alarm.warning_text = self.DETECT_MODES[self.current_mode]['warning_text']

        # 延迟初始化 YOLO 检测器（根据当前选择的模式动态加载）
        if self.detector is None:
            try:
                mode_cfg = self.DETECT_MODES[self.current_mode]
                print(f"[MainWindow] 加载检测模式: {self.current_mode}")
                print(f"[MainWindow]   模型路径: {mode_cfg['model']}")
                print(f"[MainWindow]   目标类别: {mode_cfg['target_classes']}")
                self.detector = YOLODetector(
                    model_path=mode_cfg['model'],
                    device='cpu',
                    target_classes=mode_cfg['target_classes']
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载 YOLO 模型失败:\n{str(e)}")
                self.video_label.setText("请打开视频或摄像头")
                return

        # 重置追踪器与 FPS 计数器（每次新视频源重置）
        self.tracker = ByteTrackTracker()
        self.fps_counter = FPSCounter()

        # 创建检测线程
        self.detect_thread = DetectionThread()
        self.detect_thread.set_source(source, is_camera)
        self.detect_thread.set_params(
            self.conf_spin.value(),
            self.iou_spin.value(),
            self.alarm_spin.value()
        )
        self.detect_thread.set_display_options(
            self.chk_bbox.isChecked(),
            self.chk_label.isChecked(),
            self.chk_trails.isChecked()
        )
        self.detect_thread.set_shared_objects(
            self.detector, self.tracker, self.zone,
            self.alarm, self.fps_counter
        )

        # 连接信号到槽
        self.detect_thread.update_frame_signal.connect(
            self.video_label.display_frame
        )
        self.detect_thread.update_stats_signal.connect(self._on_stats_received)
        self.detect_thread.alarm_status_signal.connect(
            self._on_alarm_status_changed
        )
        self.detect_thread.video_finished_signal.connect(
            self._on_video_finished
        )

        self.detect_thread.start()

    def _stop_detection(self):
        """安全停止检测线程"""
        # 先停止报警，避免关闭视频后警报继续响
        self.alarm._stop_alarm()
        if self.detect_thread is not None:
            self.detect_thread.stop()
            self.detect_thread.wait(3000)  # 最多等待 3 秒
            self.detect_thread = None

    # ======================== 信号槽：接收线程数据 ========================

    def _on_stats_received(self, fps, duration_str, zone_count, total_count):
        """更新右侧信息面板的统计数值"""
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_duration.setText(f"检测时长: {duration_str}")
        self.lbl_zone_count.setText(f"区域内人数: {zone_count}")
        self.lbl_total_count.setText(f"画面总人数: {total_count}")

    def _on_alarm_status_changed(self, is_alarming):
        """报警状态变化时，高亮区域内人数标签"""
        if is_alarming:
            self.lbl_zone_count.setStyleSheet(
                "color: red; font-weight: bold; font: 14px 'Microsoft YaHei';"
            )
        else:
            self.lbl_zone_count.setStyleSheet(
                "color: #ccc; font: 13px 'Microsoft YaHei';"
            )

    def _on_video_finished(self):
        """视频播放结束，重置界面"""
        self._stop_detection()
        self._reset_ui_state()
        self.video_label.setText("视频播放结束")

    # ======================== 辅助方法 ========================

    def _reset_ui_state(self):
        """重置界面按钮和状态"""
        self.btn_open_video.setText("📂 打开视频")
        self.btn_open_camera.setText("📷 打开摄像头")
        self.video_opened = False
        self.camera_opened = False
        self.drawing_mode = False
        self.btn_draw_zone.setEnabled(True)
        self.btn_finish_zone.setEnabled(False)
        self.video_label.setCursor(Qt.ArrowCursor)
        self.video_label.setText("请打开视频或摄像头")

        # 清空危险区域
        self.zone.clear()

        # 重置统计显示
        self.lbl_fps.setText("FPS: --")
        self.lbl_duration.setText("检测时长: --")
        self.lbl_zone_count.setText("区域内目标: 0")
        self.lbl_total_count.setText("检测目标总数: 0")
        self.lbl_zone_count.setStyleSheet(
            "color: #ccc; font: 13px 'Microsoft YaHei';"
        )

    def closeEvent(self, event):
        """窗口关闭时确保检测线程被正确停止"""
        self._stop_detection()
        event.accept()


# ======================================================================
# 多摄像头模式窗口
# ======================================================================

class MultiCamWindow(QMainWindow):
    """多摄像头多视频联动主窗口"""

    def __init__(self, config_path: str):
        super().__init__()
        self.setWindowTitle("多摄像头联动检测系统 - MultiCam")
        self.resize(1400, 800)

        # ---- 加载配置 ----
        self._config = MultiCamConfig(config_path)
        self._camera_ids = self._config.camera_ids

        # ---- 运行状态 ----
        self._running = False
        self._active_camera_id: str = self._camera_ids[0] if self._camera_ids else ""
        self._drawing_mode = False

        # ---- 多路管理器 ----
        self._manager = MultiCamManager(self._config)

        # ---- 告警聚合器 ----
        self._alarm_aggregator = AlarmAggregator(
            dedup_window_seconds=self._config.global_config.alarm_dedup_window_seconds
        )
        self._alarm_aggregator.new_alarm.connect(self._on_new_alarm)

        # ---- 视频标签缓存 ----
        self._video_labels: dict = {}       # camera_id → VideoLabel

        # ---- 界面 ----
        self._init_ui()
        self._apply_style()

    # ==================== UI 初始化 ====================

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ----- 左侧：视频网格区域 -----
        self._grid_widget = QWidget()
        self._grid_layout = None  # 由 _build_grid 动态创建
        self._build_video_grid()
        main_layout.addWidget(self._grid_widget, stretch=3)

        # ----- 右侧：控制面板 -----
        right_panel = QWidget()
        right_panel.setFixedWidth(280)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)

        # -- 摄像头选择 --
        cam_group = QGroupBox("当前摄像头")
        cam_layout = QVBoxLayout(cam_group)
        self._cmb_camera = QComboBox()
        self._cmb_camera.addItems(self._camera_ids)
        self._cmb_camera.currentTextChanged.connect(self._on_active_camera_changed)
        cam_layout.addWidget(self._cmb_camera)
        right_layout.addWidget(cam_group)

        # -- 操作按钮 --
        btn_group = QGroupBox("操作控制")
        btn_layout = QVBoxLayout(btn_group)

        self._btn_start = QPushButton("▶ 启动全部")
        self._btn_start.clicked.connect(self._on_start_all)
        btn_layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("⏹ 停止全部")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop_all)
        btn_layout.addWidget(self._btn_stop)

        self._btn_draw_zone = QPushButton("✏️ 绘制区域")
        self._btn_draw_zone.setEnabled(False)
        self._btn_draw_zone.clicked.connect(self._on_draw_zone)
        btn_layout.addWidget(self._btn_draw_zone)

        self._btn_finish_zone = QPushButton("✅ 绘制完成")
        self._btn_finish_zone.setEnabled(False)
        self._btn_finish_zone.clicked.connect(self._on_finish_zone)
        btn_layout.addWidget(self._btn_finish_zone)

        right_layout.addWidget(btn_group)

        # -- 参数设置 --
        param_group = QGroupBox("参数设置（当前选中摄像头）")
        param_layout = QVBoxLayout(param_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("置信度:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.1, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.5)
        self._conf_spin.valueChanged.connect(self._on_params_changed)
        row1.addWidget(self._conf_spin)
        param_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("IoU:"))
        self._iou_spin = QDoubleSpinBox()
        self._iou_spin.setRange(0.1, 1.0)
        self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setDecimals(2)
        self._iou_spin.setValue(0.5)
        self._iou_spin.valueChanged.connect(self._on_params_changed)
        row2.addWidget(self._iou_spin)
        param_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("报警阈值:"))
        self._alarm_spin = QSpinBox()
        self._alarm_spin.setRange(1, 100)
        self._alarm_spin.setValue(1)
        self._alarm_spin.valueChanged.connect(self._on_params_changed)
        row3.addWidget(self._alarm_spin)
        param_layout.addLayout(row3)

        right_layout.addWidget(param_group)

        # -- 显示选项 --
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout(display_group)

        self._chk_bbox = QCheckBox("显示检测框")
        self._chk_bbox.setChecked(True)
        self._chk_bbox.stateChanged.connect(self._on_display_changed)
        display_layout.addWidget(self._chk_bbox)

        self._chk_label = QCheckBox("显示标签")
        self._chk_label.setChecked(True)
        self._chk_label.stateChanged.connect(self._on_display_changed)
        display_layout.addWidget(self._chk_label)

        self._chk_trails = QCheckBox("显示追踪轨迹")
        self._chk_trails.setChecked(True)
        self._chk_trails.stateChanged.connect(self._on_display_changed)
        display_layout.addWidget(self._chk_trails)

        right_layout.addWidget(display_group)

        # -- 状态信息 --
        info_group = QGroupBox("检测信息")
        info_layout = QVBoxLayout(info_group)
        self._lbl_status = QLabel('就绪 — 请点击「启动全部」')
        self._lbl_status.setWordWrap(True)
        info_layout.addWidget(self._lbl_status)
        right_layout.addWidget(info_group)

        right_layout.addStretch()
        main_layout.addWidget(right_panel)

    def _build_video_grid(self):
        """根据摄像头数量动态构建网格布局"""
        if self._grid_layout is not None:
            # 清除旧布局
            while self._grid_layout.count():
                item = self._grid_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        from PyQt5.QtWidgets import QGridLayout
        n = len(self._camera_ids)
        if n == 0:
            return

        # 计算行列数
        if n <= 2:
            rows, cols = 1, n
        elif n <= 4:
            rows, cols = 2, 2
        elif n <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3

        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(4)

        # 均分列宽和行高，确保所有视频框大小一致
        for c in range(cols):
            self._grid_layout.setColumnStretch(c, 1)
        for r in range(rows):
            self._grid_layout.setRowStretch(r, 1)

        self._video_labels.clear()
        for idx, cam_id in enumerate(self._camera_ids):
            label = VideoLabel()
            label.setMinimumSize(320, 240)
            label.mouse_clicked.connect(self._make_click_handler(cam_id))
            r, c = divmod(idx, cols)
            self._grid_layout.addWidget(label, r, c)
            self._video_labels[cam_id] = label

    def _make_click_handler(self, camera_id: str):
        """为每个 VideoLabel 创建绑定了 camera_id 的点击处理器"""
        def handler(x, y):
            self._on_video_clicked(camera_id, x, y)
        return handler

    # ==================== 按钮/控件事件 ====================

    def _on_active_camera_changed(self, cam_id: str):
        self._active_camera_id = cam_id
        # 同步参数控件的值为当前选中摄像头的值
        worker = self._manager.get_worker(cam_id)
        if worker is not None:
            self._conf_spin.blockSignals(True)
            self._iou_spin.blockSignals(True)
            self._alarm_spin.blockSignals(True)
            self._conf_spin.setValue(worker.conf_thres)
            self._iou_spin.setValue(worker.iou_thres)
            self._alarm_spin.setValue(worker.alarm_threshold)
            self._conf_spin.blockSignals(False)
            self._iou_spin.blockSignals(False)
            self._alarm_spin.blockSignals(False)

    def _on_start_all(self):
        """启动所有摄像头"""
        self._running = True
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_draw_zone.setEnabled(True)

        # 启动所有 worker
        self._manager.start_all()

        # 连接每个 worker 的信号
        for cam_id, worker in self._manager.get_all_workers().items():
            video_label = self._video_labels.get(cam_id)
            if video_label:
                worker.update_frame.connect(self._make_frame_handler(cam_id))
                worker.update_stats.connect(self._on_stats_received)
                worker.alarm_event.connect(self._on_raw_alarm)
                worker.video_finished.connect(self._on_video_finished)

            # 设置显示选项
            worker.set_display_options(
                self._chk_bbox.isChecked(),
                self._chk_label.isChecked(),
                self._chk_trails.isChecked()
            )

        self._manager.all_finished.connect(self._on_all_finished)
        self._lbl_status.setText(f"运行中 — {self._manager.worker_count} 路视频")

    def _on_stop_all(self):
        """停止所有摄像头"""
        self._running = False
        self._alarm_aggregator.reset()
        self._manager.stop_all()
        self._reset_ui()

    def _on_draw_zone(self):
        cam_id = self._active_camera_id
        zone = self._manager.get_zone(cam_id)
        if zone is None:
            return
        self._drawing_mode = True
        zone.clear()
        self._btn_draw_zone.setEnabled(False)
        self._btn_finish_zone.setEnabled(False)
        label = self._video_labels.get(cam_id)
        if label:
            label.setCursor(Qt.CrossCursor)

    def _on_finish_zone(self):
        cam_id = self._active_camera_id
        zone = self._manager.get_zone(cam_id)
        if zone is None or not zone.is_ready():
            QMessageBox.warning(self, "提示", "至少需要 3 个顶点才能闭合多边形！")
            return
        zone.close()
        self._drawing_mode = False
        self._btn_draw_zone.setEnabled(True)
        self._btn_finish_zone.setEnabled(False)
        label = self._video_labels.get(cam_id)
        if label:
            label.setCursor(Qt.ArrowCursor)

    def _on_video_clicked(self, camera_id: str, x: int, y: int):
        if not self._drawing_mode:
            return
        if camera_id != self._active_camera_id:
            return
        zone = self._manager.get_zone(camera_id)
        if zone is None:
            return
        zone.add_point(x, y)
        if zone.is_ready():
            self._btn_finish_zone.setEnabled(True)

    # ==================== 参数 / 显示变更 ====================

    def _on_params_changed(self):
        cam_id = self._active_camera_id
        worker = self._manager.get_worker(cam_id)
        if worker is not None and worker.isRunning():
            worker.set_params(
                self._conf_spin.value(),
                self._iou_spin.value(),
                self._alarm_spin.value()
            )

    def _on_display_changed(self):
        for worker in self._manager.get_all_workers().values():
            if worker.isRunning():
                worker.set_display_options(
                    self._chk_bbox.isChecked(),
                    self._chk_label.isChecked(),
                    self._chk_trails.isChecked()
                )

    # ==================== 信号处理 ====================

    def _make_frame_handler(self, camera_id: str):
        """为每路视频创建帧显示回调"""
        def handle(cam_id, qt_image):
            label = self._video_labels.get(cam_id)
            if label:
                label.display_frame(qt_image)
        return handle

    def _on_stats_received(self, camera_id: str, stats: dict):
        if camera_id == self._active_camera_id:
            fps = stats.get('fps', 0)
            dur = stats.get('duration', '--')
            zc = stats.get('zone_count', 0)
            tc = stats.get('total_count', 0)
            alarming = stats.get('is_alarming', False)
            dt = stats.get('detect_time_ms', 0)
            gs = stats.get('gallery_size', 0)

            status_text = (
                f"Cam: {camera_id}\n"
                f"FPS: {fps:.1f} | 检测: {dt:.0f}ms\n"
                f"区域内: {zc} | 总人数: {tc}\n"
                f"全局库: {gs} 人 | 时长: {dur}"
            )
            if alarming:
                status_text += "\n⚠ 报警中!"
                self._lbl_status.setStyleSheet(
                    "color: red; font-weight: bold; font: 13px 'Microsoft YaHei';"
                )
            else:
                self._lbl_status.setStyleSheet(
                    "color: #ccc; font: 13px 'Microsoft YaHei';"
                )
            self._lbl_status.setText(status_text)

    def _on_raw_alarm(self, camera_id: str, alarm_data: dict):
        """原始告警事件 → 送入聚合器去重"""
        event = self._alarm_aggregator.process(
            camera_id=camera_id,
            global_person_ids=alarm_data.get('global_person_ids', []),
            zone_count=alarm_data.get('zone_count', 0),
            threshold=alarm_data.get('threshold', 1),
            timestamp=alarm_data.get('timestamp'),
        )
        # event 为 None 表示被去重或持续告警

    def _on_new_alarm(self, event: AlarmEvent):
        """去重后的新告警事件"""
        cam_list = ', '.join(event.merged_from) if event.is_merged else event.camera_id
        gid_list = ', '.join(f'#{gid}' for gid in event.global_person_ids)
        print(f"[MultiCam] ⚠ 新告警: 人员 {gid_list} 闯入 "
              f"摄像头 [{cam_list}] 区域 "
              f"(区域内 {event.zone_count} 人, 阈值 {event.threshold})")

    def _on_video_finished(self, camera_id: str):
        label = self._video_labels.get(camera_id)
        if label:
            label.setText(f"[{camera_id}] 播放结束")

    def _on_all_finished(self):
        self._lbl_status.setText("全部视频播放完毕")
        self._btn_stop.setEnabled(False)
        self._btn_start.setEnabled(True)
        self._btn_draw_zone.setEnabled(False)

    # ==================== 辅助方法 ====================

    def _reset_ui(self):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_draw_zone.setEnabled(False)
        self._btn_finish_zone.setEnabled(False)
        self._drawing_mode = False
        self._lbl_status.setText("已停止")
        self._lbl_status.setStyleSheet("color: #ccc; font: 13px 'Microsoft YaHei';")
        for label in self._video_labels.values():
            label.setText("等待启动...")
            label.setCursor(Qt.ArrowCursor)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QGroupBox {
                border: 1px solid #555; border-radius: 5px;
                margin-top: 10px; padding-top: 10px;
                color: #ddd; font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px;
            }
            QPushButton {
                background-color: #3c3c3c; border: 1px solid #555;
                border-radius: 4px; padding: 6px 12px;
                color: #ddd; font: 13px "Microsoft YaHei"; min-height: 24px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:pressed { background-color: #2a2a2a; }
            QPushButton:disabled { background-color: #333; color: #666; }
            QLabel { color: #ccc; font: 13px "Microsoft YaHei"; }
            QDoubleSpinBox, QSpinBox {
                background-color: #3c3c3c; border: 1px solid #555;
                border-radius: 3px; padding: 2px 4px;
                color: #ddd; font: 12px "Microsoft YaHei"; min-width: 70px;
            }
            QCheckBox { color: #ccc; font: 13px "Microsoft YaHei"; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QComboBox {
                background-color: #3c3c3c; border: 1px solid #555;
                border-radius: 3px; padding: 4px 8px;
                color: #ddd; font: 13px "Microsoft YaHei"; min-width: 100px;
            }
            QComboBox:hover { border-color: #777; }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c; border: 1px solid #555;
                color: #ddd; selection-background-color: #4a4a4a;
            }
        """)

    def closeEvent(self, event):
        self._manager.cleanup()
        event.accept()


def main():
    """程序入口

    python MainProgram.py                        # 单路模式（原 GUI）
    python MainProgram.py --config config/multi_cam.yaml   # 多路联动模式
    """
    import argparse

    parser = argparse.ArgumentParser(description="危险区域检测系统")
    parser.add_argument(
        '--config', type=str, default=None,
        help='多摄像头配置文件路径（YAML），不提供则使用单路模式'
    )
    args = parser.parse_args()

    # 自适应高 DPI 显示
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))

    if args.config:
        if not os.path.exists(args.config):
            print(f"[Error] 配置文件不存在: {args.config}")
            sys.exit(1)
        print(f"[Main] 多摄像头联动模式，配置: {args.config}")
        window = MultiCamWindow(args.config)
    else:
        print("[Main] 单路检测模式")
        window = MainWindow()

    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
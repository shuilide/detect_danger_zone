# coding: utf-8
"""
危险区域人员闯入检测与报警系统 - 主程序
基于 PyQt5 + YOLOv8 + ByteTrack + supervision 实现实时检测与报警

运行方式:
    python MainProgram.py
"""

import sys
import os
import time
import cv2
import numpy as np
import supervision as sv

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from UIProgram.detector import YOLODetector
from UIProgram.tracker import ByteTrackTracker
from UIProgram.zone import DangerZone
from UIProgram.alarm import AlarmSystem
from UIProgram.utils import FPSCounter, get_color, draw_detection, draw_trails, draw_zone_count


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

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # ========== 1. 目标检测 ==========
            detections_list = self.detector.detect(
                frame,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres
            )

            # 转为 supervision Detections 格式
            if detections_list:
                xyxy = np.array([d['bbox'] for d in detections_list], dtype=np.float32)
                conf = np.array([d['confidence'] for d in detections_list], dtype=np.float32)
                cls = np.array([d['class_id'] for d in detections_list], dtype=np.int64)
                sv_detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
            else:
                sv_detections = sv.Detections.empty()

            total_count = len(sv_detections)

            # ========== 2. 多目标追踪 ==========
            tracked_detections = self.tracker.update(sv_detections)
            trails = self.tracker.get_trails()

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
                            frame, bbox, track_id, 'person', conf_val, color,
                            show_label=self.show_label
                        )

            # ========== 6. 绘制轨迹 ==========
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

            # ========== 11. 帧转为 QImage 发送到主线程 ==========
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            # copy() 确保数据独立，避免被后续帧覆盖
            qt_image = QImage(frame.data, w, h, bytes_per_line,
                              QImage.Format_BGR888).copy()
            self.update_frame_signal.emit(qt_image)

        # 清理
        cap.release()
        self.video_finished_signal.emit()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("危险区域人员闯入检测与报警系统")
        self.resize(1200, 700)

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

        self.lbl_zone_count = QLabel("区域内人数: 0")
        info_layout.addWidget(self.lbl_zone_count)

        self.lbl_total_count = QLabel("画面总人数: 0")
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
        """)

    # ======================== 按钮事件 ========================

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

        # 延迟初始化 YOLO 检测器（仅首次）
        if self.detector is None:
            try:
                self.detector = YOLODetector(
                    model_path='models/yolov8n.pt', device='cpu'
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

        # 重置统计显示
        self.lbl_fps.setText("FPS: --")
        self.lbl_duration.setText("检测时长: --")
        self.lbl_zone_count.setText("区域内人数: 0")
        self.lbl_total_count.setText("画面总人数: 0")
        self.lbl_zone_count.setStyleSheet(
            "color: #ccc; font: 13px 'Microsoft YaHei';"
        )

    def closeEvent(self, event):
        """窗口关闭时确保检测线程被正确停止"""
        self._stop_detection()
        event.accept()


def main():
    """程序入口"""
    # 自适应高 DPI 显示
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
# coding: utf-8
"""
报警模块
当区域内人数达到阈值时，在界面上显示红色警告并播放警报声音
"""
import cv2
import winsound
import os
import threading
import time


class AlarmSystem:
    """报警系统，监控区域内人数并在超阈值时触发声光警告"""

    def __init__(self, alarm_sound_path='alarm.wav'):
        """
        初始化报警系统

        参数:
            alarm_sound_path: 警报音频文件路径，默认 'alarm.wav'
        """
        # 尝试多个路径查找音频文件
        self.alarm_sound_path = alarm_sound_path
        if not os.path.exists(self.alarm_sound_path):
            # 尝试在项目根目录查找
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.normpath(os.path.join(script_dir, '..', 'alarm.wav'))
            if os.path.exists(alt_path):
                self.alarm_sound_path = alt_path
            else:
                print(f"[AlarmSystem] 警告: 音频文件不存在 ({alarm_sound_path})，将使用系统蜂鸣")

        self.is_alarming = False
        self._alarm_thread = None
        self._lock = threading.Lock()

    def check_and_alarm(self, zone_count, threshold, frame):
        """
        检查是否需要触发/解除报警，并在画面上绘制警告

        参数:
            zone_count: 当前区域内人数
            threshold: 报警阈值（>= 该值时触发）
            frame: 当前画面帧（原地修改）

        返回:
            tuple: (frame, is_alarming) — 处理后的帧和当前报警状态
        """
        if zone_count >= threshold and threshold > 0:
            if not self.is_alarming:
                self._start_alarm()
            frame = self.draw_warning(frame)
            return frame, True
        else:
            if self.is_alarming:
                self._stop_alarm()
            return frame, False

    def _start_alarm(self):
        """启动警报（声音 + 状态标记）"""
        with self._lock:
            if self.is_alarming:
                return
            self.is_alarming = True
            self._alarm_thread = threading.Thread(target=self._play_alarm, daemon=True)
            self._alarm_thread.start()

    def _play_alarm(self):
        """在子线程中循环播放警报音"""
        sound_exists = os.path.exists(self.alarm_sound_path)
        while self.is_alarming:
            if sound_exists:
                try:
                    winsound.PlaySound(
                        self.alarm_sound_path,
                        winsound.SND_FILENAME | winsound.SND_ASYNC
                    )
                except Exception:
                    # 播放失败则回退到蜂鸣
                    winsound.Beep(1000, 500)
            else:
                winsound.Beep(1000, 500)
            time.sleep(1.0)

    def _stop_alarm(self):
        """停止警报"""
        with self._lock:
            self.is_alarming = False
        try:
            winsound.PlaySound(None, winsound.SND_ASYNC)
        except Exception:
            pass

    def draw_warning(self, frame):
        """
        在画面上绘制红色警告文字

        参数:
            frame: BGR 格式的 numpy 数组图像（原地修改）

        返回:
            numpy.ndarray: 绘制后的图像
        """
        h, w = frame.shape[:2]
        text = "WARNING: 区域入侵!!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (w - tw) // 2
        text_y = th + 20

        # 文字背景半透明框
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (text_x - 15, text_y - th - 15),
            (text_x + tw + 15, text_y + 15),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # 红色警报文字
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

        # 副标题：当前入侵人数
        sub_text = f"Intrusion Detected!"
        sub_scale = 0.8
        sub_thickness = 2
        (stw, sth), _ = cv2.getTextSize(sub_text, font, sub_scale, sub_thickness)
        sub_x = (w - stw) // 2
        sub_y = text_y + th + 30
        cv2.putText(frame, sub_text, (sub_x, sub_y), font, sub_scale, (0, 0, 255), sub_thickness)

        return frame
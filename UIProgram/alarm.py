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
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 查找系统可用的中文字体
def _get_chinese_font(font_size=36):
    """获取系统中可用的中文字体路径"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",    # 微软雅黑粗体
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
        "C:/Windows/Fonts/simkai.ttf",    # 楷体
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, font_size)
    # 回退到默认字体（不支持中文）
    return ImageFont.load_default()


class AlarmSystem:
    """报警系统，监控区域内人数并在超阈值时触发声光警告"""

    def __init__(self, alarm_sound_path='alarm.wav',
                 warning_text="WARNING: 危险区域人员闯入!!"):
        """
        初始化报警系统

        参数:
            alarm_sound_path: 警报音频文件路径，默认 'alarm.wav'
            warning_text: 报警时显示的警告文字
        """
        self.warning_text = warning_text

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
        在画面上绘制红色警告文字（使用 PIL 支持中文显示）

        参数:
            frame: BGR 格式的 numpy 数组图像

        返回:
            numpy.ndarray: 绘制后的图像
        """
        h, w = frame.shape[:2]

        # 将 OpenCV BGR 图像转为 PIL RGB 图像
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)

        # 主标题
        main_text = self.warning_text
        main_font = _get_chinese_font(40)

        # 测量主标题尺寸
        bbox = draw.textbbox((0, 0), main_text, font=main_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        text_x = (w - tw) // 2
        text_y = 20

        # 绘制半透明背景框
        pad = 15
        bg_x1 = text_x - pad
        bg_y1 = text_y - pad
        bg_x2 = text_x + tw + pad
        bg_y2 = text_y + th + pad

        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [bg_x1, bg_y1, bg_x2, bg_y2],
            fill=(0, 0, 0, 128)
        )
        pil_img = Image.alpha_composite(
            pil_img.convert("RGBA"), overlay
        ).convert("RGB")
        draw = ImageDraw.Draw(pil_img)

        # 红色警告主标题
        draw.text((text_x, text_y), main_text, font=main_font, fill=(255, 0, 0))

        # 副标题
        sub_text = "Intrusion Detected!"
        sub_font = _get_chinese_font(28)
        sub_bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
        sub_w = sub_bbox[2] - sub_bbox[0]
        sub_h = sub_bbox[3] - sub_bbox[1]
        sub_x = (w - sub_w) // 2
        sub_y = text_y + th + 20

        draw.text((sub_x, sub_y), sub_text, font=sub_font, fill=(255, 0, 0))

        # 转回 OpenCV BGR 格式
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # 将结果写回原 frame
        frame[...] = frame_bgr

        return frame
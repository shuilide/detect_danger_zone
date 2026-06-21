# coding: utf-8
"""
报警模块
当区域内人数达到阈值时，在界面上显示红色警告并播放警报声音
"""
import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# winsound 仅在 Windows 上可用，Linux/WSL 下回退到 console beep
try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False
    print("[AlarmSystem] winsound 不可用（非 Windows 系统），将跳过声音警报")


# 缓存已加载的字体，避免每帧重复加载
_font_cache = {}


def _get_chinese_font(font_size=36):
    """获取系统中可用的中文字体路径（带缓存）"""
    if font_size in _font_cache:
        return _font_cache[font_size]

    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simkai.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            font = ImageFont.truetype(path, font_size)
            _font_cache[font_size] = font
            return font

    font = ImageFont.load_default()
    _font_cache[font_size] = font
    return font


class AlarmSystem:
    """报警系统，监控区域内人数并在超阈值时触发声光警告

    注意：声音播放不再使用 daemon 线程（winsound 不是线程安全的），
    而是在检测线程主循环中限速调用 PlaySound，避免堆栈溢出崩溃。
    """

    def __init__(self, alarm_sound_path='alarm.wav',
                 warning_text="WARNING: 危险区域人员闯入!!"):
        self.warning_text = warning_text

        self.alarm_sound_path = alarm_sound_path
        if not os.path.exists(self.alarm_sound_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.normpath(os.path.join(script_dir, '..', 'alarm.wav'))
            if os.path.exists(alt_path):
                self.alarm_sound_path = alt_path
            else:
                print(f"[AlarmSystem] 警告: 音频文件不存在 ({alarm_sound_path})，使用系统蜂鸣")

        self.is_alarming = False
        self._last_beep_time = 0.0       # 上次播放声音的时间戳（限速用）
        self._beep_interval = 1.0        # 蜂鸣间隔（秒）

        # PIL 文字渲染缓存（避免报警期间每帧重新测量/渲染文字）
        self._banner_cache = None        # (banner_bgr, alpha_mask, banner_h)
        self._banner_text = None         # 缓存对应的 warning_text

    # ==================== 报警状态检查 ====================

    def check_and_alarm(self, zone_count, threshold, frame):
        """检查是否需要触发/解除报警，并在画面上绘制警告

        返回: (frame, is_alarming)
        """
        if zone_count >= threshold and threshold > 0:
            if not self.is_alarming:
                self._start_alarm()
            self._beep_if_needed()
            frame = self.draw_warning(frame)
            return frame, True
        else:
            if self.is_alarming:
                self._stop_alarm()
            return frame, False

    # ==================== 声音播放（限速，在当前线程中执行） ====================

    def _start_alarm(self):
        """标记报警开始"""
        self.is_alarming = True
        self._last_beep_time = 0.0

    def _stop_alarm(self):
        """标记报警结束"""
        self.is_alarming = False
        try:
            if _HAS_WINSOUND:
                winsound.PlaySound(None, winsound.SND_ASYNC)
        except Exception:
            pass

    def _beep_if_needed(self):
        """限速播放蜂鸣（在检测线程主循环中调用，不使用 daemon 线程）"""
        now = time.time()
        if now - self._last_beep_time < self._beep_interval:
            return
        self._last_beep_time = now

        if _HAS_WINSOUND:
            try:
                if os.path.exists(self.alarm_sound_path):
                    winsound.PlaySound(
                        self.alarm_sound_path,
                        winsound.SND_FILENAME | winsound.SND_ASYNC
                    )
                else:
                    winsound.Beep(1000, 500)
            except Exception:
                pass

    # ==================== 警告文字绘制（仅渲染 banner 区域，避免全帧 PIL 转换） ====================

    def draw_warning(self, frame):
        """在画面顶部叠加红色警告 banner（OpenCV + PIL 局部渲染）"""
        h, w = frame.shape[:2]

        # 检查缓存是否有效
        if self._banner_cache is None or self._banner_text != self.warning_text:
            self._banner_cache = self._render_banner(w)
            self._banner_text = self.warning_text

        banner_bgr, alpha, banner_h = self._banner_cache

        # 如果帧宽度变了（窗口缩放可能导致），按需重新渲染
        if banner_bgr.shape[1] != w:
            self._banner_cache = self._render_banner(w)
            banner_bgr, alpha, banner_h = self._banner_cache

        # 叠加 banner 到帧顶部
        if banner_h <= h and banner_bgr.shape[1] == w:
            roi = frame[:banner_h, :w]
            frame[:banner_h, :w] = (roi * (1.0 - alpha) + banner_bgr * alpha).astype(np.uint8)

        return frame

    def _render_banner(self, frame_width):
        """用 PIL 渲染警告文字到小尺寸 banner 图像（只做一次，结果被缓存）"""
        banner_h = 95
        w = frame_width

        # 创建透明背景的 PIL 图像
        banner_rgba = Image.new("RGBA", (w, banner_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(banner_rgba)

        # 半透明黑色背景
        draw.rectangle([0, 0, w, banner_h], fill=(0, 0, 0, 128))

        # 主标题（红色）
        main_font = _get_chinese_font(40)
        main_text = self.warning_text
        bbox = draw.textbbox((0, 0), main_text, font=main_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (w - tw) // 2
        draw.text((tx, 8), main_text, font=main_font, fill=(255, 0, 0))

        # 副标题
        sub_text = "Intrusion Detected!"
        sub_font = _get_chinese_font(26)
        sub_bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
        sub_w = sub_bbox[2] - sub_bbox[0]
        sub_x = (w - sub_w) // 2
        draw.text((sub_x, 55), sub_text, font=sub_font, fill=(255, 0, 0))

        # 转为 numpy BGR + alpha mask
        banner_rgb = banner_rgba.convert("RGB")
        banner_bgr = cv2.cvtColor(np.array(banner_rgb), cv2.COLOR_RGB2BGR)
        alpha = np.array(banner_rgba.split()[-1], dtype=np.float32) / 255.0
        alpha = alpha[:, :, np.newaxis]

        return banner_bgr, alpha, banner_h

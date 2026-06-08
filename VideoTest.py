# coding: utf-8
"""
视频检测脚本 - 独立运行（脱离 GUI）
支持鼠标绘制危险区域、实时检测、追踪与报警

用法:
    python VideoTest.py --video path/to/video.mp4 --model models/yolov8n.pt
    python VideoTest.py  （不传参数则弹出文件选择对话框）

快捷键:
    q / ESC  - 退出
    r        - 重置多边形区域
    t        - 切换轨迹显示
    b        - 切换检测框显示
    l        - 切换标签显示
    鼠标左键 - 添加区域顶点
    鼠标右键 - 闭合/完成多边形
"""
import cv2
import numpy as np
import sys
import os
import argparse
import time
import supervision as sv

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UIProgram.detector import YOLODetector
from UIProgram.tracker import ByteTrackTracker
from UIProgram.zone import DangerZone
from UIProgram.alarm import AlarmSystem
from UIProgram.utils import FPSCounter, get_color, draw_detection, draw_trails, draw_zone_count

# ============================================================
# 全局状态（供鼠标回调使用）
# ============================================================
_g_danger_zone: DangerZone = None
_g_window_name: str = "Video Detection"


def _mouse_callback(event, x, y, flags, param):
    """
    鼠标回调：左键添加多边形顶点，右键闭合多边形

    参数:
        event: OpenCV 鼠标事件类型
        x, y: 鼠标在窗口中的坐标
    """
    zone = _g_danger_zone
    if zone is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        # 左键：添加顶点（仅在未闭合时允许）
        if not zone.is_closed:
            zone.add_point(x, y)
            print(f"[Zone] 添加顶点: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右键：闭合多边形
        if not zone.is_closed and zone.is_ready():
            zone.close()
            print(f"[Zone] 多边形已闭合，共 {len(zone.get_points())} 个顶点")


def _open_video_dialog() -> str:
    """
    弹出文件选择对话框，让用户选择一个视频文件

    返回:
        str: 选中的视频文件路径；取消则返回空字符串
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("所有文件", "*.*"),
            ]
        )
        root.destroy()
        return file_path
    except Exception as e:
        print(f"[Error] 无法打开文件对话框: {e}")
        return ""


def _build_sv_detections(detection_list, frame_shape):
    """
    将 YOLODetector.detect() 返回的 dict 列表转换为 supervision Detections 对象

    参数:
        detection_list: YOLODetector.detect() 返回的检测结果列表
        frame_shape: 帧的 (h, w) 形状

    返回:
        sv.Detections: supervision 格式的检测结果
    """
    if not detection_list:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int32),
        )

    xyxy_list = []
    conf_list = []
    cls_list = []
    for det in detection_list:
        xyxy_list.append(det['bbox'])
        conf_list.append(det['confidence'])
        cls_list.append(det['class_id'])

    return sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        confidence=np.array(conf_list, dtype=np.float32),
        class_id=np.array(cls_list, dtype=np.int32),
    )


def main():
    # ----------------------------------------------------------
    # 命令行参数解析
    # ----------------------------------------------------------
    parser = argparse.ArgumentParser(description="视频检测脚本 - YOLOv8 + ByteTrack + 区域报警")
    parser.add_argument(
        '--video', type=str, default=None,
        help='视频文件路径（不提供则弹出文件选择对话框）'
    )
    parser.add_argument(
        '--model', type=str, default='models/yolov8n.pt',
        help='YOLOv8 模型路径，默认 models/yolov8n.pt'
    )
    parser.add_argument(
        '--conf', type=float, default=0.5,
        help='检测置信度阈值，默认 0.5'
    )
    parser.add_argument(
        '--iou', type=float, default=0.5,
        help='NMS IoU 阈值，默认 0.5'
    )
    parser.add_argument(
        '--threshold', type=int, default=1,
        help='区域报警人数阈值，默认 1'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='推理设备，默认 cpu'
    )
    args = parser.parse_args()

    # ----------------------------------------------------------
    # 视频路径处理
    # ----------------------------------------------------------
    video_path = args.video
    if not video_path:
        print("[Info] 未指定视频路径，正在打开文件选择对话框...")
        video_path = _open_video_dialog()
        if not video_path:
            print("[Error] 未选择视频文件，退出。")
            sys.exit(1)

    if not os.path.exists(video_path):
        print(f"[Error] 视频文件不存在: {video_path}")
        sys.exit(1)

    print(f"[Info] 视频文件: {video_path}")
    print(f"[Info] 模型路径: {args.model}")
    print(f"[Info] 推理设备: {args.device}")
    print(f"[Info] 置信度阈值: {args.conf}, IoU 阈值: {args.iou}")
    print(f"[Info] 报警阈值: {args.threshold} 人")

    # ----------------------------------------------------------
    # 初始化视频读取
    # ----------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] 无法打开视频: {video_path}")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Info] 视频分辨率: {width}x{height}, 帧率: {fps_video:.2f}")

    # ----------------------------------------------------------
    # 初始化各模块
    # ----------------------------------------------------------
    print("[Info] 正在加载模型...")
    detector = YOLODetector(model_path=args.model, device=args.device)

    tracker = ByteTrackTracker()
    fps_counter = FPSCounter()

    # 查找 alarm.wav 音频文件
    alarm_sound = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alarm.wav')
    if not os.path.exists(alarm_sound):
        alarm_sound = 'alarm.wav'
    alarm = AlarmSystem(alarm_sound_path=alarm_sound)

    # ----------------------------------------------------------
    # 全局状态初始化
    # ----------------------------------------------------------
    global _g_danger_zone
    _g_danger_zone = DangerZone()

    show_trails = True     # 是否显示轨迹
    show_bbox = True       # 是否显示检测框
    show_label = True      # 是否显示标签

    # ----------------------------------------------------------
    # 创建 OpenCV 窗口并绑定鼠标回调
    # ----------------------------------------------------------
    cv2.namedWindow(_g_window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_g_window_name, _mouse_callback)

    # ----------------------------------------------------------
    # 打印快捷键提示
    # ----------------------------------------------------------
    print("=" * 60)
    print("  快捷键:")
    print("    q / ESC  - 退出")
    print("    r        - 重置多边形区域")
    print("    t        - 切换轨迹显示")
    print("    b        - 切换检测框显示")
    print("    l        - 切换标签显示")
    print("    鼠标左键 - 添加区域顶点")
    print("    鼠标右键 - 闭合多边形")
    print("=" * 60)

    # ----------------------------------------------------------
    # 主循环
    # ----------------------------------------------------------
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Info] 视频播放完毕。")
                break

            frame_idx += 1
            t_start = time.time()

            # ---- 1. YOLOv8 检测 ----
            det_results = detector.detect(frame, conf_thres=args.conf, iou_thres=args.iou)
            detect_time = (time.time() - t_start) * 1000  # 毫秒

            # ---- 2. 转换为 supervision 格式 ----
            sv_dets = _build_sv_detections(det_results, frame.shape[:2])

            # ---- 3. ByteTrack 追踪 ----
            tracked_dets = tracker.update(sv_dets)
            trails = tracker.get_trails()

            # ---- 4. 区域判断 ----
            zone_count = 0
            total_count = 0

            if tracked_dets is not None and len(tracked_dets) > 0 and tracked_dets.tracker_id is not None:
                total_count = len(tracked_dets)
                for i in range(len(tracked_dets)):
                    x1, y1, x2, y2 = tracked_dets.xyxy[i]
                    # 使用目标底边中点判断是否在区域内（更贴近脚部位置，减少误判）
                    bottom_cx = int((x1 + x2) / 2)
                    bottom_cy = int(y2)
                    if _g_danger_zone.is_point_inside(bottom_cx, bottom_cy):
                        zone_count += 1

            # ---- 5. 报警检查 ----
            frame, is_alarming = alarm.check_and_alarm(zone_count, args.threshold, frame)

            # ---- 6. 绘制 ----
            # 绘制多边形区域
            _g_danger_zone.draw_zone(frame)

            # 绘制运动轨迹
            draw_trails(frame, trails, show_trails=show_trails)

            # 绘制检测框
            if show_bbox and tracked_dets is not None and len(tracked_dets) > 0 and tracked_dets.tracker_id is not None:
                for i in range(len(tracked_dets)):
                    tid = int(tracked_dets.tracker_id[i])
                    bbox = tracked_dets.xyxy[i]
                    conf_val = float(tracked_dets.confidence[i]) if tracked_dets.confidence is not None else 0.0
                    color = get_color(tid)
                    draw_detection(frame, bbox, tid, "person", conf_val, color, show_label=show_label)

            # ---- 7. 叠加信息面板 ----
            # FPS
            fps_counter.update()
            fps_counter.draw_fps(frame)

            # 检测耗时
            dt_text = f"Detect: {detect_time:.1f}ms"
            cv2.putText(
                frame, dt_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            # 区域内人数
            draw_zone_count(frame, zone_count)

            # 总人数（右上角）
            total_text = f"Total: {total_count}"
            total_color = (0, 255, 0)
            (tw, _), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            total_x = frame.shape[1] - tw - 15
            total_y = 30
            cv2.putText(
                frame, total_text, (total_x, total_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, total_color, 2
            )

            # 状态栏（底部）
            status_parts = []
            status_parts.append(f"Trails: {'ON' if show_trails else 'OFF'}")
            status_parts.append(f"BBox: {'ON' if show_bbox else 'OFF'}")
            status_parts.append(f"Label: {'ON' if show_label else 'OFF'}")
            status_parts.append(f"Alarm: {'ON' if is_alarming else 'OFF'}")
            status_text = " | ".join(status_parts)
            cv2.putText(
                frame, status_text, (10, frame.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
            )

            # 报警闪烁红框
            if is_alarming:
                border_color = (0, 0, 255)
                thickness = 4 if (frame_idx // 10) % 2 == 0 else 2
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), border_color, thickness)

            # ---- 8. 显示画面 ----
            cv2.imshow(_g_window_name, frame)

            # ---- 9. 键盘事件 ----
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' 或 ESC
                print("[Info] 用户按下退出键。")
                break
            elif key == ord('r'):
                _g_danger_zone.clear()
                print("[Info] 区域已重置。")
            elif key == ord('t'):
                show_trails = not show_trails
                print(f"[Info] 轨迹显示: {'ON' if show_trails else 'OFF'}")
            elif key == ord('b'):
                show_bbox = not show_bbox
                print(f"[Info] 检测框显示: {'ON' if show_bbox else 'OFF'}")
            elif key == ord('l'):
                show_label = not show_label
                print(f"[Info] 标签显示: {'ON' if show_label else 'OFF'}")

    except KeyboardInterrupt:
        print("\n[Info] 检测到 Ctrl+C，正在退出...")
    finally:
        # ----------------------------------------------------------
        # 清理资源
        # ----------------------------------------------------------
        cap.release()
        cv2.destroyAllWindows()
        alarm._stop_alarm()
        print("[Info] 资源已释放，程序退出。")


if __name__ == '__main__':
    main()
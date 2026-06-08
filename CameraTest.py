# coding: utf-8
"""
摄像头实时检测脚本（独立于 GUI）
使用 OpenCV 窗口显示 YOLOv8 检测 + ByteTrack 追踪 + 区域判断 + 报警
"""

import cv2
import numpy as np
import argparse
import time
import supervision as sv

from UIProgram.detector import YOLODetector
from UIProgram.tracker import ByteTrackTracker
from UIProgram.zone import DangerZone
from UIProgram.alarm import AlarmSystem
from UIProgram.utils import FPSCounter, get_color, draw_detection, draw_trails, draw_zone_count


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='摄像头实时检测脚本')
    parser.add_argument('--camera', type=int, default=0,
                        help='摄像头索引，默认 0')
    parser.add_argument('--model', type=str, default='models/yolov8n.pt',
                        help='YOLO 模型路径，默认 models/yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='置信度阈值，默认 0.5')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='NMS IoU 阈值，默认 0.5')
    parser.add_argument('--alarm-threshold', type=int, default=1,
                        help='报警阈值：区域内人数 >= 该值时触发报警，默认 1')
    parser.add_argument('--width', type=int, default=1280,
                        help='摄像头分辨率宽度，默认 1280')
    parser.add_argument('--height', type=int, default=720,
                        help='摄像头分辨率高度，默认 720')
    return parser.parse_args()


# ---------- 鼠标回调 ----------
def mouse_callback(event, x, y, flags, param):
    """
    鼠标回调函数
    左键：添加多边形顶点
    右键：闭合多边形区域
    """
    zone = param
    if event == cv2.EVENT_LBUTTONDOWN:
        zone.add_point(x, y)
        print(f"[区域] 添加顶点: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if zone.is_ready():
            zone.close()
            print(f"[区域] 多边形已闭合，共 {len(zone.get_points())} 个顶点")
        else:
            print("[区域] 至少需要 3 个顶点才能闭合")


# ---------- 主逻辑 ----------
def main():
    args = parse_args()

    # ---- 初始化各模块 ----
    print(f"[初始化] 加载模型: {args.model}")
    detector = YOLODetector(model_path=args.model, device='cpu')

    tracker = ByteTrackTracker()
    zone = DangerZone()
    alarm = AlarmSystem()
    fps_counter = FPSCounter()

    # ---- 打开摄像头 ----
    print(f"[初始化] 打开摄像头 {args.camera} (CAP_DSHOW)")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # ---- 创建窗口 & 绑定鼠标回调 ----
    window_name = 'Camera Detection - YOLOv8 + ByteTrack'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, zone)

    # ---- 显示开关 ----
    show_trails = True
    show_boxes = True
    show_labels = True

    # ---- 检测耗时（毫秒） ----
    detect_time = 0.0

    # ---- 打印操作提示 ----
    print("\n" + "=" * 40)
    print("  键盘控制")
    print("=" * 40)
    print("  q / ESC : 退出程序")
    print("  r      : 重置区域（清空所有顶点）")
    print("  t      : 切换轨迹显示")
    print("  b      : 切换检测框显示")
    print("  l      : 切换标签显示")
    print("  鼠标左键 : 添加区域顶点")
    print("  鼠标右键 : 闭合多边形区域")
    print("=" * 40 + "\n")

    # ---- 主循环 ----
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[警告] 无法读取摄像头画面，尝试重连...")
            time.sleep(0.1)
            continue

        # ========== 1. 检测 ==========
        t_start = time.time()
        detections_list = detector.detect(
            frame,
            conf_thres=args.conf,
            iou_thres=args.iou
        )
        detect_time = (time.time() - t_start) * 1000  # 转为毫秒

        # ========== 2. 转换为 supervision Detections 格式 ==========
        if detections_list:
            xyxy = np.array([d['bbox'] for d in detections_list], dtype=np.float32)
            confidence = np.array([d['confidence'] for d in detections_list], dtype=np.float32)
            class_id = np.array([d['class_id'] for d in detections_list], dtype=np.int32)
            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )
        else:
            sv_detections = sv.Detections.empty()

        # ========== 3. 追踪 ==========
        tracked_detections = tracker.update(sv_detections)

        # ========== 4. 统计区域内人数 ==========
        zone_count = 0
        if (tracked_detections is not None
                and len(tracked_detections) > 0
                and tracked_detections.tracker_id is not None
                and zone.is_closed):
            for i in range(len(tracked_detections)):
                x1, y1, x2, y2 = tracked_detections.xyxy[i]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                if zone.is_point_inside(cx, cy):
                    zone_count += 1

        # ========== 5. 总人数 ==========
        total_count = len(tracked_detections) if tracked_detections is not None else 0

        # ========== 6. 绘制检测框和标签 ==========
        if show_boxes and tracked_detections is not None and len(tracked_detections) > 0:
            if tracked_detections.tracker_id is not None:
                for i in range(len(tracked_detections)):
                    track_id = int(tracked_detections.tracker_id[i])
                    bbox = tracked_detections.xyxy[i]
                    conf = float(tracked_detections.confidence[i]) \
                        if tracked_detections.confidence is not None else 0.0
                    color = get_color(track_id)
                    draw_detection(
                        frame, bbox, track_id, 'person', conf,
                        color, show_label=show_labels
                    )

        # ========== 7. 绘制轨迹 ==========
        trails = tracker.get_trails()
        draw_trails(frame, trails, show_trails=show_trails)

        # ========== 8. 绘制区域 ==========
        zone.draw_zone(frame)

        # ========== 9. 绘制区域人数 ==========
        draw_zone_count(frame, zone_count)

        # ========== 10. 报警检查 ==========
        frame, is_alarming = alarm.check_and_alarm(
            zone_count, args.alarm_threshold, frame
        )

        # ========== 11. 绘制信息面板 ==========
        fps_counter.update()
        fps_counter.draw_fps(frame)

        # 检测耗时
        dt_text = f"Detect: {detect_time:.1f} ms"
        cv2.putText(frame, dt_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 总人数
        tc_text = f"Total: {total_count}"
        cv2.putText(frame, tc_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 区域状态
        if zone.is_closed:
            zs_text = "Zone: ACTIVE"
        else:
            n_pts = len(zone.get_points())
            zs_text = f"Zone: DRAWING ({n_pts} pts)"
        cv2.putText(frame, zs_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 显示开关状态
        h_frame = frame.shape[0]
        show_text = (
            f"Trails:{'ON' if show_trails else 'OFF'} | "
            f"Boxes:{'ON' if show_boxes else 'OFF'} | "
            f"Labels:{'ON' if show_labels else 'OFF'}"
        )
        cv2.putText(frame, show_text, (10, h_frame - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ========== 12. 显示画面 ==========
        cv2.imshow(window_name, frame)

        # ========== 13. 键盘处理 ==========
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 或 ESC
            print("[退出] 用户按下退出键")
            break
        elif key == ord('r'):
            zone.clear()
            print("[区域] 已重置所有顶点")
        elif key == ord('t'):
            show_trails = not show_trails
            print(f"[切换] 轨迹显示: {'开' if show_trails else '关'}")
        elif key == ord('b'):
            show_boxes = not show_boxes
            print(f"[切换] 检测框显示: {'开' if show_boxes else '关'}")
        elif key == ord('l'):
            show_labels = not show_labels
            print(f"[切换] 标签显示: {'开' if show_labels else '关'}")

    # ---- 清理资源 ----
    cap.release()
    cv2.destroyAllWindows()
    print("[退出] 程序结束")


if __name__ == '__main__':
    main()
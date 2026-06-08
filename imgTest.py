# coding: utf-8
"""
图片检测测试脚本
对单张图片进行人员检测，显示并保存结果
"""
import cv2
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UIProgram.detector import YOLODetector
from UIProgram.utils import get_color, draw_detection


def open_file_dialog():
    """弹出文件选择对话框，返回用户选择的图片文件路径"""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes('-topmost', True)  # 置顶对话框
    file_path = filedialog.askopenfilename(
        title="选择要检测的图片",
        filetypes=[
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("所有文件", "*.*"),
        ]
    )
    root.destroy()
    return file_path if file_path else None


def main():
    # ---- 解析命令行参数 ----
    parser = argparse.ArgumentParser(description="图片人员检测测试脚本")
    parser.add_argument(
        '--image', type=str, default=None,
        help='图片文件路径（不提供则弹出文件选择对话框）'
    )
    parser.add_argument(
        '--model', type=str, default='models/yolov8n.pt',
        help='YOLOv8 模型文件路径，默认 models/yolov8n.pt'
    )
    parser.add_argument(
        '--output', type=str, default='output.jpg',
        help='结果图片保存路径，默认 output.jpg'
    )
    parser.add_argument(
        '--conf', type=float, default=0.5,
        help='检测置信度阈值，默认 0.5'
    )
    args = parser.parse_args()

    # ---- 获取图片路径 ----
    image_path = args.image
    if not image_path:
        print("未指定 --image 参数，弹出文件选择对话框...")
        image_path = open_file_dialog()

    if not image_path:
        print("[错误] 未选择图片文件，程序退出。")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"[错误] 图片文件不存在: {image_path}")
        sys.exit(1)

    print(f"[信息] 图片路径: {image_path}")

    # ---- 读取图片 ----
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[错误] 无法读取图片: {image_path}")
        sys.exit(1)

    print(f"[信息] 图片尺寸: {frame.shape[1]}x{frame.shape[0]}")

    # ---- 初始化检测器 ----
    print(f"[信息] 加载模型: {args.model}")
    detector = YOLODetector(model_path=args.model, device='cpu')

    # ---- 执行检测 ----
    print(f"[信息] 正在检测（置信度阈值: {args.conf}）...")
    detections = detector.detect(frame, conf_thres=args.conf, iou_thres=0.5)

    person_count = len(detections)
    print(f"[信息] 检测到 {person_count} 个人")

    # ---- 绘制检测结果 ----
    for i, det in enumerate(detections):
        bbox = det['bbox']
        confidence = det['confidence']
        class_name = det['class_name']
        color = get_color(i)  # 使用索引作为 track_id 生成稳定颜色
        draw_detection(
            frame, bbox, i, class_name, confidence, color, show_label=True
        )

    # ---- 在图片上显示人员总数 ----
    count_text = f"Person Count: {person_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_color = (0, 255, 0)  # 绿色

    # 计算文字尺寸，放置在左上角
    (tw, th), baseline = cv2.getTextSize(count_text, font, font_scale, thickness)
    # 在文字下方绘制半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (tw + 20, th + baseline + 16), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(
        frame, count_text, (16, th + baseline + 8),
        font, font_scale, text_color, thickness
    )

    # ---- 保存结果 ----
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(args.output, frame)
    print(f"[信息] 结果已保存到: {args.output}")

    # ---- 显示结果 ----
    cv2.imshow("Detection Result - Press any key to exit", frame)
    print("[信息] 按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
# Tasks

- [x] Task 1: 创建项目目录结构和依赖配置
  - [x] 创建项目所有子目录（datasets/、Font/、models/、runs/、TestFiles/、UIProgram/）
  - [x] 编写 requirements.txt（ultralytics、opencv-python、PyQt5、supervision 等）
  - [x] 编写 installPackages.py 一键安装脚本
  - [x] 创建 datasets/Data/data.yaml 数据集配置文件

- [x] Task 2: 生成报警音频文件
  - [x] 使用 Python 合成 alarm.wav 报警音频文件，保存到项目根目录

- [x] Task 3: 实现 YOLOv8 检测核心模块（UIProgram/）
  - [x] 实现 UIProgram/detector.py：封装 YOLOv8 目标检测逻辑，过滤非 person 类别
  - [x] 实现 UIProgram/tracker.py：封装 ByteTrack 多目标追踪逻辑
  - [x] 实现 UIProgram/zone.py：封装多边形区域绘制与 cv2.pointPolygonTest 判断逻辑
  - [x] 实现 UIProgram/alarm.py：封装报警逻辑（红色警告 + 音频播放）
  - [x] 实现 UIProgram/utils.py：工具函数（FPS 计算、绘制标签等）

- [x] Task 4: 实现 PyQt5 主界面 MainProgram.py
  - [x] 实现 PyQt5 主窗口，包含视频显示区域、控制按钮、参数设置、信息展示
  - [x] 实现"打开视频"按钮逻辑：选择文件并启动视频检测线程
  - [x] 实现"打开摄像头"按钮逻辑：启动摄像头检测线程
  - [x] 实现"绘制区域"和"绘制完成"按钮逻辑：鼠标事件处理
  - [x] 实现三个开关复选框（显示检测框、显示标签、显示追踪轨迹）
  - [x] 实现置信度/IOU/报警阈值参数输入控件
  - [x] 实现实时信息显示（FPS、检测时长、区域人数、总人数）
  - [x] 实现检测线程：逐帧处理、YOLOv8 推理、ByteTrack 追踪、区域判断、报警触发
  - [x] 实现视频帧通过信号/槽机制更新到 QLabel 显示

- [x] Task 5: 实现 VideoTest.py 视频检测脚本
  - [x] 实现独立的视频文件检测脚本（脱离 GUI）
  - [x] 支持命令行参数传入视频路径和模型路径

- [x] Task 6: 实现 CameraTest.py 摄像头检测脚本
  - [x] 实现独立的摄像头检测脚本（脱离 GUI）
  - [x] 按 ESC 键退出

- [x] Task 7: 实现 imgTest.py 图片检测脚本
  - [x] 实现图片检测脚本，对单张图片进行人员检测并保存结果

- [x] Task 8: 实现 train_v8.py 模型训练脚本
  - [x] 加载 yolov8.yaml 配置和 yolov8n.pt 预训练权重
  - [x] 从 datasets/Data/data.yaml 读取训练数据
  - [x] 支持配置 epochs、batch、imgsz 等参数
  - [x] 训练结果输出到 runs/ 目录

# Task Dependencies
- Task 4 依赖 Task 3（主界面依赖核心模块）
- Task 5、Task 6、Task 7 依赖 Task 3（测试脚本依赖核心模块）
- Task 8 独立于其他任务
- Task 1 和 Task 2 可最先并行执行
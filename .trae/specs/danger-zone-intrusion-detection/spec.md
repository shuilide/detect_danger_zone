# 危险区域人员闯入检测与报警系统 Spec

## Why
基于 YOLOv8 深度学习框架，构建一套完整的危险区域人员闯入检测与报警系统。实现对监控画面中人员目标的实时检测、ByteTrack 多目标追踪、自定义多边形区域闯入判断与自动报警功能，保障工作场所的人员安全。

## What Changes
- 创建完整的项目目录结构（datasets/、Font/、models/、runs/、TestFiles/、UIProgram/、ultralytics/）
- 实现 MainProgram.py：基于 PyQt5 的完整可视化主程序，集成 YOLOv8 检测 + ByteTrack 追踪 + 区域闯入报警
- 实现 VideoTest.py：视频文件检测脚本
- 实现 CameraTest.py：摄像头实时检测脚本
- 实现 train_v8.py：YOLOv8 模型训练脚本
- 实现 imgTest.py：图片检测测试脚本
- 实现 installPackages.py：一键环境安装脚本
- 提供 requirements.txt：依赖库清单
- 生成报警音频文件（使用 Python 合成）

## Impact
- Affected specs: 无（全新项目）
- Affected code: 无（全新项目）

## ADDED Requirements

### Requirement: 项目目录结构
系统 SHALL 按照以下结构组织文件：
- datasets/ — 数据集文件夹
- Font/ — UI 字体文件
- models/ — 训练好的模型权重（best.pt / yolov8n.pt）
- runs/ — 模型训练日志和结果图表
- TestFiles/ — 测试图片和测试视频
- UIProgram/ — PyQt5 界面源码
- ultralytics/ — YOLOv8 框架源码

#### Scenario: 目录结构完整
- **WHEN** 项目创建完成
- **THEN** 上述所有目录均存在

### Requirement: 人员目标检测
系统 SHALL 使用 YOLOv8 进行目标检测，仅检测 "person" 类别（class_id=0），过滤所有其他类别。默认使用 yolov8n.pt 轻量模型，支持 CPU 运行。

#### Scenario: 检测到人员
- **WHEN** 画面中出现人员
- **THEN** 系统识别并绘制检测框

#### Scenario: 非人员目标被过滤
- **WHEN** 画面中出现非 person 类别的目标（如 car、dog）
- **THEN** 系统不显示其检测框，不参与追踪和计数

### Requirement: 双输入源支持
系统 SHALL 支持本地视频文件（.mp4/.avi/.mov 等）和电脑摄像头两种输入源，通过 UI 按钮切换。再次点击同一按钮可关闭当前输入源。

#### Scenario: 打开本地视频
- **WHEN** 用户点击"打开视频"按钮并选择视频文件
- **THEN** 系统开始对该视频进行检测

#### Scenario: 打开摄像头
- **WHEN** 用户点击"打开摄像头"按钮
- **THEN** 系统打开默认摄像头（index=0）进行实时检测

### Requirement: 鼠标绘制多边形危险区域
系统 SHALL 支持用户通过鼠标左键在视频画面上点击绘制任意多边形封闭区域。至少需要 3 个点才能闭合。点击"绘制区域"进入绘制模式，点击"绘制完成"闭合区域。

#### Scenario: 绘制区域
- **WHEN** 用户点击"绘制区域"按钮后在画面上用鼠标左键点击至少 3 个点，然后点击"绘制完成"
- **THEN** 系统将这些点连接成闭合多边形区域，并在画面上显示

#### Scenario: 点数不足
- **WHEN** 用户点击"绘制完成"但点数少于 3
- **THEN** 系统不闭合区域，继续保持绘制模式或给出提示

### Requirement: 区域内人数统计与报警
系统 SHALL 使用 cv2.pointPolygonTest 判断每个检测到的人员目标中心点是否在危险区域内。实时统计区域内人数和画面总人数。当区域内人数 ≥ 设定阈值时触发报警：界面显示红色警告文字 + 播放警报声音。

#### Scenario: 人员闯入触发报警
- **WHEN** 危险区域内人数达到设定阈值（默认 1）
- **THEN** 界面上叠加红色"WARNING: 区域入侵!!"警告文字，并播放报警音频

#### Scenario: 人员离开解除报警
- **WHEN** 危险区域内人数降至阈值以下
- **THEN** 报警停止，红色警告消失，音频停止

### Requirement: ByteTrack 多目标追踪
系统 SHALL 集成 ByteTrack 多目标追踪算法，给每个检测到的人员分配稳定且唯一的追踪 ID，支持显示追踪轨迹。

#### Scenario: 追踪 ID 绑定
- **WHEN** 同一个人员持续出现在画面中
- **THEN** 该人员保持相同的追踪 ID

#### Scenario: 追踪轨迹显示
- **WHEN** 用户勾选"显示追踪轨迹"复选框
- **THEN** 每个人员的运动轨迹以线条形式在画面上显示

### Requirement: 实时信息显示
系统 SHALL 在界面上实时显示：FPS（帧率）、检测时长、区域内人数、画面总人数。

#### Scenario: 视频模式信息显示
- **WHEN** 用户打开视频进行检测
- **THEN** 界面显示当前检测帧率、已检测视频时长、区域内人数、总人数

#### Scenario: 摄像头模式信息显示
- **WHEN** 用户打开摄像头进行检测
- **THEN** 界面显示当前检测帧率、实际运行时长、区域内人数、总人数

### Requirement: 可视化开关控制
系统 SHALL 提供三个复选框开关：显示检测框、显示标签、显示追踪轨迹，用户可以独立开关每个选项。

#### Scenario: 关闭检测框
- **WHEN** 用户取消勾选"显示检测框"
- **THEN** 画面中不再显示检测边界框，但检测和追踪仍在后台运行

### Requirement: PyQt5 可视化界面
系统 SHALL 使用 PyQt5 构建完整的图形界面，包含视频显示区域、控制按钮、参数设置、信息展示区域。界面布局清晰，操作直观。

#### Scenario: 主界面启动
- **WHEN** 用户运行 MainProgram.py
- **THEN** 显示完整的 PyQt5 窗口界面，包含所有控制组件

### Requirement: 模型训练脚本
系统 SHALL 提供 train_v8.py 脚本，支持从 YAML 配置文件加载数据，使用 YOLOv8 进行人员检测模型训练，可配置 epochs 和 batch 参数。

#### Scenario: 训练模型
- **WHEN** 用户配置好数据集和 data.yaml 后运行 train_v8.py
- **THEN** 系统开始训练，训练结果保存在 runs/ 目录下

### Requirement: 测试脚本
系统 SHALL 提供独立的 VideoTest.py（视频检测）、CameraTest.py（摄像头检测）、imgTest.py（图片检测）脚本，可脱离 GUI 独立运行。

### Requirement: 环境安装
系统 SHALL 提供 installPackages.py 一键安装脚本和 requirements.txt 依赖清单，方便用户快速搭建运行环境。

### Requirement: 报警音频
系统 SHALL 提供报警音频文件（.wav 格式），在触发报警时播放。可使用 Python 程序合成生成。

### Requirement: 代码质量
所有代码 SHALL 完整、可直接运行、注释清晰、支持 CPU 运行。
# -*- coding: utf-8 -*-
"""
一键环境安装脚本
自动安装项目所需的所有 Python 依赖包
支持普通 pip 安装和 Anaconda 环境安装
"""

import subprocess
import sys


def install_with_pip():
    """使用 pip 安装所有依赖包"""
    print("=" * 60)
    print("  使用 pip 安装依赖")
    print("=" * 60)

    packages = [
        "ultralytics",
        "opencv-python",
        "PyQt5",
        "supervision",
        "numpy",
        "pillow",
    ]

    for pkg in packages:
        print(f"\n正在安装: {pkg} ...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]
            )
            print(f"  [OK] {pkg} 安装成功")
        except subprocess.CalledProcessError:
            print(f"  [WARN] 清华源安装失败，尝试默认源...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"  [OK] {pkg} 安装成功")
            except subprocess.CalledProcessError as e:
                print(f"  [FAIL] {pkg} 安装失败: {e}")


def install_with_conda():
    """使用 conda + pip 安装所有依赖包"""
    print("=" * 60)
    print("  使用 Anaconda 安装依赖")
    print("=" * 60)

    print("\n1. 安装 conda 包...")
    conda_packages = ["opencv", "numpy", "pillow"]
    try:
        subprocess.check_call(
            ["conda", "install", "-c", "conda-forge"] + conda_packages + ["-y"]
        )
        print("  [OK] conda 包安装成功")
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] conda 安装失败，将使用 pip 尝试: {e}")

    print("\n2. 安装 pip 包...")
    pip_packages = ["ultralytics", "PyQt5", "supervision"]
    for pkg in pip_packages:
        print(f"\n正在安装: {pkg} ...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]
            )
            print(f"  [OK] {pkg} 安装成功")
        except subprocess.CalledProcessError:
            print(f"  [WARN] 清华源安装失败，尝试默认源...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"  [OK] {pkg} 安装成功")
            except subprocess.CalledProcessError as e:
                print(f"  [FAIL] {pkg} 安装失败: {e}")


def main():
    print("=" * 60)
    print("  危险区域人员闯入检测与报警系统 - 环境安装")
    print("=" * 60)
    print("\n请选择安装方式:")
    print("  [1] 普通 pip 安装（适用于系统 Python）")
    print("  [2] Anaconda 安装（推荐，需先激活 conda 环境）")
    
    try:
        choice = input("\n请输入选择 (1/2): ").strip()
    except KeyboardInterrupt:
        print("\n安装已取消")
        return

    if choice == "1":
        install_with_pip()
    elif choice == "2":
        install_with_conda()
    else:
        print("无效选择，使用默认 pip 安装")
        install_with_pip()

    print("\n" + "=" * 60)
    print("  环境安装完成！")
    print("  请运行 MainProgram.py 启动系统")
    print("=" * 60)


if __name__ == "__main__":
    main()
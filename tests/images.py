"""测试里编好的几种画面:全灰、摄像头占位图、正常画面。

从取帧那个缝喂进去(`ScriptedFrames`),用来测「画面均匀」这条规则——它是识别器里
唯一一条不需要模型就能验的规则(spec「Testing Decisions」)。

占位图这一份是照着现场那张来的:2026-09-17 握拳平移原型开头约 27 秒,摄像头隐私开关
关着,给出的是灰底加相机图标的占位图,整段录制作废。
"""

from __future__ import annotations

import cv2
import numpy as np

WIDTH, HEIGHT = 1280, 720


def gray(value: int = 128, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """一片纯色。隐私挡片挡住镜头、全黑都是这一种。"""
    return np.full((height, width, 3), value, np.uint8)


def placeholder(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """摄像头占位图:灰底(约 140)加中间一个相机图标。

    灰度标准差约 9,比「纯色」那一档(6)高,所以它只能靠占位图那一条认出来。
    """
    img = np.full((height, width, 3), 140, np.uint8)
    cv2.rectangle(
        img,
        (width // 2 - 180, height // 2 - 120),
        (width // 2 + 180, height // 2 + 120),
        (105, 105, 105),
        -1,
    )
    cv2.circle(img, (width // 2, height // 2), 90, (150, 150, 150), -1)
    return img


def normal(shift: int = 0, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """有内容的画面:左右渐变加两个形状,灰度标准差约 80。"""
    ramp = np.linspace(0, 255, width, dtype=np.uint8)
    img = np.dstack([np.tile(ramp, (height, 1))] * 3).astype(np.uint8)
    cv2.rectangle(img, (100 + shift, 100), (400 + shift, 400), (20, 20, 20), -1)
    cv2.circle(img, (900, 300), 120, (250, 250, 250), -1)
    return img


def run(kind: str, n: int) -> list[np.ndarray]:
    """一段同一种画面。正常画面每帧挪一点,免得像静止图。"""
    if kind == "gray":
        return [gray() for _ in range(n)]
    if kind == "black":
        return [gray(0) for _ in range(n)]
    if kind == "placeholder":
        return [placeholder() for _ in range(n)]
    if kind == "normal":
        return [normal(shift=k) for k in range(n)]
    raise ValueError(f"没有这种画面:{kind}")

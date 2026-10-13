"""识别结果那个缝里不用模型就能验的几件:矩形怎么贴边、怎么切,哪只手算最大的。"""

from __future__ import annotations

import numpy as np
import pytest

from vision_side.recognize import FrameResult, HandResult, Region

FULL = (1280, 720)


def hand(box, *, label="fist", track_id=1, confirmed=True) -> HandResult:
    return HandResult(box=box, label=label, track_id=track_id, confirmed=confirmed)


# ---- 矩形 --------------------------------------------------------------


def test_宽高要大于零():
    with pytest.raises(ValueError):
        Region(0, 0, 0, 100)


def test_本来就在画面里的不动():
    region = Region(240, 60, 800, 600)
    assert region.clamped_to(*FULL) == region


def test_超出右下角就贴着边放():
    region = Region(1000, 500, 800, 600).clamped_to(*FULL)
    assert (region.x, region.y) == (480, 120)
    # 贴边不缩小:检测区域还是 800x600
    assert (region.width, region.height) == (800, 600)


def test_超出左上角也贴着边放():
    region = Region(-200, -50, 800, 600).clamped_to(*FULL)
    assert (region.x, region.y) == (0, 0)
    assert (region.width, region.height) == (800, 600)


def test_画面比它还小时才裁():
    region = Region(0, 0, 800, 600).clamped_to(640, 480)
    assert (region.x, region.y, region.width, region.height) == (0, 0, 640, 480)


def test_切出来的就是那一块():
    image = np.zeros((720, 1280, 3), np.uint8)
    image[60:660, 240:1040] = 7
    cut = Region(240, 60, 800, 600).crop(image)
    assert cut.shape == (600, 800, 3)
    assert (cut == 7).all()


def test_左上角就是换算手框时要加的偏移():
    assert Region(240, 60, 800, 600).origin == (240, 60)


# ---- 手框 --------------------------------------------------------------


def test_手框宽就是手宽():
    assert hand((100, 100, 134, 140)).width == 34.0


def test_手框中心():
    assert hand((100, 100, 200, 300)).center == (150.0, 200.0)


# ---- 最大的那只手 ------------------------------------------------------


def test_挑最大的那只已确认的手():
    small = hand((0, 0, 20, 20), track_id=1)
    big = hand((100, 100, 200, 200), track_id=2)
    result = FrameResult(t=0.0, hands=(small, big))
    assert result.largest().track_id == 2


def test_没确认的再大也不算():
    big = hand((100, 100, 300, 300), track_id=2, confirmed=False)
    small = hand((0, 0, 20, 20), track_id=1)
    result = FrameResult(t=0.0, hands=(big, small))
    assert result.largest().track_id == 1


def test_一只已确认的都没有就是没有():
    result = FrameResult(
        t=0.0, hands=(hand((0, 0, 20, 20), confirmed=False),)
    )
    assert result.largest() is None
    assert FrameResult(t=0.0).largest() is None


def test_一帧的结果默认不是均匀画面也没送检测区域():
    result = FrameResult(t=1.5)
    assert result.t == 1.5
    assert result.hands == ()
    assert result.flat is False
    assert result.region is None


def test_区里区外分得开():
    zone = Region(520, 240, 240, 240)
    assert zone.contains((640, 360))
    assert zone.contains((520, 240))  # 边上算在里面
    assert zone.contains((760, 480))
    assert not zone.contains((519, 360))
    assert not zone.contains((640, 481))

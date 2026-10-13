"""取帧那个缝:翻转、时间戳、放完了没有,还有等第一帧。

不开摄像头(打不开的编号那一条除外,它不会真的用到设备),视频文件那几条自己写一段
临时录像来读。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision_side.frames import (
    CameraFrames,
    Frame,
    FrameError,
    FrameSource,
    ScriptedFrames,
    VideoFrames,
    first_frame,
)


def marked(n: int = 3, width: int = 64, height: int = 48) -> list[np.ndarray]:
    """左右不对称的几帧:左半边亮、右半边暗,翻转翻没翻一眼看得出。"""
    images = []
    for k in range(n):
        img = np.zeros((height, width, 3), np.uint8)
        img[:, : width // 2] = 200 + k
        images.append(img)
    return images


# ---- 翻转 --------------------------------------------------------------


def test_取出来的画面已经水平翻转过():
    images = marked(1)
    with ScriptedFrames(images) as source:
        frame = source.next_frame()
    assert frame is not None
    assert np.array_equal(frame.image, cv2.flip(images[0], 1))


def test_只翻一次():
    # 翻两次就等于没翻,拿原图比一下就知道翻了奇数次还是偶数次
    images = marked(1)
    with ScriptedFrames(images) as source:
        frame = source.next_frame()
    assert frame is not None
    assert not np.array_equal(frame.image, images[0])


# ---- 帧号和时间戳 ------------------------------------------------------


def test_帧号从零开始一帧一帧往上加():
    with ScriptedFrames(marked(3)) as source:
        frames = [source.next_frame() for _ in range(3)]
    assert [f.index for f in frames] == [0, 1, 2]


def test_编好的帧序列按三十帧每秒排时间戳():
    with ScriptedFrames(marked(3)) as source:
        frames = [source.next_frame() for _ in range(3)]
    assert [round(f.t, 4) for f in frames] == [0.0, round(1 / 30, 4), round(2 / 30, 4)]


def test_帧知道自己多大():
    with ScriptedFrames(marked(1, width=64, height=48)) as source:
        frame = source.next_frame()
    assert (frame.width, frame.height) == (64, 48)


# ---- 打开、关闭、放完 --------------------------------------------------


def test_放完了就没有下一帧():
    with ScriptedFrames(marked(2)) as source:
        assert source.has_next()
        source.next_frame()
        source.next_frame()
        assert not source.has_next()
        assert source.next_frame() is None


def test_关过了再关一次不出错():
    source = ScriptedFrames(marked(1))
    source.open()
    source.close()
    source.close()
    assert not source.is_open


def test_打开过了再打开一次不出错():
    source = ScriptedFrames(marked(2))
    source.open()
    source.next_frame()
    source.open()  # 没有副作用:不会从头开始
    assert source.next_frame().index == 1
    source.close()


def test_没打开就取帧要报错():
    source = ScriptedFrames(marked(1))
    with pytest.raises(FrameError) as got:
        source.next_frame()
    assert got.value.reason == "CAMERA NOT OPEN"


def test_打不开的摄像头编号当场报错():
    # 不会真的用到设备:没有 999 号摄像头,OpenCV 立刻说打不开
    with pytest.raises(FrameError) as got:
        CameraFrames(index=999).open()
    assert got.value.reason == "CAMERA NOT FOUND"


# ---- 等第一帧 ----------------------------------------------------------


class LateSource(FrameSource):
    """前几次读空,之后才给出画面——摄像头刚打开时就是这样。"""

    def __init__(self, empty_reads: int) -> None:
        super().__init__()
        self.left = empty_reads

    def _open(self) -> None: ...

    def _close(self) -> None: ...

    def _read(self):
        if self.left > 0:
            self.left -= 1
            return None
        return np.zeros((48, 64, 3), np.uint8), 0.0

    def has_next(self) -> bool:
        return self.is_open


def test_第一帧晚几次才来也等得到():
    frame = first_frame(LateSource(empty_reads=3), timeout_s=2.0, poll_s=0.001)
    assert isinstance(frame, Frame)


def test_等不到第一帧就报错():
    source = LateSource(empty_reads=10_000)
    with pytest.raises(FrameError) as got:
        first_frame(source, timeout_s=0.05, poll_s=0.001)
    assert got.value.reason == "NO CAMERA IMAGE"


def test_一帧都没有的画面直接报没画面():
    with pytest.raises(FrameError) as got:
        first_frame(ScriptedFrames([]), timeout_s=1.0)
    assert got.value.reason == "NO FRAMES"


# ---- 视频文件 ----------------------------------------------------------


@pytest.fixture
def clip(tmp_path):
    """写一段 5 帧的临时录像。写不出来(缺编码器)就跳过。"""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    if not writer.isOpened():
        writer.release()
        pytest.skip("这台机器写不出 mp4,跳过视频文件那几条")
    for k in range(5):
        writer.write(np.full((48, 64, 3), 40 * k, np.uint8))
    writer.release()
    return path


def test_录像一帧一帧读得出来(clip):
    with VideoFrames(clip) as source:
        frames = []
        while source.has_next():
            frame = source.next_frame()
            if frame is None:
                break
            frames.append(frame)
    assert len(frames) == 5
    assert [f.index for f in frames] == [0, 1, 2, 3, 4]


def test_录像的时间戳按视频里的位置走(clip):
    with VideoFrames(clip) as source:
        times = []
        while source.has_next():
            frame = source.next_frame()
            if frame is None:
                break
            times.append(frame.t)
    # 从 0 开始、一帧一帧往上,不是 time.monotonic()(那是个很大的数)
    assert times[0] == 0.0
    assert times == sorted(times)
    assert times[-1] == pytest.approx(4 / 30, abs=0.01)


def test_找不到录像就报错(tmp_path):
    with pytest.raises(FrameError) as got:
        VideoFrames(tmp_path / "没有这个.mp4").open()
    assert got.value.reason == "VIDEO NOT FOUND"


def test_同一段录像读两遍拿到一样的时间戳(clip):
    def times():
        with VideoFrames(clip) as source:
            out = []
            while source.has_next():
                frame = source.next_frame()
                if frame is None:
                    break
                out.append(frame.t)
            return out

    # 回放测试要的是「同一段录像、同样的代码,结果逐条一致」
    assert times() == times()

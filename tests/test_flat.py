"""「画面均匀」这条规则:从取帧那个缝喂全灰、占位图、正常画面各一段,看报得对不对。

不用模型:这条规则算的是像素,识别器只是每帧把它报上来(那一条在 `test_gestures.py`,
要模型)。时间戳来自取帧那个缝,按 30 fps 排,所以「持续约 0.5 秒」能一帧不差地数。
"""

from __future__ import annotations

from vision_side.frames import ScriptedFrames
from vision_side.recognize import UNIFORM_FOR_S, UniformWatch, is_flat

from tests import images

FPS = 30
HALF_SECOND = int(UNIFORM_FOR_S * FPS)  # 15 帧


def watch(frames: list) -> list[bool]:
    """把一段画面从取帧那个缝喂进去,逐帧记下「这一刻算不算画面均匀」。"""
    got = []
    watcher = UniformWatch()
    with ScriptedFrames(frames) as source:
        while True:
            frame = source.next_frame()
            if frame is None:
                break
            got.append(watcher.update(frame.image, frame.t))
    return got


# ---- 单帧 --------------------------------------------------------------


def test_全灰和全黑是均匀画面():
    assert is_flat(images.gray())
    assert is_flat(images.gray(0))


def test_摄像头占位图是均匀画面():
    assert is_flat(images.placeholder())


def test_正常画面不是均匀画面():
    assert not is_flat(images.normal())


# ---- 连续 0.5 秒 -------------------------------------------------------


def test_全灰要连着半秒才算():
    got = watch(images.run("gray", FPS))
    assert not any(got[:HALF_SECOND])  # 前半秒只是「这一帧灰」,还不算
    assert all(got[HALF_SECOND:])


def test_占位图要连着半秒才算():
    got = watch(images.run("placeholder", FPS))
    assert not any(got[:HALF_SECOND])
    assert all(got[HALF_SECOND:])


def test_正常画面一直不算():
    assert not any(watch(images.run("normal", FPS)))


def test_中间亮回来一帧就从头数():
    frames = images.run("gray", 12) + images.run("normal", 1) + images.run("gray", 12)
    got = watch(frames)
    # 一共 25 帧、不到 1 秒,但哪一段都没连着 15 帧,所以一次都不算
    assert not any(got)


def test_亮回来之后再暗下去还是要重新数满半秒():
    frames = images.run("normal", 5) + images.run("black", FPS)
    got = watch(frames)
    assert not any(got[: 5 + HALF_SECOND])
    assert all(got[5 + HALF_SECOND :])


def test_清掉记录之后重新数():
    watcher = UniformWatch()
    with ScriptedFrames(images.run("gray", FPS)) as source:
        for _ in range(HALF_SECOND + 1):
            frame = source.next_frame()
            got = watcher.update(frame.image, frame.t)
        assert got
        watcher.reset()
        frame = source.next_frame()
        assert not watcher.update(frame.image, frame.t)

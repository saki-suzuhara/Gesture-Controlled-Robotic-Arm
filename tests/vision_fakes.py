"""测试里编好的识别结果来源(spec「模块划分」第二个缝的第三种实现)。

不开摄像头、不跑模型、不开 RoboDK:一段排好的逐帧结果喂进识别线程,就能把接线那几条
验完——触发排进事件队列、停止手势直接叫停、按状态切换送哪块画面并清对记录、识别中断。

**手框一律按整幅画面的坐标编**,和真识别器报上来的一样(`vision_side/recognize.py`)。
送进去的是哪块画面不影响这些结果,但**送法换了要看得见**:`ScriptedRecognizer` 把每
一帧收到的 `region` 和清跟踪记录的时刻都记下来,测试据此核对切换时机。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vision_side.frames import SCRIPTED_STEP_S, Frame, FrameSource
from vision_side.recognize import (
    FrameResult,
    HandResult,
    Recognizer,
    Region,
    UniformWatch,
)

FPS = 1.0 / SCRIPTED_STEP_S
WIDTH, HEIGHT = 1280, 720


def hand(
    center: tuple[float, float],
    *,
    label: str = "fist",
    track_id: int = 1,
    confirmed: bool = True,
    width: float = 80.0,
) -> HandResult:
    """一只手,手框按中心和宽摆出来(和 `tests/test_triggers.py` 同一个写法)。"""
    cx, cy = center
    half = width / 2
    return HandResult(
        box=(round(cx - half), round(cy - half), round(cx + half), round(cy + half)),
        label=label,
        track_id=track_id,
        confirmed=confirmed,
    )


class ScriptedFrameSource(FrameSource):
    """一路取之不尽的空画面。

    识别结果是编好的,画面本身没人看,所以只给一幅小的纯色图——省内存,也省得测试
    里到处造 1280×720 的数组。时间戳按 30 fps 往前排。
    """

    def __init__(
        self, *, step_s: float = SCRIPTED_STEP_S, image: np.ndarray | None = None
    ) -> None:
        super().__init__()
        self.step_s = step_s
        self.image = (
            image if image is not None else np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        )
        """现在在放的那幅画面。中途换掉它 = 摄像头画面变了(比如隐私开关关上)。"""

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    def _read(self) -> tuple[np.ndarray, float] | None:
        # `_index` 是基类记的帧号(读完这一帧才加一),拿它排时间戳正好
        return self.image, self._index * self.step_s

    def has_next(self) -> bool:
        return self.is_open


class ScriptedRecognizer(Recognizer):
    """按排好的顺序一帧一帧吐结果,并记下每帧送的是哪块画面。

    `results` 用完以后一直重复最后一份(多半是「一只手都没有」),这样测试只用排出
    要紧的那几帧,后面爱跑多久跑多久。
    """

    def __init__(
        self, results: Sequence[FrameResult] | None = None, *, real_flat: bool = False
    ) -> None:
        self.real_flat = real_flat
        """真按画面像素算「画面均匀」,不照编好的那一位。

        验「摄像头隐私开关关上 → 识别中断」时要用它:那一档的关键就是**占位图真的被
        认成均匀**,编一个 `flat=True` 进去等于把要验的那一步跳过了。用的是真识别器
        里同一个 `UniformWatch`。
        """
        self._uniform = UniformWatch()
        self.results = list(results or [])
        self.sent: list[Region | None] = []
        """每一帧送进来的那块画面。切换时机按它核对。"""
        self.cleared_at: list[int] = []
        """第几帧的时候清过跟踪记录(按 `sent` 的长度记)。"""
        self.after: list[FrameResult] = []
        """对上以后换上的那一段。放哪儿由测试自己定,这里只是替它拿着。"""
        self._at = 0

    def restart(self, results: Sequence[FrameResult]) -> None:
        """换一段结果,从头放。

        对准要走多少帧,得看演示配置里的两个时长和判定内部怎么算,数不准;所以
        「对准那一段」和「对上以后那一段」分开排,对上了再换过来。
        """
        self.results = list(results)
        self._at = 0

    def recognize(self, frame: Frame, region: Region | None = None) -> FrameResult:
        self.sent.append(region)
        if self._at < len(self.results):
            got = self.results[self._at]
            self._at += 1
        elif self.results:
            got = self.results[-1]
        else:
            got = FrameResult(t=frame.t)
        flat = (
            self._uniform.update(frame.image, frame.t) if self.real_flat else got.flat
        )
        # 时间戳一律按这一帧的来:编结果的人只管手和标签,不用跟着算时间
        return FrameResult(t=frame.t, hands=got.hands, flat=flat, region=region)

    def clear_tracks(self) -> None:
        self.cleared_at.append(len(self.sent))


def still(seconds: float, center: tuple[float, float], **kwargs) -> list[FrameResult]:
    """举着不动的一段。"""
    return [FrameResult(t=0.0, hands=(hand(center, **kwargs),))] * round(seconds * FPS)


def sweep(
    start: tuple[float, float], end: tuple[float, float], seconds: float, **kwargs
) -> list[FrameResult]:
    """从一处匀速移到另一处。"""
    n = max(1, round(seconds * FPS))
    out = []
    for i in range(1, n + 1):
        out.append(
            FrameResult(
                t=0.0,
                hands=(
                    hand(
                        (
                            start[0] + (end[0] - start[0]) * i / n,
                            start[1] + (end[1] - start[1]) * i / n,
                        ),
                        **kwargs,
                    ),
                ),
            )
        )
    return out


def nothing(seconds: float, *, flat: bool = False) -> list[FrameResult]:
    """一只手都没有的一段。`flat=True` 就是「画面已经均匀了约 0.5 秒」。"""
    return [FrameResult(t=0.0, flat=flat)] * round(seconds * FPS)

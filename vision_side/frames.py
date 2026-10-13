"""取帧:视觉一侧的第一个缝(spec「两个缝的形状」)。

摄像头、视频文件、测试里编好的帧序列都实现这同一组方法,上层因此不知道画面是从
镜头来的、从录像来的,还是测试自己编的。

有的是:打开 / 关闭 / 取下一帧(带时间戳)/ 后面还有没有画面。

三条从实测来的规矩,写在方法的说明里,也由这一层强制:

- **画面水平翻转在这里做,只做一次**(`next_frame` 里),翻完再交给识别;后面所有
  坐标都按翻转后的画面算。不翻转时带方向的姿势标签会左右对调(2026-09-17 原型实测)。
- **打开摄像头和拿到第一帧是两件事**:本机打开约 0.1–0.5 秒,首帧还要 0.3–1.2 秒
  (2026-09-17 原型实测),所以 `next_frame()` 这一次没取到只返回 `None`,不算错;
  等首帧的那 5 秒由 `first_frame()` 数。
- **时间戳的起点按来源定**:摄像头用 `time.monotonic()`,视频文件用画面在视频里的
  位置(从 0 算)。所以回放时上层只能拿帧的时间戳互减,不能拿 `time.monotonic()`
  去减录像的时间戳——否则同一段录像每次跑出来的结果都不一样(spec「回放测试」)。
"""

from __future__ import annotations

import abc
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
"""`CameraFrames()` 不给参数时用的那一档。

**程序里不要读这三个**:摄像头编号和分辨率在演示配置的 `[vision]` 一节里
(`camera_index`、`camera_px`),调参不用改代码。这三个只是构造函数的默认值,
好让测试和一次性的小脚本 `CameraFrames()` 一写就能用。

数值来源(2026-09-17 原型实测):笔记本自带的 RGB 摄像头是 0 号;1.2 m 处 1280×720
的姿势标签正确率 51%,640×480 只有 42%,帧率同为 30 fps,所以取大的那一档。
"""

FIRST_FRAME_WAIT_S = 5.0
"""等第一帧最多等多久(spec「启动顺序」第 1 步)。实测首帧 0.3–1.2 秒,余量充足。"""

SCRIPTED_STEP_S = 1.0 / 30.0
"""编好的帧序列里,两帧之间差多少秒。按摄像头的 30 fps 算。"""


class FrameError(RuntimeError):
    """取帧这一侧的错误:摄像头打不开、视频文件读不了、等不到第一帧。

    `reason` 是预览窗口上那一行短句,**英文**(`CONTEXT.md`「预览窗口」),
    和机械臂一侧的 `ArmError.reason` 一样的用法。
    """

    def __init__(self, message: str, reason: str = "CAMERA ERROR") -> None:
        super().__init__(message)
        self.reason = reason


class Frame:
    """一帧画面和它的时间戳。

    `image` 是**已经水平翻转过**的 BGR 画面(`FrameSource.next_frame` 翻的),
    `t` 的起点按来源定(见模块说明),`index` 是这一路画面交出去的第几帧,从 0 算。

    不用 dataclass:自动生成的 `__repr__` 会把整幅画面的像素全打出来,`__eq__`
    拿 numpy 数组比较还会抛异常。
    """

    __slots__ = ("image", "t", "index")

    def __init__(self, image: np.ndarray, t: float, index: int) -> None:
        self.image = image
        self.t = t
        self.index = index

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def __repr__(self) -> str:
        return f"Frame(#{self.index}, t={self.t:.3f}, {self.width}x{self.height})"


class FrameSource(abc.ABC):
    """取帧接口(FRAME SOURCE)。

    各实现只写 `_open` / `_close` / `_read` 三个钩子;打开状态、帧号和**水平翻转**
    由这一层管,免得哪一份实现忘了翻,或者翻了两次。

    一个线程用(识别线程,spec「线程」)。
    """

    def __init__(self) -> None:
        self._open_flag = False
        self._index = 0

    @property
    def is_open(self) -> bool:
        return self._open_flag

    def open(self) -> None:
        """打开这一路画面。打不开就抛 `FrameError`。已经打开时再调没有副作用。"""
        if self._open_flag:
            return
        self._open()
        self._open_flag = True

    def close(self) -> None:
        """关掉。关过了再关一次不出错。"""
        if not self._open_flag:
            return
        self._open_flag = False
        self._close()

    def next_frame(self) -> Frame | None:
        """取下一帧,**水平翻转**之后交出去;这一次没取到就返回 `None`。

        返回 `None` 不算错:摄像头刚打开时要等几百毫秒才有第一帧,中途也可能偶尔
        读空。视频文件放完了同样是 `None`,配 `has_next()` 分得清是哪一种。
        """
        if not self._open_flag:
            raise FrameError("这一路画面还没打开", reason="CAMERA NOT OPEN")
        got = self._read()
        if got is None:
            return None
        image, t = got
        frame = Frame(cv2.flip(image, 1), t, self._index)
        self._index += 1
        return frame

    @abc.abstractmethod
    def has_next(self) -> bool:
        """后面还有没有画面。摄像头只要还开着就是有,视频文件放完了就是没有。"""

    @abc.abstractmethod
    def _open(self) -> None: ...

    @abc.abstractmethod
    def _close(self) -> None: ...

    @abc.abstractmethod
    def _read(self) -> tuple[np.ndarray, float] | None:
        """读一帧原始画面和它的时间戳;没读到返回 `None`。**翻转不在这里做。**"""

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class CameraFrames(FrameSource):
    """摄像头。默认 0 号、1280×720(见模块开头的常量)。"""

    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
    ) -> None:
        super().__init__()
        self.index = index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def _open(self) -> None:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise FrameError(f"打不开 {self.index} 号摄像头", reason="CAMERA NOT FOUND")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read(self) -> tuple[np.ndarray, float] | None:
        assert self._cap is not None
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        return image, time.monotonic()

    def has_next(self) -> bool:
        """摄像头不会放完,只要还开着就还有。"""
        return self.is_open

    @property
    def backend(self) -> str:
        """OpenCV 用的后端名字,记进日志用。本机默认是 MSMF(2026-09-17 实测)。"""
        return self._cap.getBackendName() if self._cap is not None else ""


class VideoFrames(FrameSource):
    """视频文件,回放测试用。

    时间戳取画面在视频里的位置(秒,从 0 算),不用 `time.monotonic()`:同一段录像
    每次跑出来的帧时间戳要一样,回放结果才能逐条对上(spec「回放测试」)。
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self._cap: cv2.VideoCapture | None = None
        self._count = 0
        self._fps = 0.0

    def _open(self) -> None:
        if not self.path.exists():
            raise FrameError(f"找不到录像 {self.path}", reason="VIDEO NOT FOUND")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise FrameError(f"读不了录像 {self.path}", reason="VIDEO UNREADABLE")
        self._cap = cap
        self._count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(cap.get(cv2.CAP_PROP_FPS))

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read(self) -> tuple[np.ndarray, float] | None:
        assert self._cap is not None
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        # 位置要在 read() **之后**取:这时拿到的正是刚读出来这一帧的时间
        # (读之前取的话,头两帧都报 0,再往后又慢一帧;2026-09-21 在原型录像上实测)。
        pos_s = float(self._cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if pos_s <= 0.0 and self._index > 0 and self._fps > 0.0:
            # 有的文件根本不报位置(自己用 mp4v 写出来的就不报),那就按帧号和帧率算。
            pos_s = self._index / self._fps
        return image, pos_s

    def has_next(self) -> bool:
        """放完了就没有了。文件不报总帧数时只当还有,读到空为止。"""
        if not self.is_open or self._cap is None:
            return False
        if self._count <= 0:
            return True
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) < self._count


class ScriptedFrames(FrameSource):
    """测试里编好的帧序列(全灰、摄像头占位图、正常画面……)。

    不开摄像头、不读文件,时间戳按 30 fps 一帧一帧往前排,所以「持续约 0.5 秒」
    这类规则能一帧不差地测出来。
    """

    def __init__(
        self, images: Sequence[np.ndarray], *, step_s: float = SCRIPTED_STEP_S
    ) -> None:
        super().__init__()
        self.images = list(images)
        self.step_s = step_s
        self._pos = 0

    def _open(self) -> None:
        self._pos = 0

    def _close(self) -> None:
        pass

    def _read(self) -> tuple[np.ndarray, float] | None:
        if self._pos >= len(self.images):
            return None
        image = self.images[self._pos]
        t = self._pos * self.step_s
        self._pos += 1
        return image, t

    def has_next(self) -> bool:
        return self.is_open and self._pos < len(self.images)


def first_frame(
    source: FrameSource, timeout_s: float = FIRST_FRAME_WAIT_S, poll_s: float = 0.01
) -> Frame:
    """等这一路画面的第一帧,最多等 `timeout_s` 秒(spec「启动顺序」第 1 步)。

    源还没打开就先打开。等不到就抛 `FrameError`——启动检查据此报错退出,一次都不碰
    机械臂。视频文件放完了也走这一条。
    """
    source.open()
    deadline = time.monotonic() + timeout_s
    while True:
        frame = source.next_frame()
        if frame is not None:
            return frame
        if not source.has_next():
            raise FrameError("这一路画面已经没有了", reason="NO FRAMES")
        if time.monotonic() > deadline:
            raise FrameError(
                f"等了 {timeout_s:.1f} 秒还没拿到第一帧", reason="NO CAMERA IMAGE"
            )
        time.sleep(poll_s)

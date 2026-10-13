"""提示音。用 Python 自带的 `winsound`,不装新包。

做完手势后操作者通常看着机械臂、不看屏幕,所以要紧的几件事各有一种声音:认到触发、
认到叫停、倒计时每秒一声、进入已停止连响三声(`docs/grill-decisions.md` 第一节第 4 组);
视觉一侧再加一种:启动对准成功(spec「提示音」)。

**放在单独的线程里播**:`winsound.Beep` 要等声音放完才返回,在主线程里播会把状态逻辑
和预览窗口一起卡住。这个线程只播声音,不碰别的东西;它落后了就落后了,不拖住主线程。
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

Play = Callable[[int, int], None]
"""播一声:频率(Hz)、时长(ms)。就是 `winsound.Beep` 的样子。"""


def _winsound_play() -> Play | None:
    """拿 `winsound.Beep`;拿不到(不在 Windows 上)就返回 None,静音运行。"""
    try:
        import winsound
    except ImportError:
        return None
    return winsound.Beep


#: 每一种声音:频率 Hz、时长 ms、响几声。调过一轮,高低分得开就行。
TRIGGER = (880, 120, 1)
"""认到触发手势:一声中音。"""

STOPPED = (440, 200, 1)
"""认到叫停:一声低音,和触发分得开。"""

ALIGNED = (1175, 110, 2)
"""启动对准成功:两声短高音。对准时操作者站在远处、多半看着自己的手,
所以这一声要和触发那一声分得开(spec「提示音」)。"""

COUNTDOWN = (660, 100, 1)
"""倒计时:每秒一声短音。"""

HALTED = (330, 250, 3)
"""进入已停止:连响三声低音——不看屏幕也知道机械臂停下等人了。"""

GO = (1568, 140, 1)
"""录制时「做!」段开始:一声高音(spec「录制工具」第 4 条)。

比对准那两声还高:录的时候人站在 1–1.5 m 外、眼睛多半在自己手上,这一声是
「现在开始做」的唯一提示,得从别的几种里跳出来。只有录制工具用它。"""


class Beeper:
    """提示音。每个方法把要播的声音丢进队列就返回,不等它播完。

    `play` 换成别的可以静音或者在测试里记账;不给就用 `winsound.Beep`,不在 Windows
    上时自动静音。`block=True` 只给测试用:当场播完再返回,好数响了几声。
    """

    def __init__(self, play: Play | None = None, *, block: bool = False) -> None:
        self._play = play if play is not None else _winsound_play()
        self._block = block
        self._queue: queue.SimpleQueue[tuple[int, int, int] | None] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        if self._play is not None and not block:
            self._thread = threading.Thread(
                target=self._run, name="beeps", daemon=True
            )
            self._thread.start()

    def trigger(self) -> None:
        self._send(TRIGGER)

    def stopped(self) -> None:
        self._send(STOPPED)

    def aligned(self) -> None:
        self._send(ALIGNED)

    def countdown(self) -> None:
        self._send(COUNTDOWN)

    def halted(self) -> None:
        self._send(HALTED)

    def go(self) -> None:
        """录制时「做!」段开始。只有录制工具用。"""
        self._send(GO)

    def close(self, timeout_s: float = 1.0) -> None:
        """结束播放线程。没启动过时再调没有副作用。"""
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout_s)
        self._thread = None

    def __enter__(self) -> Beeper:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- 内部 ----------------------------------------------------------

    def _send(self, sound: tuple[int, int, int]) -> None:
        if self._play is None:
            return
        if self._block:
            self._one(sound)
            return
        self._queue.put(sound)

    def _one(self, sound: tuple[int, int, int]) -> None:
        freq, ms, times = sound
        assert self._play is not None
        for _ in range(times):
            try:
                self._play(freq, ms)
            except Exception:
                # 放不出声不该把程序带倒:声音是提示,不是安全措施
                return

    def _run(self) -> None:
        while True:
            sound = self._queue.get()
            if sound is None:
                return
            self._one(sound)

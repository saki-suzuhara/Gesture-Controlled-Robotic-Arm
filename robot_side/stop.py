"""叫停通道(ADR-0012):按键线程 + 「已叫停」标记 + 机械臂接口的「停止」。

谁先发现要停,谁就直接叫停,不经过主线程——因为主线程会卡住(拖预览窗口停过
6.8 秒),一卡就读不到空格、也发不出停止。叫停时先立「已叫停」标记,再调机械臂
接口的「停止」;执行线程每段开始前看这个标记,已叫停就不再发下一段。

空格由单独的按键线程每 10 ms 用 Windows 自带的 `GetAsyncKeyState` 查一次,全局
有效:预览窗口卡住、不在前台,或者焦点在别的程序里,都照样叫停。**只在 Windows
上成立**;项目本来就只在 Windows 上跑。回车、R、q / Esc 仍然只在预览窗口里认
(第 03 票)。
"""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from robot_side.arm import Arm

VK_SPACE = 0x20
POLL_S = 0.010

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
_user32.GetAsyncKeyState.restype = ctypes.c_short


def key_is_down(vk_code: int) -> bool:
    """这个键现在按着吗。全局的,不看哪个窗口在前台。"""
    return bool(_user32.GetAsyncKeyState(vk_code) & 0x8000)


@dataclass(frozen=True)
class StopRecord:
    """一次叫停:来源,立标记的时刻,机械臂接口的「停止」返回的时刻。"""

    source: str
    flag_at: float
    stopped_at: float
    error: str | None = None


class StopSignal:
    """「已叫停」标记加上直达机械臂的停止。

    `raise_stop()` 可以从任意线程调,重复调没有副作用——底下的 `Arm.stop()`
    本身就是幂等的。
    """

    def __init__(self, arm: Arm) -> None:
        self._arm = arm
        self._flag = threading.Event()
        self._lock = threading.Lock()
        self._records: list[StopRecord] = []

    @property
    def raised(self) -> bool:
        """已经叫停了吗。执行线程发下一段之前看这个。"""
        return self._flag.is_set()

    @property
    def records(self) -> list[StopRecord]:
        """按先后记下的每一次叫停。"""
        with self._lock:
            return list(self._records)

    def raise_stop(self, source: str) -> None:
        """叫停:先立标记,再停机械臂。顺序不能反(ADR-0012)。

        `source` 记进日志,比如 "space"(按键线程)、"stop-gesture"(以后的识别线程)。
        机械臂那边报错时,错误记进 `records` 并原样抛出——标记已经立住,上层照样
        知道要停。
        """
        flag_at = time.monotonic()
        self._flag.set()
        error = None
        try:
            self._arm.stop()
        except Exception as exc:  # 记下来,不吞掉
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            with self._lock:
                self._records.append(
                    StopRecord(source, flag_at, time.monotonic(), error)
                )

    def wait(self, timeout_s: float) -> bool:
        """等这么久,中途被叫停就立刻返回;返回的是「这期间(或本来就)叫停了吗」。

        路点之间的停顿用它,不用 `time.sleep`:停顿期间机械臂本来就不动,但叫停之后
        不该还傻等着停顿走完才去回待命。
        """
        return self._flag.wait(timeout_s)

    def clear(self) -> None:
        """放下标记。回待命走完、重新接受手势之前调(第 03 票定在哪里调)。"""
        self._flag.clear()


class SpaceKeyWatcher:
    """按键线程:每 10 ms 查一次空格,按下的那一刻回调一次。

    只认按下去的那一下(从松到按),一直按着不会反复回调。回调在这个线程里跑,
    所以它做的事要短——照 ADR-0012,它做的就是叫停本身。
    """

    def __init__(self, on_press: Callable[[], None], poll_s: float = POLL_S) -> None:
        self._on_press = on_press
        self._poll_s = poll_s
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._errors: list[str] = []

    @property
    def errors(self) -> list[str]:
        """回调抛出的错误。线程不因此结束,否则就再也叫不了停了。"""
        return list(self._errors)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._watch, name="space-key", daemon=True
        )
        self._thread.start()

    def close(self, timeout_s: float = 1.0) -> None:
        """结束按键线程。没启动过时再调没有副作用。"""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout_s)
            self._thread = None

    def __enter__(self) -> SpaceKeyWatcher:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _watch(self) -> None:
        # 启动那一刻空格正按着的话,不算一次按下——等它松开再说
        was_down = key_is_down(VK_SPACE)
        while self._running.is_set():
            down = key_is_down(VK_SPACE)
            if down and not was_down:
                try:
                    self._on_press()
                except Exception as exc:  # 记下来,但线程要活着
                    self._errors.append(f"{type(exc).__name__}: {exc}")
            was_down = down
            time.sleep(self._poll_s)

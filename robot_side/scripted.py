"""编好的事件序列:不按键也能把整套状态逻辑走一遍。

和调试按键走的是同一条路——触发、回车、退出排进事件队列,叫停直接调
`StopSignal.raise_stop`(ADR-0012:叫停不走事件队列)。所以「编好的事件驱动」验的
就是真正那套逻辑,不是另一条旁路。

每一步写成「等到某个状态,再过几秒,做一件事」,而不是「开跑后第几秒做一件事」:
每段运动要走多久,取决于用的是空实现还是 RoboDK 仿真,按秒数写死的脚本换个实现就散了。

**进了已停止会自己按回车爬出来**(`MAX_RECOVERIES` 次为限)。已停止只有人按回车才
出得来,而脚本本来就是给没人看着的时候跑的:2026-09-20 用 `--script actions` 验收时,
外面按到的空格连叫两次停,脚本就在已停止里干等了两分多钟,只能杀掉。爬不出来就收摊
退出,不在那儿空转——现场真有问题时,空转比停下来更没用。

**连着真机时一次都不替人按**(`confirm_halts=False`)。2026-09-23 真机上:用户按空格
叫停了两次,脚本每次 3 秒后就自己按回车、重新开始回待命,等于跟人对着干。真机旁边
一定有人,已停止要人看过现场以后自己决定按回车还是按 q。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from robot_side.state import Event, Machine, State
from robot_side.stop import StopSignal


@dataclass(frozen=True)
class Step:
    """一步:等到 `wait_for` 这个状态,再过 `delay_s` 秒,做 `kind` 这件事。

    `wait_for` 不写就是「立刻」。`kind` 是 trigger / stop / confirm / quit;
    只有 trigger 用得上 `action`。

    **等到就算数**:一旦进过 `wait_for`,这一步就从那一刻开始数 `delay_s`,中间状态
    又变了也照做。否则动作走得快一点、状态早一步溜过去,脚本就卡在这里不动了。
    """

    kind: str
    action: str = ""
    wait_for: State | None = None
    delay_s: float = 0.0


#: 走一遍票里那两条链路的脚本。
FULL = (
    # 触发 → 执行中 → 叫停 → 倒计时 → 抬起 → 回待命 → 冷却 → 空闲
    Step("trigger", action="draw_square", wait_for=State.IDLE),
    Step("stop", wait_for=State.RUNNING, delay_s=1.5),
    # 回待命途中再叫停 → 已停止 → 回车 → 回待命 → 冷却 → 空闲
    Step("trigger", action="nod", wait_for=State.IDLE),
    Step("stop", wait_for=State.RUNNING, delay_s=1.0),
    Step("stop", wait_for=State.RETURNING, delay_s=1.0),
    Step("confirm", wait_for=State.HALTED, delay_s=1.0),
    Step("quit", wait_for=State.IDLE, delay_s=1.0),
)

#: 每个预设动作一个接一个跑一遍,不叫停。
ALL_ACTIONS = tuple(
    Step("trigger", action=name, wait_for=State.IDLE)
    for name in (
        "point_left", "point_right", "raise", "lower", "move_away", "come_closer",
        "draw_square", "nod", "bow",
    )
) + (Step("quit", wait_for=State.IDLE, delay_s=1.0),)

#: 只把启动那一段走通,一个动作都不触发,进了空闲就退出。
#:
#: 第一次连真机时用这一套:它验的是「打开工作站 → 按配置写入 → 可达性检查 → 经驱动
#: 连接 → 检查上电使能 → 回待命 → 空闲 → 干净退出」这整条链路,**但不触发任何预设
#: 动作**。机械臂已经在待命位姿附近(每轴 ±1°)时,回待命连抬起和移动都跳过,于是
#: 全程一条运动指令都不下发 —— 这是能对真机做的最小一次验证。
STARTUP_ONLY = (Step("quit", wait_for=State.IDLE, delay_s=1.0),)

SCRIPTS: dict[str, Sequence[Step]] = {
    "startup": STARTUP_ONLY,
    "full": FULL,
    "actions": ALL_ACTIONS,
}

MAX_RECOVERIES = 3
"""脚本最多替人按几次回车。

再多就说明不是偶然被叫停,而是现场真有问题——那就收摊退出,把已停止的原因留在
日志里给人看。设成 0 就是「一进已停止就收摊」。
"""


class ScriptPlayer:
    """按脚本往状态逻辑里喂事件。主循环每轮调一次 `pump()`。"""

    def __init__(
        self,
        steps: Sequence[Step],
        machine: Machine,
        signal: StopSignal,
        *,
        clock: Callable[[], float] = time.monotonic,
        log: logging.Logger | None = None,
        max_recoveries: int = MAX_RECOVERIES,
        confirm_halts: bool = True,
    ) -> None:
        """`confirm_halts=False` 时进了已停止一次都不替人按回车,等人来(连着真机时这么用)。"""
        self._steps = list(steps)
        self._machine = machine
        self._signal = signal
        self._now = clock
        self._log = log if log is not None else logging.getLogger("robot_side.scripted")
        self._max_recoveries = max_recoveries
        self._confirm_halts = confirm_halts
        self._at = 0
        self._fire_at: float | None = None
        self._recovered = 0
        self._nudged = False
        """这一次进已停止,回车按过了没有。按过就等它走出来,不重复按。"""

    @property
    def done(self) -> bool:
        return self._at >= len(self._steps)

    def pump(self) -> None:
        """看看这一步该不该做了;该做就做,然后挪到下一步。

        停在已停止、而这一步又不是在等已停止,就先替人按一次回车爬出来:已停止不按
        回车出不来,脚本又是没人看着时跑的。
        """
        if self.done:
            return
        step = self._steps[self._at]
        if self._machine.state is not State.HALTED:
            self._nudged = False
        elif step.wait_for is not State.HALTED:
            self._nudge_out_of_halted()
            return
        now = self._now()
        if self._fire_at is None:
            if step.wait_for is not None and self._machine.state is not step.wait_for:
                return  # 还没等到
            self._fire_at = now + step.delay_s  # 等到了,从这一刻开始数
        if now < self._fire_at:
            return
        self._fire(step)
        self._at += 1
        self._fire_at = None

    def _nudge_out_of_halted(self) -> None:
        """替人按一次回车。按过一次就等它走出已停止,不连着按。

        不许替人按(连着真机)时不按,记一笔,等人自己按回车或 q。
        """
        if self._nudged:
            return
        self._nudged = True
        # 这一步的进度作废:爬出已停止之后,要重新等它的 `wait_for`,而不是接着
        # halted 之前那半截倒数
        self._fire_at = None
        if not self._confirm_halts:
            self._log.warning(
                "脚本 停在已停止,连着真机不替你按回车:看过现场,按回车接着走,或者按 q 退出"
            )
            return
        if self._recovered >= self._max_recoveries:
            self._log.warning(
                "脚本 停在已停止,已经替你按过 %d 次回车;现场多半真有问题,收摊退出。"
                "原因看上面那条 HALTED",
                self._recovered,
            )
            self._machine.submit(Event.quit("script"))
            self._at = len(self._steps)
            return
        self._recovered += 1
        self._log.info(
            "脚本 停在已停止,自动按一次回车(第 %d 次,最多 %d 次)接着往下跑",
            self._recovered,
            self._max_recoveries,
        )
        self._machine.submit(Event.confirm("script"))

    def _fire(self, step: Step) -> None:
        self._log.info("脚本 第 %d 步:%s %s", self._at + 1, step.kind, step.action)
        if step.kind == "stop":
            # 叫停直达机械臂,不走事件队列(ADR-0012),和按空格走的是同一条路
            self._signal.raise_stop("script")
        elif step.kind == "trigger":
            self._machine.submit(Event.trigger(step.action, "script"))
        elif step.kind == "confirm":
            self._machine.submit(Event.confirm("script"))
        elif step.kind == "quit":
            self._machine.submit(Event.quit("script"))
        else:
            self._log.warning("脚本 认不出的一步:%r", step)

"""状态逻辑的测试台:假机械臂 + 真配置 + 记账的提示音和日志。

不开 RoboDK、不开摄像头、不弹窗口。默认用假时钟,倒计时 3 秒、冷却 2 秒都是一瞬间
跑完的;要让事件落在某一段运动中间时,改用真时钟并把各项时间调小(见 `Rig`)。
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable

import pytest

from robot_side.beeps import Beeper
from robot_side.config import load_arm, load_demo
from robot_side.startup import prepare
from robot_side.state import Event, Machine, State
from robot_side.stop import StopSignal
from tests.fakes import RecordingArm


class FakeClock:
    """假时钟:测试自己说现在几点,倒计时和冷却都不用真的等。"""

    real_time = False

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class RealClock:
    """真时钟:让事件和运动按真实先后发生,给跑整条链路的测试用。"""

    real_time = True

    def __call__(self) -> float:
        return time.monotonic()

    def advance(self, dt: float) -> None:
        time.sleep(0.002)


class _ListHandler(logging.Handler):
    def __init__(self, into: list[str]) -> None:
        super().__init__()
        self.into = into

    def emit(self, record: logging.LogRecord) -> None:
        self.into.append(record.getMessage())


class Rig:
    """一套跑得起来的状态逻辑。

    `busy_s` 是假机械臂每段运动走多久,默认 0(一问就走完);`faults` 是安排好的报错
    (`tests.fakes.Fault`);两个都是直接传给假机械臂的。`countdown_s` 和 `cooldown_s`
    把演示配置里的时间改小;给了 `countdown_s` 或 `cooldown_s` 就自动改用真时钟——
    这两样都是为「让事件落在运动中间」准备的。`segment_timeout_s` 把「一段运动等多久
    算不对劲」压小,不然测一次超时要等满那 5 秒。
    """

    def __init__(
        self,
        *,
        countdown_s: float | None = None,
        cooldown_s: float | None = None,
        job_join_s: float = 0.5,
        segment_timeout_s: float | None = None,
        arm: RecordingArm | None = None,
        reachable=None,
        check_ahead: bool = True,
        **arm_kwargs,
    ) -> None:
        self.arm_config = load_arm("noarm")
        self.demo = load_demo()
        if countdown_s is not None or cooldown_s is not None:
            run = dataclasses.replace(
                self.demo.run,
                countdown_s=(
                    self.demo.run.countdown_s if countdown_s is None else countdown_s
                ),
                cooldown_s=(
                    self.demo.run.cooldown_s if cooldown_s is None else cooldown_s
                ),
            )
            self.demo = dataclasses.replace(self.demo, run=run)
            self.clock: FakeClock | RealClock = RealClock()
        else:
            self.clock = FakeClock()

        self.arm = (
            arm
            if arm is not None
            else RecordingArm(self.arm_config.standby_joints_deg, **arm_kwargs)
        )
        prepared = prepare(self.arm, self.arm_config, self.demo)
        self.signal = StopSignal(self.arm)
        self.beeps: list[str] = []
        self.cleared = 0
        self.on_clear: Callable[[], None] | None = None
        """进冷却时除了记一笔,还要做什么。接上识别线程那几条测试会把它接到
        `VisionLoop.clear_records` 上——正式程序里接的就是这一条。"""
        self.records: list[str] = []
        log = logging.getLogger(f"test.state.{id(self)}")
        log.propagate = False
        log.setLevel(logging.DEBUG)
        log.handlers = [_ListHandler(self.records)]
        self.log = log
        self.machine = Machine(
            self.arm,
            self.signal,
            plans=prepared.by_action,
            standby_point=prepared.standby_point,
            arm_config=self.arm_config,
            demo=self.demo,
            beeps=Beeper(play=lambda *_: self.beeps.append("beep"), block=True),
            log=log,
            clock=self.clock,
            on_clear_records=self._clear,
            # 测试里不等满 5 秒:执行线程不肯收尾那几条路要跑得快
            job_join_s=job_join_s,
            segment_timeout_s=segment_timeout_s,
            reachable=reachable,
            check_ahead=check_ahead,
        )

    def _clear(self) -> None:
        self.cleared += 1
        if self.on_clear is not None:
            self.on_clear()

    # ---- 推着它走 ------------------------------------------------------

    def until(self, want: State, **kwargs) -> None:
        """一直 tick 并推进时钟,直到进入某个状态。"""
        self.pump(lambda: self.machine.state is want, what=want.name, **kwargs)

    def pump(
        self,
        ready,
        *,
        limit_s: float = 60.0,
        real_limit_s: float = 5.0,
        step: float = 0.05,
        what: str = "",
    ) -> None:
        spent = 0.0
        real_deadline = time.monotonic() + real_limit_s
        while True:
            self.machine.tick()
            if ready():
                return
            out_of_time = time.monotonic() > real_deadline or (
                not self.clock.real_time and spent > limit_s
            )
            if out_of_time:
                waited_for = what or "这一步"
                raise AssertionError(
                    f"等不到 {waited_for},停在 {self.machine.state.name};"
                    f"日志:{self.records[-6:]}"
                )
            self.clock.advance(step)
            spent += step
            time.sleep(0.001)  # 让执行线程有机会跑

    def wait_for_job(self, *, limit_s: float = 5.0) -> None:
        """等执行线程跑完,但**不调 `tick()`**。

        专门用来摆「执行线程已经出事、状态逻辑还没看到」这种局面。看的是状态逻辑
        自己的那个 `Job`,所以伸手进了私有字段——换成睡一小会儿就成了看运气的测试。
        """
        deadline = time.monotonic() + limit_s
        while time.monotonic() < deadline:
            job = self.machine._job
            if job is not None and job.done:
                return
            time.sleep(0.002)
        raise AssertionError("执行线程一直没结束")

    def to_idle(self, **kwargs) -> None:
        """从启动一路走到空闲。"""
        self.machine.start()
        self.until(State.IDLE, **kwargs)

    def trigger(self, action: str = "point_left", source: str = "key:left") -> None:
        self.machine.submit(Event.trigger(action, source))

    def close(self) -> None:
        self.machine.close()


@pytest.fixture
def rig():
    r = Rig()
    yield r
    r.close()

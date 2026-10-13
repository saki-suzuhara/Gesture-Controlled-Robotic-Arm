"""机械臂接口的空实现(NO ARM):不连任何机械臂,只把时间走掉。

不开 RoboDK 时用,`CONTEXT.md`「空实现」:

- 每段运动固定等一会儿(默认 1 秒,写在机械臂配置的 `[noarm] segment_s` 里,可改);
- **途中能被叫停打断**:`stop()` 一调,这一段立刻算走完;
- **不模拟报错**(那是测试里的假机械臂干的事):`error()` 永远是 `None`,可达性检查
  一律通过。

它不做运动学,所以「现在停在哪」是装出来的,但装得让上层走得通:

- `pose()` 报最后一次直线运动的目标;关节运动回待命位姿之后,报回待命点。
- `joints()` 报最后一次关节运动的目标;**走过一段直线之后,报成离待命位姿差一点**
  (`MOVED_OFFSET_DEG`)。不这样的话,空实现永远报「已经在待命位姿上」,回待命的
  抬起和慢速回那两段在空实现下就一次也跑不到——而那正是要在这里试出来的东西。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence

from robodk.robomath import Mat, transl

from robot_side.arm import Arm
from robot_side.config import ArmConfig
from robot_side.motion import LinearSegment, Segment
from robot_side.reach import ReachReport

SEGMENT_S = 1.0
"""没有配置时,每段运动等多久。"""

STANDBY_POINT_MM = (0.0, -400.0, 550.0)
"""没写入过配置时,`pose()` 报的位置。

工作台坐标系里的待命点(`config/arms/*.toml` 的注释里也是这个数)。空实现不做
正解,所以待命点只能这样给一个;写入配置之后仍然是它——反正没有真的运动学。
"""

MOVED_OFFSET_DEG = 5.0
"""走过一段直线之后,`joints()` 报成离待命位姿差这么多度。

只要大过 `robot_side.state.STANDBY_TOL_DEG`(每轴 ±1°)就行:让上层知道「不在
待命位姿上了」,该走完整的回待命。
"""


class NoArm(Arm):
    """空实现。`stop()` 和别的实现一样,任意线程可调、重复调没有副作用。"""

    def __init__(self, segment_s: float = SEGMENT_S) -> None:
        self.segment_s = segment_s
        self._lock = threading.Lock()
        self._connected = False
        self._moving_until = 0.0
        self._standby: list[float] = [0.0] * 6
        self._joints: list[float] = [0.0] * 6
        self._pose: Mat = transl(*STANDBY_POINT_MM)
        self.sent: list[tuple[str, tuple[float, ...]]] = []
        """发出去的每一段,按先后记下来。给演示脚本和日志看。"""

    # ---- 连接 ----------------------------------------------------------

    def connect(self) -> None:
        with self._lock:
            self._connected = True

    def disconnect(self) -> None:
        self.stop()
        with self._lock:
            self._connected = False

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    def __enter__(self) -> NoArm:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.disconnect()

    # ---- 配置和可达性 --------------------------------------------------

    def apply_config(self, config: ArmConfig) -> tuple[float, float, float]:
        """收下配置,只记住待命位姿和「每段等多久」;返回待命点。

        不做正解,所以待命点是个给定的数,不是算出来的(见 `STANDBY_POINT_MM`)。
        """
        with self._lock:
            self._standby = list(config.standby_joints_deg)
            self._joints = list(self._standby)
            self.segment_s = config.noarm_segment_s
        return STANDBY_POINT_MM

    def reach_check(
        self, plans: Mapping[str, Sequence[Segment]], standby_joints: Sequence[float]
    ) -> ReachReport:
        """一律通过:空实现没有运动学,查不了,也就不假装查得了。"""
        waypoints = sum(
            1
            for plan in plans.values()
            for segment in plan
            if isinstance(segment, LinearSegment)
        )
        return ReachReport(checked_waypoints=waypoints, checked_segments=waypoints)

    def solve_ik(self, pose: Mat, seed: Sequence[float]) -> list[float]:
        """没有运动学:报「就在 `seed` 那儿」。回待命的抬起那一段因此原地走一段时间。"""
        return [float(v) for v in seed]

    def set_speed(
        self, *, linear_mms: float | None = None, joints_degs: float | None = None
    ) -> None:
        """收下就算:每段等的时间是固定的,不按速度算。"""

    # ---- 运动 ----------------------------------------------------------

    def move_joints(self, joints: Sequence[float]) -> None:
        with self._lock:
            self._joints = [float(v) for v in joints]
            if self._joints == self._standby:
                # 回到待命位姿了,法兰也就回到待命点(空实现里只有这一种关节运动)
                self._pose = transl(*STANDBY_POINT_MM)
            self._moving_until = time.monotonic() + self.segment_s
            self.sent.append(("joints", tuple(self._joints)))

    def move_linear(self, pose: Mat) -> None:
        with self._lock:
            self._pose = pose
            # 没有运动学,只能报「不在待命位姿上了」,好让上层走完整的回待命
            self._joints = [v + MOVED_OFFSET_DEG for v in self._standby]
            self._moving_until = time.monotonic() + self.segment_s
            self.sent.append(("linear", tuple(float(v) for v in pose.Pos())))

    def is_moving(self) -> bool:
        with self._lock:
            return time.monotonic() < self._moving_until

    def stop(self) -> None:
        """立刻把这一段算走完。任意线程可调、重复调没有副作用。"""
        with self._lock:
            self._moving_until = 0.0

    # ---- 读回来 --------------------------------------------------------

    def joints(self) -> list[float]:
        """现在大概在哪(见模块说明:这是装出来的,不是算出来的)。

        和别的实现不同,**还在动时也答得上**:空实现没有真的机械臂,读它不会让
        什么东西崩掉,报的也只是目标值。
        """
        with self._lock:
            return list(self._joints)

    def pose(self) -> Mat:
        with self._lock:
            return self._pose

    # ---- 报错状态 ------------------------------------------------------

    def error(self) -> None:
        """从来没有报错:空实现不模拟报错(那是测试里的假机械臂干的事)。"""

    def clear_error(self) -> None:
        """没有可清的。"""

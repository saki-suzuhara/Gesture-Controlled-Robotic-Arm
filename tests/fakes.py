"""测试里用的假机械臂(FAKE ARM)。只存在于测试代码里(`CONTEXT.md`「假机械臂」)。

记下都被调了什么,每段运动一问就说走完了,所以状态逻辑的测试跑得很快——不开 RoboDK,
也不等真的时间。

**按安排在指定时刻报错**:`Fault` 说清「到第几次运动、在哪一刻、报哪一个」。
照着真机上那三种来:

- **下发那一下就报**(`on_send=True`):控制器当场拒了这一段,这一次调用直接抛。
  运动错误多是这一种。
- **发出去了,走到一半跳闸**(`on_send=False`):机械臂自己停下,报错挂在它身上,
  没有哪一次调用会抛——只有问 `error()` 才知道。碰撞保护是这一种。
- **一条指令都没发也会报**(`fault_now()`):人按了急停。

三种都会把报错**挂住**,直到 `clear_error()` 清掉;断开那一种清不掉(清了也接不回来)。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from robodk.robomath import Mat, transl

from robot_side.arm import Arm, ArmError
from robot_side.config import ArmConfig
from robot_side.motion import LinearSegment, Segment
from robot_side.reach import ReachFailure, ReachReport

STANDBY_POINT_MM = (0.0, -400.0, 550.0)


@dataclass(frozen=True)
class Fault:
    """安排好的一次报错。

    `at_move` 是第几次运动(0 起,关节段直线段一起数);`on_send` 见模块说明。
    """

    error: ArmError
    at_move: int = 0
    on_send: bool = True


class RecordingArm(Arm):
    """记账用的假机械臂。

    `busy_s` 是每段运动要走多久,默认 0——一问就说走完了,状态逻辑的测试因此跑得飞快;
    要让叫停落在某一段中间时才把它设大。`joints_now` 和 `pose_now` 可以在测试里直接改,
    用来摆出「不在待命位姿附近」「被人挪走过」这种局面。
    """

    def __init__(
        self,
        standby_joints: Sequence[float] = (0.0,) * 6,
        *,
        faults: Sequence[Fault] = (),
        busy_s: float = 0.0,
    ) -> None:
        self.standby_joints = tuple(standby_joints)
        self.busy_s = busy_s
        self.joints_now = list(self.standby_joints)
        self.pose_now: Mat = transl(*STANDBY_POINT_MM)
        self.faults = list(faults)
        self.moves: list[tuple[str, tuple[float, ...]]] = []
        self.speeds: list[tuple[float | None, float | None]] = []
        self.stopped = 0
        self.stopped_at: float | None = None
        self.connects = 0
        self.disconnects = 0
        self.cleared = 0
        """`clear_error()` 被调了几次。回车那一步核对用。"""
        self.reach_checks: list[tuple[str, ...]] = []
        """每次可达性检查查的是哪几个动作,按先后。"""
        self.unreachable: set[str] = set()
        """可达性检查要报「够不着」的动作。启动查完以后再往里放,就只影响出发前那一查。"""
        self.ik_answer: list[float] | None | str = "here"
        """`solve_ik` 答什么:"here"(默认)是「就在出发那一组」;给一组关节角就答它;
        给 None 就是求不出。回待命抬起那一段的几条路径靠它摆。"""
        self._fault: ArmError | None = None
        self._moving_until = 0.0
        self._lock = threading.Lock()

    # ---- 按安排报错 ----------------------------------------------------

    def schedule(self, error: ArmError, *, at_move: int = 0, on_send: bool = True) -> None:
        """安排一次报错。可以在测试跑到一半时再安排。"""
        self.faults.append(Fault(error, at_move=at_move, on_send=on_send))

    def fault_now(self, error: ArmError) -> None:
        """机械臂这一刻自己挂上一个报错,不需要谁调它(照的是人按了急停)。"""
        with self._lock:
            self._fault = error
            self._moving_until = 0.0

    def _trip(self, index: int) -> None:
        """第 `index` 次运动发出去时,该报的报、该挂的挂。"""
        for fault in list(self.faults):
            if fault.at_move != index:
                continue
            with self._lock:
                self._fault = fault.error
                if not fault.on_send:
                    self._moving_until = 0.0  # 半路跳闸:机械臂自己停下
            if fault.on_send:
                raise fault.error

    # ---- 连接 ----------------------------------------------------------

    def connect(self) -> None:
        self.connects += 1

    def disconnect(self) -> None:
        self.disconnects += 1

    @property
    def connected(self) -> bool:
        return True

    # ---- 配置和可达性 --------------------------------------------------

    def apply_config(self, config: ArmConfig) -> tuple[float, float, float]:
        self.joints_now = list(config.standby_joints_deg)
        return STANDBY_POINT_MM

    def reach_check(
        self, plans: Mapping[str, Sequence[Segment]], standby_joints: Sequence[float]
    ) -> ReachReport:
        """一律通过,除了 `unreachable` 里点名的动作。每次查了什么都记在 `reach_checks`。"""
        self.reach_checks.append(tuple(plans))
        waypoints = sum(
            1
            for plan in plans.values()
            for segment in plan
            if isinstance(segment, LinearSegment)
        )
        failures = tuple(
            ReachFailure(name, 1, plan[0].label, "测试里安排的够不着")
            for name, plan in plans.items()
            if name in self.unreachable
        )
        return ReachReport(failures, checked_waypoints=waypoints, checked_segments=waypoints)

    def solve_ik(self, pose: Mat, seed: Sequence[float]) -> list[float] | None:
        if self.ik_answer == "here":
            return [float(v) for v in seed]
        return None if self.ik_answer is None else list(self.ik_answer)

    def set_speed(
        self, *, linear_mms: float | None = None, joints_degs: float | None = None
    ) -> None:
        self.speeds.append((linear_mms, joints_degs))

    # ---- 运动 ----------------------------------------------------------

    def move_joints(self, joints: Sequence[float]) -> None:
        with self._lock:
            index = len(self.moves)
        self._trip(index)  # 抛的话这一段就没发出去,和控制器当场拒了一样
        with self._lock:
            self.joints_now = [float(v) for v in joints]
            self.moves.append(("joints", tuple(self.joints_now)))
            self._start_moving()

    def move_linear(self, pose: Mat) -> None:
        with self._lock:
            index = len(self.moves)
        self._trip(index)
        with self._lock:
            self.pose_now = pose
            self.moves.append(("linear", tuple(float(v) for v in pose.Pos())))
            self._start_moving()

    def _start_moving(self) -> None:
        """这一段开始走。半路跳闸的那一种已经把机械臂停住了,就别再让它动起来。"""
        if self._fault is None:
            self._moving_until = time.monotonic() + self.busy_s

    def is_moving(self) -> bool:
        with self._lock:
            return time.monotonic() < self._moving_until

    def stop(self) -> None:
        """立刻把这一段算走完。任意线程可调、重复调没有副作用。"""
        with self._lock:
            self.stopped += 1
            self.stopped_at = time.monotonic()
            self._moving_until = 0.0

    # ---- 读回来 --------------------------------------------------------

    def joints(self) -> list[float]:
        with self._lock:
            return list(self.joints_now)

    def pose(self) -> Mat:
        with self._lock:
            return self.pose_now

    # ---- 报错状态 ------------------------------------------------------

    def error(self) -> ArmError | None:
        with self._lock:
            return self._fault

    def clear_error(self) -> None:
        with self._lock:
            fault = self._fault
            if fault is not None and not fault.clearable:
                raise fault
            self._fault = None
            self.cleared += 1


class NeverStopsArm(RecordingArm):
    """发出去就一直说自己还在动,照的是控制器一直不回「到位」。

    等这一段走完的那个循环因此只能等到超时——「某段运动超时」那条路就是这样摆出来的。
    """

    def is_moving(self) -> bool:
        return True

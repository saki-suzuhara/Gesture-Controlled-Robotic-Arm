"""逐段下发:一次只发一段,发之前先看「已叫停」标记,每段按自己的速度发。

用一个只会记账的假机械臂,不开 RoboDK。按安排报错的那个在 `tests/fakes.py`,
超时那条式子在 `tests/test_timeout.py`。
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from robodk.robomath import Mat, transl

from robot_side.arm import Arm, ArmTimeoutError
from robot_side.executor import run_plan, settle
from robot_side.motion import JointSegment, LinearSegment
from robot_side.stop import StopSignal

STANDBY_JOINTS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


class RecordingArm(Arm):
    """记下都被调了什么,`is_moving()` 一问就说不动了,而且**当场就走到了目标**。

    「走到了」这一条是必须的:`run_plan` 不再只信「是否还在动」,还要读回位置核对
    到没到目标(`executor.wait_until_done` —— 真机上 `Busy()` 会在机械臂还在走的时候
    就说不动了)。所以假机械臂也得真「到」,不然它永远判不到位。
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.stopped = 0
        self._joints = list(STANDBY_JOINTS)
        self._pose: Mat = transl(0.0, 0.0, 0.0)

    def connect(self) -> None:
        self.calls.append(("connect",))

    def disconnect(self) -> None:
        self.calls.append(("disconnect",))

    @property
    def connected(self) -> bool:
        return True

    def apply_config(self, config):
        return (0.0, 0.0, 0.0)

    def reach_check(self, plans, standby_joints):
        raise NotImplementedError

    def set_speed(self, *, linear_mms=None, joints_degs=None) -> None:
        self.calls.append(("speed", linear_mms, joints_degs))

    def move_joints(self, joints: Sequence[float]) -> None:
        self.calls.append(("joints", tuple(joints)))
        self._joints = [float(v) for v in joints]  # 发出去就算走到了

    def move_linear(self, pose: Mat) -> None:
        self.calls.append(("linear", tuple(round(v, 3) for v in pose.Pos())))
        self._pose = pose

    def is_moving(self) -> bool:
        self.calls.append(("is_moving",))
        return False

    def stop(self) -> None:
        self.stopped += 1

    def joints(self) -> list[float]:
        return list(self._joints)

    def pose(self) -> Mat:
        return self._pose

    def error(self) -> None:
        return None

    def clear_error(self) -> None:
        pass


def linear(x, speed=50.0, pause=0.0):
    return LinearSegment(
        label=f"x={x}", pose=transl(x, 0.0, 0.0), speed=speed, pause_s=pause
    )


def joint(speed=10.0):
    return JointSegment(label="回待命位姿", joints=STANDBY_JOINTS, speed=speed)


def test_每段先设速度再发出():
    arm = RecordingArm()
    sent = run_plan(arm, StopSignal(arm), [linear(10.0), joint()], timeout_s=5.0)
    assert sent == 2
    assert arm.calls[0] == ("speed", 50.0, None)
    assert arm.calls[1] == ("linear", (10.0, 0.0, 0.0))
    assert ("speed", None, 10.0) in arm.calls
    assert ("joints", STANDBY_JOINTS) in arm.calls


def test_等上一段结束才发下一段():
    arm = RecordingArm()
    run_plan(arm, StopSignal(arm), [linear(10.0), linear(20.0)], timeout_s=5.0)
    kinds = [c[0] for c in arm.calls]
    # 发第二段之前,一定问过「还在动吗」
    assert kinds.index("is_moving") < kinds.index("linear", kinds.index("linear") + 1)


def test_已叫停就不再发下一段():
    # 叫停正好落在两段之间:机械臂本来就不动,靠「已叫停」标记把下一段拦住
    class StopAfterFirst(RecordingArm):
        def is_moving(self) -> bool:
            signal.raise_stop("test")
            return False

    arm = StopAfterFirst()
    signal = StopSignal(arm)
    plan = [linear(10.0), linear(20.0), linear(30.0)]
    sent = run_plan(arm, signal, plan, timeout_s=5.0)
    assert sent == 1
    assert [c for c in arm.calls if c[0] == "linear"] == [("linear", (10.0, 0.0, 0.0))]
    assert arm.stopped == 1


def test_路点上的停顿走完这一段才停():
    arm = RecordingArm()
    plan = [linear(10.0, pause=0.01), linear(20.0)]
    sent = run_plan(arm, StopSignal(arm), plan, timeout_s=5.0)
    assert sent == 2


def test_停顿途中被叫停就不等它停满():
    import time

    arm = RecordingArm()
    signal = StopSignal(arm)
    signal.raise_stop("test")  # 上一段刚走完就叫停了
    t0 = time.monotonic()
    sent = run_plan(arm, signal, [linear(10.0, pause=30.0)], timeout_s=5.0)
    # 第一段本来就不会发(发之前先看标记),更不会在停顿里干等
    assert sent == 0
    assert time.monotonic() - t0 < 1.0


def test_停顿走完才发下一段():
    import time

    arm = RecordingArm()
    t0 = time.monotonic()
    sent = run_plan(arm, StopSignal(arm), [linear(10.0, pause=0.05), linear(20.0)], timeout_s=5.0)
    assert sent == 2
    assert time.monotonic() - t0 >= 0.05


# ---- 「说不动了」不等于「走到了」(2026-09-22 真机实测)---------------------


class LatePoseArm(RecordingArm):
    """照真机上那种毛病摆:`is_moving()` 一问就说不动了,可位置要过几次才跟上。

    2026-09-22 在 JAKA 上实测到的正是这一种——一段 100 mm 的直线,`Busy()` 4.82 秒
    落下时机械臂还差 43.6 mm;照着它发下一段,下一段就从一个错的起点算起
    (`notes/jaka-second-connection-results.md`)。

    `sent_from` 记下每一段**发出去的那一刻**机械臂在哪儿。下一段是不是从上一段的
    终点发出去的,看它就知道。
    """

    def __init__(self, late_reads: int = 2) -> None:
        super().__init__()
        self.late_reads = late_reads
        self.sent_from: list[tuple] = []
        self._target: Mat | None = None
        self._left = 0

    def move_linear(self, pose: Mat) -> None:
        self.sent_from.append(tuple(round(v, 3) for v in self._pose.Pos()))
        self.calls.append(("linear", tuple(round(v, 3) for v in pose.Pos())))
        self._target = pose
        self._left = self.late_reads  # 先装作还没到

    def pose(self) -> Mat:
        if self._left > 0:
            self._left -= 1
            return self._pose  # 还停在上一段的终点
        if self._target is not None:
            self._pose, self._target = self._target, None
        return self._pose


def test_说不动了但还没到位时不发下一段():
    # 距离要拉得比到位容差(`ARRIVE_TOL_MM`)远得多,不然「半路」和「到了」分不开
    arm = LatePoseArm(late_reads=2)
    sent = run_plan(arm, StopSignal(arm), [linear(100.0), linear(200.0)], timeout_s=5.0)
    assert sent == 2
    # 第二段是从第一段的**终点**发出去的,不是从半路
    assert arm.sent_from == [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]


def test_一直到不了位就按超时报出来():
    """上一段被控制器悄悄丢了也走这条路——真机上出过一次(指令没执行、机械臂没动)。"""
    arm = LatePoseArm(late_reads=10**6)
    with pytest.raises(ArmTimeoutError):
        run_plan(arm, StopSignal(arm), [linear(100.0)], timeout_s=0.6)


def test_没到位但被叫停就立刻返回():
    """叫停之后这一段本来就走不完了,不该干等到超时。"""
    import time

    arm = LatePoseArm(late_reads=10**6)
    signal = StopSignal(arm)
    signal.raise_stop("test")
    t0 = time.monotonic()
    sent = run_plan(arm, signal, [linear(100.0)], timeout_s=30.0)
    assert sent == 0  # 发之前先看标记,这一段根本没发
    assert time.monotonic() - t0 < 1.0



class TrustedBusyArm(LatePoseArm):
    """「是否还在动」靠得住的那一种(只跑 RoboDK 仿真时):说不动了就是走完了。"""

    @property
    def busy_is_reliable(self) -> bool:
        return True


def test_只跑仿真时说不动了就算走完不再核对到位():
    # 仿真里 JAKA 走直线会差一点(左上角画方块姿态差过 2.14°);核对到位只会误报超时
    arm = TrustedBusyArm(late_reads=10**6)
    sent = run_plan(arm, StopSignal(arm), [linear(100.0), linear(200.0)], timeout_s=0.6)
    assert sent == 2



# ---- 走关节运动过去的那几段(点头) --------------------------------------


def _nod_segment():
    from robodk.robomath import rotx, transl

    from robot_side.motion import LinearSegment

    return LinearSegment(
        label="nod 路点 1", pose=transl(0.0, -400.0, 550.0) * rotx(3.0), speed=50.0,
        via_joints=True,
    )


def test_走关节运动的那一段发的是求出来的关节角():
    from tests.fakes import RecordingArm as FakeArm

    arm = FakeArm((0.0,) * 6)
    target = [0.0, 1.0, -2.0, 12.0, 0.5, 0.0]
    arm.ik_answer = target
    sent = run_plan(arm, StopSignal(arm), [_nod_segment()], timeout_s=1.0)
    assert sent == 1
    assert arm.moves == [("joints", tuple(target))]  # 没发直线


def test_走关节运动的那一段要甩臂就一条都不发():
    from robot_side.arm import MotionError
    from tests.fakes import RecordingArm as FakeArm

    arm = FakeArm((0.0,) * 6)
    arm.ik_answer = [90.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 逆解换了一支:J1 跳 90°
    with pytest.raises(MotionError, match="换了一支"):
        run_plan(arm, StopSignal(arm), [_nod_segment()], timeout_s=1.0)
    assert arm.moves == []


def test_走关节运动的那一段求不出关节角就一条都不发():
    from robot_side.arm import MotionError
    from tests.fakes import RecordingArm as FakeArm

    arm = FakeArm((0.0,) * 6)
    arm.ik_answer = None
    with pytest.raises(MotionError, match="求不出"):
        run_plan(arm, StopSignal(arm), [_nod_segment()], timeout_s=1.0)
    assert arm.moves == []


# ---- 按尖端速度走的关节段:发之前换算成关节速度 ----------------------------


class ConvertingArm(RecordingArm):
    """会按尖端速度换算关节速度的假机械臂:问它就答一个定好的数,并记下问了什么。"""

    def __init__(self, answer: float | None) -> None:
        super().__init__()
        self.answer = answer
        self.asked: list[tuple] = []

    def joint_speed_for(self, from_joints, to_joints, tip_mms):
        self.asked.append((tuple(from_joints), tuple(to_joints), tip_mms))
        return self.answer


TARGET = (1.0, 2.0, 3.0, 4.0, 5.0, 26.0)


def by_tip(tip_mms=50.0, speed=10.0):
    return JointSegment(label="回待命", joints=TARGET, speed=speed, tip_speed_mms=tip_mms)


def test_按尖端速度的关节段发的是换算出来的关节速度():
    arm = ConvertingArm(answer=3.0)
    run_plan(arm, StopSignal(arm), [by_tip()], timeout_s=5.0)
    assert ("speed", None, 3.0) in arm.calls
    assert arm.asked == [(STANDBY_JOINTS, TARGET, 50.0)]  # 从现在停着的地方算起


def test_算不出来就照旧用它自己写的关节速度():
    arm = ConvertingArm(answer=None)
    run_plan(arm, StopSignal(arm), [by_tip(speed=10.0)], timeout_s=5.0)
    assert ("speed", None, 10.0) in arm.calls


def test_没写尖端速度的关节段不换算():
    arm = ConvertingArm(answer=3.0)
    run_plan(arm, StopSignal(arm), [joint(speed=10.0)], timeout_s=5.0)
    assert arm.asked == []
    assert ("speed", None, 10.0) in arm.calls


def test_时限按换算以后的速度算():
    """换算出来慢了,时限就跟着长:转 20° 按 2°/s 是 10 秒,不是按 10°/s 的 2 秒。"""
    from robot_side.executor import TIMEOUT_FACTOR, TIMEOUT_SLACK_S, segment_timeout_s, settle

    arm = ConvertingArm(answer=2.0)
    settled = settle(arm, by_tip(speed=10.0))
    assert settled.speed == 2.0
    assert segment_timeout_s(arm, settled) == pytest.approx(
        20.0 / 2.0 * TIMEOUT_FACTOR + TIMEOUT_SLACK_S
    )


def test_点头那几段按原来那段直线的速度换算():
    """点头和左右上下一样快:尖端速度就是那段直线写的 50 mm/s。"""
    from tests.fakes import RecordingArm as FakeArm

    class Converting(FakeArm):
        def joint_speed_for(self, from_joints, to_joints, tip_mms):
            self.asked.append(tip_mms)
            return 7.0

    arm = Converting((0.0,) * 6)
    arm.asked = []
    arm.ik_answer = [0.0, 1.0, -2.0, 12.0, 0.5, 0.0]
    run_plan(arm, StopSignal(arm), [_nod_segment()], timeout_s=1.0)
    assert arm.asked == [50.0]
    assert arm.speeds == [(None, 7.0)]


def test_按规矩放慢估算但不少于下限():
    """真机开了程序设速度:估算 × 放慢倍数 × 2 + 5,短的那几段按下限。"""
    from robot_side.executor import SegmentTimeout, segment_timeout_s

    arm = ConvertingArm(answer=2.0)  # 转 20° 按 2°/s:估算 10 秒
    rule = SegmentTimeout(slowdown=10.0, floor_s=90.0)
    long = settle(arm, by_tip())
    assert max(rule.floor_s, segment_timeout_s(arm, long, slowdown=rule.slowdown)) == (
        pytest.approx(10.0 * 10.0 * 2 + 5.0)
    )
    arm.answer = 200.0  # 0.1 秒的一段:按下限
    short = settle(arm, by_tip())
    assert max(rule.floor_s, segment_timeout_s(arm, short, slowdown=rule.slowdown)) == 90.0


def test_给了规矩的时候按规矩等():
    """`run_plan` 收到 `SegmentTimeout` 时走的是放慢以后的估算,不是固定一个数。"""
    from robot_side.executor import SegmentTimeout

    arm = ConvertingArm(answer=3.0)
    sent = run_plan(arm, StopSignal(arm), [by_tip()], SegmentTimeout(slowdown=10.0, floor_s=1.0))
    assert sent == 1

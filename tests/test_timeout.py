"""一段运动等多久算不对劲:估算用时 × 2 + 5 秒。

估算只按「走多远 ÷ 这一段的速度」算,不算加减速,也不算真机上示教器里设的速度
——所以才乘 2 再加 5 秒。整条路(超时 → 已停止)在 `tests/test_faults.py`;
这里只盯那条式子本身。
"""

from __future__ import annotations

import math

import pytest
from robodk.robomath import rotx, transl

from robot_side.arm import MOVE_POLL_S, ArmTimeoutError
from robot_side.executor import (
    TIMEOUT_FACTOR,
    TIMEOUT_SLACK_S,
    run_plan,
    segment_timeout_s,
)
from robot_side.motion import JointSegment, LinearSegment, estimate_s
from robot_side.stop import StopSignal
from tests.fakes import NeverStopsArm, RecordingArm

STANDBY = (0.0,) * 6
ORIGIN = transl(0.0, 0.0, 0.0)


def at(pose):
    """「怎么读位姿」:`estimate_s` 收的是读的办法,不是读回来的值。"""
    return lambda: pose


def far(x: float = 100.0, speed: float = 50.0) -> LinearSegment:
    return LinearSegment(label=f"x={x}", pose=transl(x, 0.0, 0.0), speed=speed)


def turned(deg: float = 30.0, speed: float = 10.0) -> JointSegment:
    return JointSegment(
        label="回待命位姿", joints=(0.0, deg, 0.0, 0.0, 0.0, 0.0), speed=speed
    )


# ---- 估算用时 ----------------------------------------------------------


def test_直线段按走过的距离除以速度估():
    # 走到 100 mm 外,50 mm/s → 2 秒
    assert estimate_s(far(), pose=at(ORIGIN)) == pytest.approx(2.0)


def test_关节段按转得最多的那一轴估():
    # 最多的一轴转 30°,10°/s → 3 秒
    assert estimate_s(turned(), joints=lambda: STANDBY) == pytest.approx(3.0)


def test_只转姿势不挪位置的那一段也估得出():
    """点头是原地转:位置一点没变,只按距离估就成了 0。"""
    nod = LinearSegment(label="nod", pose=ORIGIN * rotx(math.radians(20.0)), speed=50.0)
    assert estimate_s(nod, pose=at(ORIGIN)) > 0.0


def test_读不到出发点就估不出():
    """`run_segments` 那种由调用方自己包起来的一段,没有出发点可读。"""
    assert estimate_s(far()) == 0.0
    assert estimate_s(turned()) == 0.0


def test_只读用得上的那一样():
    """直线段不该去读关节角,关节段不该去读位姿——每读一次都是一次 API 来回。"""

    def boom():
        raise AssertionError("这一样不该被读")

    assert estimate_s(far(), pose=at(ORIGIN), joints=boom) == pytest.approx(2.0)
    assert estimate_s(turned(), pose=boom, joints=lambda: STANDBY) == pytest.approx(3.0)


# ---- 乘二加五 ----------------------------------------------------------


def test_时限是估算用时乘二加五():
    arm = RecordingArm(STANDBY)
    arm.pose_now = ORIGIN
    assert segment_timeout_s(arm, far()) == pytest.approx(
        2.0 * TIMEOUT_FACTOR + TIMEOUT_SLACK_S
    )


def test_不动的那一段也有五秒可等():
    """终点就是现在停的地方:估算用时 0,仍然给满那 5 秒。"""
    arm = RecordingArm(STANDBY)
    arm.pose_now = ORIGIN
    assert segment_timeout_s(arm, far(x=0.0)) == pytest.approx(TIMEOUT_SLACK_S)


def test_速度写成零也不会除以零():
    arm = RecordingArm(STANDBY)
    arm.pose_now = ORIGIN
    assert segment_timeout_s(arm, far(speed=0.0)) == pytest.approx(TIMEOUT_SLACK_S)


def test_按现在停在哪算不是按上一段的终点():
    """机械臂被人挪走过,时限跟着变长;这就是「估算」要现读一次的理由。"""
    arm = RecordingArm(STANDBY)
    arm.pose_now = ORIGIN
    near = segment_timeout_s(arm, far())
    arm.pose_now = transl(-900.0, 0.0, 0.0)
    assert segment_timeout_s(arm, far()) > near


# ---- 逐段下发时用的就是这个数 ------------------------------------------


class RemembersLimits(RecordingArm):
    """记下每段等的时候被给了多少时限。

    `wait_until_not_moving` 是机械臂接口上的方法,不是谁的内部实现,所以假机械臂
    接得住它。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.limits: list[float] = []

    def wait_until_not_moving(self, timeout_s: float, poll_s: float = MOVE_POLL_S) -> None:
        self.limits.append(timeout_s)
        super().wait_until_not_moving(timeout_s, poll_s)


def test_不给时限就每段自己算():
    arm = RemembersLimits(STANDBY)
    arm.pose_now = ORIGIN
    run_plan(arm, StopSignal(arm), [far(), turned()])
    assert arm.limits == [
        pytest.approx(2.0 * TIMEOUT_FACTOR + TIMEOUT_SLACK_S),  # 直线 2 秒
        pytest.approx(3.0 * TIMEOUT_FACTOR + TIMEOUT_SLACK_S),  # 关节 3 秒
    ]


def test_给了时限就每段都用它():
    """只有测试才这么用:不然一段 0.2 秒的和一段 20 秒的会共用一个数。"""
    arm = RemembersLimits(STANDBY)
    arm.pose_now = ORIGIN
    run_plan(arm, StopSignal(arm), [far(), turned()], timeout_s=0.5)
    assert arm.limits == [0.5, 0.5]


def test_一段运动迟迟不结束就抛超时():
    arm = NeverStopsArm(STANDBY)
    arm.pose_now = ORIGIN
    with pytest.raises(ArmTimeoutError):
        run_plan(arm, StopSignal(arm), [far()], timeout_s=0.05, poll_s=0.01)

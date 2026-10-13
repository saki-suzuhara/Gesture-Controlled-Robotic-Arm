"""机械臂一侧报错的那几条路:一律进已停止,回车后清除报错、回待命。

从「按键 / 手势事件」进,看的是对机械臂接口的调用和状态变化——不看状态逻辑内部怎么写的。
不开 RoboDK:报错由测试里的假机械臂按安排报出来(`tests/fakes.py`)。

    三种报错(碰撞保护、运动错误、急停被按下)→ 已停止 → 回车 → 清除报错 → 回待命
    某段运动超时                                → 已停止
    执行前不在待命位姿                          → 已停止,一步不动
    RoboDK 断开                                 → 已停止,清不掉,要恢复就重启程序
"""

from __future__ import annotations

import pytest

from robot_side.arm import (
    CollisionError,
    EmergencyStopError,
    MotionError,
    NotConnectedError,
)
from robot_side.state import Event, State
from tests.conftest import Rig
from tests.fakes import Fault, NeverStopsArm

THREE_KINDS = [
    (CollisionError("碰撞保护:与工作台碰上了"), "COLLISION"),
    (MotionError("运动错误:这一段走不通"), "MOTION ERROR"),
    (EmergencyStopError("急停被按下"), "E-STOP"),
]


def halted_by(error, *, on_send=True, **kwargs) -> Rig:
    """摆到「执行一个预设动作时机械臂报错、已经进了已停止」。"""
    r = Rig(faults=[Fault(error, at_move=0, on_send=on_send)], **kwargs)
    r.to_idle()
    r.trigger("point_left")
    r.pump(lambda: r.machine.state is State.HALTED, what="HALTED")
    return r


# ---- 三种报错各一条路 --------------------------------------------------


@pytest.mark.parametrize("error,reason", THREE_KINDS, ids=lambda v: getattr(v, "reason", v))
def test_三种报错都进已停止(error, reason):
    r = halted_by(error)
    try:
        assert r.machine.state is State.HALTED
        assert r.machine.view.reason == reason
        assert str(error) in r.machine.view.detail  # 机械臂报上来的原文
        assert r.arm.stopped >= 1  # 进已停止时叫了停,把「已叫停」标记立住
    finally:
        r.close()


@pytest.mark.parametrize("error,reason", THREE_KINDS, ids=lambda v: getattr(v, "reason", v))
def test_回车后先清除报错再回待命(error, reason):
    r = halted_by(error)
    try:
        moved = len(r.arm.moves)
        r.machine.submit(Event.confirm("key:enter"))
        r.machine.tick()
        assert r.arm.cleared == 1  # 清在前
        assert r.machine.state is State.RETURNING  # 回待命在后
        assert len(r.arm.moves) == moved  # 倒计时没走完之前一步不动
        r.until(State.IDLE)
        assert r.arm.error() is None
    finally:
        r.close()


def test_急停那一种回车后提示重新上电使能():
    """程序只提示,不替人上电使能(用户故事 27)。"""
    r = halted_by(EmergencyStopError("急停被按下"))
    try:
        r.machine.submit(Event.confirm("key:enter"))
        r.machine.tick()
        assert "PENDANT" in r.machine.view.detail
        assert any("重新上电使能" in line for line in r.records)
    finally:
        r.close()


def test_另外两种不提示上电使能():
    r = halted_by(CollisionError("碰撞保护"))
    try:
        r.machine.submit(Event.confirm("key:enter"))
        r.machine.tick()
        assert r.machine.view.detail == ""
    finally:
        r.close()


def test_没有哪次调用会抛的那种报错也进已停止():
    """碰撞保护半路跳闸:这一段发出去了,机械臂自己停下,只有问它才知道。"""
    r = halted_by(CollisionError("碰撞保护:半路跳闸"), on_send=False)
    try:
        assert r.machine.view.reason == "COLLISION"
        assert len(r.arm.moves) >= 1  # 这一段确实发出去了
        # 带着故障没有把剩下的段发完(一个动作 = 每个路点一段 + 一段回待命位姿)
        assert len(r.arm.moves) < len(r.demo.action("point_left").waypoints) + 1
    finally:
        r.close()


def test_一条指令都没发的时候报错也进已停止():
    """人在空闲时按了急停。这一刻没有哪一次调用能把它带出来。"""
    r = Rig()
    try:
        r.to_idle()
        r.arm.fault_now(EmergencyStopError("急停被按下"))
        r.machine.tick()
        assert r.machine.state is State.HALTED
        assert r.machine.view.reason == "E-STOP"
        assert r.arm.moves == []
    finally:
        r.close()


def test_已停止之后不再一遍遍重报():
    """报错一直挂着,但连响三声只该有一次。"""
    r = halted_by(CollisionError("碰撞保护"))
    try:
        r.beeps.clear()
        for _ in range(5):
            r.machine.tick()
        assert r.beeps == []
        assert r.machine.state is State.HALTED
    finally:
        r.close()


# ---- 超时 --------------------------------------------------------------


def test_某段运动超时就当作机械臂一侧报错():
    r = Rig(arm=NeverStopsArm((0.0,) * 6), segment_timeout_s=0.05, job_join_s=1.0)
    try:
        r.to_idle()
        r.trigger("point_left")
        r.pump(lambda: r.machine.state is State.HALTED, what="HALTED")
        assert r.machine.view.reason == "TIMEOUT"
        assert "ArmTimeoutError" in r.machine.view.detail
    finally:
        r.close()


# 执行前不在待命位姿那条在 `tests/test_state.py`(第 03 票就有,这次补了画面上那行字)。


# ---- RoboDK 断开 -------------------------------------------------------


def test_断开时进已停止并提示重启程序():
    r = halted_by(NotConnectedError("和 RoboDK 断了"))
    try:
        assert r.machine.view.reason == "ARM DISCONNECTED"
        assert "RESTART THE PROGRAM" in r.machine.view.detail
    finally:
        r.close()


def test_断开之后按回车也恢复不了():
    """清不掉的报错:留在已停止,不回待命、不再命令机械臂动。"""
    r = halted_by(NotConnectedError("和 RoboDK 断了"))
    try:
        moved = len(r.arm.moves)
        for _ in range(3):
            r.machine.submit(Event.confirm("key:enter"))
            r.machine.tick()
            r.clock.advance(5.0)
            r.machine.tick()
        assert r.machine.state is State.HALTED
        assert r.machine.view.reason == "ARM DISCONNECTED"
        assert len(r.arm.moves) == moved
        assert r.arm.cleared == 0  # 清不掉就是清不掉
    finally:
        r.close()

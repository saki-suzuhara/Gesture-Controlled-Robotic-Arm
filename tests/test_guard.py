"""真机护栏:会让真机动的东西,下发之前先过这一关。

纯计算,不开 RoboDK、不连任何机械臂——护栏本身正不正确,不该靠真机来验
(上一轮就是在这里差点出事,见 `notes/jaka-first-connection-results.md`
「一个操作上的失误」)。
"""

from __future__ import annotations

import pytest
from robodk.robomath import rotz, transl

from robot_side.guard import (
    GuardRefused,
    MoveLimits,
    check_orientation_change,
    PowerState,
    check_joint_readback,
    check_joint_travel,
    check_linear_travel,
    check_pose_readback,
    check_powered,
)

STANDBY = (118.09, 81.83, 131.94, 236.22, 90.0, 28.09)


def bump(joints, axis: int, by: float):
    out = list(joints)
    out[axis] += by
    return tuple(out)


# ---- 没上电使能就不下发 --------------------------------------------------


def test_上电使能都有才放行():
    check_powered(PowerState(powered=True, enabled=True))  # 不抛就是放行


@pytest.mark.parametrize(
    "state",
    [
        PowerState(powered=False, enabled=False),
        PowerState(powered=True, enabled=False),
        PowerState(powered=False, enabled=True),
    ],
)
def test_没上电或没使能就拒绝下发(state):
    with pytest.raises(GuardRefused) as caught:
        check_powered(state)
    assert "示教器" in str(caught.value)


def test_读不出上电使能时按没有处理():
    """读不出来 = 不放行。宁可停下问人,也不能猜「大概使能了吧」。"""
    with pytest.raises(GuardRefused) as caught:
        check_powered(PowerState(powered=None, enabled=None, detail="jkrc 登录失败"))
    assert "jkrc 登录失败" in str(caught.value)


# ---- 逐轴核对:读回来的要和打算发的一模一样 ------------------------------


def test_读回来和打算发的一致就放行():
    check_joint_readback(STANDBY, tuple(STANDBY))


def test_读回来差了一轴就拒绝():
    with pytest.raises(GuardRefused) as caught:
        check_joint_readback(STANDBY, bump(STANDBY, 2, 0.5))
    assert "J3" in str(caught.value)  # 说清是哪一轴


def test_读回来的轴数不对也拒绝():
    with pytest.raises(GuardRefused):
        check_joint_readback(STANDBY, STANDBY[:5])


def test_位姿读回来一致就放行():
    pose = transl(0.0, -400.0, 550.0) * rotz(0.3)
    check_pose_readback(pose, transl(0.0, -400.0, 550.0) * rotz(0.3))


def test_位姿读回来挪了位就拒绝():
    pose = transl(0.0, -400.0, 550.0)
    with pytest.raises(GuardRefused) as caught:
        check_pose_readback(pose, transl(0.0, -400.0, 600.0))
    assert "50" in str(caught.value)  # 差了 50 mm,说出来


def test_位姿读回来转了向也拒绝():
    pose = transl(0.0, -400.0, 550.0)
    with pytest.raises(GuardRefused):
        check_pose_readback(pose, transl(0.0, -400.0, 550.0) * rotz(0.2))


# ---- 只动该动的轴、行程设上限 --------------------------------------------


LIMITS_J1 = MoveLimits(axes=(0,), max_delta_deg=2.0)


def test_只动准动的那一轴且没超行程就放行():
    check_joint_travel(STANDBY, bump(STANDBY, 0, 1.5), LIMITS_J1)


def test_动了不准动的轴就拒绝():
    with pytest.raises(GuardRefused) as caught:
        check_joint_travel(STANDBY, bump(STANDBY, 3, 0.5), LIMITS_J1)
    assert "J4" in str(caught.value)


def test_准动的那一轴超了行程也拒绝():
    with pytest.raises(GuardRefused) as caught:
        check_joint_travel(STANDBY, bump(STANDBY, 0, 2.5), LIMITS_J1)
    assert "2.5" in str(caught.value) and "2" in str(caught.value)


def test_六轴都准动时仍然逐轴卡行程():
    limits = MoveLimits(axes=(0, 1, 2, 3, 4, 5), max_delta_deg=10.0)
    check_joint_travel(STANDBY, bump(STANDBY, 5, 9.0), limits)
    with pytest.raises(GuardRefused) as caught:
        check_joint_travel(STANDBY, bump(STANDBY, 5, 11.0), limits)
    assert "J6" in str(caught.value)


def test_上一轮那次差点出事的甩动会被拦下():
    """`notes/jaka-first-connection-results.md`:指令里的目标没改成功,
    机械臂会从 J1=201° 甩到 J1=1°。护栏要拦住的就是这一下。"""
    here = (201.0, 98.0, 95.0, 80.0, -93.0, -27.0)
    stale = (1.0, 90.0, 90.0, 180.0, 90.0, 0.0)
    with pytest.raises(GuardRefused) as caught:
        check_joint_travel(here, stale, MoveLimits(axes=(0,), max_delta_deg=6.0))
    assert "J1" in str(caught.value) or "J2" in str(caught.value)


# ---- 直线段:走多远、转多少,都设上限 ------------------------------------


def test_直线走得不远就放行():
    here = transl(0.0, -400.0, 550.0)
    check_linear_travel(here, transl(0.0, -450.0, 550.0), max_mm=60.0, max_deg=1.0)


def test_直线走太远就拒绝():
    here = transl(0.0, -400.0, 550.0)
    with pytest.raises(GuardRefused) as caught:
        check_linear_travel(here, transl(0.0, -600.0, 550.0), max_mm=60.0, max_deg=1.0)
    assert "200" in str(caught.value)


def test_直线段转太多也拒绝():
    here = transl(0.0, -400.0, 550.0)
    with pytest.raises(GuardRefused):
        check_linear_travel(
            here, transl(0.0, -410.0, 550.0) * rotz(0.5), max_mm=60.0, max_deg=1.0
        )


# ---- 姿态逐个角卡上限(2026-09-22 用户对这台 JAKA 的要求)------------------

CAPS = (5.0, 5.0, 1.0)   # RX ≤ 5°、RY ≤ 5°、RZ 基本不许转


def test_姿态不变就放行():
    check_orientation_change((-178.7, 0.3, 0.0), (-178.7, 0.3, 0.0), max_deg=CAPS)


def test_小幅变化在上限内放行():
    check_orientation_change((-178.7, 0.3, 0.0), (-175.0, -4.0, 0.5), max_deg=CAPS)


def test_rz_转了就拒绝():
    """用户明说法兰末端 RZ 最好不要转。"""
    with pytest.raises(GuardRefused) as caught:
        check_orientation_change((-178.7, 0.3, 0.0), (-178.7, 0.3, 30.0), max_deg=CAPS)
    assert "RZ" in str(caught.value)


def test_rx_超过五度就拒绝():
    with pytest.raises(GuardRefused) as caught:
        check_orientation_change((-178.7, 0.3, 0.0), (-170.0, 0.3, 0.0), max_deg=CAPS)
    assert "RX" in str(caught.value)


def test_ry_超过五度就拒绝():
    with pytest.raises(GuardRefused) as caught:
        check_orientation_change((-178.7, 0.3, 0.0), (-178.7, 8.0, 0.0), max_deg=CAPS)
    assert "RY" in str(caught.value)


def test_角度差折到正负一百八再比():
    """179° 和 −179° 只差 2°,不是 358°。"""
    check_orientation_change((179.0, 0.0, 0.0), (-179.0, 0.0, 0.0), max_deg=CAPS)


def test_报错里说清是哪一次目标():
    with pytest.raises(GuardRefused) as caught:
        check_orientation_change(
            (-178.7, 0.3, 0.0), (-178.7, 0.3, 30.0), max_deg=CAPS,
            what="驱动会翻出来的那一种",
        )
    assert "驱动会翻出来的那一种" in str(caught.value)

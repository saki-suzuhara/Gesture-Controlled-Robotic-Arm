"""选待命位姿:只转 J6 对上默认姿态、核对不动点、改配置那一行。

纯计算加一次文本改写,不开 RoboDK。正解用一个手搭的:法兰位置固定,姿态是
「底座偏航 → 指针朝下(可以歪一点)→ J6 绕法兰自己的 Z 轴转」。拿真机模型核对 9-22
那一次手算的,在 `tests/test_standby_robodk.py`。
"""

from __future__ import annotations

import math
import textwrap
import tomllib

import pytest
from robodk.robomath import pose_angle_between, rotx, roty, rotz, transl

from robot_side.config import Worktable
from robot_side.motion import orientation_pose, worktable_pose
from robot_side.standby import (
    FIXED_POINT_TOL_DEG,
    STAMP_MARK,
    TILT_TOL_DEG,
    StandbyRefused,
    plan_standby,
    rewrite_standby,
)

FLANGE_MM = (46.5, -945.6, 895.1)
WORKTABLE = worktable_pose(Worktable(angle_deg=90.0))
DEFAULT = orientation_pose((180.0, 0.0, 90.0))
"""演示配置的默认姿态。工作台坐标系转了 90°,换到基坐标系里正好是 RY=RZ=0 的朝下。"""

PICKED = (187.723, 90.782, 98.739, 80.480, -89.329, -52.043)
"""9-22 用户在示教器上摆的那一组(J6 还没补)。"""


def arm(*, yaw0=30.233, tilt_x=0.0, tilt_y=0.0, j6_sign=1.0, j6_axis="z"):
    """手搭的正解:返回 (工作台坐标系里的, 基坐标系里的)。"""
    turn = {"z": rotz, "x": rotx}[j6_axis]

    def in_base(q):
        return (
            transl(*FLANGE_MM)
            * rotz(math.radians(yaw0))
            * rotx(math.radians(180.0 + tilt_x))
            * roty(math.radians(tilt_y))
            * turn(math.radians(j6_sign * q[5]))
        )

    def in_worktable(q):
        return WORKTABLE.invH() * in_base(q)

    return in_worktable, in_base


def plan(picked=PICKED, *, fixed_point=True, **geometry):
    in_worktable, in_base = arm(**geometry)
    return plan_standby(
        picked,
        flange_in_worktable=in_worktable,
        flange_in_base=in_base,
        default_orientation=DEFAULT,
        fixed_point=fixed_point,
    )


def orientation_error(result, **geometry) -> float:
    in_worktable, _ = arm(**geometry)
    got = in_worktable(result.joints)
    got.setPos([0.0, 0.0, 0.0])
    return math.degrees(pose_angle_between(got, DEFAULT))


# ---- 只转 J6,对上默认姿态 ------------------------------------------------


def test_只转_j6_就对上默认姿态():
    result = plan()
    assert result.ok, result.problems
    assert result.joints[:5] == PICKED[:5]
    assert orientation_error(result) < 1e-3
    assert result.tilt_deg < 1e-3


def test_转_j6_法兰位置一点不变():
    result = plan()
    assert result.moved_mm < 1e-6
    assert result.standby_point == pytest.approx(
        (WORKTABLE.invH() * transl(*FLANGE_MM)).Pos(), abs=1e-6
    )


def test_补完以后停在镜像的不动点上():
    result = plan()
    _, ry, rz = result.base_rpy_after
    assert abs(ry) < 1e-3
    assert abs(rz) < 1e-3


def test_补之前的偏航照实报出来():
    """指针朝正下方时,补之前 RZ 差多少,J6 就正好转多少。"""
    result = plan()
    assert abs(result.base_rpy_before[2]) > 1.0
    assert abs(result.base_rpy_before[2]) == pytest.approx(abs(result.j6_turn_deg), abs=1e-3)


def test_j6_正方向反过来也算得对():
    result = plan(j6_sign=-1.0)
    assert result.ok, result.problems
    assert orientation_error(result, j6_sign=-1.0) < 1e-3


def test_转多少取最少的那一边():
    """正解对 J6 以 360° 为周期:摆在 +300° 时不该转回去七百多度。"""
    picked = (*PICKED[:5], 300.0)
    result = plan(picked)
    assert -180.0 < result.j6_turn_deg <= 180.0
    assert result.joints[5] == pytest.approx(300.0 + result.j6_turn_deg)
    assert orientation_error(result) < 1e-3


def test_转最少的那一边出了限位就换另一边():
    """摆在 J6 = 350° 附近:转最少的那一边会冲过 360°,只能反过来多转一圈里的剩下那段。"""
    free = plan((*PICKED[:5], 350.0))
    assert 350.0 + free.j6_turn_deg > 360.0  # 不给限位时,转得最少的那一边冲过 360°
    in_worktable, in_base = arm()
    result = plan_standby(
        (*PICKED[:5], 350.0),
        flange_in_worktable=in_worktable,
        flange_in_base=in_base,
        default_orientation=DEFAULT,
        fixed_point=True,
        j6_limits=(-360.0, 360.0),
    )
    assert result.ok, result.problems
    assert -360.0 <= result.joints[5] <= 360.0
    assert result.j6_turn_deg == pytest.approx(free.j6_turn_deg - 360.0)
    assert orientation_error(result) < 1e-3


def test_两边都出限位就拒绝():
    in_worktable, in_base = arm()
    result = plan_standby(
        PICKED,
        flange_in_worktable=in_worktable,
        flange_in_base=in_base,
        default_orientation=DEFAULT,
        fixed_point=True,
        j6_limits=(-52.1, -52.0),
    )
    assert not result.ok
    assert "限位" in result.problems[0]


def test_已经对上了就不转():
    first = plan()
    again = plan(first.joints)
    assert again.j6_turn_deg == pytest.approx(0.0, abs=1e-6)
    assert again.joints == pytest.approx(first.joints)


# ---- J6 补不了的:指针没朝正下方 ------------------------------------------


def test_指针歪了超过上限就拒绝():
    result = plan(tilt_x=TILT_TOL_DEG + 1.0)
    assert not result.ok
    assert "扶正" in result.problems[0]
    assert result.tilt_deg == pytest.approx(TILT_TOL_DEG + 1.0, abs=1e-3)


def test_歪一点点照样放行():
    result = plan(tilt_x=0.5)
    assert result.ok, result.problems


def test_不在不动点上就拒绝():
    """侧着歪 1°:没超姿态上限,但 RY 离不动点 1°,镜像一翻就是 2°。"""
    result = plan(tilt_y=FIXED_POINT_TOL_DEG + 0.5)
    assert result.tilt_deg < TILT_TOL_DEG
    assert not result.ok
    assert any("不动点" in p for p in result.problems)


def test_不走_jaka_驱动的不查不动点():
    result = plan(tilt_y=FIXED_POINT_TOL_DEG + 0.5, fixed_point=False)
    assert result.ok, result.problems


def test_j6_转的不是法兰那根轴就不往下算():
    with pytest.raises(StandbyRefused, match="正解对不上"):
        plan(j6_axis="x")


# ---- 改配置:只动 [standby] joints_deg 那一行 ------------------------------

CONFIG = textwrap.dedent(
    """\
    name = "JAKA Zu 20"

    # 待命位姿上面那一大段说明
    [standby]
    joints_deg = [187.723, 90.782, 98.739, 80.480, -89.329, -82.276]

    [speed]
    max_joint_degs = 30.0         # 行尾注释
    """
)

NEW = (190.0, 91.0, 97.5, 81.25, -89.5, -80.125)


def test_只改那一行():
    out = rewrite_standby(CONFIG, NEW, stamp="2026-09-24", picked_j6=-50.0)
    before, after = CONFIG.splitlines(), out.splitlines()
    assert len(before) == len(after)
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert changed == [4]
    assert tomllib.loads(out)["standby"]["joints_deg"] == pytest.approx(list(NEW))


def test_行尾记下哪天和原来的_j6():
    out = rewrite_standby(CONFIG, NEW, stamp="2026-09-24", picked_j6=-50.0)
    line = out.splitlines()[4]
    assert f"# {STAMP_MARK} 2026-09-24" in line
    assert "-50.000" in line


def test_重写时整句换掉不往上叠():
    once = rewrite_standby(CONFIG, NEW, stamp="2026-09-24", picked_j6=-50.0)
    twice = rewrite_standby(once, PICKED, stamp="2026-09-25", picked_j6=-52.043)
    line = twice.splitlines()[4]
    assert line.count(STAMP_MARK) == 1
    assert "2026-09-25" in line


def test_保留原来的换行符():
    crlf = CONFIG.replace("\n", "\r\n")
    out = rewrite_standby(crlf, NEW, stamp="2026-09-24", picked_j6=-50.0)
    assert out.count("\r\n") == crlf.count("\r\n")


def test_找不到就不改():
    with pytest.raises(StandbyRefused, match="找不到"):
        rewrite_standby('name = "x"\n', NEW, stamp="d", picked_j6=0.0)


def test_别的节里的同名字段不算():
    text = "[other]\njoints_deg = [1, 2, 3, 4, 5, 6]\n"
    with pytest.raises(StandbyRefused, match="找不到"):
        rewrite_standby(text, NEW, stamp="d", picked_j6=0.0)


def test_写了两个就不改():
    text = CONFIG.replace(
        "[speed]", "joints_deg = [0, 0, 0, 0, 0, 0]\n\n[speed]"
    )
    with pytest.raises(StandbyRefused, match="2 个"):
        rewrite_standby(text, NEW, stamp="d", picked_j6=0.0)


def test_跨行的数组不改():
    text = "[standby]\njoints_deg = [\n  1, 2, 3,\n  4, 5, 6,\n]\n"
    with pytest.raises(StandbyRefused, match="一行"):
        rewrite_standby(text, NEW, stamp="d", picked_j6=0.0)

"""标定工作台坐标系:算角度、改配置、排验证移动。

纯计算加一次文本改写,不开 RoboDK、不连真机。真机那一段(读当前法兰位姿、慢速走
一小段)由标定脚本自己做,要人在现场。
"""

from __future__ import annotations

import math
import textwrap
import tomllib

import pytest
from robodk.robomath import pose_angle_between, rotz, transl

from robot_side.calibrate import (
    MIN_RADIUS_MM,
    TOWARD_OPERATOR_BEARING_DEG,
    CalibrationRefused,
    check_verify_move,
    normalize_deg,
    rewrite_angle,
    verify_target,
    worktable_angle_deg,
)
from robot_side.config import Worktable
from robot_side.guard import GuardRefused, PowerState
from robot_side.motion import worktable_pose


def flange_at(x: float, y: float, z: float = 550.0):
    return transl(x, y, z)


# ---- 角度:法兰挪到操作者那条线上,算工作台坐标系绕 Z 转了多少 ------------


def test_法兰正对底座负_y_方向时角度是零():
    """出厂的仿真布置:待命点在工作台坐标系里是 (0, −400, 550),也就是 −Y。"""
    assert worktable_angle_deg(flange_at(0.0, -400.0)) == pytest.approx(0.0)


def test_法兰在底座正_x_方向时角度是九十度():
    assert worktable_angle_deg(flange_at(400.0, 0.0)) == pytest.approx(90.0)


def test_法兰在底座正_y_方向时角度是一百八():
    assert abs(worktable_angle_deg(flange_at(0.0, 400.0))) == pytest.approx(180.0)


def test_算出来的角度真能把法兰换算到操作者那条线上():
    """不看公式,看结果:按算出来的角度建坐标系,法兰在里面应当落在 −Y 上。"""
    for x, y in [(400.0, 0.0), (-300.0, 300.0), (150.0, -520.0), (-600.0, -100.0)]:
        angle = worktable_angle_deg(flange_at(x, y))
        in_worktable = worktable_pose(Worktable(angle_deg=angle)).invH() * flange_at(x, y)
        px, py, _ = in_worktable.Pos()
        bearing = math.degrees(math.atan2(py, px))
        assert normalize_deg(bearing - TOWARD_OPERATOR_BEARING_DEG) == pytest.approx(
            0.0, abs=1e-6
        )


def test_法兰离底座转轴太近就拒绝():
    """离得太近时方位角基本是噪声,标出来的角度不能用。"""
    with pytest.raises(CalibrationRefused) as caught:
        worktable_angle_deg(flange_at(10.0, 10.0))
    assert str(int(MIN_RADIUS_MM)) in str(caught.value)


def test_原点有平移时从原点量方位角():
    angle = worktable_angle_deg(
        flange_at(300.0, -400.0), origin_mm=(300.0, 0.0, 0.0)
    )
    assert angle == pytest.approx(0.0)


def test_角度一律折到正负一百八之间():
    assert normalize_deg(370.0) == pytest.approx(10.0)
    assert normalize_deg(-370.0) == pytest.approx(-10.0)
    assert normalize_deg(180.0) == pytest.approx(180.0)
    assert normalize_deg(-180.0) == pytest.approx(180.0)


# ---- 改配置:只改那一个数,别的一行不动 ----------------------------------


ARM_TOML = textwrap.dedent(
    """\
    name = "JAKA Zu 20"
    robot_name = "JAKA Zu20"

    [connection]
    driver = "robodk"
    ip = ""

    # 工作台坐标系:相对底座绕 Z 轴转过的角度,加上可选的原点平移(默认 0)。
    [worktable]
    angle_deg = 0.0
    origin_mm = [0.0, 0.0, 0.0]

    [standby]
    joints_deg = [118.09, 81.83, 131.94, 236.22, 90.00, 28.09]
    """
)


def test_改角度只动那一行():
    out = rewrite_angle(ARM_TOML, 37.5, stamp="2026-09-22")
    assert "angle_deg = 37.5" in out
    # 别的都还在:注释、别的字段、别的节
    assert "# 工作台坐标系" in out
    assert "origin_mm = [0.0, 0.0, 0.0]" in out
    assert 'robot_name = "JAKA Zu20"' in out
    assert len(out.splitlines()) == len(ARM_TOML.splitlines())


def test_改完读得回来():
    out = rewrite_angle(ARM_TOML, -12.25, stamp="2026-09-22")
    assert tomllib.loads(out)["worktable"]["angle_deg"] == pytest.approx(-12.25)


def test_改完带上是什么时候标的():
    out = rewrite_angle(ARM_TOML, 37.5, stamp="2026-09-22")
    line = [x for x in out.splitlines() if x.startswith("angle_deg")][0]
    assert "2026-09-22" in line


def test_再标一次不会把上一次的戳叠上去():
    once = rewrite_angle(ARM_TOML, 10.0, stamp="2026-09-22")
    twice = rewrite_angle(once, 20.0, stamp="2026-09-23")
    line = [x for x in twice.splitlines() if x.startswith("angle_deg")][0]
    assert line.count("标定于") == 1
    assert "2026-09-22" not in twice


def test_没有工作台那一节就拒绝改():
    with pytest.raises(CalibrationRefused):
        rewrite_angle("name = \"x\"\n[standby]\nangle_deg = 1.0\n", 5.0, stamp="d")


def test_工作台那一节里没有角度也拒绝改():
    text = '[worktable]\norigin_mm = [0.0, 0.0, 0.0]\n'
    with pytest.raises(CalibrationRefused):
        rewrite_angle(text, 5.0, stamp="d")


def test_有两个角度写着就拒绝改():
    """宁可不动也不猜改哪一个。"""
    text = "[worktable]\nangle_deg = 0.0\nangle_deg = 1.0\n"
    with pytest.raises(CalibrationRefused):
        rewrite_angle(text, 5.0, stamp="d")


# ---- 验证移动:朝操作者走一小段 ------------------------------------------


def test_验证移动是朝操作者走的():
    here = transl(0.0, -400.0, 550.0)
    there = verify_target(here, 50.0)
    x, y, z = there.Pos()
    assert (x, z) == pytest.approx((0.0, 550.0))
    assert y == pytest.approx(-450.0)  # 往 −Y,也就是操作者那一边


def test_验证移动的四道护栏一起放行():
    here_pose = transl(0.0, -400.0, 550.0)
    here_joints = (118.09, 81.83, 131.94, 236.22, 90.0, 28.09)
    wanted = verify_target(here_pose)
    check_verify_move(
        power=PowerState(powered=True, enabled=True),
        here_joints=here_joints,
        here_pose=here_pose,
        wanted=wanted,
        staged=wanted,
        solved=tuple(v + 1.0 for v in here_joints),
    )


def good_move(**changed):
    here_pose = transl(0.0, -400.0, 550.0)
    here_joints = (118.09, 81.83, 131.94, 236.22, 90.0, 28.09)
    wanted = verify_target(here_pose)
    args = dict(
        power=PowerState(powered=True, enabled=True),
        here_joints=here_joints,
        here_pose=here_pose,
        wanted=wanted,
        staged=wanted,
        solved=tuple(here_joints),
    )
    args.update(changed)
    return args


def test_临发之前没使能就不下发():
    with pytest.raises(GuardRefused):
        check_verify_move(**good_move(power=PowerState(powered=True, enabled=False)))


def test_指令里读回来的目标不对就不下发():
    """坐标系写错、单位写错,都会在「读回来的和打算发的对不上」这一步显出来。"""
    with pytest.raises(GuardRefused):
        check_verify_move(**good_move(staged=transl(0.0, -900.0, 550.0)))


def test_读回来的目标要走太远就不下发():
    far = transl(0.0, -900.0, 550.0)
    with pytest.raises(GuardRefused) as caught:
        check_verify_move(**good_move(wanted=far, staged=far))
    assert "500" in str(caught.value)


def test_求不出逆解就不下发():
    with pytest.raises(GuardRefused) as caught:
        check_verify_move(**good_move(solved=None))
    assert "逆解" in str(caught.value)


def test_某一轴要甩一大圈就不下发():
    stale = (1.0, 90.0, 90.0, 180.0, 90.0, 0.0)  # 上一轮差点出事的那一组
    with pytest.raises(GuardRefused) as caught:
        check_verify_move(**good_move(solved=stale))
    assert "J1" in str(caught.value)


def test_验证移动不改姿态():
    here = transl(0.0, -400.0, 550.0) * rotz(0.4)
    there = verify_target(here, 50.0)
    assert math.degrees(pose_angle_between(here, there)) == pytest.approx(0.0, abs=1e-9)

"""把配置里的路点算成一段段运动(纯计算,不碰 RoboDK)。

只测外部行为:给定待命点和一个预设动作,拆出来的每一段是关节运动还是直线运动、
去哪里、多快、停多久。
"""

from __future__ import annotations

import math

import pytest
from robodk.robomath import Pose_2_TxyzRxyz, pose_angle_between, rotx, transl

from robot_side.config import Action, Waypoint, Worktable
from robot_side.motion import (
    DEFAULT_ORIENTATION_DEG,
    ORIGIN,
    in_range,
    joint_path_mm,
    orientation_pose,
    plan_action,
    lifted_pose,
    plan_returning,
    shifted,
    tip_position,
    waypoint_pose,
    worktable_pose,
)

STANDBY = (0.0, -400.0, 550.0)
STANDBY_JOINTS = (118.09, 81.83, 131.94, 236.22, 90.0, 28.09)
DOWN = orientation_pose(DEFAULT_ORIENTATION_DEG)


def pos(pose):
    return tuple(round(v, 6) for v in pose.Pos())


def rpy_deg(pose):
    return tuple(round(math.degrees(v), 6) for v in Pose_2_TxyzRxyz(pose)[3:])


# ---- 工作台坐标系 ------------------------------------------------------


def test_工作台坐标系不转不平移时是单位阵():
    pose = worktable_pose(Worktable())
    assert pos(pose) == (0.0, 0.0, 0.0)
    assert rpy_deg(pose) == (0.0, 0.0, 0.0)


def test_工作台坐标系绕Z转再平移():
    pose = worktable_pose(Worktable(angle_deg=90.0, origin_mm=(10.0, 20.0, 30.0)))
    assert pos(pose) == (10.0, 20.0, 30.0)
    # 绕 Z 转 90°:坐标系的 X 轴指向基坐标系的 +Y
    assert pytest.approx([0.0, 1.0, 0.0], abs=1e-9) == [pose[i, 0] for i in range(3)]


# ---- 路点 --------------------------------------------------------------


def test_路点按相对待命点的偏移算():
    pose = waypoint_pose(STANDBY, Waypoint(dx_mm=-150.0), DOWN)
    assert pos(pose) == (-150.0, -400.0, 550.0)


def test_路点不写姿态时用默认姿态():
    pose = waypoint_pose(STANDBY, Waypoint(dz_mm=100.0), DOWN)
    assert pos(pose) == (0.0, -400.0, 650.0)
    # 默认姿态 (180°, 0°, 0°):法兰 Z 轴竖直朝下
    assert pytest.approx([0.0, 0.0, -1.0], abs=1e-9) == [pose[i, 2] for i in range(3)]


def test_点头时法兰原地转位置不变():
    pose = waypoint_pose(STANDBY, Waypoint(nod_deg=15.0), DOWN)
    assert pos(pose) == (0.0, -400.0, 550.0)
    # 姿态相对默认姿态正好转过 15°(从朝下再往前转,绕过了 180°,所以不看 Rx 的数值)
    assert math.degrees(pose_angle_between(pose, DOWN)) == pytest.approx(15.0, abs=1e-6)


def test_点头时指针尖端扫出弧():
    # 法兰原地转 15°,尖端在半径 280.47 mm 的弧上:前后摆约 73 mm、抬起约 10 mm
    pointer = (0.0, 0.0, 280.47)
    still = tip_position(waypoint_pose(STANDBY, Waypoint(), DOWN), pointer)
    nodded = tip_position(waypoint_pose(STANDBY, Waypoint(nod_deg=15.0), DOWN), pointer)
    assert still == pytest.approx((0.0, -400.0, 550.0 - 280.47), abs=1e-6)
    assert nodded[1] - still[1] == pytest.approx(280.47 * math.sin(math.radians(15)), abs=1e-6)
    assert nodded[2] - still[2] == pytest.approx(
        280.47 * (1 - math.cos(math.radians(15))), abs=1e-6
    )


# ---- 把一个预设动作拆成一段段 ------------------------------------------


def action(*waypoints, name="point_left", speed=50.0, end="stay"):
    return Action(
        name=name, display=name.upper(), speed_mms=speed, waypoints=tuple(waypoints), end=end
    )


SQUARE = (
    Waypoint(dx_mm=-50.0, dz_mm=-50.0),
    Waypoint(dx_mm=50.0, dz_mm=-50.0),
    Waypoint(dx_mm=50.0, dz_mm=50.0),
    Waypoint(dx_mm=-50.0, dz_mm=50.0),
    Waypoint(dx_mm=-50.0, dz_mm=-50.0),
)


def test_停在终点的动作每个路点一段直线不回待命():
    # 2026-09-23 起做完不回待命位姿:没有最后那段关节运动
    segments = plan_action(
        action(Waypoint(dx_mm=-150.0)), start_point=STANDBY, orientation=DOWN
    )
    assert [type(s).__name__ for s in segments] == ["LinearSegment"]
    assert pos(segments[0].pose) == (-150.0, -400.0, 550.0)
    assert segments[0].speed == 50.0


def test_画方块拆成五段直线再直线回到出发点():
    segments = plan_action(
        action(*SQUARE, name="draw_square", end="back"), start_point=STANDBY, orientation=DOWN
    )
    assert [type(s).__name__ for s in segments] == ["LinearSegment"] * 6
    assert pos(segments[0].pose) == (-50.0, -400.0, 500.0)
    assert pos(segments[2].pose) == (50.0, -400.0, 600.0)
    assert pos(segments[-1].pose) == STANDBY
    assert segments[-1].speed == 50.0


def test_从别的停靠点出发路点跟着挪():
    here = (150.0, -400.0, 650.0)  # 待命点右边一步、上面一步
    segments = plan_action(
        action(*SQUARE, name="draw_square", end="back"), start_point=here, orientation=DOWN
    )
    assert pos(segments[0].pose) == (100.0, -400.0, 600.0)
    assert pos(segments[-1].pose) == here


def test_点头做完摆回默认姿态():
    nod = action(Waypoint(nod_deg=10.0), Waypoint(nod_deg=-10.0), name="nod", end="back")
    segments = plan_action(nod, start_point=STANDBY, orientation=DOWN)
    assert len(segments) == 3
    assert pos(segments[-1].pose) == STANDBY
    assert pose_angle_between(segments[-1].pose, transl(*STANDBY) * DOWN) == pytest.approx(0.0)


def test_最后一个路点本来就在出发点就不多走一段():
    there_and_back = action(Waypoint(dx_mm=-50.0), Waypoint(), name="wiggle", end="back")
    segments = plan_action(there_and_back, start_point=STANDBY, orientation=DOWN)
    assert len(segments) == 2


def test_每一段都说得出是哪个动作的哪个路点():
    segments = plan_action(
        action(Waypoint(dx_mm=-150.0), Waypoint(dx_mm=-50.0), end="back"),
        start_point=STANDBY,
        orientation=DOWN,
    )
    assert segments[0].label == "point_left 路点 1"
    assert segments[1].label == "point_left 路点 2"
    assert "出发点" in segments[2].label


def test_路点上的停顿跟着那一段():
    segments = plan_action(
        action(Waypoint(dx_mm=-150.0, pause_s=0.4), end="back"),
        start_point=STANDBY,
        orientation=DOWN,
    )
    assert segments[0].pause_s == 0.4
    assert segments[1].pause_s == 0.0


# ---- 停靠点和活动范围 --------------------------------------------------


def test_停在终点的动作把停靠点挪一步回原处的不挪():
    assert shifted(ORIGIN, action(Waypoint(dx_mm=150.0))) == (150.0, 0.0, 0.0)
    assert shifted((150.0, 0.0, 0.0), action(Waypoint(dz_mm=100.0))) == (150.0, 0.0, 100.0)
    square = action(*SQUARE, name="draw_square", end="back")
    assert shifted((150.0, 0.0, 100.0), square) == (150.0, 0.0, 100.0)


def test_连挪几步不攒浮点误差():
    step = action(Waypoint(dx_mm=0.1))
    here = ORIGIN
    for _ in range(3):
        here = shifted(here, step)
    assert here == (0.3, 0.0, 0.0)


def test_活动范围边上算在里面出去一点就不算():
    limits = (300.0, 0.0, 200.0)
    assert in_range((300.0, 0.0, -200.0), limits)
    assert in_range(ORIGIN, limits)
    assert not in_range((450.0, 0.0, 0.0), limits)
    assert not in_range((0.0, 0.0, 300.0), limits)
    assert not in_range((0.0, 10.0, 0.0), limits)  # Y 给的是 0:一点都不许挪


# ---- 回待命 ------------------------------------------------------------


LIFTED = (118.0, 80.0, 128.0, 239.0, 90.0, 28.0)


def test_回待命先抬起再回去两段都是关节运动():
    # 2026-09-23:抬起那一段原来是直线;经 RoboDK 的 JAKA 驱动下发直线姿态会被镜像
    segments = plan_returning(
        LIFTED, lift_mm=50.0, standby_joints=STANDBY_JOINTS, return_joint_degs=10.0
    )
    assert [type(s).__name__ for s in segments] == ["JointSegment", "JointSegment"]
    assert segments[0].joints == LIFTED
    assert segments[1].joints == STANDBY_JOINTS
    assert segments[0].speed == segments[1].speed == 10.0


def test_回待命两段都带着尖端速度():
    segments = plan_returning(
        LIFTED,
        lift_mm=50.0,
        standby_joints=STANDBY_JOINTS,
        return_joint_degs=10.0,
        tip_speed_mms=50.0,
    )
    assert [s.tip_speed_mms for s in segments] == [50.0, 50.0]


def test_回待命抬起那一段说得出自己是抬起():
    segments = plan_returning(
        LIFTED, lift_mm=50.0, standby_joints=STANDBY_JOINTS, return_joint_degs=10.0
    )
    assert "抬起" in segments[0].label


def test_抬起是沿工作台竖直往上姿态不变():
    here = transl(123.0, -300.0, 480.0) * rotx(math.pi)
    up = lifted_pose(here, 50.0)
    assert pos(up) == (123.0, -300.0, 530.0)
    assert rpy_deg(up) == rpy_deg(here)


def test_路点自己写了姿态就不用默认姿态():
    pose = waypoint_pose(
        STANDBY, Waypoint(dx_mm=-150.0, orientation_deg=(180.0, 0.0, 90.0)), DOWN
    )
    assert pos(pose) == (-150.0, -400.0, 550.0)
    # 姿态换了一组:法兰 Z 轴仍朝下,X 轴不再是默认姿态那个方向
    assert pytest.approx([0.0, 0.0, -1.0], abs=1e-9) == [pose[i, 2] for i in range(3)]
    assert pytest.approx([0.0, -1.0, 0.0], abs=1e-9) == [pose[i, 0] for i in range(3)]


def test_路点自己写的姿态上还能再点头():
    pose = waypoint_pose(
        STANDBY, Waypoint(orientation_deg=(180.0, 0.0, 90.0), nod_deg=15.0), DOWN
    )
    plain = waypoint_pose(STANDBY, Waypoint(orientation_deg=(180.0, 0.0, 90.0)), DOWN)
    assert pos(pose) == pos(plain)
    assert math.degrees(pose_angle_between(pose, plain)) == pytest.approx(15.0, abs=1e-6)



# ---- 点头那几段走关节运动(2026-09-23 真机上点头被姿态镜像转反) ----------


def test_只转姿态的路点下发时走关节运动平移的照旧走直线():
    nod = action(Waypoint(nod_deg=10.0), Waypoint(nod_deg=-10.0), name="nod", end="back")
    segments = plan_action(nod, start_point=STANDBY, orientation=DOWN)
    # 回出发点那段目标是默认姿态,但要从歪着的转回来:也走关节运动,好按尖端速度走
    assert [s.via_joints for s in segments] == [True, True, True]
    square = plan_action(
        action(*SQUARE, name="draw_square", end="back"), start_point=STANDBY, orientation=DOWN
    )
    assert not any(s.via_joints for s in square)


# ---- 关节运动时尖端走多远 --------------------------------------------------


def turning_base(q):
    """J1 转、法兰伸出去 500 mm:法兰和尖端一起画弧。"""
    from robodk.robomath import rotz

    return rotz(math.radians(q[0])) * transl(500.0, 0.0, 0.0)


def nodding_wrist(q):
    """法兰钉在原地,只绕自己的 X 轴点头:法兰不动,尖端画弧。"""
    return transl(0.0, -900.0, 850.0) * rotx(math.radians(q[3]))


def test_转底座时量的是法兰画的弧():
    path = joint_path_mm(turning_base, [0.0] * 6, [90.0, 0, 0, 0, 0, 0], (0.0, 0.0, 0.0))
    assert path == pytest.approx(500.0 * math.pi / 2, rel=1e-3)


def test_点头时法兰不动量的是尖端画的弧():
    path = joint_path_mm(
        nodding_wrist, [0.0] * 6, [0, 0, 0, 20.0, 0, 0], (0.0, 0.0, 280.0)
    )
    assert path == pytest.approx(280.0 * math.radians(20.0), rel=1e-3)


def test_不动就是零():
    assert joint_path_mm(turning_base, [10.0] * 6, [10.0] * 6, (0.0, 0.0, 280.0)) == 0.0

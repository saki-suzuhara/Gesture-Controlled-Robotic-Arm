"""可达性检查:启动时逐路点求逆解、每段直线单独检查,不通过就说清哪个动作的哪个路点。

测试里的运动学是假的(不开 RoboDK):按安排让某个路点求不出逆解、某段直线走不通,
看报出来的话对不对,以及检查过程中有没有动过机械臂。
"""

from __future__ import annotations

import math

import pytest
from robodk.robomath import rotx, transl

from robot_side.config import Action, Waypoint
from robot_side.motion import (
    DEFAULT_ORIENTATION_DEG,
    JointSegment,
    orientation_pose,
    plan_action,
)
from robot_side.reach import check_reach

STANDBY = (0.0, -400.0, 550.0)
STANDBY_JOINTS = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
DOWN = orientation_pose(DEFAULT_ORIENTATION_DEG)


class FakeKinematics:
    """假运动学:默认样样都过;按 x 坐标安排某个路点不收敛、某段直线走不通。

    只实现可达性检查要用的两件事。它没有「动起来」的方法,所以检查里一旦想动机械臂,
    测试就会以 AttributeError 失败。
    """

    def __init__(self, ik_fail_at=(), movel_rc=None):
        self.ik_fail_at = set(ik_fail_at)
        self.movel_rc = movel_rc or {}
        self.ik_calls = []
        self.movel_calls = []

    def solve_ik(self, pose, seed):
        x = round(pose.Pos()[0], 3)
        self.ik_calls.append((x, tuple(seed)))
        if x in self.ik_fail_at:
            return None
        # 假的「关节角」:把 x 塞进第一个轴,后面断言初值有没有接上一个点
        return [x, 0.0, 0.0, 0.0, 0.0, 0.0]

    def linear_test(self, from_joints, pose):
        x = round(pose.Pos()[0], 3)
        self.movel_calls.append((tuple(from_joints), x))
        return self.movel_rc.get(x, 0)


def plan(*waypoints, name="point_left"):
    """一个动作拆成的段。按「停在终点」拆:只有路点那几段,不多一段回出发点。"""
    action = Action(
        name=name,
        display=name.upper(),
        speed_mms=50.0,
        waypoints=tuple(waypoints),
        end="stay",
    )
    return plan_action(action, start_point=STANDBY, orientation=DOWN)


def test_全过时没有不过的路点():
    kin = FakeKinematics()
    report = check_reach(kin, {"point_left": plan(Waypoint(dx_mm=-150.0))}, STANDBY_JOINTS)
    assert report.ok
    assert report.failures == ()


def test_求不出逆解时指出是哪个动作的哪个路点():
    kin = FakeKinematics(ik_fail_at=[-150.0])
    report = check_reach(kin, {"point_left": plan(Waypoint(dx_mm=-150.0))}, STANDBY_JOINTS)
    assert not report.ok
    (failure,) = report.failures
    assert failure.action == "point_left"
    assert failure.waypoint == 1
    assert "逆解" in failure.why
    assert "point_left" in report.describe()
    assert "路点 1" in report.describe()


def test_直线走不通和够不着分得开():
    kin = FakeKinematics(movel_rc={-150.0: -1})
    report = check_reach(kin, {"a": plan(Waypoint(dx_mm=-150.0), name="a")}, STANDBY_JOINTS)
    assert "直线走不通" in report.failures[0].why

    kin = FakeKinematics(movel_rc={-150.0: -2})
    report = check_reach(kin, {"a": plan(Waypoint(dx_mm=-150.0), name="a")}, STANDBY_JOINTS)
    assert "够不着" in report.failures[0].why


def test_每段直线都单独查():
    kin = FakeKinematics()
    check_reach(
        kin,
        {"sq": plan(Waypoint(dx_mm=-50.0), Waypoint(dx_mm=50.0), Waypoint(dx_mm=10.0), name="sq")},
        STANDBY_JOINTS,
    )
    assert [x for _, x in kin.movel_calls] == [-50.0, 50.0, 10.0]


def test_下一段的起点接着上一个点求不读机械臂位置():
    kin = FakeKinematics()
    check_reach(
        kin,
        {"sq": plan(Waypoint(dx_mm=-50.0), Waypoint(dx_mm=50.0), name="sq")},
        STANDBY_JOINTS,
    )
    # 第一个路点以待命位姿作初值,第二个以第一个路点求出来的关节角作初值
    assert kin.ik_calls[0][1] == STANDBY_JOINTS
    assert kin.ik_calls[1][1][0] == -50.0
    # 每段直线也从上一个点的关节角出发
    assert kin.movel_calls[0][0] == STANDBY_JOINTS
    assert kin.movel_calls[1][0][0] == -50.0


def test_一个动作不过就不再往下查这个动作():
    kin = FakeKinematics(ik_fail_at=[-50.0])
    report = check_reach(
        kin,
        {"sq": plan(Waypoint(dx_mm=-50.0), Waypoint(dx_mm=50.0), name="sq")},
        STANDBY_JOINTS,
    )
    assert len(report.failures) == 1
    assert [x for x, _ in kin.ik_calls] == [-50.0]


def test_几个动作都不过时每个都报出来():
    kin = FakeKinematics(ik_fail_at=[-150.0], movel_rc={150.0: -1})
    report = check_reach(
        kin,
        {
            "point_left": plan(Waypoint(dx_mm=-150.0)),
            "point_right": plan(Waypoint(dx_mm=150.0), name="point_right"),
            "raise": plan(Waypoint(dz_mm=100.0), name="raise"),
        },
        STANDBY_JOINTS,
    )
    assert [f.action for f in report.failures] == ["point_left", "point_right"]
    assert "point_left" in report.describe() and "point_right" in report.describe()


def test_关节段不进直线检查():
    # 关节运动回待命位姿(回待命那一段):不求逆解、不查直线
    kin = FakeKinematics()
    back = JointSegment(label="回待命", joints=STANDBY_JOINTS, speed=10.0)
    check_reach(kin, {"point_left": [*plan(Waypoint(dx_mm=-150.0)), back]}, STANDBY_JOINTS)
    assert len(kin.movel_calls) == 1
    assert len(kin.ik_calls) == 1


def test_点头这种只转姿态的路点也查():
    kin = FakeKinematics()
    report = check_reach(
        kin, {"nod": plan(Waypoint(nod_deg=15.0), Waypoint(nod_deg=-15.0), name="nod")},
        STANDBY_JOINTS,
    )
    assert report.ok
    assert len(kin.movel_calls) == 2


def test_报出来的话里有为什么不过():
    kin = FakeKinematics(movel_rc={-150.0: -1})
    report = check_reach(kin, {"point_left": plan(Waypoint(dx_mm=-150.0))}, STANDBY_JOINTS)
    text = report.describe()
    assert "point_left" in text
    assert "直线走不通" in text


def test_位姿原样交给运动学():
    kin = FakeKinematics()
    check_reach(kin, {"point_left": plan(Waypoint(dx_mm=-150.0))}, STANDBY_JOINTS)
    want = transl(-150.0, -400.0, 550.0) * rotx(math.pi)
    got_x, _ = kin.ik_calls[0]
    assert got_x == pytest.approx(want.Pos()[0])


def test_关了碰撞检查还报碰撞时只提示不算够不着():
    # 第 9 条:关了还报碰撞就记日志、提示,不当作够不着
    kin = FakeKinematics(movel_rc={-150.0: 3})
    report = check_reach(kin, {"point_left": plan(Waypoint(dx_mm=-150.0))}, STANDBY_JOINTS)
    assert report.ok
    assert report.collisions and report.collisions[0].action == "point_left"
    assert "碰撞" in report.describe()


def test_逆解接不上上一个点时退回用待命位姿作初值():
    # 上一个路点把构型带到了别处,接着它求不出来;不能因此就说够不着
    class SeedPicky(FakeKinematics):
        def solve_ik(self, pose, seed):
            x = round(pose.Pos()[0], 3)
            self.ik_calls.append((x, tuple(seed)))
            if x == 50.0 and tuple(seed) != STANDBY_JOINTS:
                return None
            return [x, 0.0, 0.0, 0.0, 0.0, 0.0]

    kin = SeedPicky()
    report = check_reach(
        kin,
        {"sq": plan(Waypoint(dx_mm=-50.0), Waypoint(dx_mm=50.0), name="sq")},
        STANDBY_JOINTS,
    )
    assert report.ok
    assert kin.ik_calls[-1][1] == STANDBY_JOINTS


def test_两个初值都求不出才算够不着():
    kin = FakeKinematics(ik_fail_at=[50.0])
    report = check_reach(
        kin,
        {"sq": plan(Waypoint(dx_mm=-50.0), Waypoint(dx_mm=50.0), name="sq")},
        STANDBY_JOINTS,
    )
    assert not report.ok
    # 第二个路点:接着上一个点求一次,再用待命位姿作初值求一次,两次都不成
    assert [seed for x, seed in kin.ik_calls if x == 50.0] == [
        (-50.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        STANDBY_JOINTS,
    ]


def test_第一个路点的初值本来就是待命位姿不用再退回():
    kin = FakeKinematics(ik_fail_at=[-150.0])
    check_reach(kin, {"point_left": plan(Waypoint(dx_mm=-150.0))}, STANDBY_JOINTS)
    assert len(kin.ik_calls) == 1


def test_退回用待命位姿作初值时要留一条注记():
    # 不能默默放过:接下来喂给直线检查的关节角,未必是机械臂真走到的那一组
    class SeedPicky(FakeKinematics):
        def solve_ik(self, pose, seed):
            x = round(pose.Pos()[0], 3)
            self.ik_calls.append((x, tuple(seed)))
            if x == 50.0 and tuple(seed) != STANDBY_JOINTS:
                return None
            return [x, 0.0, 0.0, 0.0, 0.0, 0.0]

    report = check_reach(
        SeedPicky(),
        {"sq": plan(Waypoint(dx_mm=-50.0), Waypoint(dx_mm=50.0), name="sq")},
        STANDBY_JOINTS,
    )
    assert report.ok
    assert [n.waypoint for n in report.notes] == [2]
    assert "构型接不上" in report.describe()


def test_一路顺的时候没有注记():
    report = check_reach(
        FakeKinematics(),
        {"sq": plan(Waypoint(dx_mm=-50.0), Waypoint(dx_mm=50.0), name="sq")},
        STANDBY_JOINTS,
    )
    assert report.notes == ()

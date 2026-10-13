"""活动范围全查一遍(连真机之前用,ADR-0014):哪些停靠点查到了、出发关节角怎么接。

用假机械臂,不开 RoboDK。假机械臂的可达性检查一律通过,除了测试里点名的动作。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from robot_side.config import load_arm, load_demo
from robot_side.motion import ORIGIN, Segment
from robot_side.reach import ReachReport
from robot_side.startup import check_range, prepare, range_points
from tests.fakes import RecordingArm

DEMO = load_demo()
ARM_CONFIG = load_arm("noarm")


def prepared_arm(arm: RecordingArm | None = None):
    arm = arm if arm is not None else RecordingArm(ARM_CONFIG.standby_joints_deg)
    return arm, prepare(arm, ARM_CONFIG, DEMO)


def test_活动范围里的停靠点都走得到():
    # 左右各两步、上下各两步、前后各一步:5 × 5 × 3 个停靠点,待命点在最前
    points = range_points(DEMO)
    assert len(points) == 75
    assert points[0] == ORIGIN
    assert (-300.0, 100.0, 200.0) in points
    assert all(abs(x) <= 300 and abs(y) <= 100 and abs(z) <= 200 for x, y, z in points)


def test_全过时每个停靠点出发的每个动作都查过():
    arm, prepared = prepared_arm()
    seen = []
    checked = check_range(
        arm, ARM_CONFIG, DEMO, prepared, on_progress=lambda done, total: seen.append((done, total))
    )
    assert checked.rest_points == 75
    assert checked.failures == ()
    assert all(checked.reachable.values())
    # 待命点那一轮启动时查过,不再查;其余 74 个各查一次
    assert len(arm.reach_checks) == 1 + 74
    assert seen[0] == (1, 75) and seen[-1] == (75, 75)
    # 边上的停靠点:再往外那一步出了活动范围,不查
    assert ((300.0, 0.0, 0.0), "point_right") not in checked.reachable
    assert checked.reachable[((300.0, 0.0, 0.0), "point_left")]


def test_某一步不过它通往的停靠点就不查():
    arm, prepared = prepared_arm()
    arm.unreachable.add("raise")  # 启动那一轮查完以后才放:待命点往上那一步是过的
    checked = check_range(arm, ARM_CONFIG, DEMO, prepared)
    assert checked.reachable[(ORIGIN, "raise")]
    assert not checked.reachable[((0.0, 0.0, 100.0), "raise")]
    # 往上只走得到一步:最上面那一层(Z +200,5 × 3 个)一个都没查
    assert checked.rest_points == 60
    assert not any(z == 200.0 for (x, y, z), _ in checked.reachable)
    assert "够不着" in checked.describe()


class ChainArm(RecordingArm):
    """可达性检查报出每个动作走完时的「关节角」,并记下每次从哪组关节角出发。

    关节角编成 (x, y, z, 0, 0, 0):x、y、z 是查的时候出发的那组加上这个动作挪的那一步,
    看得出下一个停靠点的出发关节角是不是接着走到它的那一步来的。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.starts: list[tuple[float, ...]] = []

    def reach_check(
        self, plans: Mapping[str, Sequence[Segment]], standby_joints: Sequence[float]
    ) -> ReachReport:
        report = super().reach_check(plans, standby_joints)
        self.starts.append(tuple(standby_joints))
        x, y, z = standby_joints[:3]
        ends = {}
        for name in plans:
            dx, dy, dz = DEMO.action(name).shift_mm
            ends[name] = (x + dx, y + dy, z + dz, 0.0, 0.0, 0.0)
        return ReachReport(
            report.failures,
            report.checked_waypoints,
            report.checked_segments,
            end_joints=ends,
        )


def test_下一个停靠点的出发关节角接着走到它的那一步():
    arm = ChainArm((0.0,) * 6)
    _, prepared = prepared_arm(arm)
    check_range(arm, ARM_CONFIG, DEMO, prepared)
    # 每个停靠点出发时的关节角,正好是从待命位姿一步一步走过去累出来的那组
    s0, s1, s2 = ARM_CONFIG.standby_joints_deg[:3]
    assert {(round(a, 6), round(b, 6), round(c, 6)) for a, b, c, *_ in arm.starts[1:]} == {
        (round(s0 + x, 6), round(s1 + y, 6), round(s2 + z, 6))
        for x, y, z in range_points(DEMO)
        if (x, y, z) != ORIGIN
    }


# ---- 存下来,配置不变就不再查(2026-09-23 用户要的) ------------------------


def cached(arm, prepared, tmp_path, demo=DEMO, arm_config=ARM_CONFIG):
    from robot_side.range_cache import check_range_cached

    station = tmp_path / "station.rdk"
    if not station.exists():
        station.write_bytes(b"model v1")
    return check_range_cached(
        arm, arm_config, demo, prepared, cache_dir=tmp_path / "cache", station_file=station
    )


def test_第二次启动用存下来的结果不再查(tmp_path):
    arm, prepared = prepared_arm()
    first, from_cache = cached(arm, prepared, tmp_path)
    assert not from_cache
    checks = len(arm.reach_checks)
    second, from_cache = cached(arm, prepared, tmp_path)
    assert from_cache
    assert len(arm.reach_checks) == checks  # 一次都没再查
    assert second.reachable == first.reachable
    assert second.rest_points == first.rest_points


def test_够不着的那几处也原样存下来(tmp_path):
    arm, prepared = prepared_arm()
    arm.unreachable.add("raise")
    first, _ = cached(arm, prepared, tmp_path)
    second, from_cache = cached(arm, prepared, tmp_path)
    assert from_cache
    assert second.failures == first.failures
    assert not second.reachable[((0.0, 0.0, 100.0), "raise")]


def test_活动范围改了就重新查(tmp_path):
    import dataclasses

    arm, prepared = prepared_arm()
    cached(arm, prepared, tmp_path)
    checks = len(arm.reach_checks)
    smaller = dataclasses.replace(
        DEMO, run=dataclasses.replace(DEMO.run, range_mm=(150.0, 0.0, 100.0))
    )
    report, from_cache = cached(arm, prepared, tmp_path, demo=smaller)
    assert not from_cache
    assert len(arm.reach_checks) > checks
    assert report.rest_points == 9


def test_只改速度停顿不重新查(tmp_path):
    # 够不够得着和走多快无关。2026-09-24 用户在真机上调速度,每改一次就重查 5 分钟
    import dataclasses

    arm, prepared = prepared_arm()
    cached(arm, prepared, tmp_path)
    checks = len(arm.reach_checks)
    faster = dataclasses.replace(
        DEMO,
        actions=tuple(
            dataclasses.replace(
                a,
                speed_mms=a.speed_mms * 4,
                display=a.display + " FAST",
                waypoints=tuple(dataclasses.replace(w, pause_s=w.pause_s + 0.5) for w in a.waypoints),
            )
            for a in DEMO.actions
        ),
        run=dataclasses.replace(DEMO.run, return_speed_mms=DEMO.run.return_speed_mms * 3),
    )
    _, from_cache = cached(arm, prepared, tmp_path, demo=faster)
    assert from_cache
    assert len(arm.reach_checks) == checks


def test_路点改了就重新查(tmp_path):
    import dataclasses

    arm, prepared = prepared_arm()
    cached(arm, prepared, tmp_path)
    first = DEMO.actions[0]
    wider = dataclasses.replace(
        first,
        waypoints=(dataclasses.replace(first.waypoints[0], dx_mm=first.waypoints[0].dx_mm * 2),),
    )
    moved = dataclasses.replace(DEMO, actions=(wider, *DEMO.actions[1:]))
    _, from_cache = cached(arm, prepared, tmp_path, demo=moved)
    assert not from_cache


def test_待命位姿改了就重新查(tmp_path):
    import dataclasses

    arm, prepared = prepared_arm()
    cached(arm, prepared, tmp_path)
    moved = dataclasses.replace(
        ARM_CONFIG, standby_joints_deg=tuple(v + 1.0 for v in ARM_CONFIG.standby_joints_deg)
    )
    _, from_cache = cached(arm, prepared, tmp_path, arm_config=moved)
    assert not from_cache


def test_工作站文件改了就重新查(tmp_path):
    arm, prepared = prepared_arm()
    cached(arm, prepared, tmp_path)
    (tmp_path / "station.rdk").write_bytes(b"model v2")
    _, from_cache = cached(arm, prepared, tmp_path)
    assert not from_cache


def test_控制柜IP不同照样用存下来的(tmp_path):
    # 仿真里跑一次攒下的,真机上照用:IP 和查出来什么无关
    import dataclasses

    arm, prepared = prepared_arm()
    cached(arm, prepared, tmp_path)
    real = dataclasses.replace(ARM_CONFIG, ip="192.168.1.200")
    _, from_cache = cached(arm, prepared, tmp_path, arm_config=real)
    assert from_cache


def test_存的文件坏了就重新查不拿它放行(tmp_path):
    arm, prepared = prepared_arm()
    cached(arm, prepared, tmp_path)
    for path in (tmp_path / "cache").glob("range-*.json"):
        path.write_text("{ 不是 json", encoding="utf-8")
    _, from_cache = cached(arm, prepared, tmp_path)
    assert not from_cache

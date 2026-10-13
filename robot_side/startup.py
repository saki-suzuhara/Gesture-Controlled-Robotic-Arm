"""启动顺序里属于第 02 票的两步:按配置写入,然后可达性检查。

已定的启动顺序(`docs/grill-decisions.md` 第二节)是:打开工作站 → 按机械臂配置写入
工作台坐标系和待命位姿 → 可达性检查(这时还没连真机)→ 要连真机才经驱动连接 →
回待命 → 空闲。这里做中间那两步,外加回待命这一段的排布;状态逻辑是第 03 票。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from robot_side.arm import Arm
from robot_side.config import END_STAY, ArmConfig, DemoConfig
from robot_side.motion import (
    ORIGIN,
    Offset,
    Segment,
    in_range,
    lifted_pose,
    orientation_pose,
    plan_action,
    plan_returning,
    shifted,
)
from robot_side.reach import ReachFailure, ReachReport


@dataclass(frozen=True)
class StartupPlans:
    """写完配置、查完可达性之后手上有的东西:待命点、每个预设动作拆好的段、检查结果。"""

    standby_point: tuple[float, float, float]
    """待命位姿下法兰在工作台坐标系里的位置;停靠点都相对它量。"""
    by_action: dict[str, list[Segment]]
    """**从待命点出发**的每个预设动作。从别的停靠点出发的,状态逻辑到时候再排、再查。"""
    report: ReachReport

    @property
    def ok(self) -> bool:
        return self.report.ok


def prepare(arm: Arm, arm_config: ArmConfig, demo: DemoConfig) -> StartupPlans:
    """按配置写入,把六个预设动作拆成一段段,再查一遍可达性。

    查的是**从待命点出发**的那一轮。动作做完不回待命以后,从别的停靠点出发的要等
    机械臂真停到那儿才知道要不要查(每段 `MoveL_Test` 要 0.1–0.4 秒,活动范围里
    二十几个停靠点全查一遍要一两分钟),由状态逻辑在出发前查。

    **检查不通过时机械臂一步没动**:这一路只写坐标系和待命位姿、只求逆解和查直线,
    没有一处下发运动。
    """
    standby_point = arm.apply_config(arm_config)
    orientation = orientation_pose(demo.default_orientation_deg)
    plans = {
        action.name: plan_action(action, start_point=standby_point, orientation=orientation)
        for action in demo.actions
    }
    report = arm.reach_check(plans, arm_config.standby_joints_deg)
    return StartupPlans(standby_point=standby_point, by_action=plans, report=report)


LIFT_MAX_JOINT_DEG = 15.0
"""抬起那一段每一轴最多转多少度,超了就不走。

抬起 50 mm 各轴只转几度(仿真里 JAKA 在活动范围里最多约 8°)。转得多了,多半是逆解
换到了另一支构型——照着关节运动走过去,机械臂会大幅甩一下。真机脚本的护栏规矩:
只让该动的动、限行程(`real-arm-script-guardrails`)。
"""


class LiftRefused(Exception):
    """回待命的抬起那一段排不出来,或者排出来要甩臂。**一步不动**,由状态逻辑停下等人。"""


def returning(arm: Arm, arm_config: ArmConfig, demo: DemoConfig) -> list[Segment]:
    """回待命的两段:先抬起,再回待命位姿,**两段都是关节运动**(`motion.plan_returning`)。

    抬起以后的关节角从机械臂**现在停着的地方**求:读一次位姿和关节角,沿工作台 +Z 抬
    `lift_mm`,接着现在的关节角求逆解。所以只在它不动时调。程序启动后第一次移到待命
    位姿也走这一套(倒计时是状态逻辑的事)。

    求不出来、或者某一轴要转超过 `LIFT_MAX_JOINT_DEG`,抛 `LiftRefused`,一段都不排。
    """
    here = arm.joints()
    target = lifted_pose(arm.pose(), demo.run.lift_mm)
    lift_joints = arm.solve_ik(target, here)
    if lift_joints is None:
        raise LiftRefused(
            f"抬起 {demo.run.lift_mm:.0f} mm 求不出关节角(够不着或在奇异点上);"
            "在示教器上把机械臂挪回 Home 附近再按回车"
        )
    swing = max(abs(a - b) for a, b in zip(lift_joints, here, strict=True))
    if swing > LIFT_MAX_JOINT_DEG:
        raise LiftRefused(
            f"抬起 {demo.run.lift_mm:.0f} mm 要让某一轴转 {swing:.1f}°(上限 "
            f"{LIFT_MAX_JOINT_DEG:.0f}°),多半是逆解换了一支构型,不走;"
            "在示教器上把机械臂挪回 Home 附近再按回车"
        )
    return plan_returning(
        lift_joints,
        lift_mm=demo.run.lift_mm,
        standby_joints=arm_config.standby_joints_deg,
        return_joint_degs=arm_config.return_joint_degs,
        tip_speed_mms=demo.run.return_speed_mms,
    )


# ---- 活动范围全查一遍(连真机之前用) ----------------------------------


@dataclass(frozen=True)
class RangeFailure:
    """从某个停靠点出发,某个预设动作不过。"""

    rest: Offset
    failure: ReachFailure

    def __str__(self) -> str:
        x, y, z = self.rest
        return f"从待命点 ({x:+.0f}, {y:+.0f}, {z:+.0f}) mm 出发:{self.failure}"


@dataclass(frozen=True)
class RangeReport:
    """活动范围里每个停靠点出发的每个预设动作,过没过。

    `reachable` 的键是 (停靠点, 动作名)。**没在里面的就是没查过**:那个停靠点走不到
    (通往它的那一步本身就不过),或者做完会出活动范围。状态逻辑把没查过的一律当够不着。
    """

    reachable: dict[tuple[Offset, str], bool]
    rest_points: int
    """查过的停靠点有几个(含待命点)。"""
    checked_segments: int
    failures: tuple[RangeFailure, ...]

    def describe(self) -> str:
        passed = sum(self.reachable.values())
        line = (
            f"活动范围查完:{self.rest_points} 个停靠点、{len(self.reachable)} 个组合"
            f"({passed} 个过)、{self.checked_segments} 段直线"
        )
        if not self.failures:
            return line
        return "\n".join([line + ";这几处够不着,到时候不动(CAN'T REACH):"] + [
            f"  - {f}" for f in self.failures
        ])


def range_points(demo: DemoConfig) -> list[Offset]:
    """活动范围里走得到的停靠点,待命点在前、由近到远。纯计算,不查够不够得着。"""
    seen = [ORIGIN]
    queue = deque([ORIGIN])
    while queue:
        here = queue.popleft()
        for action in demo.actions:
            if action.end != END_STAY:
                continue
            there = shifted(here, action)
            if in_range(there, demo.run.range_mm) and there not in seen:
                seen.append(there)
                queue.append(there)
    return seen


def check_range(
    arm: Arm,
    arm_config: ArmConfig,
    demo: DemoConfig,
    prepared: StartupPlans,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> RangeReport:
    """把活动范围里每个停靠点出发的每个预设动作都查一遍。**连真机之前做。**

    平时(只跑仿真)不用它:从别的停靠点出发的,状态逻辑第一次用到时才查。但那一查
    要在 RoboDK 里挪一下模型再放回去,连着真机时这么做安不安全没验过——所以连真机
    之前在这里一次查完,连上以后运行中就不再查(2026-09-23 用户定,ADR-0014)。

    从待命点往外一步一步走(和机械臂真走过去一样):一个停靠点的出发关节角,是走到
    它的那一步查完时的关节角(`ReachReport.end_joints`),不是从待命位姿重新求。
    某一步不过,它通往的停靠点就不从这条路走;别的路也走不到的,整个停靠点不查。

    待命点那一轮 `prepare` 已经查过,直接用它的结果。`on_progress(查完几个, 一共几个)`
    每查完一个停靠点报一次,好让画面上写进度——JAKA 一段直线要 0.3–0.6 秒,整个
    活动范围要查几百段。
    """
    orientation = orientation_pose(demo.default_orientation_deg)
    standby_point = prepared.standby_point
    total = len(range_points(demo))
    reachable: dict[tuple[Offset, str], bool] = {
        (ORIGIN, name): True for name in prepared.by_action
    }
    joints_at: dict[Offset, tuple[float, ...]] = {
        ORIGIN: tuple(arm_config.standby_joints_deg)
    }
    queue: deque[Offset] = deque()
    failures: list[RangeFailure] = []
    segments = prepared.report.checked_segments

    def step_out(here: Offset, report: ReachReport) -> None:
        """这一轮过了的「停在终点」的动作,把它们通往的停靠点排进来。"""
        for action in demo.actions:
            if action.end != END_STAY or not reachable.get((here, action.name)):
                continue
            there = shifted(here, action)
            if in_range(there, demo.run.range_mm) and there not in joints_at:
                joints_at[there] = report.end_joints.get(action.name, joints_at[here])
                queue.append(there)

    step_out(ORIGIN, prepared.report)
    done = 1
    if on_progress is not None:
        on_progress(done, total)
    while queue:
        here = queue.popleft()
        start = tuple(p + o for p, o in zip(standby_point, here, strict=True))
        plans = {
            action.name: plan_action(action, start_point=start, orientation=orientation)
            for action in demo.actions
            if in_range(shifted(here, action), demo.run.range_mm)
        }
        report = arm.reach_check(plans, joints_at[here])
        segments += report.checked_segments
        failed = {f.action for f in report.failures}
        for name in plans:
            reachable[(here, name)] = name not in failed
        failures += [RangeFailure(here, f) for f in report.failures]
        step_out(here, report)
        done += 1
        if on_progress is not None:
            on_progress(done, total)
    return RangeReport(
        reachable=reachable,
        rest_points=len(joints_at),
        checked_segments=segments,
        failures=tuple(failures),
    )

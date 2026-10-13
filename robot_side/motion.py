"""把配置里的路点算成一段段运动。纯计算,不碰 RoboDK,所以测试里不用开它。

约定(ADR-0013、ADR-0003、`CONTEXT.md`):

- 位姿都是**法兰**在**工作台坐标系**里的位姿;控制器保持工具 0。
- 路点写成相对**出发点**的偏移。出发点是动作开始时机械臂停着的**停靠点**,停靠点
  记成相对**待命点**的偏移;待命点是待命位姿下法兰落在工作台坐标系里的位置,由待命
  位姿求正解得出,不单独写进配置。
- 一个预设动作 = 逐个路点的直线段;`back` 的再加一段直线回到出发点。**做完不回待命
  位姿**(2026-09-23 用户定)。目标姿态不是默认姿态的那几段(点头)下发时走关节运动
  (`LinearSegment.via_joints`)。
- 回待命 = 先沿工作台坐标系 +Z 抬起,再回待命位姿,两段都是关节运动;程序启动后第一次
  移到待命位姿也走这一套。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from robodk.robomath import Mat, pose_angle_between, rotx, roty, rotz, transl

from robot_side.config import (
    DEFAULT_ORIENTATION_DEG,
    END_BACK,
    Action,
    Waypoint,
    Worktable,
)

__all__ = [
    "DEFAULT_ORIENTATION_DEG",
    "ORIGIN",
    "JointSegment",
    "LinearSegment",
    "Offset",
    "Segment",
    "estimate_s",
    "in_range",
    "joint_path_mm",
    "lifted_pose",
    "orientation_pose",
    "plan_action",
    "plan_returning",
    "shifted",
    "tip_position",
    "waypoint_pose",
    "worktable_pose",
]

Offset = tuple[float, float, float]
"""停靠点相对待命点的偏移,工作台坐标系里的 mm。"""

ORIGIN: Offset = (0.0, 0.0, 0.0)
"""待命点本身。"""

RANGE_TOL_MM = 1e-6
"""判「在不在活动范围里」时容许的浮点误差。150 + 150 就该等于 300。"""


@dataclass(frozen=True, kw_only=True)
class Segment:
    """一段运动。

    `speed` 直线段是 mm/s、关节段是 °/s。真机上要不要真的发出去,看机械臂配置的
    `program_sets_speed`(不发的话,快慢全看示教器)。
    `label` 是这一段的名字,报错和日志里用来说清「哪个动作的哪个路点」。
    `pause_s` 是走完这一段之后的停顿(`CONTEXT.md`「预设动作」:段与段之间可以停顿)。
    """

    label: str
    speed: float
    pause_s: float = 0.0


@dataclass(frozen=True, kw_only=True)
class LinearSegment(Segment):
    """一段直线运动,走到工作台坐标系里的这个法兰位姿。

    `via_joints` 为真的那几段**不走直线**:下发时先求出这个位姿的关节角,再用关节运动
    过去(`robot_side.executor.send`)。点头那几段就是这样——它们只转姿态、目标姿态不是
    默认姿态,而经 RoboDK 的 JAKA 驱动下发直线时姿态会被镜像((RX, RY, RZ) 执行成
    (RX, −RY, −RZ)):2026-09-23 真机上点头 +10° 被执行成 −9.89°,到位核对差 20°,
    干等到有人叫停。关节运动下发即落点,不受镜像影响。
    """

    pose: Mat
    via_joints: bool = False


@dataclass(frozen=True, kw_only=True)
class JointSegment(Segment):
    """一段关节运动,走到这组关节角。

    `tip_speed_mms` 给了,就是「指针尖端要走多快」(mm/s),和直线段的 `speed` 一个量纲:
    发之前按这一段尖端要走多远现算出关节速度,换进 `speed`(`robot_side.executor.settle`)。
    算不出来(没有运动学的实现)就照旧用 `speed`。

    为什么不直接写 °/s:同样 13 °/s,J1 转起来伸出去一米的法兰每秒走两百多毫米,J6 转起来
    法兰原地不动。写 °/s 的关节段和写 mm/s 的直线段快慢根本对不上——9-23 真机上回待命、
    点头看着比左右上下快得多,就是这么来的。
    """

    joints: tuple[float, ...]
    tip_speed_mms: float | None = None


ROTATION_DEGS = 10.0
"""估「只转姿势、不挪位置」那种直线段时,按多少度每秒算。

原地点头就是这一种:法兰位置一点不变,只绕工作台 X 轴转(ADR-0013 的后果)。只按
走过的距离估的话它是 0 秒,时限就只剩那 5 秒。取 10°/s 是照着回待命那段关节运动在
仿真里用的速度来的——只用来估时限,**不是下发给机械臂的速度**。
"""


def estimate_s(
    segment: Segment,
    *,
    pose: Callable[[], Mat] | None = None,
    joints: Callable[[], Sequence[float]] | None = None,
) -> float:
    """估这一段大概要走多久(秒)。估不出来就返回 0。

    直线段按法兰走过的直线距离(mm ÷ 段速度)和转过的角度(° ÷ `ROTATION_DEGS`)
    取大的那个;关节段按转得最多的那一轴(° ÷ 段速度)。

    `pose` 和 `joints` 传的是**怎么读**(两个不带参数的函数),不是读回来的值:哪一样
    都只在真用得上时才读,直线段不读关节角、关节段不读位姿。读一次机械臂是一次 API
    来回,不该白读。传 `None` 就是这一样读不出来,那就估不出、返回 0。

    **只是个粗估。**加减速没算,姿态插值没算;真机上还要乘示教器上的速度倍率(没开
    `program_sets_speed` 时程序连速度本身都不设),配置里的速度只对仿真原样成立。所以拿它
    定超时的那条式子还要乘 2 再加 5 秒(`robot_side.executor.segment_timeout_s`)。
    """
    if segment.speed <= 0:
        return 0.0
    if isinstance(segment, LinearSegment):
        if pose is None:
            return 0.0
        from_pose = pose()
        moved_mm = math.dist(from_pose.Pos(), segment.pose.Pos())
        turned_deg = math.degrees(pose_angle_between(from_pose, segment.pose))
        return max(moved_mm / segment.speed, turned_deg / ROTATION_DEGS)
    if isinstance(segment, JointSegment):
        if joints is None:
            return 0.0
        turned = max(
            abs(a - b) for a, b in zip(joints(), segment.joints, strict=True)
        )
        return turned / segment.speed
    return 0.0


def worktable_pose(worktable: Worktable) -> Mat:
    """工作台坐标系相对机械臂基坐标系的位姿:绕 Z 转过一个角度,再加原点平移。"""
    return transl(*worktable.origin_mm) * rotz(math.radians(worktable.angle_deg))


def orientation_pose(rpy_deg: Sequence[float]) -> Mat:
    """由 (Rx, Ry, Rz) 三个角度算出姿态,按 X → Y → Z 的顺序相乘。"""
    rx, ry, rz = (math.radians(v) for v in rpy_deg)
    return rotx(rx) * roty(ry) * rotz(rz)


def waypoint_pose(
    start_point: Sequence[float], waypoint: Waypoint, default_orientation: Mat
) -> Mat:
    """一个路点的法兰位姿:出发点加偏移,再摆上姿态。

    姿态按 `CONTEXT.md`「路点」:路点自己写了姿态就用它(工作台坐标系下的绝对姿态),
    没写就用默认姿态;点头角在这之上再绕工作台 X 轴转。

    点头绕的是过**法兰**、平行工作台 X 轴的轴,所以法兰位置不变;指针尖端会扫出
    半径等于指针长度的弧(ADR-0013 的后果)。
    """
    x, y, z = start_point
    orientation = (
        default_orientation
        if waypoint.orientation_deg is None
        else orientation_pose(waypoint.orientation_deg)
    )
    return (
        transl(x + waypoint.dx_mm, y + waypoint.dy_mm, z + waypoint.dz_mm)
        * rotx(math.radians(waypoint.nod_deg))
        * orientation
    )


def tip_position(flange_pose: Mat, pointer_offset_mm: Sequence[float]) -> tuple[float, ...]:
    """指针尖端在同一个坐标系里的位置。只作参考(离人多远、伸到哪),不参与运动。"""
    return tuple((flange_pose * transl(*pointer_offset_mm)).Pos())


PATH_SAMPLES = 24
"""量关节运动的尖端路径时,沿途取几段。点头那种弧线 24 段量出来和真弧长差不到千分之一。"""


def joint_path_mm(
    flange_pose: Callable[[Sequence[float]], Mat],
    from_joints: Sequence[float],
    to_joints: Sequence[float],
    pointer_offset_mm: Sequence[float],
    *,
    samples: int = PATH_SAMPLES,
) -> float:
    """一段关节运动里,指针尖端和法兰走过的路,取长的那个(mm)。只算正解,不动机械臂。

    关节运动是各轴按同一个比例从起点插到终点,所以沿这条插值取点、正解、把相邻两点的
    距离加起来。取长的那个:点头时法兰不动、尖端扫弧;没有指针(偏移是 0)时两者一样。
    """
    start = [float(v) for v in from_joints]
    end = [float(v) for v in to_joints]
    offset = transl(*pointer_offset_mm)
    tips, flanges = [], []
    for i in range(samples + 1):
        s = i / samples
        pose = flange_pose([a + (b - a) * s for a, b in zip(start, end, strict=True)])
        flanges.append(pose.Pos())
        tips.append((pose * offset).Pos())
    tip = sum(math.dist(a, b) for a, b in zip(tips, tips[1:]))
    flange = sum(math.dist(a, b) for a, b in zip(flanges, flanges[1:]))
    return max(tip, flange)


def shifted(offset: Offset, action: Action) -> Offset:
    """做完这个动作以后停在哪儿(相对待命点)。四舍五入到微米,好拿来当字典的键。"""
    x, y, z = offset
    dx, dy, dz = action.shift_mm
    return (round(x + dx, 6), round(y + dy, 6), round(z + dz, 6))


def in_range(offset: Offset, range_mm: Sequence[float]) -> bool:
    """这个停靠点在不在活动范围里(每个方向离待命点都不超过那一个数)。"""
    return all(abs(o) <= r + RANGE_TOL_MM for o, r in zip(offset, range_mm, strict=True))


def plan_action(
    action: Action,
    *,
    start_point: Sequence[float],
    orientation: Mat,
) -> list[Segment]:
    """把一个预设动作拆成一段段:从 `start_point` 出发逐个路点直线走。

    `start_point` 是出发点在工作台坐标系里的位置(待命点加上停靠点的偏移)。`back` 的
    最后再加一段直线回到出发点、摆回默认姿态;最后一个路点本来就在那儿就不加。
    **不回待命位姿**:那是回待命的事(`plan_returning`)。
    """
    segments: list[Segment] = [
        LinearSegment(
            label=f"{action.name} 路点 {i + 1}",
            pose=waypoint_pose(start_point, wp, orientation),
            speed=action.speed_mms,
            pause_s=wp.pause_s,
            # 目标姿态不是默认姿态的(点头、自己写了姿态的)走关节运动,绕开姿态镜像
            via_joints=wp.nod_deg != 0.0 or wp.orientation_deg is not None,
        )
        for i, wp in enumerate(action.waypoints)
    ]
    last = action.waypoints[-1]
    at_start = (
        (last.dx_mm, last.dy_mm, last.dz_mm, last.nod_deg) == (0.0, 0.0, 0.0, 0.0)
        and last.orientation_deg is None
    )
    if action.end == END_BACK and not at_start:
        segments.append(
            LinearSegment(
                label=f"{action.name} 回到出发点",
                pose=waypoint_pose(start_point, Waypoint(), orientation),
                speed=action.speed_mms,
                # 从歪着的姿态转回来(点头、鞠躬的最后一段)也走关节运动:走直线的话,
                # 平移只有几十毫米、却要转二三十度,控制器按直线速度排时间,指针就甩得飞快
                via_joints=last.nod_deg != 0.0 or last.orientation_deg is not None,
            )
        )
    return segments


def lifted_pose(from_pose: Mat, lift_mm: float) -> Mat:
    """沿工作台坐标系 +Z 抬起 `lift_mm` 以后的法兰位姿,姿态不变。"""
    return transl(0.0, 0.0, lift_mm) * from_pose


def plan_returning(
    lift_joints: Sequence[float],
    *,
    lift_mm: float,
    standby_joints: Sequence[float],
    return_joint_degs: float,
    tip_speed_mms: float | None = None,
) -> list[Segment]:
    """回待命的两段:先抬起,再回到待命位姿,**两段都是关节运动**。

    `tip_speed_mms` 是指针尖端走多快(演示配置的 `return_speed_mms`),关节速度发之前现算;
    `return_joint_degs` 只在算不出来时用(`JointSegment` 上的说明)。

    `lift_joints` 是抬起 `lift_mm` 以后的关节角(`lifted_pose` 求逆解得来,见
    `robot_side.startup.returning`)。抬起那一段原来是直线,2026-09-23 改成关节运动:
    经 RoboDK 的 JAKA 驱动下发直线时姿态会被镜像(`(RX, RY, RZ)` 执行成
    `(RX, −RY, −RZ)`),从任意位姿出发的那一段直线避不开它;关节运动下发即落点。
    抬起 50 mm 各轴只转几度,关节插值的路径和直线差不了多少。倒计时是状态逻辑的事。
    """
    return [
        JointSegment(
            label=f"回待命:抬起 {lift_mm:.0f} mm",
            joints=tuple(lift_joints),
            speed=return_joint_degs,
            tip_speed_mms=tip_speed_mms,
        ),
        JointSegment(
            label="回待命:回到待命位姿",
            joints=tuple(standby_joints),
            speed=return_joint_degs,
            tip_speed_mms=tip_speed_mms,
        ),
    ]

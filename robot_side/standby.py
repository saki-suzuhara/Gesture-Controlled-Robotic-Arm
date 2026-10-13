"""选待命位姿:从示教器上摆好的那组关节角,算出配置里该写的那一组。纯计算,不碰 RoboDK。

**为什么不能照抄示教器上的数**(`config/arms/jaka.toml` 待命位姿上面那段说明):

- 待命位姿的姿态要和演示配置里的**默认姿态**对上。不对上的话,每个动作的第一段都要
  先转过去——9-22 那次用户摆的 Home 偏航差了 120°,J6 要白转大半圈。
- 经 RoboDK 的 JAKA 驱动下发直线运动时,姿态会被镜像:基坐标系里 `(RX, RY, RZ)` 执行成
  `(RX, −RY, −RZ)`(`notes/jaka-second-connection-results.md`)。`RY = 0、RZ = 0` 是它的
  **不动点**,待命位姿必须停在那儿,左右上下这些直线动作才不会被拧歪。

**J6 能补的只有偏航。** J6 绕的就是法兰自己的 Z 轴,法兰原点在这根轴上,所以转它法兰
位置一点不变(9-22 实测 0.000000 mm),指针是一根杆,绕自己转在现场看不出区别。偏航之外
的那部分——指针没朝正下方——J6 补不了,要人在示教器上扶正。所以这里只转 J6,转完还对不上
的就拒绝,不去动别的轴:动别的轴会挪法兰位置,那就不是人选的那个 Home 了。

9-22 那一次就是这么手算的:用户摆的 J6 是 −52.043,转 −30.233° 到 −82.276,基坐标系
姿态落在 `RX=−179.335 RY=0.091 RZ=0.000`。`plan_standby` 做的是同一件事。

改配置只改 `[standby] joints_deg` **那一行**,注释和别的字段一行不动(和
`robot_side.calibrate.rewrite_angle` 同一个规矩)。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from robodk.robomath import Mat, pose_2_xyzrpw, pose_angle_between

from robot_side.calibrate import normalize_deg
from robot_side.config import POWER_CHECK_JAKA, ArmConfig

J6 = 5
"""J6 在关节角里的下标。"""

TILT_TOL_DEG = 2.0
"""J6 补完以后,待命位姿和默认姿态最多还能差多少度。超了就拒绝,请人扶正。

差的这部分每个动作开头都要转一次;2° 以内指针尖端几乎不动。9-22 现场摆的 Home 差
0.67°(演示配置里把它扶正成 180 / 0 的那一点),在这里面。
"""

FIXED_POINT_TOL_DEG = 0.5
"""基坐标系里 RY、RZ 离不动点(0)最多多远。

镜像把 RY 翻成 −RY:差 0.5° 执行出来就差 1°,离直线段到位核对的 2°
(`robot_side.executor.ARRIVE_TOL_POSE_DEG`)还有一倍的余量。9-22 那个 Home 的 RY 是 0.091。
"""

SOLVE_TOL_DEG = 1e-4
MAX_ITERS = 12
PROBE_DEG = 1.0
"""量 J6 转向用的那一小下。只在正解里转,机械臂不动。"""


class StandbyRefused(Exception):
    """这一次选不了待命位姿。调用方停下,配置不动。"""


@dataclass(frozen=True)
class StandbyPlan:
    """算出来的待命位姿,和给人看的那几个数。"""

    picked: tuple[float, ...]
    """示教器上摆的那一组(读回来的)。"""
    joints: tuple[float, ...]
    """补过 J6 的那一组,要写进配置的就是它。"""
    j6_turn_deg: float
    """J6 补了多少度,折到 (−180, 180]。"""
    moved_mm: float
    """补 J6 前后法兰位置差多少。应当是 0,不是 0 说明 J6 轴没过法兰原点。"""
    tilt_deg: float
    """补完以后和默认姿态还差多少度(J6 补不了的那部分)。"""
    base_rpy_before: tuple[float, float, float]
    base_rpy_after: tuple[float, float, float]
    """基坐标系里的姿态,`pose_2_xyzrpw` 的 ZYX 顺序,也就是控制柜报的那一套。"""
    standby_point: tuple[float, float, float]
    """新待命点:法兰在工作台坐标系里的位置。"""
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def needs_fixed_point(config: ArmConfig) -> bool:
    """这台机械臂的待命位姿要不要停在姿态镜像的不动点上。

    镜像是经 RoboDK 的 **JAKA 驱动**才有的(`robot_side.real.executed_rpy`)。认 JAKA 用的是
    上电使能那一栏:只有 JAKA 走 jkrc 查上电使能。别的机械臂怎么接还没定,不管这一条。
    """
    return config.power_check == POWER_CHECK_JAKA


def plan_standby(
    picked: Sequence[float],
    *,
    flange_in_worktable: Callable[[Sequence[float]], Mat],
    flange_in_base: Callable[[Sequence[float]], Mat],
    default_orientation: Mat,
    fixed_point: bool,
    j6_limits: tuple[float, float] | None = None,
) -> StandbyPlan:
    """只转 J6,让待命位姿的姿态对上默认姿态;再核对转完还差多少、在不在不动点上。

    `flange_in_worktable` / `flange_in_base` 是正解(`RoboDKArm.flange_pose` 和
    `flange_pose_in_base`),只算不动。`default_orientation` 是演示配置里的默认姿态
    (`motion.orientation_pose`),工作台坐标系。`j6_limits` 是 J6 的上下限;给了就只在
    限位里挑。

    偏航是按正解一步步逼近的,不假设 J6 往哪边转算正:先让 J6 在正解里转一小下,
    量出法兰绕自己 Z 轴转了多少、往哪边转,再按这个比例补。
    """
    picked = tuple(float(v) for v in picked)
    q = list(picked)
    sign = _j6_direction(q, flange_in_worktable)
    for _ in range(MAX_ITERS):
        yaw = _yaw_to(flange_in_worktable(q), default_orientation)
        if abs(yaw) < SOLVE_TOL_DEG:
            break
        q[J6] += yaw / sign
    problems = []
    # 正解对 J6 以 360° 为周期:取转得最少、又不出限位的那一个
    turn = normalize_deg(q[J6] - picked[J6])
    choices = sorted((turn, turn - 360.0, turn + 360.0), key=abs)
    if j6_limits is not None:
        low, high = j6_limits
        inside = [t for t in choices if low <= picked[J6] + t <= high]
        if inside:
            turn = inside[0]
        else:
            problems.append(
                f"J6 要转到 {picked[J6] + turn:.3f}°,出了限位 [{low:g}, {high:g}]°;"
                "在示教器上把 J6 往回转一圈再来"
            )
    q[J6] = picked[J6] + turn
    joints = tuple(q)

    before = flange_in_worktable(picked)
    after = flange_in_worktable(joints)
    moved = math.dist(before.Pos(), after.Pos())
    tilt = math.degrees(pose_angle_between(_rotation(after), _rotation(default_orientation)))
    rpy_before = _base_rpy(flange_in_base(picked))
    rpy_after = _base_rpy(flange_in_base(joints))

    if tilt > TILT_TOL_DEG:
        problems.append(
            f"J6 补完以后姿态和默认姿态还差 {tilt:.2f}°(上限 {TILT_TOL_DEG:g}°),"
            "多半是指针没朝正下方——这部分 J6 补不了。在示教器上把法兰扶正再来"
        )
    if fixed_point:
        ry, rz = rpy_after[1], rpy_after[2]
        if abs(ry) > FIXED_POINT_TOL_DEG or abs(rz) > FIXED_POINT_TOL_DEG:
            problems.append(
                f"基坐标系里 RY={ry:+.3f}° RZ={rz:+.3f}°,不在姿态镜像的不动点上"
                f"(两个都要在 ±{FIXED_POINT_TOL_DEG:g}° 以内)。左右上下这些直线动作会被"
                "拧歪;在示教器上把指针扶到正下方再来"
            )
    x, y, z = after.Pos()
    return StandbyPlan(
        picked=picked,
        joints=joints,
        j6_turn_deg=turn,
        moved_mm=moved,
        tilt_deg=tilt,
        base_rpy_before=rpy_before,
        base_rpy_after=rpy_after,
        standby_point=(float(x), float(y), float(z)),
        problems=tuple(problems),
    )


def _j6_direction(q: Sequence[float], fk: Callable[[Sequence[float]], Mat]) -> float:
    """J6 转 +1° 时,法兰绕自己 Z 轴转了多少度(应当是 ±1)。转不出来就是正解有问题。"""
    probe = list(q)
    probe[J6] += PROBE_DEG
    turned = _yaw_to(fk(q), fk(probe))
    if abs(abs(turned) - PROBE_DEG) > 0.05 * PROBE_DEG:
        raise StandbyRefused(
            f"J6 转 {PROBE_DEG:g}° 时法兰绕自己的轴只转了 {turned:.4f}°,"
            "不像是法兰自己那根轴;正解对不上,不往下算"
        )
    return turned / PROBE_DEG


def _yaw_to(current: Mat, target: Mat) -> float:
    """从 `current` 到 `target` 要绕 `current` 自己的 Z 轴转多少度(只看偏航那一部分)。"""
    m = _rotation(current).inv() * _rotation(target)
    return math.degrees(math.atan2(m[1, 0], m[0, 0]))


def _rotation(pose: Mat) -> Mat:
    only = Mat(pose)
    only.setPos([0.0, 0.0, 0.0])
    return only


def _base_rpy(pose: Mat) -> tuple[float, float, float]:
    rx, ry, rz = pose_2_xyzrpw(pose)[3:6]
    return (normalize_deg(rx), normalize_deg(ry), normalize_deg(rz))


# ---- 改配置:只动 [standby] joints_deg 那一行 ---------------------------

STAMP_MARK = "set_standby 写于"
"""那一行后面记一笔是哪天、从示教器上的哪个 J6 补过来的。重写时整句换掉,不往上叠。"""

_SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_JOINTS = re.compile(r"^(\s*joints_deg\s*=\s*)(\[[^\]\n]*\])(\s*(?:#.*)?)$")
_JOINTS_START = re.compile(r"^\s*joints_deg\s*=")


def rewrite_standby(text: str, joints: Sequence[float], *, stamp: str, picked_j6: float) -> str:
    """把 `[standby]` 里的 `joints_deg` 改成新值,**别的一行都不动**。

    找不到、写了不止一个、或者写成了跨行的数组,一律拒绝改:这份配置是人手写的,
    拿不准改哪一行时,不动比猜着改好。
    """
    lines = text.splitlines(keepends=True)
    section = ""
    hits = []
    for index, line in enumerate(lines):
        found = _SECTION.match(line)
        if found:
            section = found.group(1).strip()
            continue
        if section == "standby" and _JOINTS_START.match(line):
            hits.append(index)
    if not hits:
        raise StandbyRefused(
            "配置里找不到 [standby] 一节下的 joints_deg,没有改;"
            "照 config/arms/jaka.toml 的样子补上这一节再来"
        )
    if len(hits) > 1:
        raise StandbyRefused(
            f"[standby] 里写了 {len(hits)} 个 joints_deg,不知道该改哪一个,没有改"
        )
    index = hits[0]
    found = _JOINTS.match(lines[index].rstrip("\r\n"))
    if found is None:
        raise StandbyRefused(
            "[standby] 的 joints_deg 不是写在一行里的 [..],没有改;改成一行再来"
        )
    head = found.group(1)
    ending = lines[index][len(lines[index].rstrip("\r\n")) :]
    values = ", ".join(f"{v:.3f}" for v in joints)
    lines[index] = (
        f"{head}[{values}]  # {STAMP_MARK} {stamp},示教器上摆的 J6 是 {picked_j6:.3f}{ending}"
    )
    return "".join(lines)

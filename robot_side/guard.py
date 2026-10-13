"""真机护栏:凡是会让真机动的东西,下发之前都要过这一关。

四条,来自 spec「安全相关」第 51 条和 `docs/grill-decisions.md` 第二节:

- **没上电使能就不下发**(用户故事 27)。读不出上电使能状态时**也不下发**——
  猜「大概使能了吧」和猜错的代价不对称。
- **从指令里把目标读回来逐轴核对。**不是核对「我算出来的目标」,而是核对
  **指令里真正指向的那个目标**:上一轮出事就出在这儿(见下)。
- **只动该动的轴。**
- **行程设上限。**每一轴一个上限;直线段另外卡走过的距离和转过的角度。

为什么护栏要长成这样(`notes/jaka-first-connection-results.md`「一个操作上的失误」):
那次程序的运动目标是在连接**之前**用仿真默认姿态建的,连上之后才去改目标点;护栏
只校验了算出来的目标值,**没有回头核对指令里真正指向的点**。目标要是没改成功,
机械臂会从 J1 = 201° 甩到 J1 = 1°,远超当时的 6° 护栏。没出事是运气,不是护栏起
了作用。所以这里的每一条都收「读回来的那个值」,不收「本来打算发的那个值」。

这个模块**不碰 RoboDK、不连任何机械臂**:护栏正不正确不该靠真机来验
(`tests/test_guard.py` 全程不开 RoboDK)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from robodk.robomath import Mat, pose_angle_between

AXIS_MATCH_TOL_DEG = 1e-4
"""读回来的目标和打算发的目标,逐轴容许差多少度。

这是「同一个数经了一趟 RoboDK 再读回来」的容差,不是运动精度的容差,所以取得很小。
"""

POSE_MATCH_TOL_MM = 0.01
POSE_MATCH_TOL_DEG = 0.01
"""位姿读回来时的容差。同上,只容得下往返一趟的舍入。"""

STILL_TOL_DEG = 1e-4
"""小到这个程度就算「这一轴没动」,不去卡它准不准动。

和 `AXIS_MATCH_TOL_DEG` 数值一样,但说的是两件事:那一个是「读回来的和打算发的
对不对得上」,这一个是「这一轴这一次要不要动」。共用一个常量的话,哪天要调其中
一个就会连另一个一起调歪。
"""

AXIS_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")
"""六个轴在报错里怎么称呼。示教器上就是这么标的,人照着找得到。"""


class GuardRefused(Exception):
    """护栏不放行。**调用方一律停下,不许降级重试。**"""


@dataclass(frozen=True)
class PowerState:
    """控制柜上电和使能的状态。

    `None` 是**读不出来**,不是「没有」:两者都不放行,但报出来的话不一样,人才知道
    是去示教器上按一下,还是去查为什么读不到。`detail` 是读不出来的原因原文。
    """

    powered: bool | None
    enabled: bool | None
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.powered is True and self.enabled is True

    def describe(self) -> str:
        return f"上电 {_yes_no(self.powered)},使能 {_yes_no(self.enabled)}"


def _yes_no(value: bool | None) -> str:
    return "读不出来" if value is None else ("有" if value else "没有")


@dataclass(frozen=True)
class MoveLimits:
    """这一次运动准动哪几轴、每轴最多转多少度。

    `axes` 是轴号(0 起)。写全六轴也不等于放行:每一轴照样卡 `max_delta_deg`。
    """

    axes: tuple[int, ...]
    max_delta_deg: float

    def allows(self, axis: int) -> bool:
        return axis in self.axes


def check_powered(state: PowerState) -> None:
    """没上电使能就不下发(用户故事 27)。程序只检查,不替人上电、不替人使能。"""
    if state.ready:
        return
    why = state.describe()
    if state.detail:
        why = f"{why}({state.detail})"
    raise GuardRefused(
        f"机械臂没有上电使能,不下发任何运动:{why}。"
        "请到示教器上上电、使能;程序不替你做(用户故事 27)"
    )


def check_joint_readback(
    intended: Sequence[float], readback: Sequence[float]
) -> None:
    """把**指令里读回来的**目标和打算发的逐轴核对。对不上就不发。"""
    intended = tuple(float(v) for v in intended)
    readback = tuple(float(v) for v in readback)
    if len(intended) != len(readback):
        raise GuardRefused(
            f"从指令里读回来的目标是 {len(readback)} 轴,打算发的是 {len(intended)} 轴,"
            "对不上,不下发"
        )
    for axis, (want, got) in enumerate(zip(intended, readback, strict=True)):
        if abs(want - got) > AXIS_MATCH_TOL_DEG:
            raise GuardRefused(
                f"{_axis(axis)}:指令里读回来的是 {got:.4f}°,打算发的是 {want:.4f}°,"
                f"差 {abs(want - got):.4f}°,不下发"
            )


def check_pose_readback(intended: Mat, readback: Mat) -> None:
    """位姿版的逐项核对:位置差多少 mm、姿态差多少度,都要在往返容差之内。"""
    moved_mm = math.dist(readback.Pos(), intended.Pos())
    turned_deg = math.degrees(pose_angle_between(readback, intended))
    if moved_mm > POSE_MATCH_TOL_MM or turned_deg > POSE_MATCH_TOL_DEG:
        raise GuardRefused(
            f"从指令里读回来的位姿和打算发的对不上:差 {moved_mm:.3f} mm、"
            f"{turned_deg:.3f}°,不下发"
        )


def check_joint_travel(
    from_joints: Sequence[float], target: Sequence[float], limits: MoveLimits
) -> None:
    """只动该动的轴,每轴不超行程上限。

    `target` 应当是**已经核对过的**那一组(先过 `check_joint_readback`),不然卡的
    是一个没人核对过的数。
    """
    from_joints = tuple(float(v) for v in from_joints)
    target = tuple(float(v) for v in target)
    if len(from_joints) != len(target):
        raise GuardRefused(
            f"现在是 {len(from_joints)} 轴、目标是 {len(target)} 轴,对不上,不下发"
        )
    for axis, (now, want) in enumerate(zip(from_joints, target, strict=True)):
        delta = abs(want - now)
        if delta <= STILL_TOL_DEG:
            continue  # 这一轴没动
        if not limits.allows(axis):
            raise GuardRefused(
                f"{_axis(axis)} 要动 {delta:.3f}°,但这一次只准动 "
                f"{_axis_list(limits.axes)},不下发"
            )
        if delta > limits.max_delta_deg:
            raise GuardRefused(
                f"{_axis(axis)} 要动 {delta:.3f}°,超过这一次的行程上限 "
                f"{limits.max_delta_deg:g}°,不下发"
            )


def check_orientation_change(
    from_rpy: Sequence[float],
    to_rpy: Sequence[float],
    *,
    max_deg: Sequence[float],
    what: str = "",
) -> None:
    """姿态相对现在变了多少,**逐个角**卡上限。超了就不下发。

    `from_rpy` / `to_rpy` 是同一个坐标系里的 (RX, RY, RZ),单位度;`max_deg` 是这三个
    各自准变多少。角度差折到 (−180, 180] 再比,免得 179° 和 −179° 被算成差 358°。

    和 `check_linear_travel` 的转角上限不同:那一条卡的是「整体转了多少」,这一条卡的是
    **哪一个角**转了多少。现场往往只对其中某一个角有要求——2026-09-22 用户对这台 JAKA
    的要求就是「法兰末端 RZ 最好不要转,RX / RY 不超过 5°」,那是一条只有站在机械臂
    旁边的人才提得出来的约束,整体转角卡不住它。

    `what` 是报错里对这一次目标的称呼,比如「下发的目标」「驱动会翻出来的那一种」。
    """
    names = ("RX", "RY", "RZ")
    for i, (a, b, cap) in enumerate(zip(from_rpy, to_rpy, max_deg, strict=True)):
        delta = abs(_fold(float(b) - float(a)))
        if delta > cap:
            where = f"{what}的 " if what else ""
            raise GuardRefused(
                f"{where}{names[i]} 要变 {delta:.2f}°,超过上限 {cap:g}°,不下发"
            )


def _fold(angle: float) -> float:
    """把角度差折到 (−180, 180]。"""
    folded = math.fmod(angle, 360.0)
    if folded <= -180.0:
        folded += 360.0
    elif folded > 180.0:
        folded -= 360.0
    return folded


def check_linear_travel(
    from_pose: Mat, to_pose: Mat, *, max_mm: float, max_deg: float
) -> None:
    """直线段:走过的距离和转过的角度各设一个上限。

    逐轴那一条(`check_joint_travel`)卡的是关节;这一条卡的是法兰在空间里走多远
    ——同一段运动两头都卡住,算错了哪一头都拦得下。
    """
    moved_mm = math.dist(from_pose.Pos(), to_pose.Pos())
    turned_deg = math.degrees(pose_angle_between(from_pose, to_pose))
    if moved_mm > max_mm:
        raise GuardRefused(
            f"这一段直线要走 {moved_mm:.1f} mm,超过上限 {max_mm:g} mm,不下发"
        )
    if turned_deg > max_deg:
        raise GuardRefused(
            f"这一段直线要转 {turned_deg:.1f}°,超过上限 {max_deg:g}°,不下发"
        )


def _axis(index: int) -> str:
    return AXIS_NAMES[index] if index < len(AXIS_NAMES) else f"第 {index + 1} 轴"


def _axis_list(axes: Sequence[int]) -> str:
    return "、".join(_axis(a) for a in axes) if axes else "(一轴都不准动)"

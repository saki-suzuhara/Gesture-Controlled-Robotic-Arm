"""可达性检查:启动时把演示配置里所有路点和直线段查一遍,通过了才让机械臂动。

规矩(spec「可达性检查」):

- **每个路点求逆解,并用正解核对、反复迭代**——JAKA 的逆解是数值迭代的,不收敛也会
  返回,所以不能把「返回不为空」当作够得着。这一层由运动学那一侧做(见 `Kinematics`)。
- **每段直线单独检查**(中途过奇异点、超关节限位都要报出来)。
- **下一段起点的关节角接着上一个点求**,不读回机械臂位置。
- 不通过时指出是**哪个预设动作的哪个路点、为什么**,而且**全程不移动机械臂**——这个
  模块只会调 `Kinematics` 的那两件事,它们都不动机械臂。
- 不查会不会碰到东西:碰撞检查在启动时关掉(`CONTEXT.md`「可达性检查」)。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from robodk.robomath import Mat

from robot_side.motion import LinearSegment, Segment

MOVEL_REASONS = {
    -1: "这一段直线走不通(中途过奇异点或超关节限位)",
    -2: "这一段直线够不着",
}


class Kinematics(Protocol):
    """可达性检查用得上的两件事。两件都不动机械臂。"""

    def solve_ik(self, pose: Mat, seed: Sequence[float]) -> list[float] | None:
        """求逆解,用正解核对、反复迭代;核对不过就返回 None(算够不着)。"""

    def linear_test(self, from_joints: Sequence[float], pose: Mat) -> int:
        """从这组关节角出发的一段直线走不走得通。

        返回 RoboDK `MoveL_Test` 的原值:0 走得通,−1 走不通(奇异点、超限位),
        −2 够不着,**大于 0 是碰撞编号**(碰撞检查已经关掉,按说不会出现)。
        """


@dataclass(frozen=True)
class ReachFailure:
    """一个不过的地方:哪个预设动作、第几个路点、为什么。"""

    action: str
    waypoint: int
    label: str
    why: str

    def __str__(self) -> str:
        return f"{self.action} 路点 {self.waypoint}:{self.why}"


@dataclass(frozen=True)
class ReachReport:
    """一次可达性检查的结果。

    `collisions` 是关掉碰撞检查之后仍然报出碰撞的段。按第 02 票第 9 条,这只记日志
    并提示,**不当作够不着**,所以它不影响 `ok`。

    `notes` 是过了、但过得不干净的地方,最要紧的一种是「接着上一个点求不出逆解,
    退回用待命位姿作初值」:那说明这一段前后的构型接不上,检查用的关节角未必是机械臂
    真走到的那一组。同样不影响 `ok`,但一定要说出来,不能默默放过。
    """

    failures: tuple[ReachFailure, ...] = ()
    checked_waypoints: int = 0
    checked_segments: int = 0
    collisions: tuple[ReachFailure, ...] = ()
    notes: tuple[ReachFailure, ...] = ()
    end_joints: dict[str, tuple[float, ...]] = field(default_factory=dict)
    """每个查过了的动作走完时的关节角(接着逆解一路求下来的那一组)。

    活动范围那一查(`robot_side.startup.check_range`)拿它当下一个停靠点的出发关节角:
    机械臂真走过去时关节角也是这么一路接过来的,不是从待命位姿重新求。没过的动作
    不在里面;没做运动学的实现(空实现、假机械臂)干脆不给。
    """

    @property
    def ok(self) -> bool:
        return not self.failures

    def describe(self) -> str:
        if self.ok:
            lines = [
                (
                    f"可达性检查通过:{self.checked_waypoints} 个路点、"
                    f"{self.checked_segments} 段直线"
                )
            ]
        else:
            lines = [f"可达性检查不通过({len(self.failures)} 处),机械臂一步没动:"]
            lines += [f"  - {f}" for f in self.failures]
        if self.notes:
            lines.append("过了,但这几处要留意:")
            lines += [f"  - {n}" for n in self.notes]
        if self.collisions:
            lines.append("碰撞检查已经关掉,但这几段仍然报了碰撞(只作提示,不算够不着):")
            lines += [f"  - {c}" for c in self.collisions]
        return "\n".join(lines)


def check_reach(
    kinematics: Kinematics,
    plans: Mapping[str, Sequence[Segment]],
    standby_joints: Sequence[float],
) -> ReachReport:
    """把每个预设动作的每个路点和每段直线查一遍。

    一个动作里第一处不过就停下,不再往下查这个动作——后面的路点要接着它的关节角求,
    接不上再报只会是一串跟着来的噪声。其余动作照常查完,好让一次就看全。
    """
    failures: list[ReachFailure] = []
    collisions: list[ReachFailure] = []
    notes: list[ReachFailure] = []
    end_joints: dict[str, tuple[float, ...]] = {}
    waypoints = segments = 0
    for name, plan in plans.items():
        joints = tuple(standby_joints)
        index = 0
        passed = True
        for segment in plan:
            if not isinstance(segment, LinearSegment):
                continue  # 关节段:回的是已知够得着的待命位姿
            index += 1
            rc = kinematics.linear_test(joints, segment.pose)
            segments += 1
            solved = kinematics.solve_ik(segment.pose, joints)
            if solved is None and joints != tuple(standby_joints):
                # 接着上一个点求不出来,不一定是够不着:上一个路点可能把构型带到了
                # 另一支,数值迭代从那儿收不回来。退回用待命位姿作初值再求一次——
                # 但要记一条,因为接下来喂给直线检查的关节角就未必是机械臂真走到的
                # 那一组了(原型里这也是一条注记,不是直接放过)。
                solved = kinematics.solve_ik(segment.pose, standby_joints)
                if solved is not None:
                    notes.append(
                        ReachFailure(
                            name,
                            index,
                            segment.label,
                            "接着上一个路点求不出逆解,退回用待命位姿作初值才求出来"
                            "(这一段前后的构型接不上)",
                        )
                    )
            waypoints += 1
            if solved is None:
                failures.append(
                    ReachFailure(name, index, segment.label, "逆解不收敛(够不着或在奇异点上)")
                )
                passed = False
                break
            if rc > 0:
                # 碰撞检查已经关掉还报碰撞:记下来、提示,但不算够不着(第 02 票第 9 条)
                collisions.append(
                    ReachFailure(name, index, segment.label, f"报了碰撞(编号 {rc})")
                )
            elif rc != 0:
                failures.append(
                    ReachFailure(
                        name, index, segment.label, MOVEL_REASONS.get(rc, f"直线检查返回 {rc}")
                    )
                )
                passed = False
                break
            joints = tuple(solved)
        if passed:
            end_joints[name] = joints
    return ReachReport(
        tuple(failures), waypoints, segments, tuple(collisions), tuple(notes), end_joints
    )

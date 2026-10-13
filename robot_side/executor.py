"""逐段下发:一次只发一段,发下一段之前先看「已叫停」标记。

三条规矩合在一起(spec「机械臂接口的形状」、ADR-0012):

- **一次只下发一段。**发出后轮询「是否还在动」,等它变成「不动」才发下一段。
  上一段还在走时发下一段,会占着连接的锁一直等,同一条连接上的叫停也就卡住了;
  而且程序崩了的话,机械臂最多走完当前这一段。
- **叫停可能正好落在两段之间。**这时机械臂本来就不动,「停止」没什么可停的,
  所以要靠「已叫停」标记把下一段拦住。
- **机械臂身上挂着报错就不再发下一段。**碰撞保护半路跳闸那种,没有哪一次调用会抛
  (`robot_side.arm.Arm.error`);带着故障接着发下一段,是最不该做的事。

怎么把一个预设动作拆成一段段运动,是 `robot_side.motion`;每段等多久算超时在下面的
`segment_timeout_s`。这里只管「发一段、等一段、发之前先看标记」这个循环。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import math
import time

from robodk.robomath import pose_angle_between

from robot_side.arm import (
    MOVE_POLL_S,
    Arm,
    ArmBusyError,
    ArmTimeoutError,
    MotionError,
    joint_drift,
)
from robot_side.motion import JointSegment, LinearSegment, Segment, estimate_s
from robot_side.stop import StopSignal

TIMEOUT_FACTOR = 2.0
TIMEOUT_SLACK_S = 5.0
"""一段运动等多久算不对劲:**估算用时 × 2 + 5 秒**(用户故事 19)。

乘 2 加 5 是给估算留的余地:估算里没有加减速,而且真机上还要乘示教器上的速度倍率,
程序读不到(没开 `program_sets_speed` 时连速度本身都不设)。示教器上调得比配置里写的慢一半
以上时,这条式子会把「走得慢」当成「不对劲」——上真机前照演示前检查清单核对一遍
速度(第 05 票)。
"""

ARRIVE_TOL_DEG = 0.5
"""关节段算「到位」的容差:每轴差不超过这么多度。

真机上关节运动下发即落点(2026-09-22 实测差 0.002–0.006°),0.5° 是宽裕的。
"""

ARRIVE_TOL_MM = 10.0
ARRIVE_TOL_POSE_DEG = 2.0
"""直线段算「到位」的容差。**这两个数是按真机实测定的,不是拍的。**

2026-09-22 在 JAKA 上跑六向方盒(每段 100 mm,共 10 段)量到的:

- **真到位时,离目标 1.4–5.2 mm。** 里面有约 1.9 mm 是 RoboDK 的名义 DH 和真机出厂
  标定 DH 的差(同一组关节角正解出来就差这么多),剩下的是定位误差。
- 要挡住的那两种差得远:`Busy()` 早落时读到的半路位置 **43.6 mm**;上一段被控制器
  悄悄丢了、机械臂根本没动 **100 mm**。

所以 8–40 mm 之间都分得开这两类。取 10 mm,两边都留足余量 —— 先取 5 mm 试过,
那一轮里有一段落在 5.2 mm 上,刚好卡在门槛外判不到位,白等了一个超时。

姿态那 2°:同一轮里真到位时 RX 最多飘 0.67°,而原地点头是 ±10°,分得开。
"""

ARRIVE_POLL_S = 0.5
"""多久核对一次到位。**比 `MOVE_POLL_S` 慢得多,是故意的**,见 `wait_until_done`。"""


SegmentReport = Callable[[int, "Segment | None"], None]
"""每发一段之前报一次:这是第几段(0 起)、是哪一段。

预览窗口要显示「第几段」,日志要记下每段运动(第 03 票)。**在发出之前**报,所以
报到的就是正要走的那一段,不是刚走完的那一段;被叫停拦下的那一段不会被报。
"""

SegmentCall = Callable[[], None]
"""发一段运动的调用:调一次机械臂接口的关节运动或直线运动,发出即返回。

(排好的那种「一段」是 `robot_side.motion.Segment`,给 `run_plan` 用;这个是
`run_segments` 用的,调用方自己把要发什么包成一个调用。)
"""


@dataclass(frozen=True)
class SegmentTimeout:
    """每段等多久算不对劲的规矩:按估算用时,乘 `slowdown`,再 × 2 + 5 秒,不少于 `floor_s`。

    真机上开了 `program_sets_speed` 时用它(`robot_side.real.real_segment_timeout`):程序发的
    速度还要乘示教器倍率,倍率程序读不到,就按「倍率至少多少」把估算放慢。固定一个数不行——
    一段从远处回待命的关节运动,程序速度下要一分多钟,倍率 13% 时是好几分钟。
    """

    slowdown: float = 1.0
    floor_s: float = 0.0


def segment_timeout_s(arm: Arm, segment: Segment, *, slowdown: float = 1.0) -> float:
    """这一段等多久算不对劲:估算用时 × 2 + 5 秒。`slowdown` 是估算再放慢几倍(`SegmentTimeout`)。

    出发点是**现读**的:直线段读一次法兰位姿,关节段读一次关节角(`estimate_s` 只读
    用得上的那一样)。这两个只能在机械臂不动时读(`robot_side.arm`),而算这个数的时刻
    正好是上一段已经走完、这一段还没发出去——整条循环里唯一能读的那一下。

    不按「上一段的终点」推,是因为机械臂可能根本不在那儿:程序刚启动、上一段被叫停在
    半路、或者有人把它挪过。

    **这一次读占不了多少时间。**RoboDK 5.9 仿真上实测(2026-09-20,50 次):查「还在
    不在动」中位 0.20 ms、最大 1.57 ms,读关节角 0.49 / 2.64 ms,读位姿 0.37 / 1.86 ms;
    算一次时限是这两样各一次,合起来中位约 0.6 ms、最坏约 4 ms。这段时间占着的是叫停
    也要用的那条连接的锁,但比非阻塞 `MoveL` 自己占的 63–146 ms 小一个量级,叫停用时
    的预算(0.5 秒以内)不受影响。
    """
    estimate = estimate_s(segment, pose=arm.pose, joints=arm.joints)
    return estimate * slowdown * TIMEOUT_FACTOR + TIMEOUT_SLACK_S


def arrived(arm: Arm, segment: Segment | None) -> bool:
    """这一段是不是真的走到目标了。读一次位置,所以**只在机械臂不动时调**。

    估不出目标的那一种(`run_segments` 自己包的调用)返回 True —— 没有目标可核对,
    只能信「是否还在动」。
    """
    try:
        if isinstance(segment, JointSegment):
            return joint_drift(arm.joints(), segment.joints) <= ARRIVE_TOL_DEG
        if isinstance(segment, LinearSegment):
            here = arm.pose()
            moved = math.dist(here.Pos(), segment.pose.Pos())
            turned = math.degrees(pose_angle_between(here, segment.pose))
            return moved <= ARRIVE_TOL_MM and turned <= ARRIVE_TOL_POSE_DEG
    except ArmBusyError:
        return False  # 正好撞上它又动起来了:那就还没到位
    return True


def wait_until_done(
    arm: Arm,
    signal: StopSignal,
    segment: Segment | None,
    timeout_s: float,
    poll_s: float = MOVE_POLL_S,
) -> None:
    """等这一段**真的**走完:机械臂说不动了,**而且读回来确实到了目标**。

    **为什么不能只信「是否还在动」**(2026-09-22 真机实测,
    `notes/jaka-second-connection-results.md`):经 RoboDK 的 JAKA 驱动,`Busy()` 两头
    都不准,而且危险的是「早」那一头——它会在机械臂还在走、甚至还没开始走的时候就说
    「不动了」。实测里一段 100 mm 的直线,`Busy()` 4.82 秒落下时机械臂还差 43.6 mm;
    照着它发下一段,下一段就从一个错的起点算起,越差越远(那一次是护栏把它拦住的)。

    多核对「到没到目标」这一下,还顺带治住另外两种:上一段被控制器悄悄丢了(实测有过
    一次,指令根本没执行、机械臂一动没动,`Busy()` 照样真了 20 秒),以及走到一半停住
    ——两种都会等不到位,最后按超时报出来,而不是默默当成走完了。

    **被叫停就立刻返回**:叫停之后这一段本来就走不完了,调用方按「没走完」处理。

    **一个已知的风险,说明白:**核对到位要读一次位置,而我们是在「`Busy()` 说不动」
    的时候读的——可 `Busy()` 恰恰会在机械臂还在动时这么说。笔记里记过一次
    `robot.Joints()` 在运动中把 RoboDK 5.9 崩掉(2026-09-20,高频轮询)。所以这里
    **核对得很慢**(`ARRIVE_POLL_S`,半秒一次),而不是跟着 `poll_s` 每 20 ms 读一次
    ——把暴露压到最小。真要根治,得换一条读位置的通道,那是另一件事。
    """
    deadline = time.monotonic() + timeout_s
    left = timeout_s  # 第一次给足额:后面几轮才按剩下的算
    while True:
        if signal.raised:
            return  # 被叫停:这一段不会走完了,由调用方处理
        # 先按老办法等它说「不动了」——`wait_until_not_moving` 是机械臂接口上的方法,
        # 时限和轮询都归它管,超了由它抛 `ArmTimeoutError`
        arm.wait_until_not_moving(left, poll_s)
        # 只跑仿真时「不动了」就是走完了,不再核对(`Arm.busy_is_reliable`)
        if signal.raised or arm.busy_is_reliable or arrived(arm, segment):
            return
        # 它说不动了,可没到目标 —— 多半还在走(`Busy()` 早落)。慢慢再核对,别急着读
        if time.monotonic() > deadline:
            raise ArmTimeoutError(f"一段运动超过 {timeout_s:.1f} 秒还没走到目标")
        time.sleep(ARRIVE_POLL_S)
        left = max(deadline - time.monotonic(), 0.0)


VIA_JOINTS_MAX_DEG = 45.0
"""「走关节运动过去」的那一段,每一轴最多转多少度,超了就不走。

点头在 ±10° 之间来回,J4 一次要转二十来度;逆解换到另一支构型时是上百度的跳变。
45° 拦的是后一种:照着换了构型的关节角走过去,机械臂会大幅甩一下。
"""

VIA_JOINTS_DEGS = 10.0
"""「走关节运动过去」那一段算不出尖端速度时的关节速度(°/s)。和估时限用的
`ROTATION_DEGS` 一样。算得出时按原来那段直线的 `speed` 换算(`settle`)。"""


def joints_for(arm: Arm, segment: LinearSegment) -> list[float]:
    """这一段目标位姿的关节角:接着现在的关节角求逆解,再过一道甩臂护栏。

    求不出、或者某一轴要转超过 `VIA_JOINTS_MAX_DEG`,抛 `MotionError`——一条都不发,
    由状态逻辑进已停止。只在机械臂不动时调(要读一次关节角)。
    """
    here = arm.joints()
    joints = arm.solve_ik(segment.pose, here)
    if joints is None:
        raise MotionError(f"{segment.label}:求不出关节角(够不着或在奇异点上),没发")
    swing = max(abs(a - b) for a, b in zip(joints, here, strict=True))
    if swing > VIA_JOINTS_MAX_DEG:
        raise MotionError(
            f"{segment.label}:某一轴要转 {swing:.1f}°(上限 {VIA_JOINTS_MAX_DEG:.0f}°),"
            "多半是逆解换了一支构型,没发"
        )
    return joints


def settle(arm: Arm, segment: Segment) -> Segment:
    """发之前把这一段定下来:要走关节运动的求出关节角,要按尖端速度走的算出关节速度。

    返回**实际要发的**那一段——算时限、等它走完、核对到位,都按它来。要读一次关节角,
    所以只在机械臂不动时调(正好是「上一段走完、这一段还没发」那一下)。

    - `via_joints` 的直线段不走直线:求出关节角,换成一段关节运动(`LinearSegment` 上的
      说明),尖端速度就是原来那段直线的 `speed`——点头和左右上下一样快;
    - 给了 `tip_speed_mms` 的关节段:按尖端要走多远换算成 °/s(`Arm.joint_speed_for`),
      算不出来就照旧用它自己写的 °/s。
    """
    if isinstance(segment, LinearSegment) and segment.via_joints:
        segment = JointSegment(
            label=segment.label,
            joints=tuple(joints_for(arm, segment)),
            speed=VIA_JOINTS_DEGS,
            pause_s=segment.pause_s,
            tip_speed_mms=segment.speed,
        )
    if isinstance(segment, JointSegment) and segment.tip_speed_mms is not None:
        degs = arm.joint_speed_for(arm.joints(), segment.joints, segment.tip_speed_mms)
        if degs is not None:
            segment = replace(segment, speed=degs)
    return segment


def send(arm: Arm, segment: Segment) -> Segment:
    """发出一段运动:先定下来(`settle`),再设这一段的速度、非阻塞发出。
    返回**实际发出去的**那一段。速度真机上发不发得出去,看机械臂配置的 `program_sets_speed`。
    """
    segment = settle(arm, segment)
    _dispatch(arm, segment)
    return segment


def _dispatch(arm: Arm, segment: Segment) -> None:
    """设速度、发出去。`segment` 得是已经定下来的(`settle` 过的)。"""
    if isinstance(segment, LinearSegment):
        arm.set_speed(linear_mms=segment.speed)
        arm.move_linear(segment.pose)
        return
    if isinstance(segment, JointSegment):
        arm.set_speed(joints_degs=segment.speed)
        arm.move_joints(segment.joints)
        return
    raise TypeError(f"认不出的一段运动:{segment!r}")


def run_segments(
    arm: Arm,
    signal: StopSignal,
    segments: Iterable[SegmentCall],
    timeout_s: float,
    poll_s: float = MOVE_POLL_S,
    on_segment: SegmentReport | None = None,
) -> int:
    """逐段走完,返回真正发出去的段数。

    已叫停时立刻返回,**不再发下一段**——所以返回的段数小于总段数,就说明是被
    叫停拦下的。某一段超过 `timeout_s` 还没结束会抛 `ArmTimeoutError`;这里的每一段
    是调用方自己包起来的一个调用,估不出要走多久,所以时限由调用方给。
    """
    steps = ((call, 0.0, None) for call in segments)
    return _run(arm, signal, steps, timeout_s, poll_s, on_segment)


def run_plan(
    arm: Arm,
    signal: StopSignal,
    plan: Sequence[Segment],
    timeout_s: float | SegmentTimeout | None = None,
    poll_s: float = MOVE_POLL_S,
    on_segment: SegmentReport | None = None,
) -> int:
    """按 `robot_side.motion` 拆出来的段逐段走完,返回真正发出去的段数。

    和 `run_segments` 同一个循环,只是每段先定下来(`settle`)、按自己的速度发,走完照
    配置停顿。

    时限**每段单算**:`segment_timeout_s`(估算用时 × 2 + 5 秒),按**定下来以后**的那一段
    算——换算出来的关节速度慢,时限就跟着长。`timeout_s` 给一个数就每段都用它(测试,和
    没开 `program_sets_speed` 的真机);给 `SegmentTimeout` 就按它放慢估算(开了的真机)。
    """
    steps = ((None, segment.pause_s, segment) for segment in plan)
    return _run(arm, signal, steps, timeout_s, poll_s, on_segment)


def _run(
    arm: Arm,
    signal: StopSignal,
    steps: Iterable[tuple[SegmentCall | None, float, Segment | None]],
    timeout_s: float | SegmentTimeout | None,
    poll_s: float,
    on_segment: SegmentReport | None = None,
) -> int:
    sent = 0
    for call, pause_s, segment in steps:
        if signal.raised:
            return sent
        error = arm.error()
        if error is not None:
            # 带着故障不再发下一段;抛出去由上层进已停止(不是被叫停拦下的那一种)
            raise error
        if on_segment is not None:
            on_segment(sent, segment)
        if call is None and segment is not None:
            # 排好的一段:先定下来(求关节角、换算速度),时限和到位核对都按定下来的算
            segment = settle(arm, segment)
        # 时限要在**发出之前**算:算它得读一次「现在停在哪」,发出去之后就读不了了
        if isinstance(timeout_s, (int, float)):
            limit = timeout_s
        elif segment is not None:
            rule = timeout_s or SegmentTimeout()
            limit = max(rule.floor_s, segment_timeout_s(arm, segment, slowdown=rule.slowdown))
        else:
            limit = TIMEOUT_SLACK_S  # 估不出来的一段:只给那 5 秒兜底
        if call is None and segment is not None:
            _dispatch(arm, segment)
        else:
            call()
        sent += 1
        wait_until_done(arm, signal, segment, limit, poll_s)
        if pause_s and signal.wait(pause_s):
            # 停顿途中被叫停:不等它停满,下一段本来也不会发了
            return sent
    return sent

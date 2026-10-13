"""核对程序设的速度在真机上真的受控(换上包过的驱动以后,第一次上真机先跑这个)。

    uv run python -m robot_side.scripts.speed_check --arm ../jaka-real/jaka.toml --real
    uv run python -m robot_side.scripts.speed_check --arm jaka          # 仿真排练

**为什么要先跑这个**:包过的驱动(`robot_side/drivers/jaka_keep_speed.py`)只在仿真和离线
测试里验过。真机上第一次设速度,先用最小的行程验证速度到底受不受控,耗时不对就停下
(`real-arm-script-guardrails`)。9-20 那一次就是连跑五段才发现速度设不了。

同一段运动用两个速度各走一遍,比耗时。速度真的受控时,慢的那一遍明显更久(J6 约三倍、
直线约两倍);没受控(原驱动那样每行都重置成 100)时两遍差不多一样快。两组:

1. **只转 J6**:+10°(关节 30 °/s)再 −10°(10 °/s)转回来。J6 绕指针自己的轴转,尖端几乎不动。
2. **沿工作台 +Z 直线抬 50 mm**(50 mm/s)再落回来(25 mm/s)。只在姿态落在镜像不动点上
   时做(待命位姿、停靠点都在),不然直线会被镜像拧歪、到位核对过不去。

每一段都过护栏:从指令里把目标读回来核对、只动该动的轴、限行程;倒计时之前查上电使能
(查完顺带热身)。按空格随时叫停。走完回到出发的地方,不回待命、不下使能。

实际耗时 = 行程 ÷(程序设的速度 × 示教器上的速度倍率)。倍率 13% 时,第 1 组约 2.6 秒和
7.7 秒,第 2 组约 7.7 秒和 15 秒;再加 `Busy()` 落下晚的那 1–2.6 秒。
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from robodk.robomath import pose_2_xyzrpw, transl

from robot_side.arm import ArmError
from robot_side.beeps import Beeper
from robot_side.config import ConfigError, load_arm_config, resolve_arm_config
from robot_side.executor import wait_until_done
from robot_side.guard import (
    GuardRefused,
    MoveLimits,
    PowerState,
    check_joint_readback,
    check_joint_travel,
    check_linear_travel,
    check_pose_readback,
    check_powered,
)
from robot_side.motion import JointSegment, LinearSegment
from robot_side.real import (
    RealArmRefused,
    check_power_then_warm,
    check_real_allowed,
    connect_real,
)
from robot_side.robodk_arm import KEEP_SPEED_DRIVER, RoboDKArm
from robot_side.runlog import LOG_DIR
from robot_side.station import STATION_FILE
from robot_side.stop import SpaceKeyWatcher, StopSignal

J6_TURN_DEG = 10.0
J6_FAST_DEGS, J6_SLOW_DEGS = 30.0, 10.0
"""差三倍,不是两倍:真机上 `Busy()` 会晚落 1–2.6 秒,差两倍时最坏只剩 1.36 倍,判不清。
30 °/s 也是 JAKA 配置里的关节速度上限。"""
J6_LIMITS = MoveLimits(axes=(5,), max_delta_deg=J6_TURN_DEG + 0.5)
"""第 1 组只准动 J6,最多 10.5°。开跑前定好,不按结果调。"""

LIFT_MM = 50.0
LIFT_FAST_MMS, LIFT_SLOW_MMS = 50.0, 25.0
LIFT_TRAVEL_MM = LIFT_MM + 5.0
LIFT_TURN_DEG = 1.0
LIFT_LIMITS = MoveLimits(axes=(0, 1, 2, 3, 4, 5), max_delta_deg=15.0)
"""第 2 组:直线走不超过 55 mm、转不超过 1°,每轴不超过 15°。"""

FIXED_POINT_TOL_DEG = 0.5
"""基坐标系 RY、RZ 离 0 多远以内才做第 2 组(`robot_side.standby.FIXED_POINT_TOL_DEG`)。"""

MOVE_TIMEOUT_S = 90.0
COUNTDOWN_S = 3
CONTROLLED_RATIO = 1.4
"""慢的那一遍至少是快的这么多倍,才算速度受控。理论上 J6 是 3、直线是 2;`Busy()` 晚落的
那一两秒会把它往 1 拉,倍率 13% 时最坏也在 1.6 左右。没受控时就是 1 左右。"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="核对程序设的速度在真机上真的受控")
    parser.add_argument(
        "--arm", required=True, help="机械臂配置:jaka,或填了 IP 的那份 .toml。**必须明写**"
    )
    parser.add_argument(
        "--real", action="store_true", help="连真机。不加就在 RoboDK 仿真里排练一遍"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_arm_config(resolve_arm_config(args.arm))
        if args.real:
            check_real_allowed(config)
    except ConfigError as exc:
        print(f"配置有问题,什么也没连:\n  {exc}")
        return 2
    except RealArmRefused as exc:
        print(f"不连真机:\n  {exc}")
        return 2
    # 这一支就是来验包过的驱动的:不管配置里写没写,都换上它
    config = dataclasses.replace(config, program_sets_speed=True)

    print(f"机械臂配置:{config.path}({config.name})")
    print(f"驱动:{KEEP_SPEED_DRIVER}(程序设的速度会发给控制柜)")
    if args.real:
        print(
            "\n连真机。先照 docs/demo-checklist.md 过一遍:人和笔记本站开,急停在手里;"
            "\n查一眼有没有上一次留下的 JAKADriver.py / jaka_keep_speed.py 进程。"
        )
    else:
        print("\n没加 --real:在 RoboDK 仿真里排练,不碰真机。")

    started = datetime.now()
    with RoboDKArm(STATION_FILE, config.robot_name) as arm:
        arm.apply_config(config)
        if args.real:
            try:
                # 只连不动;查上电使能和热身放到你确认之后(两件事要成对,见 `real.warm_up`)
                connect_real(arm, config.ip or "")
            except (RealArmRefused, ArmError) as exc:
                print(f"\n停在连真机这一步,一条运动都没发:\n  {exc}")
                return 2
        try:
            code = _check(args, arm, config)
        except (ArmError, GuardRefused) as exc:
            print(f"\n停下了,没做完:\n  {exc}")
            code = 1
    if args.real:
        _show_driver_log(started)
    return code


def _check(args, arm: RoboDKArm, config) -> int:
    here = tuple(arm.joints())
    fixed = _at_fixed_point(arm, here)
    print(f"\n现在的关节角:{_fmt(here)}")
    print("要做的:")
    print(
        f"  1. J6 +{J6_TURN_DEG:g}°(关节 {J6_FAST_DEGS:g} °/s),再转回来({J6_SLOW_DEGS:g} °/s)"
    )
    if fixed:
        print(
            f"  2. 沿工作台 +Z 抬 {LIFT_MM:g} mm({LIFT_FAST_MMS:g} mm/s),"
            f"再落回来({LIFT_SLOW_MMS:g} mm/s)"
        )
    else:
        print("  2. 不做:姿态不在镜像的不动点上,直线会被拧歪。先回待命再跑这一支")
    print("  每一遍的实际速度 = 上面的数 × 示教器上的速度倍率。")
    if not _confirm("确认开始?"):
        print("没动。")
        return 0

    power = _power_now(args, arm, config)
    print(f"  上电使能:{power.describe()}")
    check_powered(power)

    signal = StopSignal(arm)
    beeps = Beeper()
    results: dict[str, tuple[float, float]] = {}
    try:
        with SpaceKeyWatcher(lambda: signal.raise_stop("space")):
            print("  按空格可以随时叫停。")
            for left in range(COUNTDOWN_S, 0, -1):
                print(f"  {left} ...", flush=True)
                beeps.countdown()
                time.sleep(1.0)
                if signal.raised:
                    print("  倒计时里被叫停了,一步没动。")
                    return 0
            results["J6"] = _j6_pair(arm, signal, here)
            if fixed and not signal.raised:
                results["直线"] = _lift_pair(arm, signal)
    finally:
        beeps.close()
    if signal.raised:
        print("\n被叫停了,没做完。机械臂停在原地。")
        return 1
    return _verdict(results)


def _j6_pair(arm: RoboDKArm, signal: StopSignal, here) -> tuple[float, float]:
    turned = list(here)
    turned[5] += J6_TURN_DEG
    fast = _joint_move(arm, signal, here, turned, J6_FAST_DEGS, "J6 转过去")
    slow = _joint_move(arm, signal, turned, list(here), J6_SLOW_DEGS, "J6 转回来")
    return fast, slow


def _joint_move(arm, signal, start, target, degs, label) -> float:
    if signal.raised:
        return math.nan
    readback = arm.stage_joint_target(target)
    check_joint_readback(target, readback)
    check_joint_travel(start, readback, J6_LIMITS)
    arm.set_speed(joints_degs=degs)
    t0 = time.monotonic()
    arm.move_joints_staged()
    segment = JointSegment(label=label, joints=tuple(readback), speed=degs)
    wait_until_done(arm, signal, segment, MOVE_TIMEOUT_S)
    spent = time.monotonic() - t0
    print(f"  {label}:{spent:.2f} 秒(发的是 {degs:g} °/s)")
    return spent


def _lift_pair(arm: RoboDKArm, signal: StopSignal) -> tuple[float, float]:
    start = arm.pose()
    up = transl(0.0, 0.0, LIFT_MM) * start
    fast = _linear_move(arm, signal, up, LIFT_FAST_MMS, "直线抬起")
    slow = _linear_move(arm, signal, start, LIFT_SLOW_MMS, "直线落回")
    return fast, slow


def _linear_move(arm, signal, target, mms, label) -> float:
    if signal.raised:
        return math.nan
    here_joints = arm.joints()
    here_pose = arm.pose()
    staged = arm.stage_target(target)
    check_pose_readback(target, staged)
    check_linear_travel(here_pose, staged, max_mm=LIFT_TRAVEL_MM, max_deg=LIFT_TURN_DEG)
    solved = arm.solve_ik(staged, here_joints)
    if solved is None:
        raise GuardRefused(f"{label}:从读回来的目标求不出逆解,不下发")
    check_joint_travel(here_joints, solved, LIFT_LIMITS)
    arm.set_speed(linear_mms=mms)
    t0 = time.monotonic()
    arm.move_linear_staged()
    segment = LinearSegment(label=label, pose=staged, speed=mms)
    wait_until_done(arm, signal, segment, MOVE_TIMEOUT_S)
    spent = time.monotonic() - t0
    print(f"  {label}:{spent:.2f} 秒(发的是 {mms:g} mm/s)")
    return spent


def _verdict(results: dict[str, tuple[float, float]]) -> int:
    print("\n结果(慢的那一遍 ÷ 快的那一遍,受控时 J6 约 3、直线约 2,没受控时约 1):")
    ok = True
    for name, (fast, slow) in results.items():
        ratio = slow / fast if fast > 0 else math.nan
        good = ratio >= CONTROLLED_RATIO
        ok = ok and good
        print(f"  {name}:{fast:.2f} 秒 → {slow:.2f} 秒,{ratio:.2f} 倍 {'受控' if good else '没受控'}")
    if ok:
        print(
            "\n速度受控。可以在填了 IP 的那份机械臂配置里写上 `program_sets_speed = true`"
            "\n([speed] 一节),main.py --real 就按演示配置里的速度走;快慢总开关还是示教器倍率。"
        )
        return 0
    print(
        "\n速度没受控。别开 program_sets_speed,照旧在示教器上调;"
        "把这一段输出和 data/logs 里的 jaka-driver-*.log 一起留着查。"
    )
    return 1


def _at_fixed_point(arm: RoboDKArm, joints) -> bool:
    _, _, _, _, ry, rz = pose_2_xyzrpw(arm.flange_pose_in_base(joints))
    fold = lambda a: (a + 180.0) % 360.0 - 180.0  # noqa: E731
    return abs(fold(ry)) <= FIXED_POINT_TOL_DEG and abs(fold(rz)) <= FIXED_POINT_TOL_DEG


def _power_now(args, arm, config) -> PowerState:
    """查一次上电使能,**查完顺带热身**(`real.check_power_then_warm`)。仿真里直接当有。"""
    if not args.real:
        return PowerState(powered=True, enabled=True, detail="仿真,没有上电使能这回事")
    return check_power_then_warm(arm, config)


def _show_driver_log(since: datetime) -> None:
    """把这一次包过的驱动记下的几行打出来:SPEED 收到了什么、每段按什么速度走的。"""
    lines = driver_log_lines(since)
    print("\n包过的驱动这一次记下的(data/logs/jaka-driver-*.log):")
    if not lines:
        print("  一行都没有——驱动多半没换成包过的那个。速度的结论不作数。")
    for line in lines:
        print(f"  {line}")


def driver_log_lines(since: datetime, log_dir: Path = LOG_DIR) -> list[str]:
    """包过的驱动在 `since` 之后记下的那几行(它每天一个文件,行首是时分秒)。"""
    path = log_dir / f"jaka-driver-{since:%Y%m%d}.log"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    start = f"{since:%H:%M:%S}"
    return [line for line in text.splitlines() if line[:8] >= start]


def _fmt(joints) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in joints) + "]"


def _confirm(question: str) -> bool:
    """要人手敲 yes 才算数。回车、y、随手一个键都不算。"""
    try:
        answer = input(f"{question} 敲 yes 确认,别的都算不做:").strip().lower()
    except EOFError:
        return False
    return answer == "yes"


if __name__ == "__main__":
    sys.exit(main())

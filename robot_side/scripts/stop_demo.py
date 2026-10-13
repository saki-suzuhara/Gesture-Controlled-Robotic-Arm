"""人工演示:发一段运动,按空格停住。

    uv run python -m robot_side.scripts.stop_demo [--robot "JAKA Zu20"] [--step 2]

只动 RoboDK 仿真里的模型,不连真机。两步:
1. 一段直线运动,看着 RoboDK 的窗口按空格;
2. 一段关节运动,**先切到别的窗口再按空格**——空格由单独的按键线程每 10 ms 用
   Windows 自带的接口查,全局有效,预览窗口不在前台照样停得住(ADR-0012)。

每一步跑完报告:从按下空格到机械臂不动用了多久,停在这一段的百分之几。
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable

from robot_side.arm import MOVE_POLL_S
from robot_side.robodk_arm import JAKA, RoboDKArm
from robot_side.scripts.stop_check import LINE_MM, PASS_S, plan_test_move
from robot_side.station import STATION_FILE
from robot_side.stop import POLL_S, SpaceKeyWatcher, StopSignal

LINE_SPEED_MMS = 15.0
JOINT_DEG = 90.0
JOINT_SPEED_DEGS = 10.0
MOVE_TIMEOUT_S = 120.0


def countdown(seconds: int, text: str) -> None:
    for left in range(seconds, 0, -1):
        print(f"  {text} {left} ...", flush=True)
        time.sleep(1.0)


def run_step(
    arm: RoboDKArm,
    signal: StopSignal,
    title: str,
    prompt: str,
    dispatch: Callable[[], None],
    expected_s: float,
    progress: Callable[[], float],
    step: int,
) -> None:
    """发一段运动,主线程轮询等它结束;按键线程随时可以叫停。"""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(f"  {prompt}")
    print(f"  这一段不叫停的话要走约 {expected_s:.0f} 秒")
    countdown(3, "开始前")

    signal.clear()
    started = time.monotonic()
    dispatch()
    dispatched = time.monotonic()
    print(f"  运动已发出(下发本身用了 {dispatched - started:.3f} 秒),按空格试试")

    last_tick = 0.0
    while arm.is_moving():
        if time.monotonic() - dispatched > MOVE_TIMEOUT_S:
            print("  这一段太久没结束,停下")
            signal.raise_stop("demo/超时")
            break
        elapsed = time.monotonic() - dispatched
        if elapsed - last_tick >= 1.0:
            last_tick = elapsed
            print(f"  走了 {elapsed:>4.1f} 秒 ...", flush=True)
        time.sleep(MOVE_POLL_S)
    not_moving = time.monotonic()

    done = progress()
    if not signal.raised:
        # 没按到就等于这一步什么也没验到,说清楚怎么只重跑这一步
        print(f"  没有按空格,这一段走完了({done:.0%})——这一步没验到")
        print(
            f"  只重跑这一步:uv run python -m robot_side.scripts.stop_demo "
            f"--step {step}"
        )
        return
    record = signal.records[-1]
    stop_s = not_moving - record.flag_at
    verdict = "过" if stop_s <= PASS_S and done < 0.99 else "不过"
    print(
        f"  叫停来自 {record.source};按下到停住 {stop_s:.3f} 秒"
        f"(过关线 {PASS_S} 秒),停在这一段的 {done:.0%} 处 —— {verdict}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按空格叫停的人工演示")
    parser.add_argument("--robot", default=JAKA, help="工作站里机械臂的名字")
    parser.add_argument(
        "--step",
        type=int,
        choices=(1, 2),
        action="append",
        help="只跑某一步(可以给多次);不给就两步都跑",
    )
    args = parser.parse_args(argv)
    steps = set(args.step or (1, 2))

    print("只动 RoboDK 仿真里的模型,不连真机。RoboDK 没开的话这就去开。")
    with RoboDKArm(STATION_FILE, robot_name=args.robot) as arm:
        signal = StopSignal(arm)
        arm.set_speed(linear_mms=LINE_SPEED_MMS)
        move = plan_test_move(arm)
        from_joints = move.from_joints
        origin = arm.pose().Pos()
        print(
            f"机械臂 {arm.robot_name};出发关节角 "
            f"{[round(v, 1) for v in from_joints]},直线方向 {move.direction_name}"
        )

        with SpaceKeyWatcher(lambda: signal.raise_stop("space")) as watcher:
            print(f"按键线程已启动,每 {POLL_S * 1000:.0f} ms 查一次空格")

            # 第 1 步:直线运动,看着 RoboDK 的窗口按
            if 1 in steps:
                run_step(
                    arm, signal,
                    "第 1 步:直线运动 —— 看着 RoboDK 的窗口,随时按空格",
                    f"机械臂会沿直线走 {LINE_MM:.0f} mm。按下空格,它应当停在半路。",
                    lambda: arm.move_linear(move.target),
                    LINE_MM / LINE_SPEED_MMS,
                    lambda: math.dist(arm.pose().Pos(), origin) / LINE_MM,
                    step=1,
                )

            # 第 2 步:关节运动,切到别的窗口再按
            if 2 in steps:
                arm.set_joints(from_joints)
                arm.set_speed(joints_degs=JOINT_SPEED_DEGS)
                target_joints = list(from_joints)
                target_joints[0] += JOINT_DEG
                run_step(
                    arm, signal,
                    "第 2 步:关节运动 —— 运动发出后先切到别的窗口,再按空格",
                    "这一步要证明空格是全局的:焦点不在这个终端上,照样停得住。"
                    "\n  提示:等下面打印出「运动已发出」再切窗口,有 9 秒够用。",
                    lambda: arm.move_joints(target_joints),
                    JOINT_DEG / JOINT_SPEED_DEGS,
                    lambda: abs(arm.joints()[0] - from_joints[0]) / JOINT_DEG,
                    step=2,
                )

            arm.set_joints(from_joints)
            if watcher.errors:
                print(f"\n按键线程记下的错误:{watcher.errors}")

    print("\n演示结束,已断开;仿真速度和原来激活的工作站都已恢复。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

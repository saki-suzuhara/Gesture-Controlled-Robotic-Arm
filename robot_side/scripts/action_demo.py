"""在仿真里按配置把六个预设动作跑一遍(第 02 票的验收)。

    uv run python -m robot_side.scripts.action_demo [--arm jaka|nova5]
                                                    [--actions point_left,nod]
                                                    [--check-only]

做的事,照已定的启动顺序:

1. 读两份 TOML(写错的东西在这一步就报出来,程序不连 RoboDK);
2. 连上 RoboDK 仿真、打开工作站,按机械臂配置把**工作台坐标系和待命位姿**写进去
   (界面上改过的数会被盖掉),并关掉碰撞检查;
3. **可达性检查**:逐路点求逆解并用正解核对、反复迭代,每段直线单独查;不通过就指出
   是哪个预设动作的哪个路点、为什么,然后退出——全程一步不动;
4. 回待命:先沿工作台坐标系直线抬起 50 mm,再关节运动回待命位姿(启动后第一次也走这套);
5. 逐个预设动作:一次只发一段,等上一段结束才发下一段。预设动作做完不回待命位姿
   (2026-09-23 起),所以每个动作之后这里自己回一次待命,下一个照旧从待命位姿出发
   ——第 3 步查的就是从待命点出发的那一轮。

**这一支永远只动仿真**,没有连真机的开关:它是第 02 票的验收脚本,拿来把六个预设
动作在仿真里过一遍。要连真机走 `state_demo --real --arm <哪一台>`(第 05 票)。
空格随时叫停(第 01 票那条路),叫停后
收尾回待命;倒计时和整套状态逻辑是第 03 票。演示配置里的速度只喂给仿真。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from robot_side.arm import ArmError, joint_drift
from robot_side.config import (
    ConfigError,
    load_arm_config,
    load_demo,
    load_demo_config,
    resolve_arm_config,
)
from robot_side.executor import run_plan
from robot_side.motion import tip_position
from robot_side.robodk_arm import RoboDKArm
from robot_side.startup import prepare, returning
from robot_side.station import STATION_FILE
from robot_side.stop import SpaceKeyWatcher, StopSignal


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 RoboDK 仿真里跑六个预设动作")
    parser.add_argument(
        "--arm", default="jaka", help="机械臂配置:jaka / nova5,或者一份 .toml 的路径"
    )
    parser.add_argument("--demo", default="", help="换一份演示配置(.toml 的路径)")
    parser.add_argument("--actions", default="", help="只跑这几个动作,逗号分隔;默认全跑")
    parser.add_argument(
        "--check-only", action="store_true", help="只做写入和可达性检查,一步不动"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        arm_config = load_arm_config(resolve_arm_config(args.arm))
        demo = load_demo_config(Path(args.demo)) if args.demo else load_demo()
    except ConfigError as exc:
        print(f"配置有问题,没有连 RoboDK:\n  {exc}")
        return 2

    print(f"机械臂配置:{arm_config.path.name}({arm_config.name})")
    print(f"演示配置:  {demo.path.name}")
    print(
        f"工作台坐标系:绕 Z {arm_config.worktable.angle_deg:g}°,"
        f"原点平移 {tuple(arm_config.worktable.origin_mm)} mm"
    )
    print(f"待命位姿:  {list(arm_config.standby_joints_deg)}")
    print(
        f"回待命:抬起 {demo.run.lift_mm:g} mm 再关节运动回去,"
        f"指针尖端 {demo.run.return_speed_mms:g} mm/s"
    )
    print("这一支只跑仿真:速度照演示配置里写的走。\n")

    wanted = [a.strip() for a in args.actions.split(",") if a.strip()]
    names = wanted or [a.name for a in demo.actions]
    unknown = [n for n in names if n not in {a.name for a in demo.actions}]
    if unknown:
        print(f"演示配置里没有这些动作:{unknown}")
        return 2

    with RoboDKArm(STATION_FILE, arm_config.robot_name) as arm:
        before = arm.joints()
        prepared = prepare(arm, arm_config, demo)
        tip = tip_position(
            arm.flange_pose(arm_config.standby_joints_deg), arm_config.pointer_offset_mm
        )
        print(
            f"已写入 RoboDK:工作台坐标系 + 待命位姿;工具保持 0。\n"
            f"待命点(法兰,工作台坐标系):"
            f"({prepared.standby_point[0]:.1f}, {prepared.standby_point[1]:.1f}, "
            f"{prepared.standby_point[2]:.1f}) mm"
        )
        print(
            f"待命位姿下指针尖端在:({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f}) mm"
            "(只作参考,不参与运动)\n"
        )

        print(prepared.report.describe())
        after = arm.joints()
        moved = joint_drift(before, after)
        print(f"检查期间机械臂动了 {moved:.4f}°(应当是 0)\n")
        if not prepared.ok:
            return 1
        if args.check_only:
            print("--check-only:到此为止,一步没动。")
            return 0

        signal = StopSignal(arm)
        with SpaceKeyWatcher(lambda: signal.raise_stop("space")):
            print("按空格可以随时叫停。\n")
            print("回待命(启动后第一次)...", flush=True)
            if not _run(arm, signal, returning(arm, arm_config, demo), "回待命"):
                return _stopped(arm, signal, arm_config, demo)

            for name in names:
                action = demo.action(name)
                print(f"\n{action.display}({name}),{len(action.waypoints)} 个路点:", flush=True)
                if not _run(arm, signal, prepared.by_action[name], name):
                    return _stopped(arm, signal, arm_config, demo)
                if not _run(arm, signal, returning(arm, arm_config, demo), "回待命"):
                    return _stopped(arm, signal, arm_config, demo)

        print(f"\n{len(names)} 个预设动作都走完了,机械臂停在待命位姿。")
        return 0


def _run(arm: RoboDKArm, signal: StopSignal, plan, what: str) -> bool:
    """走完一段计划;被叫停就返回 False。"""
    t0 = time.monotonic()
    try:
        # 每段等多久算不对劲由 `run_plan` 自己算:估算用时 × 2 + 5 秒
        sent = run_plan(arm, signal, plan)
    except ArmError as exc:
        print(f"  机械臂一侧出错:{exc}")
        raise
    for segment in plan[:sent]:
        print(f"  {segment.label}")
    print(f"  {what}:发出 {sent}/{len(plan)} 段,用时 {time.monotonic() - t0:.1f} s")
    if sent < len(plan):
        print("  被叫停了,没再发下一段。")
        return False
    return True


def _stopped(arm: RoboDKArm, signal: StopSignal, arm_config, demo) -> int:
    """叫停之后的收尾:放下标记,走一遍回待命(倒计时是第 03 票)。"""
    for record in signal.records:
        print(f"  叫停来自 {record.source},用了 {record.stopped_at - record.flag_at:.3f} s")
    signal.clear()
    print("\n收尾:回待命 ...", flush=True)
    run_plan(arm, signal, returning(arm, arm_config, demo))
    print("机械臂停在待命位姿。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

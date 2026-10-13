"""选待命位姿(Home):读示教器上摆好的姿态,补好 J6,你确认了才写进机械臂配置。

    uv run python -m robot_side.scripts.set_standby --arm ../jaka-real/jaka.toml --real
    uv run python -m robot_side.scripts.set_standby --arm jaka --from-joints 187.723,90.782,98.739,80.480,-89.329,-52.043

**这一支一步都不动机械臂。** 连真机时只做三件事:经驱动连上、读一次关节角、立刻断开。
后面的计算和可达性检查都在 RoboDK 仿真里做——可达性检查要挪仿真里的模型,连着真机时
那样挪会怎样没验过(ADR-0014),所以先断开再查。

步骤:

1. 在示教器上把机械臂挪到你想要的待命位置,**指针朝正下方**;
2. 跑上面第一行。它读回关节角,只转 J6 把姿态对上演示配置的默认姿态——JAKA 上这同时
   也就落在姿态镜像的不动点上(`robot_side.standby`);转 J6 法兰位置不变;
3. 拿补好的那一组在仿真里查一遍六个预设动作够不够得着;
4. 把前后的数都摆给你看,你敲 yes 才把 `[standby] joints_deg` 那一行写进配置,注释不动。

写完以后下次启动 `main.py --real`:活动范围按新待命点重查一遍(配置变了,存下来的结果
不再用,JAKA 约 80 秒);启动回待命会把机械臂从你摆的位置抬 50 mm 再转到新待命位姿——
和你摆的只差 J6 那一下。

不加 `--real` 就读 RoboDK 仿真里模型现在的关节角;`--from-joints` 先把模型摆到给定的
那一组再读,拿来排练「示教器上摆成这样会怎样」。
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from datetime import date

from robot_side.arm import ArmError
from robot_side.config import ConfigError, load_arm_config, load_demo, resolve_arm_config
from robot_side.motion import orientation_pose
from robot_side.real import RealArmRefused, check_real_allowed, connect_real
from robot_side.robodk_arm import RoboDKArm
from robot_side.standby import (
    StandbyRefused,
    needs_fixed_point,
    plan_standby,
    rewrite_standby,
)
from robot_side.startup import prepare
from robot_side.station import STATION_FILE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从示教器上摆好的姿态选待命位姿(Home),确认后写进机械臂配置"
    )
    parser.add_argument(
        "--arm",
        required=True,
        help="机械臂配置:jaka / nova5,或一份 .toml 的路径(连真机时是填了 IP 的那份副本)。"
        "**必须明写**,没有默认值;写进的就是这一份",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="连真机读当前关节角(读完就断开,一步不动)。不加就读 RoboDK 仿真里的模型",
    )
    parser.add_argument(
        "--from-joints",
        type=_joints,
        help="只跑仿真时用:先把模型摆到这组关节角(逗号隔开的六个数)再读",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只算给你看,不写配置",
    )
    args = parser.parse_args(argv)
    if args.real and args.from_joints is not None:
        parser.error("--from-joints 只用于仿真排练,不能和 --real 一起用")
    return args


def _joints(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"读不懂 {text!r}:要逗号隔开的六个数") from None
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise argparse.ArgumentTypeError(f"要六个数,读到的是 {len(values)} 个")
    return values


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_arm_config(resolve_arm_config(args.arm))
        demo = load_demo()
        if args.real:
            check_real_allowed(config)
    except ConfigError as exc:
        print(f"配置有问题,什么也没连:\n  {exc}")
        return 2
    except RealArmRefused as exc:
        print(f"不连真机:\n  {exc}")
        return 2

    print(f"机械臂配置:{config.path}({config.name})")
    print(f"现在配置里的待命位姿:{_fmt(config.standby_joints_deg)}")
    if args.real:
        print(
            "\n连真机,但一步不动:连上 → 读一次关节角 → 断开。"
            "\n先确认机械臂已经在示教器上摆好、停稳了,指针朝正下方。"
        )
    else:
        print("\n没加 --real:读 RoboDK 仿真里的模型,不碰真机。")

    with RoboDKArm(STATION_FILE, config.robot_name) as arm:
        arm.apply_config(config)  # 按已定顺序,写配置排在连驱动之前(用户故事 29)
        try:
            picked = _read_picked(args, arm, config)
        except (RealArmRefused, ArmError) as exc:
            print(f"\n没读到关节角,配置没动:\n  {exc}")
            return 2
        return _choose(args, arm, config, demo, picked)


def _read_picked(args, arm: RoboDKArm, config) -> tuple[float, ...]:
    if not args.real:
        if args.from_joints is not None:
            arm.set_joints(args.from_joints)  # 只摆仿真里的模型
        return tuple(arm.joints())
    connect_real(arm, config.ip or "")
    try:
        return tuple(arm.joints())  # 连着真机时会先从驱动把模型同步过来
    finally:
        # 读完就断:后面查可达性要挪仿真里的模型,连着真机时不这么做(ADR-0014)
        arm.disconnect_driver()
        print("读完了,已断开真机驱动。后面都在仿真里算。")


def _choose(args, arm: RoboDKArm, config, demo, picked: tuple[float, ...]) -> int:
    low, high = arm.joint_limits()
    try:
        plan = plan_standby(
            picked,
            flange_in_worktable=arm.flange_pose,
            flange_in_base=arm.flange_pose_in_base,
            default_orientation=orientation_pose(demo.default_orientation_deg),
            fixed_point=needs_fixed_point(config),
            j6_limits=(low[5], high[5]),
        )
    except StandbyRefused as exc:
        print(f"\n算不了,配置没动:\n  {exc}")
        return 1

    old_point = arm.flange_pose(config.standby_joints_deg).Pos()
    print("\n读到的(示教器上摆的):")
    print(f"  关节角:{_fmt(plan.picked)}")
    print(f"  基坐标系姿态:{_rpy(plan.base_rpy_before)}")
    print(f"\nJ6 补 {plan.j6_turn_deg:+.3f}°,要写进配置的是:")
    print(f"  关节角:{_fmt(plan.joints)}")
    print(f"  基坐标系姿态:{_rpy(plan.base_rpy_after)}")
    print(f"  和默认姿态还差 {plan.tilt_deg:.2f}°;转 J6 让法兰挪了 {plan.moved_mm:.3f} mm")
    print(
        f"  待命点(法兰,工作台坐标系):{_point(old_point)} → {_point(plan.standby_point)},"
        f"挪了 {math.dist(old_point, plan.standby_point):.1f} mm"
    )
    if not plan.ok:
        print("\n这个位姿当不了待命位姿,配置没动:")
        for problem in plan.problems:
            print(f"  - {problem}")
        return 1

    print("\n在仿真里查一遍:从新待命点出发,六个预设动作够不够得着……")
    prepared = prepare(
        arm, dataclasses.replace(config, standby_joints_deg=plan.joints), demo
    )
    if not prepared.ok:
        print(f"够不着,配置没动:\n{prepared.report.describe()}")
        return 1
    print(f"  都够得着:{prepared.report.describe()}")

    if args.dry_run:
        print("\n--dry-run:到此为止,配置没动。")
        return 0
    if not _confirm(f"把这一组写进 {config.path.name} 的 [standby] joints_deg?"):
        print("没写。配置没动。")
        return 0
    # 按字节读写,不经文本模式:原来的换行符(LF 还是 CRLF)原样留着
    text = config.path.read_bytes().decode("utf-8")
    try:
        written = rewrite_standby(
            text, plan.joints, stamp=f"{date.today():%Y-%m-%d}", picked_j6=plan.picked[5]
        )
    except StandbyRefused as exc:
        print(f"\n没写成:\n  {exc}")
        return 1
    config.path.write_bytes(written.encode("utf-8"))
    print(f"已写进 {config.path}(只改了 joints_deg 那一行;上面那段说明写的还是原来的 Home)。")
    print(
        "\n下次启动 main.py --real:"
        "\n  1. 活动范围按新待命点重查一遍(配置变了,存下来的结果不再用,JAKA 约 80 秒);"
        f"\n  2. 启动回待命:从你摆的位置先抬 50 mm,再转到新待命位姿(J6 {plan.j6_turn_deg:+.1f}°)。"
    )
    return 0


def _fmt(joints) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in joints) + "]"


def _rpy(rpy) -> str:
    rx, ry, rz = rpy
    return f"RX={rx:+.3f} RY={ry:+.3f} RZ={rz:+.3f}"


def _point(p) -> str:
    return "(" + ", ".join(f"{v:.1f}" for v in p) + ")"


def _confirm(question: str) -> bool:
    """要人手敲 yes 才算数。回车、y、随手一个键都不算。"""
    try:
        answer = input(f"{question} 敲 yes 确认,别的都算不做:").strip().lower()
    except EOFError:
        return False
    return answer == "yes"


if __name__ == "__main__":
    sys.exit(main())

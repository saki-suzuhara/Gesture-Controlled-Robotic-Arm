"""标定工作台坐标系:读一次当前法兰位姿,算出角度,你确认了才写进机械臂配置。

    uv run python -m robot_side.scripts.calibrate_worktable --arm jaka --real
    uv run python -m robot_side.scripts.calibrate_worktable --arm jaka --dry-run

**这是一次性的工具,不属于主程序流程**(spec「模块划分」)。它做四件事:

1. 读当前法兰位姿(在**基坐标系**里,不是工作台坐标系——那正是还没定的东西);
2. 算出工作台坐标系绕 Z 转过的角度,**先显示给你**(用户故事 34);
3. 你敲 yes 确认,才把这一个数写进机械臂配置(只改那一行,注释一行不动);
4. 写完做一次**验证移动**:按回车确认 → 倒计时 3 秒 → 朝操作者慢速走 50 mm
   (用户故事 35)。

**标之前先把机械臂手动挪好:**从底座转轴看过去,法兰要和操作位在同一个方位上,
离转轴远一点(至少 `calibrate.MIN_RADIUS_MM`)。标出来的角度就是这条线和工作台
坐标系 −Y 的关系。

**上真机之前**(`docs/demo-checklist.md`):人和笔记本在底座转轴 2.0 m 以外;实体
急停拿在手里;**在示教器上先把速度和加速度调到慢速**——程序不设速度、不管倍率
(ADR-0011),这一段走多快完全看示教器上设的值。

**写配置那一下会先把真机驱动断开再连回来:**写配置要调 `setPoseFrame` /
`setPoseTool`,而 RoboDK 的 JAKA 驱动里有 `set_tool_id` / `set_user_frame_id`,
连着写有可能改掉控制柜的工具 0 和 0 号坐标系(用户故事 29)。断开—写—再连,和已定
启动顺序里「按配置写入」排在「经驱动连接」之前是同一条道理。连回来时会再查一次
上电使能,并顺带热身一次(两件事必须成对,见 `real.check_power_then_warm`)。

**这一支为什么不跑可达性检查:**可达性检查是拿工作台坐标系去算路点的,而这会儿
那个角度恰恰还是错的,查出来的结果没有意义。它唯一会动的那一下由护栏兜着
(`robot_side.guard`):从指令里把目标读回来核对、逐轴限行程、直线限距离和转角、
倒计时之前查一次上电使能。主程序那边的顺序不变,可达性检查照旧在连真机之前。

不加 `--real` 就只动 RoboDK 仿真里的模型:整套流程照走,好在没有真机的时候排练。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path


from robot_side.arm import ArmError
from robot_side.beeps import Beeper
from robot_side.calibrate import (
    VERIFY_COUNTDOWN_S,
    VERIFY_MOVE_MM,
    CalibrationRefused,
    check_verify_move,
    rewrite_angle,
    verify_target,
    worktable_angle_deg,
)
from robot_side.config import ConfigError, load_arm_config, resolve_arm_config
from robot_side.guard import GuardRefused, PowerState
from robot_side.real import (
    RealArmRefused,
    bring_up_real,
    check_power_then_warm,
    check_real_allowed,
)
from robot_side.robodk_arm import RoboDKArm
from robot_side.station import STATION_FILE
from robot_side.stop import SpaceKeyWatcher, StopSignal

MOVE_TIMEOUT_S = 60.0
"""等这一段走完最多等多久。真机速度由示教器决定,所以给得宽。"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="标定工作台坐标系的角度,确认后写进机械臂配置"
    )
    parser.add_argument(
        "--arm",
        required=True,
        help="机械臂配置:jaka / nova5,或一份 .toml 的路径。**必须明写**,没有默认值",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="连真机读当前法兰位姿。不加就只读 RoboDK 仿真里的模型(排练用)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只算给你看,不写配置、不做验证移动",
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

    print(f"机械臂配置:{config.path}({config.name})")
    print(f"现在配置里的工作台坐标系角度:{config.worktable.angle_deg:g}°")
    print(f"原点平移:{tuple(config.worktable.origin_mm)} mm")
    if args.real:
        print(
            "\n连真机。先照 docs/demo-checklist.md 过一遍:"
            "人和笔记本在底座转轴 2.0 m 以外,实体急停在手,"
            "示教器上已把速度和加速度调到慢速(程序不设速度)。"
        )
    else:
        print("\n没加 --real:只读 RoboDK 仿真里的模型,不碰真机。")

    with RoboDKArm(STATION_FILE, config.robot_name) as arm:
        arm.apply_config(config)  # 先按现有配置写进去,读回来的位姿才有个已知的出发点
        if args.real:
            try:
                bring_up_real(arm, config)
            except (RealArmRefused, GuardRefused) as exc:
                print(f"\n停在连真机这一步,一条运动都没发:\n  {exc}")
                return 2
        return _calibrate(args, arm, config)


def _calibrate(args, arm: RoboDKArm, config) -> int:
    joints = arm.joints()  # 只在不动时读(`robot_side.arm`)
    in_base = arm.flange_pose_in_base(joints)
    x, y, z = in_base.Pos()
    print("\n读到的当前姿态:")
    print(f"  关节角:{[round(v, 3) for v in joints]}")
    print(f"  法兰(基坐标系):({x:.1f}, {y:.1f}, {z:.1f}) mm")

    try:
        angle = worktable_angle_deg(in_base, config.worktable.origin_mm)
    except CalibrationRefused as exc:
        print(f"\n标不了:\n  {exc}")
        return 1

    was = config.worktable.angle_deg
    print("\n算出来的工作台坐标系角度:")
    print(f"  {angle:.4f}°(配置里现在是 {was:g}°,差 {angle - was:+.4f}°)")
    print(
        "  意思是:按这个角度建坐标系,现在法兰所在的那条线就是工作台坐标系的 −Y,"
        "\n  也就是「从底座看向操作者」的方向。"
    )
    if args.dry_run:
        print("\n--dry-run:到此为止,配置没动,机械臂一步没动。")
        return 0

    if not _confirm(f"把 {angle:.4f}° 写进 {config.path.name}?"):
        print("没写。配置没动,机械臂一步没动。")
        return 0

    text = config.path.read_text(encoding="utf-8")
    try:
        config.path.write_text(
            rewrite_angle(text, angle, stamp=f"{date.today():%Y-%m-%d}"),
            encoding="utf-8",
        )
    except CalibrationRefused as exc:
        print(f"\n没写成:\n  {exc}")
        return 1
    print(f"已写进 {config.path}(只改了 angle_deg 那一行)。")

    # 把新角度写进 RoboDK,验证移动才走在新坐标系里。**写之前先把真机驱动断开**:
    # 写配置会调 `setPoseFrame` / `setPoseTool`,连着真机时可能被驱动 push 到控制柜,
    # 改掉那里的工具 0 和 0 号坐标系(`RoboDKArm.apply_config`、用户故事 29)。
    # 这也正是已定的启动顺序里「按配置写入」排在「经驱动连接」前面的原因。
    updated = load_arm_config(config.path)
    if args.real:
        print("\n先断开真机驱动再写配置(不改控制柜的工具 0 和 0 号坐标系)……")
        arm.disconnect_driver()
    arm.apply_config(updated)
    if args.real:
        print("写完了,重新连上真机(会再查一次上电使能)……")
        try:
            bring_up_real(arm, updated)
        except (RealArmRefused, GuardRefused) as exc:
            print(f"\n配置已经写好了,但没能连回真机,验证移动没做:\n  {exc}")
            return 1
    return _verify_move(args, arm, updated)


def _verify_move(args, arm: RoboDKArm, config) -> int:
    """验证移动:按键确认 → 倒计时 3 秒 → 朝操作者慢速走一小段(用户故事 35)。"""
    print(
        f"\n验证移动:沿工作台坐标系 −Y(朝操作者)走 {VERIFY_MOVE_MM:g} mm。"
        "\n角度标对了,机械臂就是朝你这条线走过来;走岔了说明角度不对。"
    )
    if args.real:
        print(
            "  真机会动。再确认一遍:人和笔记本站开了,急停在手里,"
            "示教器上的速度是慢的。"
        )
    if not _confirm("确认开始?"):
        print("没动。")
        return 0

    # **查上电使能要排在倒计时之前,而且查完必须热身**(`real.check_power_then_warm`)。
    # 2026-09-22 真机实测:查使能要用 jkrc 登录控制柜,那一下会把 RoboDK 驱动的会话
    # 搅了,之后第一条运动指令要等 20–24 秒才执行。原来是「倒计时完、临发之前再查一次」
    # ——那样数完 3、2、1 之后机械臂还要杵二十秒才动,人早以为没反应走开了。
    #
    # 代价说明白:这样一来,倒计时那三秒里人按了急停就查不出来了。那几秒靠的是你手里
    # 的急停本身,不是这一次检查——真按下去,控制柜会拒了这一段,程序按机械臂报错处理。
    power = _power_now(args, arm, config)
    print(f"  上电使能:{power.describe()}")

    beeps = Beeper()
    signal = StopSignal(arm)
    try:
        with SpaceKeyWatcher(lambda: signal.raise_stop("space")):
            print("  按空格可以随时叫停。")
            for left in range(VERIFY_COUNTDOWN_S, 0, -1):
                print(f"  {left} ...", flush=True)
                beeps.countdown()
                time.sleep(1.0)
                if signal.raised:
                    print("  倒计时里被叫停了,一步没动。")
                    return 0
            return _send_guarded(args, arm, config, signal, power)
    except (ArmError, GuardRefused) as exc:
        print(f"\n没走成:\n  {exc}")
        return 1
    finally:
        beeps.close()


def _send_guarded(args, arm: RoboDKArm, config, signal: StopSignal, power: PowerState) -> int:
    """带护栏地发出那一段:核对过了才下发(`robot_side.guard`)。

    顺序照 spec 第 51 条,每一步卡住的都是**读回来的那个值**:

    1. `power`:倒计时**之前**查的那一次上电使能(查完顺带热身了,见 `_verify_move`);
    2. 把目标放进 RoboDK,再从那个目标身上**读回来**,和打算发的逐项核对;
    3. 拿**读回来的**那个目标求逆解,逐轴核对行程;
    4. 直线段另外卡走过的距离和转过的角度。
    """
    here_joints = arm.joints()
    here_pose = arm.pose()  # 工作台坐标系里,只在不动时读
    wanted = verify_target(here_pose, VERIFY_MOVE_MM)
    staged = arm.stage_target(wanted)  # 放进去,再从那个目标身上读回来
    solved = arm.solve_ik(staged, here_joints)  # 求的是**读回来的**那个目标
    check_verify_move(
        power=power,
        here_joints=here_joints,
        here_pose=here_pose,
        wanted=wanted,
        staged=staged,
        solved=solved,
    )
    print(f"  护栏过了:逐轴要动 {_deltas(here_joints, solved)}")

    if signal.raised:
        print("  已叫停,不下发。")
        return 0

    # 速度只喂给仿真:真机走多快完全看示教器上设的值(ADR-0011)
    arm.set_speed(linear_mms=demo_lift_speed(config))
    arm.move_linear_staged()
    arm.wait_until_not_moving(MOVE_TIMEOUT_S)

    after = arm.joints()
    print(f"  走完了。逐轴实际动了 {_deltas(here_joints, after)}")
    print(
        "\n标定完成。配置里的角度已经是新的;机械臂停在原地,没有回待命、没有下使能。"
    )
    return 0


def _power_now(args, arm, config) -> PowerState:
    """查一次上电使能,**查完顺带热身**(两件事必须成对,见 `real.check_power_then_warm`)。

    仿真里没有上电使能这回事,也没有那笔延迟,所以直接当有。
    """
    if not args.real:
        return PowerState(powered=True, enabled=True, detail="仿真,没有上电使能这回事")
    return check_power_then_warm(arm, config)


def demo_lift_speed(config) -> float:
    """验证移动在仿真里用多慢。**只对仿真有效**,真机的速度在示教器上调。"""
    return min(25.0, config.max_linear_mms)


def _deltas(before, after) -> str:
    return "、".join(f"{a - b:+.3f}°" for b, a in zip(before, after, strict=True))


def _confirm(question: str) -> bool:
    """要人手敲 yes 才算数。回车、y、随手一个键都不算。"""
    try:
        answer = input(f"{question} 敲 yes 确认,别的都算不做:").strip().lower()
    except EOFError:
        return False
    return answer == "yes"


if __name__ == "__main__":
    sys.exit(main())

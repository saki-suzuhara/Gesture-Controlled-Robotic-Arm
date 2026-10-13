"""把整套状态逻辑跑一遍,不开摄像头(第 03 票的验收)。

    uv run python -m robot_side.scripts.state_demo --arm noarm --debug
    uv run python -m robot_side.scripts.state_demo --arm jaka  --debug
    uv run python -m robot_side.scripts.state_demo --arm noarm --script full
    uv run python -m robot_side.scripts.state_demo --arm jaka  --real --debug

走的是「触发 → 执行中 → 叫停 → 倒计时 → 抬起 → 回待命 → 冷却 → 空闲」,以及
「回待命途中再叫停 → 已停止 → 回车 → 回待命」。事件有两个来源,走的是同一套逻辑:

- **调试按键**(`--debug`):方向键 = 握拳平移四个方向,P = 比 V,L = 竖拇指;
  按的是触发手势,对应哪个预设动作查演示配置。不加 `--debug` 就没有这几个键,
  免得正式演示时被误按。
- **编好的事件序列**(`--script full` / `--script actions`):不用人按键,自己走完。

`--arm noarm` 用空实现:**不开 RoboDK**,每段固定等 1 秒、途中能被叫停打断,上面
整套照样走得通。`--arm jaka` / `--arm nova5` 走 RoboDK 仿真。

**默认只动仿真。** 加了 `--real` 才连真机,而且这时 `--arm` 必须明写是哪一台
(用户故事 1、2)。连真机时按已定顺序走:打开工作站 → 按配置写入 → 可达性检查
(还没连真机,不通过就不碰)→ 经驱动连接 → **检查上电使能**(只查不做,没有就停在
这一步、一条运动都不发)→ 回待命 → 空闲。上真机之前照 `docs/demo-checklist.md`
过一遍:人和笔记本在底座转轴 2.0 m 以外,实体急停在手,示教器上先把速度和加速度
调到慢速——机械臂配置没写 `program_sets_speed = true` 时**程序不设速度**(ADR-0011);
写了的话按演示配置里的速度发,实际再乘示教器倍率。

空格随时叫停,预览窗口卡住、不在前台、正被拖着都算(ADR-0012);回车只在已停止时
有效;q / Esc 或关掉窗口就退出。日志同时打到终端和 `data/logs/`。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from robot_side.arm import NotConnectedError, open_arm
from robot_side.beeps import Beeper
from robot_side.config import (
    ArmConfig,
    ConfigError,
    load_arm_config,
    load_demo,
    load_demo_config,
    resolve_arm_config,
)
from robot_side.guard import GuardRefused
from robot_side.preview import PreviewWindow, debug_key_pairs
from robot_side.range_cache import check_range_cached
from robot_side.real import (
    LINK_POLL_S,
    LinkMonitor,
    RealArmRefused,
    bring_up_real,
    check_real_allowed,
    choose_arm,
    real_segment_timeout,
)
from robot_side.runlog import close as close_log
from robot_side.runlog import start as start_log
from robot_side.scripted import SCRIPTS, ScriptPlayer
from robot_side.startup import prepare
from robot_side.state import Event, Machine, State
from robot_side.stop import SpaceKeyWatcher, StopSignal

LOOP_S = 0.02
"""主循环每轮歇多久。倒计时和画面按这个刷新;叫停不走这里,不受它影响。"""

SIM_DEFAULT_ARM = "noarm"
"""`--arm` 不写时用哪一份。

这一支默认空实现,不开 RoboDK:改状态逻辑时省一次开 RoboDK(用户故事 40)。
**加了 `--real` 就没有默认值可用**,必须明说是哪一台(用户故事 2)。
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把整套状态逻辑跑一遍,不开摄像头")
    parser.add_argument(
        "--arm",
        default="",
        help=(
            "机械臂配置:noarm(空实现,不开 RoboDK)/ jaka / nova5,或一份 .toml 的"
            f"路径。不写就是 {SIM_DEFAULT_ARM}(空实现,只在仿真里动);加了 --real 就必须写"
        ),
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help=(
            "连真机。不加就只在 RoboDK 仿真里动。加了必须同时用 --arm 明确指定是哪一台;"
            "上真机之前照 docs/demo-checklist.md 过一遍"
        ),
    )
    parser.add_argument("--demo", default="", help="换一份演示配置(.toml 的路径)")
    parser.add_argument(
        "--debug", action="store_true", help="打开调试用的触发按键(方向键、P、L)"
    )
    parser.add_argument(
        "--script",
        default="",
        choices=["", *SCRIPTS],
        help="改用编好的事件序列驱动,不用人按键",
    )
    parser.add_argument(
        "--no-window", action="store_true", help="不开预览窗口,只看终端(调脚本时用)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        # 默认只在仿真里动:加了 --real 才连真机,而且必须明说是哪一台(用户故事 1、2)
        which = choose_arm(args.arm, real=args.real, default=SIM_DEFAULT_ARM)
        arm_config = load_arm_config(resolve_arm_config(which))
        demo = load_demo_config(Path(args.demo)) if args.demo else load_demo()
        if args.real:
            check_real_allowed(arm_config)
    except ConfigError as exc:
        print(f"配置有问题,什么也没连:\n  {exc}")
        return 2
    except RealArmRefused as exc:
        print(f"不连真机:\n  {exc}")
        return 2

    log, _ = start_log("state")
    log.info("机械臂配置 %s(%s),驱动 %s", arm_config.path.name, arm_config.name, arm_config.driver)
    log.info("演示配置 %s,%d 个预设动作", demo.path.name, len(demo.actions))
    if args.debug:
        log.info("调试按键已打开:%s", ",".join(debug_key_pairs(demo)))
    if args.script:
        log.info("由编好的事件序列驱动:%s", args.script)

    arm = open_arm(arm_config)
    try:
        # 已定的启动顺序:打开工作站 → 按配置写入 → 可达性检查(这时还没连真机)
        #   → 要连真机才经驱动连接 → 检查上电使能 → 回待命 → 空闲
        arm.connect()
        prepared = prepare(arm, arm_config, demo)
        log.info("%s", prepared.report.describe())
        if not prepared.ok:
            log.error("可达性检查不通过,机械臂一步没动。")
            return 1  # 不通过就不碰真机(第 05 票第 2 条)
        reachable = None
        if args.real:
            # 连真机之前把活动范围全查完,连上以后运行中不再查(ADR-0014:那一查要挪
            # RoboDK 里的模型,连着真机时没验过)。配置和模型都没变就用上次存下来的结果
            checked, _ = check_range_cached(
                arm,
                arm_config,
                demo,
                prepared,
                on_progress=lambda done, total: (done % 5 == 0 or done == total)
                and log.info("查活动范围:%d / %d 个停靠点(还没连真机,机械臂不动)", done, total),
                log=log,
            )
            log.info("%s", checked.describe())
            reachable = checked.reachable
            try:
                bring_up_real(arm, arm_config, log=log)
            except (RealArmRefused, GuardRefused) as exc:
                log.error("停在连真机这一步,一条运动都没发:%s", exc)
                return 2
        return _run(args, arm, arm_config, demo, prepared, log, reachable)
    finally:
        arm.disconnect()  # 先停住当前运动再断开(机械臂接口保证)
        close_log(log)


def _run(args, arm, arm_config, demo, prepared, log, reachable=None) -> int:
    signal = StopSignal(arm)
    monitor = _link_monitor(args, arm, arm_config, signal, log)
    beeps = Beeper()
    machine = Machine(
        arm,
        signal,
        plans=prepared.by_action,
        standby_point=prepared.standby_point,
        arm_config=arm_config,
        demo=demo,
        beeps=beeps,
        log=log,
        on_clear_records=lambda: log.info("清空判定记录(视觉部分接上之前只记一条)"),
        # 真机上每段等多久另算(`real.real_segment_timeout`)
        segment_timeout_s=real_segment_timeout(arm_config) if args.real else None,
        reachable=reachable,
        check_ahead=not args.real,
    )
    player = (
        ScriptPlayer(
            SCRIPTS[args.script],
            machine,
            signal,
            log=log,
            # 连着真机时进了已停止一次都不替人按回车:人在现场,由人决定
            confirm_halts=not args.real,
        )
        if args.script
        else None
    )
    window = None
    if not args.no_window:
        window = PreviewWindow(
            demo,
            on_confirm=lambda source: machine.submit(Event.confirm(source)),
            on_quit=lambda source: machine.submit(Event.quit(source)),
            on_home=lambda source: machine.submit(Event.home(source)),
            on_trigger=(
                (lambda action, source: machine.submit(Event.trigger(action, source)))
                if args.debug
                else None
            ),
        )

    try:
        # 按键线程:空格全局有效,预览窗口卡住、不在前台都照样叫停(ADR-0012)
        with SpaceKeyWatcher(lambda: signal.raise_stop("space")) as keys:
            machine.start()
            next_link_look = 0.0
            while not machine.finished:
                machine.tick()
                if monitor is not None and time.monotonic() >= next_link_look:
                    if _look_at_link(monitor, machine):
                        next_link_look = time.monotonic() + LINK_POLL_S
                if player is not None:
                    player.pump()
                if window is not None:
                    window.update(machine.view)
                time.sleep(LOOP_S)
            for error in keys.errors:
                log.warning("按键线程里出过错:%s", error)
    finally:
        if window is not None:
            window.close()
        machine.close()
        beeps.close()
    log.info("退出。机械臂停在原地,没有自动回待命。")
    return 0


QUIET_STATES = (State.IDLE, State.COOLDOWN, State.HALTED)
"""机械臂肯定停着的那几个状态。只在这几个状态里看真机那条连接(`real.LINK_POLL_S`)。"""


def _link_monitor(args, arm, arm_config, signal, log) -> LinkMonitor | None:
    """连着真机才盯梢。判定掉线就挂上「断开」这个报错,再叫停。

    挂报错是为了让状态逻辑下一轮 `tick()` 看见并进已停止(用户故事 31);叫停是
    ADR-0012 那条规矩:谁先发现要停,谁就直接调机械臂接口的「停止」。
    """
    if not args.real:
        return None

    def lost(why: str) -> None:
        log.error("%s", why)
        arm.raise_fault(NotConnectedError(why))
        signal.raise_stop("link")

    return LinkMonitor(arm, arm_config.ip or "", on_lost=lost)


def _look_at_link(monitor: LinkMonitor, machine: Machine) -> bool:
    """机械臂停着的时候才看一眼真机的连接;返回「这一轮看了没有」。

    运动中不看:那一下要占着叫停也要用的那条连接的锁,而运动中真掉线了,下一条
    指令自己就会报出来(`robot_side.robodk_arm` 翻成「断开」)。
    """
    if machine.state not in QUIET_STATES:
        return False
    monitor.poll()
    return True


if __name__ == "__main__":
    sys.exit(main())

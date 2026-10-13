"""正式程序:站着做手势,机械臂做对应的预设动作(`main.py` 调的就是这里)。

启动顺序照 spec「启动顺序」,一步都不能换:

1. 打开摄像头;
2. 最多等约 5 秒拿到第一帧;
3. 要求连续约 1 秒都有识别结果,且画面不是均匀的;
4. **启动对准**:站到操作位,把拳头举起来停约 2 秒,程序放下检测区域和起点区;
5. **这时才连机械臂** → 按配置写入 → 可达性检查 → 回待命 → 冷却 → 空闲。
   连真机(`--real`)时在可达性检查和回待命之间多几步,顺序照机器人一侧第 05 票:
   **把活动范围全查一遍(还没连真机)→ 经驱动连接控制柜 → 检查上电使能(只查不做)
   → 热身**(一条零行程的关节运动,付掉连上之后第一条指令那约 20 秒)。

**前四步一次都不碰机械臂。**哪一步不对就报错退出,并在屏幕和日志里写清是哪一步
——摄像头打不开、迟迟没画面、识别不正常、对不准,四种各有各的退出码和说明。

线程分工(spec「线程」):

- **识别线程**:取帧 → 识别 → 触发判定 → 结果搁给主线程;认出触发手势排进状态
  逻辑的事件队列,认出停止手势**就地叫停**(`vision_side.loop`);
- **主线程**:状态逻辑、预览窗口、键盘、识别中断判定,外加把最新一帧画出来;
- **执行线程、按键线程、提示音线程**:机器人一侧本来就有,照旧。

用法:

    uv run python main.py                      # 空实现,不开 RoboDK(先把视觉这半边跑通)
    uv run python main.py --arm jaka           # RoboDK 仿真
    uv run python main.py --arm nova5 --debug  # 再加上调试按键
    uv run python main.py --arm <填了控制柜 IP 的 jaka.toml 副本> --real   # 真机

`--arm` 不给就是空实现:会动的那两档要明写是哪一台,和机器人一侧的规矩一致。
**加了 `--real` 就必须用 `--arm` 明写是哪一台**,而且那份配置要填了控制柜 IP、查得了
上电使能(`robot_side.real.check_real_allowed`);IP 不进仓库,填在仓库外的副本里。
上真机之前照 `docs/demo-checklist.md` 过一遍。

预览窗口怎么摆(`--window`):仿真时默认**并排**——预览窗口占左半屏,RoboDK 挪到
右半屏,好看得见仿真里的机械臂;连真机和空实现默认**整屏**(连真机时看的是真机)。
给了就照给的。
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from robot_side.arm import Arm, NotConnectedError, open_arm
from robot_side.beeps import Beeper
from robot_side.config import (
    ArmConfig,
    ConfigError,
    DemoConfig,
    load_arm_config,
    load_demo,
    load_demo_config,
    over_linear_cap,
    resolve_arm_config,
)
from robot_side.guard import GuardRefused
from robot_side.preview import (
    DEBUG_KEY_NAMES,
    FULL,
    LAYOUTS,
    SIDE,
    PreviewWindow,
    debug_key_pairs,
)
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
from robot_side.startup import RangeReport, StartupPlans, prepare
from robot_side.state import Event, Machine, State, View
from robot_side.stop import SpaceKeyWatcher, StopSignal
from vision_side.frames import (
    CameraFrames,
    FrameError,
    FrameSource,
    VideoFrames,
    first_frame,
)
from vision_side.gestures import GestureRecognizer, ModelsMissing
from vision_side.loop import (
    LoopHealth,
    RobotSide,
    VisionLoop,
    VisionLost,
    VisionThread,
    VisionUpdate,
    check_vision,
)
from vision_side.overlay import Painter, legend_poses, load_legend_images

LOOP_S = 0.02
"""主循环每轮歇多久。倒计时和画面按这个刷新;叫停不走这里,不受它影响。"""

ALIGN_HEADLINE = "ALIGN"
"""对准那一步屏幕上的大字。"""

NOT_CONNECTED = "ARM NOT CONNECTED"
"""对准期间那一行。写出来是为了让人看见:这一步还没碰过机械臂。"""

CONNECTING_HEADLINE = "CONNECTING"
"""对上以后、连机械臂和可达性检查那几秒的大字。"""

CONNECTING = "OPENING ARM AND CHECKING REACH - ARM DOES NOT MOVE"
"""连机械臂那几秒的那一行。这一步只写配置、求逆解,机械臂一步不动(`prepare`)。"""

CHECKING_RANGE = "CHECKING RANGE {done} / {total} - ARM DOES NOT MOVE"
"""连真机之前把活动范围全查一遍时的那一行(`check_range`)。还没连真机。"""

BRINGING_UP = "CONNECTING TO CONTROLLER, CHECKING POWER, WARMING UP (ABOUT 20 S)"
"""经驱动连控制柜、查上电使能、热身那一段的那一行。热身是一条零行程的关节运动。"""

DEFAULT_ARM = "noarm"
"""`--arm` 不写时用哪一份:空实现,不开 RoboDK。加了 `--real` 就没有默认值可用。"""

QUIET_STATES = (State.IDLE, State.COOLDOWN, State.HALTED)
"""机械臂肯定停着的那几个状态。连着真机时只在这几个状态里看那条连接
(`robot_side.real.LINK_POLL_S`:运动中看要抢叫停也要用的那把锁)。"""

T = TypeVar("T")

EXIT_OK = 0
"""跑完了,正常退出。"""
EXIT_VISION = 1
"""摄像头或识别不对。**一次都没碰机械臂。**"""
EXIT_CONFIG = 2
"""配置有问题。什么也没连。"""
EXIT_MODELS = 3
"""两个 ONNX 模型不在。什么也没连。"""
EXIT_REACH = 4
"""可达性检查不通过。机械臂连上了,但一步没动。"""
EXIT_REAL = 5
"""连真机那一步停下了(驱动连不上、没上电使能)。**一条运动都没发。**"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="站着做手势,机械臂做对应的预设动作")
    parser.add_argument(
        "--arm",
        default="",
        help=(
            "机械臂配置:noarm(空实现,不开 RoboDK)/ jaka / nova5,或一份 .toml 的路径。"
            f"不写就是 {DEFAULT_ARM};加了 --real 就必须写"
        ),
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help=(
            "连真机。不加就只在 RoboDK 仿真里动。加了必须同时用 --arm 明写是哪一台,"
            "那份配置要填了控制柜 IP;上真机之前照 docs/demo-checklist.md 过一遍"
        ),
    )
    parser.add_argument("--demo", default="", help="换一份演示配置(.toml 的路径)")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="摄像头编号。不给就照演示配置里的 camera_index",
    )
    parser.add_argument(
        "--video", type=Path, default=None, help="放录像,不开摄像头(调画面时用)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="打开调试用的触发按键(方向键、P、L)"
    )
    parser.add_argument(
        "--window",
        choices=LAYOUTS,
        default=None,
        help="预览窗口:full(全屏)/ side(左半屏,右半屏留给 RoboDK)。"
        "不给就是:仿真并排,连真机和空实现全屏",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="自测用:跑这么多秒自动退出(几个查看脚本也都有这个)",
    )
    return parser.parse_args(argv)


def which_arm(args: argparse.Namespace) -> str:
    """这一次用哪一份机械臂配置。加了 `--real` 就必须明写(`real.choose_arm`)。"""
    return choose_arm(args.arm, real=args.real, default=DEFAULT_ARM)


def layout_for(window: str | None, arm_config: ArmConfig, *, real: bool = False) -> str:
    """预览窗口怎么摆。`--window` 给了就照给的。

    不给就看机械臂:仿真走 RoboDK 的**并排**——看的就是 RoboDK 里那台,全屏置顶把它
    整个挡住了(2026-09-23 用户实跑);**连真机**时看的是真机,照旧**整屏**;空实现没
    什么可看的,也是整屏。
    """
    if window is not None:
        return window
    if real:
        return FULL
    return SIDE if arm_config.driver == "robodk" else FULL


def while_pumping(
    work: Callable[[], T], pump: Callable[[], None], *, every_s: float = LOOP_S
) -> T:
    """在单独的线程上做 `work`,做完之前主线程每隔 `every_s` 调一次 `pump`。

    `work` 做完返回它的结果;它抛的错**原样在这里抛出来**,和在主线程上直接调一样。
    `pump` 至少调一次。
    """
    done: list[T] = []
    failed: list[BaseException] = []

    def run() -> None:
        try:
            done.append(work())
        except BaseException as exc:  # 带回主线程再抛
            failed.append(exc)

    thread = threading.Thread(target=run, name="connect", daemon=True)
    thread.start()
    while True:
        pump()
        if not thread.is_alive():
            break
        time.sleep(every_s)
    thread.join()
    if failed:
        raise failed[0]
    return done[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        # 默认只在仿真里动:加了 --real 才连真机,而且必须明说是哪一台(机器人一侧第 05 票)
        arm_config = load_arm_config(resolve_arm_config(which_arm(args)))
        demo = load_demo_config(Path(args.demo)) if args.demo else load_demo()
        if args.real:
            check_real_allowed(arm_config)
    except ConfigError as exc:
        print(f"配置有问题,什么也没连:\n  {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except RealArmRefused as exc:
        print(f"不连真机,什么也没连:\n  {exc}", file=sys.stderr)
        return EXIT_CONFIG

    log, path = start_log("demo")
    log.info("日志 %s", path)
    log.info(
        "机械臂配置 %s(%s),驱动 %s", arm_config.path.name, arm_config.name, arm_config.driver
    )
    log.info("演示配置 %s,%d 个预设动作", demo.path.name, len(demo.actions))
    over = over_linear_cap(demo, arm_config)
    if over:
        log.warning(
            "速度 %s 超过机械臂配置的上限 %g mm/s,实际按 %g 发(再乘示教器倍率);"
            "要更快调示教器倍率",
            "、".join(f"{name} {mms:g}" for name, mms in over),
            arm_config.max_linear_mms,
            arm_config.max_linear_mms,
        )
    if args.debug:
        log.info("调试按键已打开:%s", ",".join(debug_key_pairs(demo)))
    if args.real:
        log.warning("连真机:%s,控制柜 %s", arm_config.name, arm_config.ip)
    log.info("预览窗口 %s", layout_for(args.window, arm_config, real=args.real))
    try:
        return App(args, arm_config, demo, log).run()
    finally:
        close_log(log)


class App:
    """整个正式程序。启动那几步和主循环都在这儿。"""

    def __init__(
        self,
        args: argparse.Namespace,
        arm_config: ArmConfig,
        demo: DemoConfig,
        log: logging.Logger,
    ) -> None:
        self._args = args
        self._arm_config = arm_config
        self._demo = demo
        self._log = log
        self._real = bool(args.real)
        self._layout = layout_for(args.window, arm_config, real=self._real)
        self._note = CONNECTING
        """连机械臂那一段屏幕上那一行。连的线程往这里写,主线程画;读写一个名字是原子的。"""
        self._monitor: LinkMonitor | None = None
        self._machine: Machine | None = None
        self._signal: StopSignal | None = None
        self._loop: VisionLoop | None = None
        self._window: PreviewWindow | None = None
        self._painter: Painter | None = None
        self._thread: VisionThread | None = None
        self._lost = VisionLost(demo.run.vision_lost_after_s)
        self._health = LoopHealth("主循环", log)
        self._deadline = (
            None if args.seconds is None else time.monotonic() + args.seconds
        )
        self._last: VisionUpdate | None = None
        """最近一帧识别结果。没有新的时候接着画它,画面才不会一闪一闪。"""
        self._quit = False

    # ---- 整条流程 ------------------------------------------------------

    def run(self) -> int:
        """跑完整个程序。返回退出码。"""
        try:
            recognizer = GestureRecognizer()
        except ModelsMissing as exc:
            self._log.error("模型不在:%s", exc)
            print(f"模型不在,什么也没连:\n  {exc}", file=sys.stderr)
            return EXIT_MODELS

        source = self._open_source()
        try:
            with source:
                # 第 1–3 步:开摄像头、等第一帧、要求识别连着跑满约 1 秒。
                # 这三步一次都不碰机械臂,不对就报错退出。
                frame = first_frame(source)
                self._log.info(
                    "启动检查 第一帧 %dx%d%s",
                    frame.width,
                    frame.height,
                    f",后端 {source.backend}" if isinstance(source, CameraFrames) else "",
                )
                check_vision(
                    source,
                    recognizer,
                    gap_s=self._demo.run.vision_lost_after_s,
                    log=self._log,
                )
                return self._with_vision(source, recognizer, (frame.width, frame.height))
        except FrameError as exc:
            # 摄像头打不开、迟迟没画面、识别不正常:说清是哪一步,一次都没碰机械臂
            self._log.error("启动检查 不通过(%s):%s", exc.reason, exc)
            print(f"{exc.reason}:{exc}\n机械臂一次都没碰。", file=sys.stderr)
            return EXIT_VISION

    def _open_source(self) -> FrameSource:
        """这一路画面从哪儿来。

        摄像头编号照**这一次用的那份**演示配置(`--demo` 换过就是换过的那份),
        命令行给了 `--camera` 才盖过它。
        """
        if self._args.video is not None:
            return VideoFrames(self._args.video)
        width, height = self._demo.vision.camera_px
        index = (
            self._demo.vision.camera_index
            if self._args.camera is None
            else self._args.camera
        )
        return CameraFrames(index, width, height)

    def _with_vision(
        self, source: FrameSource, recognizer: GestureRecognizer, size: tuple[int, int]
    ) -> int:
        """识别线程和预览窗口都起来之后的那一段。"""
        robot = RobotSide(self._demo, log=self._log)
        beeps = Beeper()
        self._loop = VisionLoop(
            source,
            recognizer,
            self._demo,
            frame_size=size,
            robot=robot,
            beeps=beeps,
            log=self._log,
        )
        self._window = self._open_window()
        self._painter = Painter(
            self._demo,
            px_per_cm=self._window.px_per_cm,
            images=load_legend_images(legend_poses(self._demo), log=self._log),
            # 调试按键画进图例每行最右边,底下的按键说明就不再列一遍
            keys=DEBUG_KEY_NAMES if self._args.debug else None,
        )
        # 图例整屏时放右侧整列、并排时放画面下面(预览窗口定),摄像头画面那块就不再给它
        # 留地方(`_pump` 里 legend=False)
        self._window.set_legend(self._painter)
        self._thread = VisionThread(self._loop, log=self._log)
        try:
            with self._thread:
                # 第 4 步:启动对准。对不准就停在这一步,按 q / Esc 退出
                if not self._align():
                    return EXIT_VISION
                # 第 5 步:这时才连机械臂
                return self._with_arm(robot, beeps)
        finally:
            # 主循环卡没卡,写进日志。对准那一段没对上就退出时也要有这一行
            self._log.info("%s", self._health.describe())
            if self._window is not None:
                self._window.close()
            beeps.close()

    def _open_window(self) -> PreviewWindow:
        return PreviewWindow(
            self._demo,
            on_confirm=self._on_confirm,
            on_quit=self._on_quit,
            on_realign=self._on_realign,
            on_home=self._on_home,
            on_trigger=self._on_debug_trigger if self._args.debug else None,
            title="GESTURE DEMO",
            layout=self._layout,
        )

    def _align(self) -> bool:
        """站着把拳头举起来停住,程序放下两个区域。对上了返回 True。"""
        assert self._loop is not None
        self._log.info("对准 开始:站到操作位,把拳头举起来停住")
        view = View(
            # `View.state` 只决定大字的颜色。对准不是状态逻辑的五个状态之一
            # (机械臂还没连上),这里借冷却那一档的灰色用
            State.COOLDOWN,
            headline=ALIGN_HEADLINE,
            reason=NOT_CONNECTED,
        )
        while not self._quit:
            started = time.perf_counter()
            self._pump(view)
            self._health.tick(time.perf_counter() - started)
            if self._loop.alignment is not None:
                return True
            if self._dead():
                return False
            time.sleep(LOOP_S)
        self._log.warning("对准 没对上就退出了;机械臂一次都没碰")
        return False

    def _with_arm(self, robot: RobotSide, beeps: Beeper) -> int:
        """连机械臂 → 写入配置 → 可达性检查 → 回待命 → 空闲 → 主循环。"""
        arm = open_arm(self._arm_config)
        try:
            try:
                connected = self._connect(arm)
            except (RealArmRefused, GuardRefused) as exc:
                self._log.error("停在连真机这一步,一条运动都没发:%s", exc)
                print(f"停在连真机这一步,一条运动都没发:\n  {exc}", file=sys.stderr)
                return EXIT_REAL
            if connected is None:
                return EXIT_OK
            prepared, checked = connected
            if not prepared.ok:
                self._log.error("可达性检查不通过,机械臂一步没动。")
                return EXIT_REACH

            self._signal = StopSignal(arm)
            self._machine = Machine(
                arm,
                self._signal,
                plans=prepared.by_action,
                standby_point=prepared.standby_point,
                arm_config=self._arm_config,
                demo=self._demo,
                beeps=beeps,
                log=self._log,
                # 进冷却前状态逻辑清一次判定记录;真正清是识别线程下一帧自己做
                on_clear_records=self._on_clear_records,
                # 真机上每段等多久另算(`real.real_segment_timeout`);活动范围连真机
                # 之前查完了,运行中不再查——那一查要挪 RoboDK 里的模型(ADR-0014)
                segment_timeout_s=(
                    real_segment_timeout(self._arm_config) if self._real else None
                ),
                reachable=checked.reachable if checked is not None else None,
                check_ahead=not self._real,
            )
            robot.attach(self._machine, self._signal)
            if self._real:
                self._monitor = self._link_monitor(arm, self._signal)
            return self._main_loop()
        finally:
            arm.disconnect()  # 先停住当前运动再断开(机械臂接口保证)

    def _connect(self, arm: Arm) -> tuple[StartupPlans, RangeReport | None] | None:
        """连机械臂 → 按配置写入 → 可达性检查;连真机时再加上活动范围全查 → 经驱动
        连控制柜 → 查上电使能 → 热身。**放在单独的线程上做**,主线程照旧刷画面。

        这几步冷启动 RoboDK 要十几秒,开着的也要约 6 秒(2026-09-23 实跑:对上以后到
        可达性检查通过 6 秒);连真机时活动范围全查还要一两分钟,热身约 20 秒。原来在
        主线程上做,这段时间预览窗口一动不动,看着像卡死在对准那一步。

        **热身之前机械臂一步不动**:写配置、求逆解、查直线都只碰 RoboDK 里的模型,而且
        都在经驱动连真机之前。热身那一条是零行程的关节运动,护栏保证一轴都不动。

        可达性检查不过就不往下连真机(返回的 `prepared.ok` 是 False)。驱动连不上、没上电
        使能,`RealArmRefused` / `GuardRefused` 原样抛给调用方。连上了就把 RoboDK 挪到预览
        窗口右边(并排时)。连的时候按了 q / Esc:连不能半路掐断,等它做完再退,返回 `None`。
        """
        self._log.info("连机械臂:%s", self._arm_config.name)
        moved = False

        def pump() -> None:
            nonlocal moved
            started = time.perf_counter()
            self._pump(View(State.COOLDOWN, headline=CONNECTING_HEADLINE, reason=self._note))
            self._health.tick(time.perf_counter() - started)
            if not moved and arm.connected:
                moved = True
                self._make_room()

        def progress(done: int, total: int) -> None:
            self._note = CHECKING_RANGE.format(done=done, total=total)
            if done % 5 == 0 or done == total:
                self._log.info("查活动范围:%d / %d 个停靠点(还没连真机,机械臂不动)", done, total)

        def work() -> tuple[StartupPlans, RangeReport | None]:
            arm.connect()
            prepared = prepare(arm, self._arm_config, self._demo)
            self._log.info("%s", prepared.report.describe())
            if not prepared.ok or not self._real:
                return prepared, None
            # 连真机之前把活动范围全查完:连上以后运行中不再查(ADR-0014)。配置和模型
            # 都没变就用上次存下来的结果,不再等那 80 秒(`robot_side.range_cache`)
            started = time.monotonic()
            checked, _ = check_range_cached(
                arm, self._arm_config, self._demo, prepared, on_progress=progress, log=self._log
            )
            self._log.info("%s(%.0f 秒)", checked.describe(), time.monotonic() - started)
            self._note = BRINGING_UP
            bring_up_real(arm, self._arm_config, log=self._log)
            return prepared, checked

        connected = while_pumping(work, pump)
        if not moved:
            self._make_room()
        if self._quit:
            self._log.info("退出:连机械臂的时候按的")
            return None
        return connected

    def _link_monitor(self, arm: Arm, signal: StopSignal) -> LinkMonitor:
        """盯着真机那条连接。判定掉线就挂上「断开」这个报错,再叫停。

        挂报错是为了让状态逻辑下一轮 `tick()` 看见并进已停止;叫停是 ADR-0012 那条
        规矩:谁先发现要停,谁就直接调机械臂接口的「停止」。和 `state_demo --real` 同一套。

        连真机时 `arm` 一定是 `RoboDKArm`(`check_real_allowed` 只放行 RoboDK 驱动的配置),
        驱动那几件(`driver_state`、`raise_fault`)只有它有。
        """

        def lost(why: str) -> None:
            self._log.error("%s", why)
            arm.raise_fault(NotConnectedError(why))
            signal.raise_stop("link")

        return LinkMonitor(arm, self._arm_config.ip or "", on_lost=lost)

    def _make_room(self) -> None:
        """并排时把 RoboDK 的窗口挪到预览窗口右边。整屏或者不走 RoboDK 时什么也不做。"""
        assert self._window is not None
        if self._layout != SIDE or self._arm_config.driver != "robodk":
            return
        # 走 RoboDK 时 `open_arm` 已经 import 过它了,这里不多拖一个依赖进来
        from robot_side.robodk_arm import ROBODK_EXE

        if self._window.make_room(ROBODK_EXE.name):
            self._log.info("RoboDK 的窗口摆到了预览窗口右边")
        else:
            self._log.info("没找到 RoboDK 的窗口,没挪;自己把它拖到右边")

    def _main_loop(self) -> int:
        """主循环:状态逻辑、识别中断、预览窗口。"""
        machine = self._machine
        assert machine is not None and self._signal is not None
        signal = self._signal
        try:
            # 按键线程:空格全局有效,预览窗口卡住、不在前台都照样叫停(ADR-0012)
            with SpaceKeyWatcher(lambda: signal.raise_stop("space")) as keys:
                machine.start()
                next_link_look = 0.0
                while not machine.finished and not self._quit:
                    started = time.perf_counter()
                    machine.tick()
                    # 连着真机:机械臂停着的时候隔一会儿看一眼那条连接
                    if (
                        self._monitor is not None
                        and machine.state in QUIET_STATES
                        and time.monotonic() >= next_link_look
                    ):
                        self._monitor.poll()
                        next_link_look = time.monotonic() + LINK_POLL_S
                    self._pump(machine.view)
                    self._health.tick(time.perf_counter() - started)
                    time.sleep(LOOP_S)
                for error in keys.errors:
                    self._log.warning("按键线程里出过错:%s", error)
        finally:
            machine.close()
        self._log.info("退出。机械臂停在原地,没有自动回待命。")
        return EXIT_OK

    # ---- 每一轮 --------------------------------------------------------

    def _pump(self, view: View) -> None:
        """一轮里主线程该做的:取最新一帧、判识别中断、告诉识别线程状态、画一帧。"""
        assert self._loop is not None and self._painter is not None
        assert self._thread is not None and self._window is not None

        update = self._thread.take()
        if update is not None:
            self._last = update
        lost = self._lost.update(update)
        if lost:
            self._on_lost()
        self._loop.set_state(self._machine.state if self._machine is not None else None)
        image = self._painter.paint(
            self._window.panel_size, self._last, lost=lost, legend=False
        )
        self._window.update(view, image)
        if self._deadline is not None and time.monotonic() > self._deadline:
            self._on_quit("--seconds")

    def _on_lost(self) -> None:
        """识别中断:执行中或回待命途中进已停止,空闲时只在预览窗口报错。"""
        machine = self._machine
        if machine is None:
            return  # 还没连机械臂,画面上报一下就够
        event = self._lost.halt_event(machine.state)
        if event is None:
            return
        self._log.error("识别中断 %s,机械臂停下等人", self._lost.why)
        machine.submit(event)

    def _dead(self) -> bool:
        """识别线程没了吗。没了就别在对准那一步干等着(spec「线程」)。"""
        assert self._thread is not None
        if self._thread.alive or self._thread.error is None:
            return False
        self._log.error("识别线程出错,停在对准这一步;机械臂一次都没碰")
        print(f"识别线程出错:{self._thread.error}\n机械臂一次都没碰。", file=sys.stderr)
        return True

    # ---- 窗口上的按键 --------------------------------------------------

    def _on_confirm(self, source: str) -> None:
        if self._machine is not None:
            self._machine.submit(Event.confirm(source))

    def _on_quit(self, source: str) -> None:
        self._quit = True
        if self._machine is not None:
            self._machine.submit(Event.quit(source))
        else:
            self._log.info("退出:来自 %s(还没连机械臂)", source)

    def _on_home(self, source: str) -> None:
        """按 H 回待命。空闲、冷却时才有效,由状态逻辑判(还没连机械臂就不理)。"""
        if self._machine is not None:
            self._machine.submit(Event.home(source))

    def _on_realign(self, source: str) -> None:
        """按 R 重新对准。**只在空闲时**——执行中改了起点区,手上那一次就没了准。"""
        assert self._loop is not None
        machine = self._machine
        if machine is not None and machine.state is not State.IDLE:
            self._log.info("重新对准 在 %s 时按下,只有空闲时才有效,忽略", machine.state.name)
            return
        self._log.info("重新对准:来自 %s;机械臂停在待命位姿不动", source)
        self._loop.realign()

    def _on_debug_trigger(self, action: str, source: str) -> None:
        if self._machine is not None:
            self._machine.submit(Event.trigger(action, source))

    def _on_clear_records(self) -> None:
        assert self._loop is not None
        self._loop.clear_records()


if __name__ == "__main__":
    sys.exit(main())

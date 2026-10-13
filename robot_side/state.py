"""状态逻辑:空闲、冷却、执行中、回待命(含倒计时)、已停止。

整条链路(`CONTEXT.md`「运行状态」、spec「状态逻辑」):

    启动 → 回待命(倒计时 → 抬起 → 慢速回)→ 冷却 → 空闲
    空闲 + 触发 → 执行中 → 走完(停在新的停靠点,不回待命)→ 冷却 → 空闲
    空闲 / 冷却 + 按 H → 回待命 → 冷却 → 空闲
    执行中 + 叫停 → 回待命 → 冷却 → 空闲
    回待命 + 叫停 → 已停止 + 回车 → 清除报错 → 回待命 → 冷却 → 空闲
    机械臂一侧报错 / 某段超时 / 不在停靠点 / RoboDK 断开 → 已停止
    外面报上来的故障(识别中断)→ 已停止

**动作做完不回待命位姿**(2026-09-23 用户定):停在哪儿就是**停靠点**,下一个动作从
那儿算起。停靠点记成相对待命点的偏移;做完要停到活动范围外面的,那一次不动
(OUT OF RANGE)。从待命点出发的动作启动时就查过可达性;从别的停靠点出发的,第一次
做之前在执行线程上查一遍(`Job` 的 `check`),查过的记住,够不着就不动(CAN'T REACH)。

机械臂一侧的报错从三处进来:下发那一下抛出来的(执行线程接住)、主线程
问机械臂时抛出来的、以及**没有哪一次调用会抛、只挂在机械臂身上的**那种(急停被按下、
控制柜自己报碰撞)——最后这种只有每轮问一次 `Arm.error()` 才知道。三处一律进已停止。

**叫停不走事件队列**(ADR-0012)。谁先发现要停,谁就直接调机械臂接口的「停止」;
这里只是每次 `tick()` 看一眼 `StopSignal` 上多了几条记录,然后切状态、刷新画面。
所以主线程卡住(拖预览窗口卡过 6.8 秒)也不影响机械臂停下来,只影响画面什么时候更新。

按记录条数、而不是按「已叫停」标记来认叫停:标记会被我们自己放下(每次发运动之前),
条数只增不减,连着按两下空格也漏不掉。

线程分工:主线程管这里和预览窗口,执行线程逐段下发(`Job`),按键线程只管空格
(`robot_side.stop`),提示音另有一个线程(`robot_side.beeps`)。
"""

from __future__ import annotations

import enum
import logging
import math
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from robot_side.arm import Arm, ArmError, joint_drift
from robot_side.beeps import Beeper
from robot_side.config import HOME_ACTION, ArmConfig, DemoConfig
from robot_side.executor import SegmentTimeout, run_plan
from robot_side.motion import (
    ORIGIN,
    Offset,
    Segment,
    in_range,
    orientation_pose,
    plan_action,
    shifted,
)
from robot_side.reach import ReachReport
from robot_side.startup import LiftRefused, returning
from robot_side.stop import StopSignal

STANDBY_TOL_DEG = 1.0
"""执行预设动作前核对「还在停靠点上」的容差,每轴 ±1°。

启动时也用它判断要不要跳过抬起和移动(`docs/grill-decisions.md` 第二节)。
"""

OUT_OF_RANGE = "OUT OF RANGE"
"""做完要停到活动范围外面:这一次不动,屏幕上那一行。"""

CANT_REACH = "CAN'T REACH"
"""从这个停靠点出发够不着(可达性检查不过):这一次不动,屏幕上那一行。"""

CHECKING = "CHECKING REACH"
"""第一次从这个停靠点出发,先查可达性那一会儿,执行进度那一行。"""

JOB_JOIN_S = 5.0
"""等执行线程收尾多久。叫停之后它本来就该很快返回。"""


class State(enum.Enum):
    """五个状态。名字就是预览窗口和日志里显示的英文名(`CONTEXT.md`)。"""

    IDLE = "IDLE"
    COOLDOWN = "COOLDOWN"
    RUNNING = "RUNNING"
    RETURNING = "RETURNING"
    HALTED = "HALTED"


@dataclass(frozen=True)
class Event:
    """一件从外面进来的事。**叫停不在这里**(ADR-0012:它直达机械臂)。

    `source` 记进日志,比如 "key:left"(调试按键)、"script"(编好的事件序列),
    以后是识别线程报上来的手势。
    """

    kind: str
    action: str = ""
    source: str = ""

    @staticmethod
    def trigger(action: str, source: str) -> Event:
        return Event("trigger", action=action, source=source)

    @staticmethod
    def confirm(source: str) -> Event:
        return Event("confirm", source=source)

    @staticmethod
    def home(source: str) -> Event:
        """回待命。笔记本上按 H;只在空闲、冷却时有效(2026-09-23 用户定)。"""
        return Event("home", source=source)

    @staticmethod
    def quit(source: str) -> Event:
        return Event("quit", source=source)

    @staticmethod
    def halt(reason: str, source: str) -> Event:
        """外面发现的故障:停下等人。`reason` 是预览窗口上那一行短句,**英文**。

        机械臂一侧自己的报错不走这里(那几种由 `_check_arm_error` 和执行线程报上来)。
        这一条留给**机械臂之外**的故障:识别中断(VISION LOST)就从这里进来——看不见
        停止手势的时候,机械臂不该还在动(视觉一侧第 04 票)。
        """
        return Event("halt", action=reason, source=source)


@dataclass(frozen=True)
class View:
    """预览窗口要显示的东西。**一律英文**(`CONTEXT.md`「预览窗口」)。"""

    state: State
    headline: str
    """大字那一行:IDLE / COOLDOWN 2 / RUNNING / RETURNING IN 3 / HALTED。"""
    action: str = ""
    """正在执行哪个预设动作,用演示配置里的英文名。"""
    step: str = ""
    """执行进度,比如 STEP 2 / 6。空闲和冷却时写机械臂停在哪儿,比如 AT X +150 Z +100 MM。"""
    reason: str = ""
    """为什么到了这个状态,短句英文,比如 STOPPED BY SPACE、ARM ERROR。"""
    detail: str = ""
    """报错原文一类的细节;机械臂报上来什么就是什么,可能不是英文。"""


@dataclass(frozen=True)
class Rest:
    """机械臂现在停着的停靠点:相对待命点的偏移,加上停下那一刻读回来的关节角。

    关节角是拿来核对「还在不在这儿」的(被人在示教器上挪过就进已停止);偏移是下一个
    动作的出发点。
    """

    offset: Offset
    joints: tuple[float, ...]

    @property
    def at_standby(self) -> bool:
        return self.offset == ORIGIN


class Job:
    """在执行线程上走完一段计划:一次只发一段,发之前先看「已叫停」标记。

    出了错不往外抛——执行线程出错当作机械臂一侧报错,由状态逻辑切到已停止
    (`docs/grill-decisions.md` 第二节)。
    """

    def __init__(
        self,
        arm: Arm,
        signal: StopSignal,
        plan: Sequence[Segment],
        what: str,
        log: logging.Logger,
        timeout_s: float | SegmentTimeout | None = None,
        check: Callable[[], ReachReport] | None = None,
    ) -> None:
        """`timeout_s` 不给就每段自己算(估算用时 × 2 + 5 秒);给一个数就每段都用它
        (测试、没开 `program_sets_speed` 的真机);给 `SegmentTimeout` 就按它放慢估算。

        `check` 给了就先查可达性,过了才下发;不过就一段都不发(`unreachable`)。
        查放在执行线程上:JAKA 一段要 0.3 秒左右,放主线程上画面会卡。
        """
        self.plan = list(plan)
        self.what = what
        self.sent = 0
        self.index = 0
        """正在走第几段(1 起);还没发出第一段时是 0。"""
        self.error: BaseException | None = None
        self.reach: ReachReport | None = None
        """出发前查的结果;没查就是 None。"""
        self.checking = False
        """正在查可达性。"""
        self._check = check
        self._arm = arm
        self._signal = signal
        self._log = log
        self._timeout_s = timeout_s
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"exec-{what}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cut_short(self) -> bool:
        """没把计划走完——被叫停拦下了,或者中途报错。"""
        return self.sent < len(self.plan)

    @property
    def unreachable(self) -> bool:
        """出发前查了可达性、没过,所以一段都没发。"""
        return self.reach is not None and not self.reach.ok

    def join(self, timeout_s: float = JOB_JOIN_S) -> bool:
        """等执行线程结束,返回它**是不是真的结束了**。

        返回的不是「等够了没有」:超时了线程还活着,调用方就绝不能再起一个,否则
        两个线程会同时往同一条连接上下发。
        """
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def _run(self) -> None:
        try:
            if self._check is not None:
                self.checking = True
                try:
                    self.reach = self._check()
                finally:
                    self.checking = False
                if not self.reach.ok:
                    return
            self.sent = run_plan(
                self._arm,
                self._signal,
                self.plan,
                self._timeout_s,
                on_segment=self._report,
            )
        except BaseException as exc:  # 执行线程出错 = 机械臂一侧报错
            self.error = exc
        finally:
            self._done.set()

    def _report(self, index: int, segment: Segment | None) -> None:
        self.index = index + 1
        label = segment.label if segment is not None else "?"
        self._log.info("  运动 %d/%d %s", self.index, len(self.plan), label)


class Machine:
    """状态逻辑本体。

    主线程每轮调一次 `tick()`:看叫停、看执行线程、收事件、按时间推进。`submit()`
    可以从别的线程调(事件排进队列),`tick()` 只在主线程调。
    """

    def __init__(
        self,
        arm: Arm,
        signal: StopSignal,
        *,
        plans: Mapping[str, Sequence[Segment]],
        standby_point: Sequence[float],
        arm_config: ArmConfig,
        demo: DemoConfig,
        beeps: Beeper | None = None,
        log: logging.Logger | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_clear_records: Callable[[], None] | None = None,
        segment_timeout_s: float | SegmentTimeout | None = None,
        job_join_s: float = JOB_JOIN_S,
        reachable: Mapping[tuple[Offset, str], bool] | None = None,
        check_ahead: bool = True,
    ) -> None:
        """`reachable` 是事先查好的「从哪个停靠点出发的哪个动作过没过」
        (`robot_side.startup.check_range`);`check_ahead=False` 时运行中**一次都不查**,
        没查过的组合一律当够不着——连着真机时就这么用(ADR-0014:出发前那一查要在
        RoboDK 里挪模型,连着真机时没验过)。
        """
        self._arm = arm
        self._signal = signal
        self._plans = dict(plans)
        """从待命点出发的每个预设动作,启动时查过可达性。"""
        self._standby_point = tuple(standby_point)
        self._orientation = orientation_pose(demo.default_orientation_deg)
        self._arm_config = arm_config
        self._demo = demo
        # 不给提示音就静音。`block=True` 是为了**不起那个播放线程**:没声音可播,
        # 起一个线程挂在队列上,谁也不会去关它。
        self._beeps = (
            beeps if beeps is not None else Beeper(play=lambda *_: None, block=True)
        )
        self._log = log if log is not None else logging.getLogger("robot_side.state")
        self._now = clock
        self._on_clear_records = on_clear_records
        self._segment_timeout_s = segment_timeout_s
        self._job_join_s = job_join_s

        self._state = State.IDLE
        self._phase = ""
        """RETURNING 分两段:countdown 和 moving。别的状态里,有活干就是 moving。"""
        self._started = False
        self._finished = False
        self._reason = ""
        self._detail = ""
        self._action = ""
        self._deadline = 0.0
        self._next_beep_at = 0.0
        self._job: Job | None = None
        self._stops_seen = len(signal.records)
        self._events: queue.SimpleQueue[Event] = queue.SimpleQueue()

        self._rest: Rest | None = None
        """机械臂停在哪个停靠点。只有停稳了、知道在哪儿时才有(进冷却那一刻记下);
        执行中、回待命、已停止时是 None。空闲一定有:只有进冷却才到得了空闲。"""
        self._start: Offset = ORIGIN
        """手上这个执行线程是从哪个停靠点出发的。够不着、一段没发时,机械臂还在这儿。"""
        self._target: Offset = ORIGIN
        """手上这个执行线程走完以后停在哪儿。"""
        self._reachable: dict[tuple[Offset, str], bool] = {
            (ORIGIN, name): True for name in self._plans
        }
        self._reachable.update(reachable or {})
        """从哪个停靠点出发的哪个动作查过、过没过。从待命点出发的启动时查过。"""
        self._check_ahead = check_ahead
        """没查过的组合,出发前在执行线程上查一次(仿真);不许查就当够不着(真机)。"""
        self._checking: tuple[Offset, str] | None = None
        """手上这个执行线程出发前要查的是哪一对。"""
        self._notice = ""
        """空闲、冷却时那一行提示(OUT OF RANGE / CAN'T REACH)。下一次动起来就清掉。"""

    # ---- 外面看得到的 --------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def finished(self) -> bool:
        """收到退出了吗。主循环看这个决定要不要收摊。"""
        return self._finished

    def submit(self, event: Event) -> None:
        """排一件事进来。可以从别的线程调;`tick()` 时才真正处理。"""
        self._events.put(event)

    def start(self) -> None:
        """启动后第一次回待命:和叫停之后走的是同一套(用户故事 7)。"""
        if self._started:
            return
        self._started = True
        self._log.info(
            "启动:机械臂配置 %s,演示配置 %s,倒计时 %.0f 秒,冷却 %.0f 秒",
            self._arm_config.path.name,
            self._demo.path.name,
            self._demo.run.countdown_s,
            self._demo.run.cooldown_s,
        )
        self._enter_returning("STARTUP", why="启动后第一次回待命")

    def tick(self) -> None:
        """推一轮。顺序有讲究,见各个 `_check_*` 上的说明。

        这一轮里有几处要直接问机械臂(在不在待命位姿、现在停在哪)。机械臂那边报错
        就进已停止,而不是把主循环带倒——机械臂在动、程序却挂了,是最不该发生的事。
        RoboDK 中途被关掉走的也是这一条(`robot_side.robodk_arm` 把它翻成「断开」)。
        """
        if not self._started or self._finished:
            return
        try:
            self._check_stops()
            self._check_arm_error()
            self._check_job()
            self._check_events()
            self._advance()
        except ArmError as exc:
            # 主线程问机械臂时报的错:RoboDK 中途被关掉走的就是这一条
            self._halt_on_error(exc, why="主线程问机械臂时报错")

    def close(self) -> None:
        """收尾:等执行线程结束。不自动回待命、不断开机械臂——那是调用方的事。"""
        if self._job is not None:
            if not self._job.join(self._job_join_s):
                self._log.warning(
                    "退出时 %s 的执行线程还没结束;机械臂由调用方断开时停住",
                    self._job.what,
                )
            self._job = None

    @property
    def view(self) -> View:
        """预览窗口要显示的东西。"""
        resting = self._state in (State.IDLE, State.COOLDOWN)
        return View(
            state=self._state,
            headline=self._headline(),
            action=self._action,
            step=self._step(),
            reason=self._reason or (self._notice if resting else ""),
            detail=self._detail,
        )

    # ---- 一轮里的四件事 ------------------------------------------------

    def _check_stops(self) -> None:
        """先看叫停:它已经发生了,机械臂已经在停了,这里只是跟上去切状态。"""
        records = self._signal.records
        if len(records) <= self._stops_seen:
            return
        new = records[self._stops_seen :]
        self._stops_seen = len(records)
        for record in new:
            self._log.info(
                "叫停 来自 %s,立标记到机械臂停住用了 %.3f 秒%s",
                record.source,
                record.stopped_at - record.flag_at,
                f",机械臂报了 {record.error}" if record.error else "",
            )
        source = new[-1].source.upper()

        if self._state is State.RUNNING:
            self._beeps.stopped()
            self._enter_returning(f"STOPPED BY {source}", why=f"执行中被 {source} 叫停")
        elif self._state is State.RETURNING:
            # 不写 STOPPED TWICE:启动后第一次回待命也是回待命,那时被叫停只是第一次
            self._enter_halted(
                f"STOPPED WHILE RETURNING ({source})", why="回待命途中被叫停,停下等人"
            )
        elif self._state is State.HALTED:
            # 已经停着了。标记不放下:放下了执行线程就可能接着发下一段。
            self._log.info("叫停 已经是 HALTED,不起作用")
        else:
            # 空闲、冷却:机械臂本来就停着,叫停不起作用。把标记放下,
            # 免得下一个预设动作一开始就被它拦住。
            self._log.info("叫停 在 %s 时按下,机械臂本来就停着,不起作用", self._state.name)
            self._halt_if_trouble(self._release_stop())

    def _check_arm_error(self) -> None:
        """再问一次机械臂身上挂着什么报错(用户故事 18)。

        碰撞保护跳闸、急停被按下这些,发生时我们这边可能一条指令都没发,没有哪一次
        调用会把它抛出来——只有问才知道。问的是挂着的那个,不是一次 API 来回
        (`Arm.error`),所以每轮问一次不贵。

        **已停止时不问**:那个错要等回车才清,每轮都问会一直重报、一直连响三声。
        """
        if self._state is State.HALTED:
            return
        error = self._arm.error()
        if error is not None:
            self._halt_on_error(error, why="机械臂身上挂着报错")

    def _check_job(self) -> None:
        """再看执行线程。只看当前状态自己起的那一个(`_phase` 是 moving 时)。"""
        job = self._job
        if job is None or self._phase != "moving" or not job.done:
            return
        if job.error is not None:
            error = job.error
            self._job = None  # 看过了,别在下一次确认时又报一遍
            # 执行线程自己出错也当作机械臂一侧报错(`docs/grill-decisions.md` 第二节);
            # 某段运动超时抛的 `ArmTimeoutError` 也从这里进来
            self._halt_on_error(error, why=f"{job.what} 的执行线程出错")
            return
        if self._checking is not None and job.reach is not None:
            self._reachable[self._checking] = job.reach.ok
            self._checking = None
        if job.unreachable:
            # 一段都没发,机械臂还在原来的停靠点上:不是故障,这一次不动就是了
            assert job.reach is not None
            self._log.warning(
                "%s 从 %s 出发够不着,这一次不动:\n%s",
                job.what,
                _where(self._start),
                job.reach.describe(),
            )
            self._notice = f"{CANT_REACH}: {self._action}"
            self._enter_cooldown(self._start)
            return
        if job.cut_short:
            # 叫停正好落在两段之间,`_check_stops` 还没轮到。按叫停处理,不当走完。
            if self._state is State.RUNNING:
                self._enter_returning("STOPPED", why="发下一段之前发现已叫停")
            else:
                self._enter_halted("STOPPED", why="回待命途中发现已叫停")
            return
        self._log.info("%s 走完 %d 段,停在 %s", job.what, job.sent, _where(self._target))
        self._enter_cooldown(self._target)

    def _check_events(self) -> None:
        """收事件。执行中除叫停外一律忽略,既不打断也不排队(用户故事 12)。"""
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            if event.kind == "quit":
                self._log.info("退出:来自 %s", event.source)
                self._finished = True
                return
            if event.kind == "confirm":
                self._on_confirm(event)
            elif event.kind == "home":
                self._on_home(event)
            elif event.kind == "trigger":
                self._on_trigger(event)
            elif event.kind == "halt":
                self._on_halt(event)
            else:
                self._log.warning("认不出的事件:%r", event)

    def _advance(self) -> None:
        """按时间推进:倒计时和冷却。"""
        now = self._now()
        if self._state is State.RETURNING and self._phase == "countdown":
            while self._next_beep_at <= now and self._next_beep_at < self._deadline:
                self._beeps.countdown()
                self._next_beep_at += 1.0
            if now >= self._deadline:
                self._start_returning_move()
        elif self._state is State.COOLDOWN and now >= self._deadline:
            self._go(State.IDLE, "", why="冷却结束,开始接受触发手势")

    # ---- 事件 ----------------------------------------------------------

    def _on_trigger(self, event: Event) -> None:
        if event.action == HOME_ACTION:
            # 回待命手势(rock):和按 H 走同一条路,只在空闲、冷却时有效
            self._on_home(event)
            return
        if self._state is not State.IDLE:
            self._log.info(
                "触发 %s(来自 %s)在 %s 时到达,忽略——不打断也不排队",
                event.action,
                event.source,
                self._state.name,
            )
            return
        try:
            action = self._demo.action(event.action)
        except KeyError:
            self._log.warning("触发 演示配置里没有叫 %r 的预设动作,忽略", event.action)
            return
        if action.name not in self._plans:
            self._log.warning("触发 %s 没有排好的段,忽略", action.name)
            return
        rest = self._rest
        assert rest is not None, "空闲时一定知道停在哪儿(只有进冷却才到得了空闲)"

        # 做完要停到活动范围外面:这一次不动,不响、不切状态,屏幕上说一声
        target = shifted(rest.offset, action)
        if not in_range(target, self._demo.run.range_mm):
            self._log.info(
                "触发 %s(来自 %s):做完会停在 %s,出了活动范围 ±%s mm,这一次不动",
                action.name,
                event.source,
                _where(target),
                "/".join(f"{r:g}" for r in self._demo.run.range_mm),
            )
            self._notice = f"{OUT_OF_RANGE}: {action.display}"
            return

        # 执行预设动作前核对机械臂还在停靠点上(每轴 ±1°),不在就进已停止
        drift = joint_drift(self._arm.joints(), rest.joints)
        if drift > STANDBY_TOL_DEG:
            where = "standby" if rest.at_standby else "where it stopped"
            self._enter_halted(
                "NOT AT STANDBY" if rest.at_standby else "NOT WHERE IT STOPPED",
                why=f"要执行 {action.name},但机械臂离停靠点 {_where(rest.offset)} 差了 {drift:.2f}°",
                detail=f"{drift:.2f} deg from {where}",
            )
            return

        key = (rest.offset, action.name)
        if rest.at_standby:
            plan = self._plans[action.name]
        else:
            start = tuple(p + o for p, o in zip(self._standby_point, rest.offset, strict=True))
            plan = plan_action(action, start_point=start, orientation=self._orientation)
        check: Callable[[], ReachReport] | None = None
        if key not in self._reachable and not self._check_ahead:
            # 连着真机:运行中不查(那一查要挪模型),没事先查过的就不动
            self._log.info(
                "触发 %s(来自 %s):从 %s 出发没事先查过,运行中不查,这一次不动",
                action.name,
                event.source,
                _where(rest.offset),
            )
            self._notice = f"{CANT_REACH}: {action.display}"
            return
        if key not in self._reachable:
            # 第一次从这个停靠点出发做这个动作:在执行线程上先查一遍,查过的记住

            def check() -> ReachReport:
                return self._arm.reach_check({action.name: plan}, rest.joints)

        elif not self._reachable[key]:
            self._log.info(
                "触发 %s(来自 %s):从 %s 出发够不着(之前查过),这一次不动",
                action.name,
                event.source,
                _where(rest.offset),
            )
            self._notice = f"{CANT_REACH}: {action.display}"
            return

        self._beeps.trigger()
        self._action = action.display
        self._notice = ""
        self._go(
            State.RUNNING,
            "",
            why=(
                f"触发 {action.name}(来自 {event.source}),{len(plan)} 段,"
                f"从 {_where(rest.offset)} 到 {_where(target)}"
                + (",先查可达性" if check is not None else "")
            ),
        )
        self._checking = key if check is not None else None
        self._start_job(plan, action.name, start=rest.offset, target=target, check=check)

    def _on_home(self, event: Event) -> None:
        """按 H 回待命。只在空闲、冷却时有效;已经在待命点上就不动。

        走的是和叫停之后同一套回待命:倒计时、先抬起、再慢速回(用户故事 7)。
        执行中要回待命就叫停——叫停本来就会回待命。
        """
        if self._state not in (State.IDLE, State.COOLDOWN):
            self._log.info(
                "回待命 按 H(来自 %s)在 %s 时,只有空闲、冷却时才有效,忽略",
                event.source,
                self._state.name,
            )
            return
        rest = self._rest
        if rest is not None and rest.at_standby:
            self._log.info("回待命 按 H(来自 %s):已经在待命位姿上,不动", event.source)
            return
        self._notice = ""
        self._action = ""
        self._enter_returning(
            "HOME",
            why=f"按 H 回待命(来自 {event.source}),从 {_where(rest.offset) if rest else '?'} 回去",
        )

    def _on_halt(self, event: Event) -> None:
        """外面报上来的故障:进已停止。已经停着了就不再报一遍。

        `_enter_halted` 本来就会立「已叫停」标记、调机械臂接口的停止,所以机械臂正在
        动的时候收到这一件,它当场就停住了。
        """
        if self._state is State.HALTED:
            self._log.info(
                "停机 %s(来自 %s)到达,已经是 HALTED,不起作用", event.action, event.source
            )
            return
        self._enter_halted(
            event.action, why=f"外面报上来的故障:{event.action}(来自 {event.source})"
        )

    def _on_confirm(self, event: Event) -> None:
        """回车只在已停止时有效,其他状态下忽略。

        顺序照用户故事 17:**先清除报错**、必要时提示重新上电使能,再走回待命流程。
        清不掉就留在已停止——连接已经断了的话,清了也接不回来,要恢复只能重启程序。
        """
        if self._state is not State.HALTED:
            self._log.info("回车 在 %s 时按下,只有 HALTED 时才有效,忽略", self._state.name)
            return
        try:
            hint = self._clear_arm_error()
        except ArmError as exc:
            self._halt_on_error(exc, why="按回车想恢复,但机械臂一侧的报错清不掉")
            return
        self._drop_reported_job_error()
        self._enter_returning(
            "CONFIRMED",
            why=f"已停止后按回车确认(来自 {event.source})",
            detail=hint,
        )

    def _clear_arm_error(self) -> str:
        """清除机械臂一侧的报错,返回还要提示人做什么(英文短句,没有就是空串)。

        清不掉就把挂着的那个错**原样抛出**(`Arm.clear_error`),由调用方留在已停止。
        急停那种清完还要人去示教器上重新上电使能:**程序只提示,不替人做**
        (用户故事 27)。
        """
        error = self._arm.error()
        if error is None:
            # 没挂着东西也照样清一次:回车这一步的顺序是定死的(用户故事 17),
            # 而且各实现都保证「没有可清的」时清了没有副作用
            self._log.info("确认:机械臂一侧没有挂着的报错,清一次就走回待命")
            self._arm.clear_error()
            return ""
        self._log.info("确认:先清除机械臂一侧的报错(%s: %s)", type(error).__name__, error)
        self._arm.clear_error()
        if not error.needs_power_on:
            return ""
        self._log.warning("确认:清完了,但要你去示教器上重新上电使能;程序不替你做")
        return error.hint

    def _drop_reported_job_error(self) -> None:
        """回车 = 把这一次故障整件事认掉,包括执行线程报上来的那个错。

        下发那一下抛出来的错,机械臂那边也挂住了;两边说的是同一件事,已停止时已经
        报过一次。不在这里认掉的话,回待命动之前 `_release_stop` 会拿它再报一遍,
        人就要按两次回车。

        **线程还没收尾就不动它**:那是另一回事(`_release_stop` 的 EXEC STUCK),
        照旧留给它处理。
        """
        job = self._job
        if job is None or not job.join(self._job_join_s):
            return
        if job.error is not None:
            self._log.info("确认:%s 的执行线程报的错已经报过,认掉", job.what)
            self._job = None

    # ---- 进各个状态 ----------------------------------------------------

    def _go(self, state: State, reason: str, *, why: str, detail: str = "") -> None:
        self._log.info("状态 %s → %s:%s", self._state.name, state.name, why)
        self._state = state
        self._reason = reason
        self._detail = detail
        self._phase = ""

    def _enter_returning(self, reason: str, *, why: str, detail: str = "") -> None:
        """回待命:先原地倒计时,倒完才抬起、才慢速回去。"""
        self._go(State.RETURNING, reason, why=why, detail=detail)
        self._phase = "countdown"
        now = self._now()
        self._deadline = now + self._demo.run.countdown_s
        self._next_beep_at = now

    def _start_returning_move(self) -> None:
        """倒计时走完:已经在待命位姿附近就只倒计时,不多动一次(用户故事 9)。

        先等上一个执行线程收尾:下面要读关节角和位姿,而**运动中读位置会让 RoboDK
        崩进程**(`robot_side.arm` 上的说明),读到的也不会是位置曲线。它没能收尾、
        或者报了错,就停下等人,不往下动。
        """
        trouble = self._release_stop()
        if trouble is not None:
            self._halt_if_trouble(trouble)
            return
        here = self._arm.joints()
        drift = joint_drift(here, self._arm_config.standby_joints_deg)
        self._log.info(
            "回待命 从关节角 [%s] 出发,离待命位姿最多差 %.2f°",
            ", ".join(f"{v:.2f}" for v in here),
            drift,
        )
        # 停在别的停靠点上(按 H 回来的)就一定要走:关节角差得少,不等于没挪过地方
        away = self._rest is not None and not self._rest.at_standby
        if drift <= STANDBY_TOL_DEG and not away:
            self._log.info(
                "回待命 已经在待命位姿附近(差 %.2f°),跳过抬起和移动", drift
            )
            self._enter_cooldown(ORIGIN)
            return
        try:
            plan = returning(self._arm, self._arm_config, self._demo)
        except LiftRefused as exc:
            # 一段都没排、一步没动:停下等人去示教器上把它挪回 Home 附近
            self._enter_halted("CAN'T LIFT", why=f"回待命排不出来:{exc}", detail=str(exc))
            return
        self._log.info(
            "回待命 抬起再慢速回去,%d 段(都是关节运动);抬起到 [%s]",
            len(plan),
            ", ".join(f"{v:.2f}" for v in plan[0].joints),  # type: ignore[attr-defined]
        )
        self._phase = "moving"
        self._start_job(plan, "回待命", start=ORIGIN, target=ORIGIN)

    def _enter_cooldown(self, rest: Offset) -> None:
        """进空闲之前先清空判定记录,再冷却——挡住拼接误触发(用户故事 20)。

        `rest` 是机械臂现在停着的停靠点。停稳了,这时读一次关节角记下来,下一个动作
        出发前拿它核对「还在不在这儿」。
        """
        trouble = self._release_stop()
        if trouble is not None:
            self._halt_if_trouble(trouble)
            return
        self._rest = Rest(rest, tuple(self._arm.joints()))
        self._action = ""
        if self._on_clear_records is not None:
            self._on_clear_records()
        self._go(
            State.COOLDOWN,
            "",
            why=f"清空判定记录,冷却 {self._demo.run.cooldown_s:.0f} 秒",
        )
        self._deadline = self._now() + self._demo.run.cooldown_s

    def _halt_if_trouble(self, trouble: tuple[str, str] | None) -> bool:
        """`_release_stop` 报回来的事,一律进已停止。返回「有没有出事」。"""
        if trouble is None:
            return False
        reason, why = trouble
        self._enter_halted(reason, why=why, detail=why)
        return True

    def _halt_on_error(self, error: BaseException, *, why: str) -> None:
        """机械臂一侧报错:一律进已停止,画面上写清是哪一种、接下来要做什么。

        哪一种和那句提示都从报错自己身上来(`ArmError.reason` / `.hint`),所以这里
        不用挨个 `isinstance`;不是机械臂接口抛的那种(执行线程里的 bug 一类)就按
        一般的 ARM ERROR 处理。
        """
        reason = error.reason if isinstance(error, ArmError) else ArmError.reason
        hint = error.hint if isinstance(error, ArmError) else ""
        detail = f"{type(error).__name__}: {error}"
        self._enter_halted(
            reason, why=f"{why}:{error}", detail=f"{detail} — {hint}" if hint else detail
        )

    def _enter_halted(self, reason: str, *, why: str, detail: str = "") -> None:
        """已停止:原地不动、不认任何手势,等人按回车。

        再叫停一次,为的是把「已叫停」标记立住:执行线程每段开始前看它,立住了就
        不会再发下一段。机械臂接口的「停止」本来就是幂等的,重复调没有副作用。
        """
        try:
            self._signal.raise_stop("halted")
        except Exception as exc:
            self._log.warning("进 HALTED 时机械臂的「停止」报错:%s", exc)
        self._stops_seen = len(self._signal.records)
        self._rest = None  # 停在哪儿不知道了;回车以后回待命,回到了才重新记
        self._go(State.HALTED, reason, why=why, detail=detail)
        self._beeps.halted()

    # ---- 执行线程 ------------------------------------------------------

    def _start_job(
        self,
        plan: Sequence[Segment],
        what: str,
        *,
        start: Offset,
        target: Offset,
        check: Callable[[], ReachReport] | None = None,
    ) -> None:
        """起一个执行线程走这段计划。上一个没收好尾就不起,停下等人。

        先等上一个收尾,再放下「已叫停」标记——反过来的话,上一个还在跑的执行线程
        会看到标记已经放下,接着把它本来不该发的那一段发出去。

        放下标记和线程起来之间还有个极窄的缝:这中间来的叫停,机械臂那边照停不误,
        这里要等到下一轮 `tick()` 才看见(`_release_stop` 不动 `_stops_seen`,
        就是为了这一下不会被漏掉),那时按叫停处理。
        """
        trouble = self._release_stop()
        if trouble is not None:
            self._halt_if_trouble(trouble)
            return
        self._phase = "moving"
        self._start = start
        self._target = target
        self._rest = None  # 要动了;走完进冷却时再记下停在哪儿
        self._job = Job(
            self._arm,
            self._signal,
            plan,
            what,
            self._log,
            timeout_s=self._segment_timeout_s,
            check=check,
        )
        self._job.start()

    def _release_stop(self) -> tuple[str, str] | None:
        """等执行线程收尾,再把「已叫停」标记放下。

        出了这两种事就**不放标记**,返回一句说明,由调用方进已停止:

        - 线程没在 `JOB_JOIN_S` 内结束:标记一放下,它就会接着发下一段,成了两个
          线程同时往一条连接上下发。宁可停下等人。
        - 线程报了错:错要是没人看,机械臂就会带着故障又被命令动起来。正常情况下
          `_check_job` 已经看过了(看过就把 `_job` 放掉),这里接住的是它没轮到的
          那种——叫停和报错撞在同一轮里。

        **不动 `_stops_seen`。**放标记只是把标记放下,不代表哪一次叫停被处理过;
        `_check_stops` 上次数到哪儿就是哪儿,这中间来的叫停下一轮照样看得见。
        """
        job = self._job
        if job is not None:
            if not job.join(self._job_join_s):
                return (
                    "EXEC STUCK",
                    f"{job.what} 的执行线程 {self._job_join_s:g} 秒还没结束",
                )
            if job.error is not None:
                self._job = None  # 看过了,别在下一次确认时又报一遍
                return ("ARM ERROR", f"{job.what} 出错:{job.error}")
        self._signal.clear()
        return None

    # ---- 画面上的字 ----------------------------------------------------

    def _headline(self) -> str:
        if self._state is State.RETURNING and self._phase == "countdown":
            return f"RETURNING IN {self._left()}"
        if self._state is State.COOLDOWN:
            return f"COOLDOWN {self._left()}"
        return self._state.value

    def _left(self) -> int:
        return max(0, math.ceil(self._deadline - self._now()))

    def _step(self) -> str:
        if self._state in (State.IDLE, State.COOLDOWN) and self._rest is not None:
            return _where(self._rest.offset, english=True)
        job = self._job
        if job is None or self._phase != "moving" or not job.plan:
            return ""
        if job.checking:
            return CHECKING
        return f"STEP {job.index} / {len(job.plan)}"


def _where(offset: Offset, *, english: bool = False) -> str:
    """停靠点写成一句话:日志里是「待命点」或「待命点 X+150 Z+100」,屏幕上是英文。"""
    if offset == ORIGIN:
        return "AT STANDBY" if english else "待命点"
    parts = " ".join(
        f"{axis} {value:+.0f}" for axis, value in zip("XYZ", offset, strict=True) if value
    )
    return f"AT {parts} MM" if english else f"待命点 {parts} mm"

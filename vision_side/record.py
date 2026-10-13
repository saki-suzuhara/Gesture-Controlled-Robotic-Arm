"""录制工具:按提示录一段,录完立刻在屏幕上出表(spec「录制工具」)。

**在正式代码里,和识别代码一起维护**(第 38 条 user story):录的时候走的是正式程序
那一套——同一个取帧、同一个识别器、同一次启动对准、同一套判定、同一个预览窗口,
只多两样:屏幕上的提示,和把每一帧写进 `data/recordings/`。因为是同一套,录下来的
数据才和正式程序的画面范围一致,回放出来的数才说明得了问题。

三档用法:

- **取景检查**(`--check`):只显示画面和识别结果,**不录像**;同屏放出示范图,
  可以边调屏幕倾角边试做。按 q / Esc 退出。
- **录制**:启动对准 → 按提示一条一条做 → 录完立刻出表。
- 两档都会在开始前**拦住摄像头占位图**:隐私开关关着时不让白录一遍(第 41 条)。

线程分工(spec「录制工具」最后一条):

- **识别线程**:取帧 → 识别 → 写逐帧日志和视频 → 结果搁给主线程;
- **主线程**:显示、推提示、记时间表。

两条各挂一个 `LoopHealth`,退出时把「卡过几次」写进日志和 `meta.json`。上一轮录制
循环卡过 6 次 0.3–1.4 秒,不在读帧和识别,只剩显示和等按键;这一票把两件事分开了,
还卡不卡有数才说得清。

**录像里有用户本人,只存本地,不提交、不分发**:`data/` 整个被 git 忽略。
"""

from __future__ import annotations

import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from robot_side.beeps import Beeper
from robot_side.config import FIST_MOVE, DemoConfig
from robot_side.preview import PreviewWindow
from robot_side.state import State, View
from vision_side.align import Alignment
from vision_side.frames import Frame, FrameSource, first_frame
from vision_side.loop import (
    LoopHealth,
    RobotSide,
    VisionLoop,
    VisionThread,
    VisionUpdate,
    check_vision,
)
from vision_side.overlay import Painter, legend_poses, load_legend_images
from vision_side.prompts import (
    FREE,
    GO,
    LEAD_S,
    PREP,
    STOP_GESTURE,
    Plan,
    PromptView,
    Rep,
    Segment,
    Timetable,
)
from vision_side.recognize import FrameResult, Recognizer, Region
from vision_side.recordings import (
    DETECTION_REGION,
    FULL_FRAME,
    Meta,
    RunWriter,
    Run,
    alignment_to_json,
    run_dir,
)
from vision_side.replay import Report, replay_log
from vision_side.triggers import DIRECTIONS

LOOP_S = 0.02
"""主循环每轮歇多久。和正式程序同一档。"""

LEAD_TEXT = "WALK TO THE MARK"
"""开头那段,人从电脑前走回操作位。"""

CHECK_HEADLINE = "FRAMING CHECK"
CHECK_TEXT = "TILT THE SCREEN, TRY THE GESTURES - NOTHING IS BEING RECORDED"
"""取景检查那一档的字。写明没在录,免得以为录上了(第 40 条 user story)。"""

DONE_HEADLINE = "DONE"
BLANK_REASON = "CAMERA BLANK - CHECK THE PRIVACY SHUTTER"
"""认出摄像头占位图时那一行。拦住开始,不白录一遍(第 41 条)。"""

TABLE_S = 120.0
"""录完那张表在屏幕上挂多久(没人按 q 的话)。

录的人站在 1–1.5 m 外,走回电脑前要几秒,看完一屏数字要几十秒;挂两分钟够了,
再久就该自己退出——表已经写进日志和 `replay-log.json`,不会丢。
"""


# ---- 套在识别器外面的那一层 --------------------------------------------


class Recorder(Recognizer):
    """录的时候套在识别器外面:结果照样往上交,顺手写进逐帧日志和视频。

    做成一层套子而不是改 `VisionLoop`,为的是**录的时候走的就是正式程序那一条路**
    ——同一个循环、同一次对准、同一套判定,录制只是在识别结果这个缝上多接了一支笔。

    `arm()` 之前什么也不写:启动检查和启动对准都在录制开始之前,那几百帧不该进数据。
    """

    def __init__(self, inner: Recognizer, writer: RunWriter | None = None) -> None:
        self._inner = inner
        self._writer = writer
        self._armed = False
        self.written = 0
        self.t0: float | None = None
        """录的第一帧的时间戳。逐帧日志和提示时间表都从它算 0。"""

    @property
    def armed(self) -> bool:
        return self._armed

    def attach(self, writer: RunWriter) -> None:
        """接上要写的那一份。启动对准过了、真要开录时才接。"""
        self._writer = writer

    def arm(self) -> None:
        """从下一帧起开始写。没接上要写的那一份时什么也不做。"""
        self._armed = self._writer is not None

    def disarm(self) -> None:
        """不再写了。录完先收这一下,再去收尾那一份(主线程调)。

        识别线程可能正好有一帧在手上,所以这一下**不保证**立刻停——真正兜住的是
        `RunWriter` 关掉以后直接丢帧那一条。这里只是让后面的帧不必白走一趟。
        """
        self._armed = False

    def recognize(self, frame: Frame, region: Region | None = None) -> FrameResult:
        result = self._inner.recognize(frame, region)
        if self._armed and self._writer is not None:
            if self.t0 is None:
                self.t0 = frame.t
            # 时间戳一律从录的第一帧算 0:提示时间表和逐帧日志才对得上,
            # 而且换台机器、换一路视频跑出来的时间轴都一样(回放要逐条一致)
            shifted = Frame(frame.image, frame.t - self.t0, frame.index)
            # 收尾之后来的帧会被丢掉(`RunWriter` 关了就不写),那几帧不该算进来
            # ——「录完 N 帧」得和 `frames.jsonl` 里真有的行数对得上
            if self._writer.write(shifted, _shift(result, self.t0)) is not None:
                self.written += 1
        return result

    def clear_tracks(self) -> None:
        self._inner.clear_tracks()


def _shift(result: FrameResult, t0: float) -> FrameResult:
    return FrameResult(
        t=result.t - t0, hands=result.hands, flat=result.flat, region=result.region
    )


# ---- 提示走到哪儿了 ----------------------------------------------------


class Schedule:
    """按计划一段一段往前走,顺手把每一段真正占了哪段时间记进时间表。

    **时间按识别线程报上来的帧时间算**,不按 `time.monotonic()`:显示卡一下,提示
    就该跟着晚,而时间表记下的必须是画面上真正发生的时刻——回放拿它当标准答案。

    一个线程用(主线程)。
    """

    def __init__(self, plan: Plan, *, lead_s: float = LEAD_S) -> None:
        self.plan = plan
        self.lead_s = lead_s
        self.timetable = Timetable()
        self._t0: float | None = None
        self._at = 0
        """走到第几段了。把全部提示摊平成一串段,按它数。"""
        self._flat: list[tuple[Rep, Segment]] = [
            (rep, seg) for rep in plan.reps for seg in rep.segments
        ]
        self._starts: list[float] = []
        acc = lead_s
        for _rep, seg in self._flat:
            self._starts.append(acc)
            acc += seg.seconds
        self._total = acc
        self._opened = -1
        """时间表上开着的是第几段。换段才另起一行,「做!」那一声也只在换过去时响。"""

    @property
    def total_s(self) -> float:
        return self._total

    @property
    def done(self) -> bool:
        return self._at >= len(self._flat)

    def start(self, t: float) -> None:
        self._t0 = t

    def update(self, t: float, beeps: Beeper | None = None) -> PromptView | None:
        """推到这一刻。返回屏幕上该显示什么;开头那段返回 `None` 的话就没法提示人
        走过去,所以那一段也返回一份(`seg` 是空的)。"""
        if self._t0 is None:
            self.start(t)
        assert self._t0 is not None
        # 老实写 `is not None`:`self._t0 or t` 在起点正好是 0.0 时会当成「还没开始」,
        # 提示就永远停在第一段上(测试里第一帧的时间戳就是 0.0)
        elapsed = t - self._t0
        if elapsed < self.lead_s:
            return PromptView(
                headline=self.plan.name.upper(),
                text=LEAD_TEXT,
                seg="",
                left_s=max(0.0, self.lead_s - elapsed),
                progress=0.0,
            )
        # 走到哪一段了:按计划里的秒数找,换段了就在时间表上另起一行
        while self._at < len(self._flat) and elapsed >= self._end_of(self._at):
            self._at += 1
            self._skipped(t)
        if self.done:
            self.timetable.close(t)
            return None
        rep, segment = self._flat[self._at]
        if self._opened != self._at:
            self._opened = self._at
            self.timetable.open(rep, segment, t)
            if segment.name == GO and beeps is not None:
                beeps.go()  # 「做!」那一声高音
        else:
            self.timetable.close(t)
        return PromptView(
            headline=rep.label,
            text=segment.text,
            seg=segment.name,
            left_s=max(0.0, self._end_of(self._at) - elapsed),
            progress=min(1.0, elapsed / max(1e-6, self._total)),
        )

    def rep_now(self) -> Rep | None:
        """现在在哪一条提示上。开头那段和走完了都是 `None`。"""
        if self.done or self._t0 is None:
            return None
        return self._flat[self._at][0]

    def _skipped(self, t: float) -> None:
        """整整一段都没显示出来:在时间表上留一行长度为 0 的,别当它没发生过。

        主循环卡得比一段还久时会这样(上一轮卡过 1.4 秒,而一段 2–3 秒,再慢一点就
        跨过去了)。不留这一行的话,**跨过去的要是「做!」那一段**,这一条提示就一行
        `go` 都没有,`Timetable.reps()` 只好退回拿准备段的开头当「做!」——于是操作者
        在准备段里冒出来的那一次会被算成**认出**,识别率凭空变高(第 05 票复查)。
        """
        gone = self._at - 1
        if gone < 0 or self._opened == gone or gone >= len(self._flat):
            return
        rep, segment = self._flat[gone]
        self._opened = gone
        self.timetable.open(rep, segment, t)

    def _end_of(self, index: int) -> float:
        return self._starts[index] + self._flat[index][1].seconds


# ---- 示范图和箭头 ------------------------------------------------------


def demo_for(
    rep: Rep | None,
    alignment: Alignment,
    demo: DemoConfig,
    images: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, tuple[tuple[float, float], tuple[float, float]] | None]:
    """这一条提示配哪张示范图、箭头画在哪儿。

    **箭头的起点和终点画在真实动作的位置**(spec「录制工具」第 1 条):起点是起点区
    中心——判定要求停住段落在起点区里,所以手本来就该在那儿;终点是朝那个方向移过
    `move_hand_widths` 个手宽之后的位置,也就是判定要求的最短距离。照着箭头做出来的
    那一下,正好够触发,不多不少。
    """
    if rep is None:
        return images.get("fist"), None
    if rep.kind == STOP_GESTURE:
        return images.get("palm"), None
    if rep.kind == FREE:
        return None, None
    if rep.key in DIRECTIONS:
        ux, uy = DIRECTIONS[rep.key]
        reach = alignment.hand_width * demo.vision.move_hand_widths
        cx, cy = alignment.start_zone.center
        return images.get("fist"), ((cx, cy), (cx + ux * reach, cy + uy * reach))
    if rep.kind == FIST_MOVE:
        return images.get("fist"), None  # 往前推、敲两下:画面里画不出箭头,全靠那一句说明
    return images.get(rep.key), None


# ---- 整条流程 ----------------------------------------------------------


class RecordSession:
    """录一段,或者只看取景。主线程跑这个。

    `plan` 是 `None` 就是取景检查那一档:不建 `RunWriter`,一帧也不写。
    """

    def __init__(
        self,
        source: FrameSource,
        recognizer: Recognizer,
        demo: DemoConfig,
        log: logging.Logger,
        *,
        plan: Plan | None,
        folder: Path,
        video: bool = True,
        seconds: float | None = None,
        table_s: float = TABLE_S,
        window: PreviewWindow | None = None,
        beeps: Beeper | None = None,
    ) -> None:
        self._source = source
        self._inner = recognizer
        self._demo = demo
        self._log = log
        self._plan = plan
        self._folder = folder
        self._want_video = video
        self._window = window
        self._beeps = beeps
        self._health = LoopHealth("录制主循环", log)
        self._table_s = table_s
        self._deadline = None if seconds is None else time.monotonic() + seconds
        self._quit = False
        self._last: VisionUpdate | None = None
        self._writer: RunWriter | None = None
        self._recorder: Recorder | None = None
        self._painter: Painter | None = None
        self._thread: VisionThread | None = None
        self._loop: VisionLoop | None = None
        self._schedule: Schedule | None = None
        self._images: dict[str, np.ndarray] = {}
        self.run: Run | None = None
        """录完的那一份。取景检查那一档是 `None`。"""

    # ---- 跑 ------------------------------------------------------------

    def run_session(self) -> int:
        """开摄像头 → 启动检查 → 启动对准 → 按提示录 → 出表。返回退出码。"""
        with self._source:
            frame = first_frame(self._source)
            self._log.info("启动检查 第一帧 %dx%d", frame.width, frame.height)
            # **拦住摄像头占位图**:隐私开关关着时这里就报错退出,不白录一遍
            check_vision(
                self._source,
                self._inner,
                gap_s=self._demo.run.vision_lost_after_s,
                log=self._log,
            )
            return self._with_vision((frame.width, frame.height))

    def _with_vision(self, size: tuple[int, int]) -> int:
        self._recorder = Recorder(self._inner, None)
        robot = RobotSide(self._demo, log=self._log)
        self._loop = VisionLoop(
            self._source,
            self._recorder,
            self._demo,
            frame_size=size,
            robot=robot,
            beeps=self._beeps,
            log=self._log,
        )
        close_window = self._window is None
        if self._window is None:
            self._window = PreviewWindow(
                self._demo,
                on_confirm=lambda _s: None,
                on_quit=self._on_quit,
                on_realign=self._on_realign,
                title="RECORD" if self._plan is not None else "FRAMING CHECK",
            )
        self._images = load_legend_images(legend_poses(self._demo), log=self._log)
        self._painter = Painter(
            self._demo, px_per_cm=self._window.px_per_cm, images=self._images
        )
        # 图例放到预览窗口给它的地方,和正式程序一样(`vision_side.app`):画在画面那块里
        # 只有中间一条那么高,十几行放不下(`_tick` 里 legend=False)
        self._window.set_legend(self._painter)
        self._thread = VisionThread(self._loop, log=self._log)
        try:
            with self._thread:
                if not self._align():
                    return 1
                if self._plan is None:
                    self._framing_check()
                    return 0
                return self._record(size)
        finally:
            self._log.info("%s", self._health.describe())
            if close_window and self._window is not None:
                self._window.close()

    def _align(self) -> bool:
        """启动对准。**录制前也做**,把区域记进日志(spec 第 3 条)。"""
        assert self._loop is not None
        self._log.info("对准 开始:站到操作位,把拳头举起来停住")
        view = View(State.COOLDOWN, headline="ALIGN", reason="NOT RECORDING YET")
        while not self._quit:
            self._tick(view, prompt=None)
            if self._loop.alignment is not None:
                return True
            if self._dead():
                return False
            time.sleep(LOOP_S)
        self._log.warning("对准 没对上就退出了,什么也没录")
        return False

    def _framing_check(self) -> None:
        """取景检查:只显示画面和识别结果,不录像;同屏放出示范图。"""
        assert self._loop is not None
        self._log.info("取景检查 开始。按 q / Esc 退出;一帧也不会写下来")
        alignment = self._loop.alignment
        assert alignment is not None
        image, _arrow = demo_for(None, alignment, self._demo, self._images)
        view = View(State.IDLE, headline=CHECK_HEADLINE, reason="NOTHING IS RECORDED")
        while not self._quit and not self._dead():
            prompt = PromptView(
                headline=CHECK_HEADLINE,
                text=CHECK_TEXT,
                seg=PREP,
                left_s=0.0,
                progress=0.0,
                demo=image,
            )
            self._tick(view, prompt=prompt)
            time.sleep(LOOP_S)

    def _record(self, size: tuple[int, int]) -> int:
        """按提示录一遍。"""
        assert self._loop is not None and self._recorder is not None
        plan = self._plan
        assert plan is not None
        alignment = self._loop.alignment
        assert alignment is not None

        if self._blank_now():
            # 对准到开录之间隐私开关被关上了:这时开录就是白录一遍
            self._log.error("录制 %s,不开录", BLANK_REASON)
            print(BLANK_REASON, file=sys.stderr)
            return 1

        path = run_dir(plan.name, folder=self._folder)
        send = FULL_FRAME if plan.name == "stop" else DETECTION_REGION
        meta = Meta(
            plan=plan.name,
            started=datetime.now().isoformat(timespec="seconds"),
            frame_size=size,
            send=send,
            alignment=alignment_to_json(alignment),
            alignment_source="recorded",
            vision=_vision_json(self._demo),
            versions=_versions(),
        )
        self._writer = RunWriter(path, meta, video=self._want_video)
        self._recorder.attach(self._writer)
        # 叫停那一组按整幅画面录:执行中送的就是整幅画面(ADR-0008)
        self._loop.set_state(State.RUNNING if send == FULL_FRAME else State.IDLE)
        self._schedule = Schedule(plan)
        self._log.info("录制 %s → %s", plan.describe(), path)

        view = View(State.IDLE, headline="RECORD", reason=plan.name.upper())
        armed = False
        try:
            while not self._quit and not self._dead():
                if not armed and self._last is not None:
                    self._recorder.arm()
                    armed = True
                now = self._now()
                if now is None:
                    # 还没写下第一帧:提示不能开始走,不然时间表会比逐帧日志早一截
                    self._tick(view, prompt=None)
                    time.sleep(LOOP_S)
                    continue
                prompt = self._schedule.update(now, self._beeps)
                if prompt is None:
                    break
                self._tick(view, prompt=self._decorate(prompt, alignment))
                time.sleep(LOOP_S)
            self._schedule.timetable.close(self._now() or 0.0)
        finally:
            # 先不再写,再收尾:识别线程还在跑,顺序反过来会让它在收尾途中接着追加
            self._recorder.disarm()
            if self._writer is not None:
                self._writer.finish(
                    self._schedule.timetable,
                    health={
                        "main": self._health.describe(),
                        "main_slow": self._health.slow,
                        "vision": self._loop_health(),
                    },
                )
        self.run = Run(path)
        self._log.info("录完 %d 帧 → %s", self._recorder.written, path)
        self._show_table(view)
        return 0

    # ---- 每一轮 --------------------------------------------------------

    def _tick(self, view: View, *, prompt: PromptView | None) -> None:
        assert self._painter is not None and self._window is not None
        assert self._thread is not None
        started = time.perf_counter()
        update = self._thread.take()
        if update is not None:
            self._last = update
        image = self._painter.paint(
            self._window.panel_size, self._last, prompt=prompt, legend=False
        )
        self._window.update(view, image)
        self._health.tick(time.perf_counter() - started)
        if self._deadline is not None and time.monotonic() > self._deadline:
            self._on_quit("--seconds")

    def _decorate(self, prompt: PromptView, alignment: Alignment) -> PromptView:
        """把示范图和箭头配上去。"""
        assert self._schedule is not None
        image, arrow = demo_for(
            self._schedule.rep_now(), alignment, self._demo, self._images
        )
        return PromptView(
            headline=prompt.headline,
            text=prompt.text,
            seg=prompt.seg,
            left_s=prompt.left_s,
            progress=prompt.progress,
            demo=image,
            arrow=arrow,
        )

    def _show_table(self, view: View) -> None:
        """**录完立即在屏幕上出表**,不用回电脑翻文件(spec 第 6 条)。

        表是拿刚写下去的逐帧日志跑一遍回放算出来的——和以后回放测试算的是同一套数,
        所以屏幕上看到的就是往后要比的那个基准。
        """
        assert self.run is not None and self._window is not None
        try:
            report = replay_log(self.run, self._demo, log=self._log)
        except Exception as exc:  # 出表失败不该让录下来的数据跟着没了
            self._log.warning("出表 算不出来:%s: %s", type(exc).__name__, exc)
            return
        report.save(self.run.path / "replay-log.json")
        text = report.table()
        self._log.info("录完出表\n%s", text)
        print(text)
        self._on_screen(view, report)

    def _on_screen(self, view: View, report: Report) -> None:
        """把表画到预览窗口上,等人按 q / Esc。站在操作位就能看见。"""
        assert self._window is not None
        done = View(
            State.IDLE,
            headline=DONE_HEADLINE,
            action=f"{report.hits}/{len(report.gesture_reps)}  "
            f"误触发 {report.false_triggers}  误叫停 {report.false_stops}",
            step=f"识别率 {report.rate:.0%}",
            reason=str(self.run.path if self.run is not None else ""),
            detail="Q / ESC = QUIT",
        )
        # 录的时候按 q 提前收了工,表照样要出——数据已经写下去了,人得看见它
        self._quit = False
        deadline = time.monotonic() + self._table_s
        while not self._quit and time.monotonic() < deadline:
            if self._deadline is not None and time.monotonic() > self._deadline:
                break
            self._window.update(done)
            time.sleep(LOOP_S)

    # ---- 小事 ----------------------------------------------------------

    def _now(self) -> float | None:
        """现在是录下来这条时间轴上的第几秒。还没写下第一帧时是 `None`。

        **提示时间表和逐帧日志必须同一条轴**:回放拿时间表当标准答案,按帧的时间戳
        去里面查这一刻该做什么。两边差一截——比如时间表按摄像头开机算、日志从录的
        第一帧算 0——每一条的窗口都会错位,算出来的数全不作数。

        所以两边都减去**录下来的第一帧**那个时间戳(`Recorder.t0`,由识别线程在真
        写下第一帧时记下)。启动对准占掉的那一两秒因此落在 0 之前,不在轴上。
        """
        if self._recorder is None or self._recorder.t0 is None or self._last is None:
            return None
        return max(0.0, self._last.t - self._recorder.t0)

    def _blank_now(self) -> bool:
        """这一刻画面是不是一片均匀(摄像头占位图、全黑)。"""
        return self._last is not None and self._last.result.flat

    def _loop_health(self) -> str:
        """识别线程卡没卡。`VisionLoop` 自己不数,这里按逐帧日志里的间隔算。"""
        if self._recorder is None or self._recorder.written == 0:
            return ""
        return f"识别线程 写下 {self._recorder.written} 帧(每帧间隔见 frames.jsonl 的 iv)"

    def _dead(self) -> bool:
        assert self._thread is not None
        if self._thread.alive or self._thread.error is None:
            return False
        self._log.error("识别线程出错:%s", self._thread.error)
        return True

    def _on_quit(self, source: str) -> None:
        self._quit = True
        self._log.info("退出:来自 %s", source)

    def _on_realign(self, source: str) -> None:
        assert self._loop is not None
        if self._recorder is not None and self._recorder.armed:
            self._log.info("重新对准 已经在录了,忽略(区域一改,这一段就没了准)")
            return
        self._log.info("重新对准:来自 %s", source)
        self._loop.realign()


def _vision_json(demo: DemoConfig) -> dict:
    """录的时候用的那一份判定参数。事后要知道这批数是拿什么参数录的。"""
    params = demo.vision
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in vars(params).items()
    }


def _versions() -> dict:
    import cv2

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }

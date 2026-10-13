"""识别线程和它接到机器人一侧的那几条线(第 04 票的自动测试)。

接的是编好的识别结果(`tests/vision_fakes.py`),**不开摄像头也不开 RoboDK**。
一帧一步地驱动(`VisionLoop.step()`),所以先后关系是定死的,不看线程调度的运气;
真起线程那一条单独验(`VisionThread`)。

覆盖票里点名的四件事:

- 触发排进状态逻辑的事件队列(和调试按键同一条);
- 停止手势就地叫停(和空格同一条,ADR-0012);
- 按状态切换送哪块画面,切换时清跟踪记录和判定记录,**停止手势的连续帧计数不清**;
- 识别中断:执行中进已停止,空闲时只报错。
"""

from __future__ import annotations

import logging
import time

import pytest

from robot_side.config import load_demo
from robot_side.state import Event, State
from robot_side.stop import StopSignal
from tests.conftest import Rig
from tests.fakes import RecordingArm
from tests.vision_fakes import (
    FPS,
    ScriptedFrameSource,
    ScriptedRecognizer,
    hand,
    nothing,
    still,
    sweep,
)
from vision_side.frames import FrameError
from vision_side.loop import RobotSide, VisionLoop, VisionLost, VisionThread, check_vision
from vision_side.recognize import FrameResult

DEMO = load_demo()
PARAMS = DEMO.vision
CENTER = (640.0, 360.0)
HAND = 80.0
"""对准量出来的手宽。编好的手框宽就是它,所以起点区 240 见方、移动要过 120 px。"""


def log() -> logging.Logger:
    quiet = logging.getLogger("test.vision.quiet")
    quiet.propagate = False
    quiet.handlers = []
    return quiet


class Events:
    """只记事件的假状态逻辑。看的是「排出去了什么」,不看它之后怎么走。"""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def submit(self, event: Event) -> None:
        self.events.append(event)

    @property
    def actions(self) -> list[str]:
        return [e.action for e in self.events if e.kind == "trigger"]


def wired(*, machine=None, arm: RecordingArm | None = None) -> tuple[RobotSide, StopSignal]:
    """一套接好的 `RobotSide`,外加它用的叫停通道。"""
    signal = StopSignal(arm if arm is not None else RecordingArm())
    robot = RobotSide(DEMO, log=log())
    robot.attach(machine, signal)
    return robot, signal


def build(results=(), *, robot: RobotSide | None = None) -> tuple[VisionLoop, ScriptedRecognizer]:
    """一套跑得起来的识别线程。`results` 是**对上以后**才放的那一段,由 `align()` 换上。"""
    recognizer = ScriptedRecognizer(align_script())
    recognizer.after = list(results)
    source = ScriptedFrameSource()
    source.open()
    loop = VisionLoop(
        source,
        recognizer,
        DEMO,
        frame_size=(1280, 720),
        robot=robot if robot is not None else RobotSide(DEMO, log=log()),
        log=log(),
    )
    return loop, recognizer


def align_script() -> list[FrameResult]:
    """把拳头举着不动,直到对准成立。两段合起来 2 秒,多给半秒余量。"""
    return still(PARAMS.hold_s + PARAMS.measure_s + 0.5, CENTER, width=HAND)


def run(loop: VisionLoop, frames: int) -> list:
    return [loop.step() for _ in range(frames)]


def align(loop: VisionLoop, recognizer: ScriptedRecognizer) -> None:
    """一直走到对上为止,然后换上要测的那一段结果。

    对准要走多少帧不去数(得看演示配置里那两个时长和判定内部怎么算),对上了就换过来
    ——不然后面那一段会接在对准剩下的那几帧后面,差几帧就全错位。
    """
    for _ in range(round((PARAMS.hold_s + PARAMS.measure_s + 1.0) * FPS)):
        loop.step()
        if loop.alignment is not None:
            recognizer.restart(recognizer.after)
            return
    raise AssertionError("编好的结果没能让它对上")


def fist_move_left() -> list[FrameResult]:
    """一次向左的握拳平移:先在起点区里停住,再往左移过 1.5 个手宽。"""
    reach = PARAMS.move_hand_widths * HAND
    end = (CENTER[0] - reach - 10, CENTER[1])
    return still(PARAMS.settle_s + 0.2, CENTER, width=HAND) + sweep(
        CENTER, end, 0.3, width=HAND
    )


def palm(seconds: float) -> list[FrameResult]:
    """张开手掌(停止手势),按整幅画面算,在哪儿做都算。"""
    return still(seconds, (200.0, 200.0), label="palm", track_id=9, width=HAND)


# ---- 对准 --------------------------------------------------------------


def test_对上之前送的是对准自己要的那块画面():
    loop, recognizer = build()
    align(loop, recognizer)
    assert loop.alignment is not None
    # 第一帧一定是整幅画面(对准先在整幅画面里找拳头)
    assert recognizer.sent[0] is None
    # 对上的时候送的是检测区域
    assert recognizer.sent[-1] == loop.alignment.detection_region


def test_对上以后量到的手宽就是编好的那个():
    loop, recognizer = build()
    align(loop, recognizer)
    assert loop.alignment is not None
    assert loop.alignment.hand_width == pytest.approx(HAND, abs=1.0)


def test_对准期间认出触发手势也不排事件():
    # 对准还没成立,判定根本没建起来;这一条挡住「举着拳头晃一下就动机械臂」
    machine = Events()
    robot, _ = wired(machine=machine)
    loop, recognizer = build(fist_move_left(), robot=robot)
    run(loop, round(PARAMS.hold_s * FPS / 2))  # 还没对上就先走一段
    assert machine.events == []


def test_按R重新对准期间不接受触发手势():
    machine = Events()
    robot, _ = wired(machine=machine)
    loop, recognizer = build(fist_move_left() + fist_move_left(), robot=robot)
    align(loop, recognizer)
    loop.set_state(State.IDLE)
    loop.realign()
    # 主线程只是留个话,真正从头对是识别线程下一帧自己做的
    loop.step()
    assert loop.alignment is None
    run(loop, round(2.0 * FPS))
    assert machine.actions == []


# ---- 触发排进事件队列 --------------------------------------------------


def test_认出触发手势就排进状态逻辑的事件队列():
    rig = Rig()
    try:
        rig.to_idle()
        robot = RobotSide(DEMO, log=log())
        robot.attach(rig.machine, rig.signal)
        loop, recognizer = build(fist_move_left(), robot=robot)
        align(loop, recognizer)
        loop.set_state(State.IDLE)
        for _ in range(round(2.0 * FPS)):
            loop.step()
            rig.machine.tick()
            if rig.machine.state is State.RUNNING:
                break
        assert rig.machine.state is State.RUNNING
        # 左移对的是「指左」,对应关系查的是演示配置,不在代码里写死
        assert rig.machine.view.action == DEMO.action("point_left").display
    finally:
        rig.close()


def test_触发手势对哪个动作照演示配置查():
    machine = Events()
    robot, _ = wired(machine=machine)
    loop, recognizer = build(fist_move_left(), robot=robot)
    align(loop, recognizer)
    loop.set_state(State.IDLE)
    run(loop, round(2.0 * FPS))
    assert machine.actions == ["point_left"]
    assert "fist_move" in machine.events[0].source
    assert "left" in machine.events[0].source


def test_演示配置里没有对应的触发手势就不排事件():
    machine = Events()
    robot, _ = wired(machine=machine)
    assert robot.to_event(_trigger("pose_hold", "dislike")) is None
    assert robot.to_event(_trigger("fist_move", "left")) is not None


def test_没连机械臂时认出触发手势只记一笔不排事件():
    # 启动对准在连机械臂之前(spec「启动顺序」),这段时间里认出来也没处排
    loop, recognizer = build(fist_move_left())  # robot 没 attach
    align(loop, recognizer)
    loop.set_state(None)
    updates = [u for u in run(loop, round(2.0 * FPS)) if u is not None]
    assert any(u.trigger is not None for u in updates)  # 认是认出来了


def _trigger(kind: str, key: str):
    from vision_side.triggers import Trigger

    return Trigger(kind=kind, key=key, t=1.0, since=0.5, track_id=1)


# ---- 停止手势直接叫停 --------------------------------------------------


def test_认到停止手势就地叫停不经过主线程():
    arm = RecordingArm()
    robot, signal = wired(arm=arm)
    loop, recognizer = build(palm(1.0), robot=robot)
    align(loop, recognizer)
    loop.set_state(State.RUNNING)
    # 一帧都不让主线程插手:step() 自己就该把机械臂停了
    for _ in range(DEMO.run.stop_gesture_frames):
        loop.step()
    assert arm.stopped >= 1
    assert [r.source for r in signal.records] == ["stop-gesture"]


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def test_叫停写进日志是哪只手什么姿势在哪儿():
    # 用户故事 37。2026-09-23 实跑:四次叫停紧跟在「拳头往上」后面,日志里看不出是谁的手
    robot, _ = wired()
    loop, recognizer = build(palm(1.0), robot=robot)
    align(loop, recognizer)
    loop.set_state(State.RUNNING)
    quiet = log()
    lines = _Lines()
    quiet.addHandler(lines)
    quiet.setLevel(logging.INFO)
    try:
        run(loop, DEMO.run.stop_gesture_frames)
    finally:
        quiet.removeHandler(lines)
        quiet.setLevel(logging.NOTSET)
    said = [line for line in lines.lines if line.startswith("叫停")]
    assert len(said) == 1
    assert "PALM 手 #9 在 (200, 200) 宽 80 px" in said[0]


def test_停止手势连续帧数不够就不叫停():
    arm = RecordingArm()
    robot, _ = wired(arm=arm)
    frames = DEMO.run.stop_gesture_frames
    script = [FrameResult(t=0.0, hands=(hand((200.0, 200.0), label="palm"),))] * (
        frames - 1
    ) + nothing(0.2)
    loop, recognizer = build(script, robot=robot)
    align(loop, recognizer)
    loop.set_state(State.RUNNING)
    run(loop, frames - 1)
    assert arm.stopped == 0


def test_一次停止手势只叫停一次():
    # 手一直举着,`stop` 会一帧一帧都是真;每帧都叫一次的话记录会越堆越多,
    # 执行中那一次叫停之后紧接着的几条会被当成「回待命途中又被叫停」,直接进已停止
    arm = RecordingArm()
    robot, signal = wired(arm=arm)
    loop, recognizer = build(palm(1.0), robot=robot)
    align(loop, recognizer)
    loop.set_state(State.RUNNING)
    run(loop, round(1.0 * FPS))
    assert len(signal.records) == 1


def test_手放下再举起来算新的一次():
    arm = RecordingArm()
    robot, signal = wired(arm=arm)
    loop, recognizer = build(palm(0.3) + nothing(0.3) + palm(0.3), robot=robot)
    align(loop, recognizer)
    loop.set_state(State.RUNNING)
    run(loop, round(0.9 * FPS))
    assert len(signal.records) == 2


def test_空闲时也照样叫停起不起作用由状态逻辑决定():
    # 空闲、冷却、已停止时叫停不起作用(机械臂本来就停着),但连续帧计数照常走、
    # 叫停照常发 —— 判断「起不起作用」是状态逻辑的事(第 03 票),不是识别线程的
    arm = RecordingArm()
    robot, signal = wired(arm=arm)
    loop, recognizer = build(palm(0.5), robot=robot)
    align(loop, recognizer)
    loop.set_state(State.IDLE)
    run(loop, round(0.5 * FPS))
    assert len(signal.records) == 1


def test_叫停时机械臂报错不把识别线程带倒():
    # 带倒了就再也叫不了停了。照按键线程的办法:记下来,线程接着活着
    class Angry(RecordingArm):
        def stop(self) -> None:
            super().stop()
            raise RuntimeError("控制器不理")

    arm = Angry()
    robot, _ = wired(arm=arm)
    loop, recognizer = build(palm(1.0), robot=robot)
    align(loop, recognizer)
    loop.set_state(State.RUNNING)
    run(loop, round(0.5 * FPS))
    assert arm.stopped >= 1  # 标记立住了、停也发出去了


# ---- 按状态切换送哪块画面 ----------------------------------------------


def test_空闲和冷却送检测区域():
    loop, recognizer = build(nothing(1.0))
    align(loop, recognizer)
    assert loop.alignment is not None
    region = loop.alignment.detection_region
    for state in (State.IDLE, State.COOLDOWN):
        loop.set_state(state)
        recognizer.sent.clear()
        run(loop, 3)
        assert recognizer.sent == [region] * 3, state


def test_执行中回待命已停止送整幅画面():
    loop, recognizer = build(nothing(2.0))
    align(loop, recognizer)
    for state in (State.RUNNING, State.RETURNING, State.HALTED):
        loop.set_state(state)
        recognizer.sent.clear()
        run(loop, 3)
        assert recognizer.sent == [None] * 3, state


def test_触发成立以后从下一帧起送整幅画面():
    rig = Rig()
    try:
        rig.to_idle()
        robot = RobotSide(DEMO, log=log())
        robot.attach(rig.machine, rig.signal)
        loop, recognizer = build(fist_move_left() + nothing(1.0), robot=robot)
        align(loop, recognizer)
        loop.set_state(State.IDLE)
        at = None
        for _ in range(round(2.0 * FPS)):
            update = loop.step()
            rig.machine.tick()
            loop.set_state(rig.machine.state)
            if update is not None and update.trigger is not None:
                # 认出触发的**那一帧**送的还是检测区域(判定就是在这块上做的)
                assert recognizer.sent[-1] is not None
                at = len(recognizer.sent)
            elif at is not None:
                assert recognizer.sent[at] is None  # 下一帧就是整幅画面
                break
        else:
            raise AssertionError("编好的结果没触发")
        assert at is not None
    finally:
        rig.close()


def test_进空闲的那一帧就切回检测区域():
    loop, recognizer = build(nothing(1.0))
    align(loop, recognizer)
    loop.set_state(State.RUNNING)
    run(loop, 2)
    loop.set_state(State.IDLE)
    recognizer.sent.clear()
    loop.step()
    assert recognizer.sent == [loop.alignment.detection_region]


def test_切换送法时清跟踪记录():
    loop, recognizer = build(nothing(1.0))
    align(loop, recognizer)
    loop.set_state(State.IDLE)
    run(loop, 2)
    before = len(recognizer.cleared_at)
    loop.set_state(State.RUNNING)
    loop.step()
    assert len(recognizer.cleared_at) == before + 1
    loop.step()
    assert len(recognizer.cleared_at) == before + 1  # 没换就不再清


def test_切换送法时清触发判定记录():
    # 停住段成立、READY 亮着;换了送法记录就清掉,READY 跟着灭
    loop, recognizer = build(still(2.0, CENTER, width=HAND))
    align(loop, recognizer)
    loop.set_state(State.IDLE)
    ready = False
    for _ in range(round(1.0 * FPS)):
        update = loop.step()
        if update is not None and update.judge is not None and update.judge.ready:
            ready = True
            break
    assert ready, "编好的结果没让 READY 亮起来"
    loop.set_state(State.RUNNING)
    update = loop.step()
    assert update is not None and update.judge is not None
    assert not update.judge.ready


def test_切换送法时不清停止手势的连续帧计数():
    # 清了反而可能晚停约 0.1 秒(spec「送哪块画面」)。
    # 空闲时先数两帧张开的手,切到整幅画面,第三帧就该叫停。
    arm = RecordingArm()
    robot, _ = wired(arm=arm)
    frames = DEMO.run.stop_gesture_frames
    loop, recognizer = build(palm(1.0), robot=robot)
    align(loop, recognizer)
    loop.set_state(State.IDLE)
    run(loop, frames - 1)
    assert arm.stopped == 0
    loop.set_state(State.RUNNING)  # 换送法,跟踪记录和判定记录都清
    loop.step()
    assert arm.stopped == 1, "停止手势的连续帧计数被清掉了,晚了一帧才停"


def test_状态逻辑清判定记录时识别线程跟着清():
    # 进冷却前状态逻辑会清一次判定记录(第 03 票的 on_clear_records),
    # 但它在主线程上;真正清是识别线程下一帧自己做,不伸手进别的线程的东西
    loop, recognizer = build(still(2.0, CENTER, width=HAND))
    align(loop, recognizer)
    loop.set_state(State.IDLE)
    for _ in range(round(1.0 * FPS)):
        update = loop.step()
        if update is not None and update.judge is not None and update.judge.ready:
            break
    loop.clear_records()
    update = loop.step()
    assert update is not None and update.judge is not None
    assert not update.judge.ready


# ---- 识别中断 ----------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def watcher(clock: Clock) -> VisionLost:
    return VisionLost(DEMO.run.vision_lost_after_s, clock=clock)


def test_超过半秒没有新结果算识别中断():
    clock = Clock()
    watch = watcher(clock)
    assert not watch.update(None)
    clock.t += DEMO.run.vision_lost_after_s + 0.01
    assert watch.update(None)


def test_新结果一来就不算识别中断了():
    clock = Clock()
    watch = watcher(clock)
    clock.t += 1.0
    assert watch.update(None)
    loop, recognizer = build()
    align(loop, recognizer)
    assert not watch.update(loop.step())


def test_画面均匀算识别中断():
    clock = Clock()
    watch = watcher(clock)
    loop, recognizer = build(nothing(1.0, flat=True))
    align(loop, recognizer)
    assert watch.update(loop.step())


def test_画面正常只是没有手不算识别中断():
    clock = Clock()
    watch = watcher(clock)
    loop, recognizer = build(nothing(1.0))
    align(loop, recognizer)
    for _ in range(round(1.0 * FPS)):
        assert not watch.update(loop.step())


def lost_watcher() -> VisionLost:
    """一只已经算识别中断了的表。"""
    clock = Clock()
    watch = watcher(clock)
    clock.t += DEMO.run.vision_lost_after_s + 0.01
    assert watch.update(None)
    return watch


def test_识别中断在执行中和回待命途中排一件停机():
    for state in (State.RUNNING, State.RETURNING):
        event = lost_watcher().halt_event(state)
        assert event is not None and event.kind == "halt", state
        assert event.action == "VISION LOST"


def test_识别中断在空闲时只报错不惊动机械臂():
    # 空闲、冷却时机械臂本来就停着;已停止时也不必再报一遍
    for state in (State.IDLE, State.COOLDOWN, State.HALTED):
        assert lost_watcher().halt_event(state) is None, state


def test_同一次识别中断只排一件停机():
    clock = Clock()
    watch = watcher(clock)
    clock.t += 1.0
    watch.update(None)
    assert watch.halt_event(State.RUNNING) is not None
    assert watch.halt_event(State.RUNNING) is None  # 每轮都排的话已停止会被反复重报


def test_画面回来以后再断一次还会再排一件停机():
    clock = Clock()
    watch = watcher(clock)
    clock.t += 1.0
    watch.update(None)
    assert watch.halt_event(State.RUNNING) is not None
    loop, recognizer = build()
    align(loop, recognizer)
    assert not watch.update(loop.step())  # 画面回来了
    clock.t += 1.0
    assert watch.update(None)
    assert watch.halt_event(State.RUNNING) is not None


def test_识别中断整条走到已停止():
    rig = Rig()
    try:
        rig.to_idle()
        rig.trigger()
        rig.until(State.RUNNING)
        clock = Clock()
        watch = watcher(clock)
        clock.t += 1.0
        assert watch.update(None)
        event = watch.halt_event(rig.machine.state)
        assert event is not None
        rig.machine.submit(event)
        rig.until(State.HALTED)
        assert rig.machine.view.reason == "VISION LOST"
    finally:
        rig.close()


# ---- 真起一条线程 ------------------------------------------------------


def test_识别线程跑起来主线程取得到最新的一帧():
    loop, _ = build()
    with VisionThread(loop, log=log()) as thread:
        got = None
        deadline = time.monotonic() + 3.0
        while got is None and time.monotonic() < deadline:
            got = thread.take()
            time.sleep(0.005)
    assert got is not None
    assert got.image is not None


def test_取过一次没有新的就是空():
    # 识别中断就是靠这个数「多久没有新结果」的
    loop, _ = build()
    thread = VisionThread(loop, log=log())
    thread.start()
    try:
        deadline = time.monotonic() + 3.0
        while thread.take() is None and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        thread.close()  # 线程停了,不会再有新的
    thread.take()  # 它停下之前可能又放了一帧,先取走
    assert thread.take() is None


def test_识别线程出错就结束错留在那儿等主线程报():
    # 出错时主线程收不到新结果,按识别中断处理(spec「线程」),不让程序悄悄停住
    class Broken:
        def step(self):
            raise RuntimeError("识别炸了")

    thread = VisionThread(Broken(), log=log())
    thread.start()
    deadline = time.monotonic() + 3.0
    while thread.alive and time.monotonic() < deadline:
        time.sleep(0.005)
    thread.close()
    assert isinstance(thread.error, RuntimeError)
    assert thread.take() is None


def test_识别线程死了主线程按识别中断处理():
    clock = Clock()
    watch = watcher(clock)
    assert not watch.update(None)
    clock.t += DEMO.run.vision_lost_after_s + 0.01
    assert watch.update(None)  # 线程没了 = 再没有新结果


# ---- 启动检查第 2 步 ---------------------------------------------------


def check(results, **kwargs):
    source = ScriptedFrameSource()
    source.open()
    check_vision(source, ScriptedRecognizer(results), log=log(), **kwargs)


def test_识别连着跑满一秒就过():
    check(nothing(2.0))


def test_画面均匀就不让往下走():
    # 摄像头隐私开关关着:这时一次都不该碰机械臂
    with pytest.raises(FrameError) as caught:
        check(nothing(2.0, flat=True))
    assert caught.value.reason == "CAMERA BLANK"


def test_画面里没有人也算有识别结果():
    # 「有识别结果」不是「看得见手」
    check(nothing(2.0))


def test_取不到画面就报等不到():
    class Empty(ScriptedFrameSource):
        def _read(self):
            return None

    source = Empty()
    source.open()
    with pytest.raises(FrameError) as caught:
        check_vision(source, ScriptedRecognizer(), timeout_s=0.05, log=log())
    assert caught.value.reason == "NO CAMERA IMAGE"


def test_画面放完了就报没有画面():
    class Ended(ScriptedFrameSource):
        def _read(self):
            return None

        def has_next(self) -> bool:
            return False

    source = Ended()
    source.open()
    with pytest.raises(FrameError) as caught:
        check_vision(source, ScriptedRecognizer(), log=log())
    assert caught.value.reason == "NO FRAMES"


# ---- 方向箭头按住一会儿(复查时发现只闪一帧) --------------------------


def drive_to_trigger(loop: VisionLoop, recognizer, frames: int = 90):
    """一直走到认出触发手势那一帧,返回那一帧。"""
    for _ in range(frames):
        update = loop.step()
        if update is not None and update.trigger is not None:
            return update
    raise AssertionError("编好的结果没触发")


def test_认出方向以后箭头按住一会儿不是只闪一帧():
    # 判定那边每帧开头就把方向清掉,只在认出来那一帧报;30 fps 下就是 33 ms,
    # 屏幕上根本看不见(spec 第 21 条 user story)
    from vision_side.loop import ARROW_S

    loop, recognizer = build(fist_move_left() + nothing(3.0))
    align(loop, recognizer)
    loop.set_state(State.IDLE)
    at = drive_to_trigger(loop, recognizer)
    assert at.judge is not None and at.judge.direction == "left"

    later = [loop.step() for _ in range(round(ARROW_S * FPS * 0.5))]
    assert later[-1].judge.direction == "left", "箭头半路就没了"

    gone = [loop.step() for _ in range(round(ARROW_S * FPS))]
    assert gone[-1].judge.direction is None, "箭头一直不撤"


def test_箭头记着是哪只手做的():
    # 触发那一下锁定已经解除(ADR-0010),拿「画面里最大的手」会画到旁人手上
    loop, recognizer = build(fist_move_left() + nothing(3.0))
    align(loop, recognizer)
    loop.set_state(State.IDLE)
    at = drive_to_trigger(loop, recognizer)
    assert at.trigger is not None
    assert at.judge.direction_track == at.trigger.track_id


def test_重新对准把箭头也撤掉():
    loop, recognizer = build(fist_move_left() + nothing(3.0))
    align(loop, recognizer)
    loop.set_state(State.IDLE)
    drive_to_trigger(loop, recognizer)
    loop.realign()
    update = loop.step()
    assert update is not None and update.judge is None  # 回到对准那一段了


# ---- 触发以后不等主线程就切整幅画面 ------------------------------------


def test_触发的下一帧就送整幅画面不等主线程():
    # 主线程会卡住(拖预览窗口卡过 6.8 秒),那期间机械臂已经在动,而这边还只送
    # 检测区域的话,站在区外的人做停止手势就看不见了(ADR-0008、ADR-0012)
    loop, recognizer = build(fist_move_left() + nothing(3.0))
    align(loop, recognizer)
    loop.set_state(State.IDLE)
    drive_to_trigger(loop, recognizer)
    assert recognizer.sent[-1] is not None  # 触发那一帧判的还是检测区域
    recognizer.sent.clear()
    loop.step()  # 主线程一步没动,set_state 还是 IDLE
    assert recognizer.sent == [None], "等主线程传状态才切,卡住时就漏停了"


def test_冷却时认出的触发手势不切整幅画面():
    # 冷却时状态逻辑本来就会忽略这一件,切过去就是白切:紧接着那一秒正好是人重做
    # 手势的时候,而整幅画面里拳头只有约 21 px、检测区域里约 34 px(ADR-0008),
    # 白切反而让重做更难认。2026-09-22 站着验收时 17 次手势里有 5 次落在冷却上
    loop, recognizer = build(fist_move_left() + nothing(2.0))
    align(loop, recognizer)
    loop.set_state(State.COOLDOWN)
    drive_to_trigger(loop, recognizer)
    run(loop, 3)
    assert recognizer.sent[-1] is not None, "白切到整幅画面了"


def test_空闲时触发但状态逻辑没收下过一会儿自己退回检测区域():
    # 演示配置里没有对应的预设动作之类:空闲时认出来就切过去了,但状态一直不变。
    # 多送一秒整幅画面不会错过什么,但不能一直送下去——检测区域那点分辨率优势
    # 是握拳平移认得准的全部理由
    from vision_side.loop import AFTER_TRIGGER_FULL_FRAME_S

    loop, recognizer = build(fist_move_left() + nothing(4.0))
    align(loop, recognizer)
    loop.set_state(State.IDLE)  # 主线程一直不动,状态卡在空闲
    drive_to_trigger(loop, recognizer)
    run(loop, 2)
    assert recognizer.sent[-1] is None  # 先切过去
    run(loop, round((AFTER_TRIGGER_FULL_FRAME_S + 0.3) * FPS))
    assert recognizer.sent[-1] is not None, "白切了还一直不退回来"


# ---- 认出来但没受理时,屏幕上要报一下 ----------------------------------


def test_冷却时认出的手势会报出来让人知道白做了():
    # 不报的话人做完什么也没发生,只能再做一次;2026-09-22 站着验收时 17 次里有 5 次
    loop, recognizer = build(fist_move_left() + nothing(2.0))
    align(loop, recognizer)
    loop.set_state(State.COOLDOWN)
    at = drive_to_trigger(loop, recognizer)
    assert at.ignored is not None
    assert (at.ignored.kind, at.ignored.key) == ("fist_move", "left")
    assert at.ignored.state is State.COOLDOWN


def test_执行中和已停止时认出的手势也报():
    for state in (State.RUNNING, State.HALTED):
        loop, recognizer = build(fist_move_left() + nothing(2.0))
        align(loop, recognizer)
        loop.set_state(state)
        at = drive_to_trigger(loop, recognizer)
        assert at.ignored is not None and at.ignored.state is state, state


def test_空闲时认出的手势不报白做():
    # 这一件会被受理,没什么好报的
    loop, recognizer = build(fist_move_left() + nothing(2.0))
    align(loop, recognizer)
    loop.set_state(State.IDLE)
    at = drive_to_trigger(loop, recognizer)
    assert at.ignored is None


def test_报完一会儿自己撤掉():
    from vision_side.loop import NOTICE_S

    loop, recognizer = build(fist_move_left() + nothing(4.0))
    align(loop, recognizer)
    loop.set_state(State.COOLDOWN)
    drive_to_trigger(loop, recognizer)
    half = [loop.step() for _ in range(round(NOTICE_S * FPS * 0.5))]
    assert half[-1].ignored is not None, "还没看清就撤了"
    gone = [loop.step() for _ in range(round(NOTICE_S * FPS))]
    assert gone[-1].ignored is None, "一直挂着不撤"


# ---- 循环卡没卡 --------------------------------------------------------


def test_循环卡过就记一条也数进去():
    from vision_side.loop import LoopHealth

    health = LoopHealth("测试循环", log(), slow_s=0.2)
    for _ in range(10):
        health.tick(0.01)
    health.tick(0.5)
    assert health.slow == 1
    assert "11 轮" in health.describe()
    assert "卡过 1 次" in health.describe()


def test_一轮都没跑过也说得出话():
    from vision_side.loop import LoopHealth

    assert "一轮都没跑过" in LoopHealth("测试循环", log()).describe()

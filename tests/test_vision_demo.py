"""整条链路走一遍:手势 → 状态逻辑 → 假机械臂(第 04 票验收第 1 条的自动版)。

**不开摄像头、不开 RoboDK、不弹窗口**:识别结果是编好的,机械臂是记账用的假机械臂
(一问就说走完了,所以跑得飞快)。走的是真的识别线程、真的状态逻辑、真的叫停通道。

票里验收第 1 条是人站在摄像头前跑的——六个触发手势各做一次,执行中叫停,回待命途中
再叫停。这一份把**除了「人真的做手势」以外**的每一段都跑了:手势做出来该怎么判,由
第 03 票的测试管;这里管的是判出来之后接线对不对。

六个手势的画面是照真实动作编的:握拳平移先在起点区里停住 0.3 秒再移过 1.5 个手宽,
姿势保持在起点区里把同一个姿势保持满 1 秒。参数一律用真的演示配置。
"""

from __future__ import annotations

import logging
import time

import pytest

from robot_side.config import HOME_ACTION, load_demo
from robot_side.state import Event, State
from tests.conftest import Rig
from tests import images
from tests.vision_fakes import (
    FPS,
    ScriptedFrameSource,
    ScriptedRecognizer,
    nothing,
    still,
    sweep,
)
from vision_side.loop import VISION_LOST, RobotSide, VisionLoop, VisionLost
from vision_side.recognize import FrameResult

DEMO = load_demo()
PARAMS = DEMO.vision
CENTER = (640.0, 360.0)
HAND = 80.0

STEP_S = 1.0 / FPS
"""一帧多久。机器人一侧的假时钟也按这个往前走,倒计时和冷却才跟画面对得上。"""

LIMIT = 3000
"""一段最多推多少帧。六个动作各有 3 秒倒计时加 2 秒冷却,余量给足。"""


def quiet() -> logging.Logger:
    log = logging.getLogger("test.demo.quiet")
    log.propagate = False
    log.handlers = []
    return log


def fist_move(direction: str) -> list[FrameResult]:
    """一次握拳平移:起点区里停住 0.3 秒,再朝这个方向移过 1.5 个手宽。

    往前是原地把手框推大,往后是往下点两下(见 `push`、`knock`)。
    """
    if direction == "forward":
        return push()
    if direction == "back":
        return knock()
    reach = PARAMS.move_hand_widths * HAND + 12
    dx, dy = {
        "left": (-reach, 0.0),
        "right": (reach, 0.0),
        "up": (0.0, -reach),
        "down": (0.0, reach),
    }[direction]
    end = (CENTER[0] + dx, CENTER[1] + dy)
    return (
        still(PARAMS.settle_s + 0.2, CENTER, width=HAND)
        + sweep(CENTER, end, 0.3, width=HAND)
        + nothing(0.6)  # 手放下:免得收手那一下被当成反方向
    )


def push() -> list[FrameResult]:
    """一次往前推:起点区里停住,再在 0.3 秒内把手框原地推大到门槛以上。"""
    big = HAND * (PARAMS.push_size_ratio + 0.2)
    n = round(0.3 * FPS)
    growing = [
        still(1.0 / FPS, CENTER, width=HAND + (big - HAND) * i / n)[0] for i in range(1, n + 1)
    ]
    return (
        still(PARAMS.settle_s + 0.2, CENTER, width=HAND)
        + growing
        + still(0.3, CENTER, width=big)
        + nothing(0.6)
    )


def knock() -> list[FrameResult]:
    """一次敲两下:起点区里停住,再往下点两下,每下比门槛深一点。"""
    low = (CENTER[0], CENTER[1] + (PARAMS.knock_hand_widths + 0.2) * HAND)
    once = sweep(CENTER, low, 0.15, width=HAND) + sweep(low, CENTER, 0.15, width=HAND)
    return still(PARAMS.settle_s + 0.2, CENTER, width=HAND) + once + once + nothing(0.6)


def pose_hold(pose: str) -> list[FrameResult]:
    """一次姿势保持:起点区里把同一个姿势保持满 1 秒,然后松开。"""
    return still(PARAMS.pose_hold_s + 0.3, CENTER, label=pose, width=HAND) + nothing(
        PARAMS.release_s + 0.4
    )


def palm(seconds: float) -> list[FrameResult]:
    """张开手掌(停止手势)。按整幅画面算,在哪儿做都算。"""
    return still(seconds, (220.0, 220.0), label="palm", track_id=7, width=HAND)


def gesture_for(trigger) -> list[FrameResult]:
    return fist_move(trigger.key) if trigger.kind == "fist_move" else pose_hold(trigger.key)


class Demo:
    """接好的一整套:识别线程 + 状态逻辑 + 假机械臂。"""

    def __init__(self, *, busy_s: float = 0.0) -> None:
        self.rig = Rig(busy_s=busy_s)
        # 每段运动要真走一会儿的时候,得让执行线程有机会跑;每段一问就走完时不用等
        self._sleep_s = 0.001 if busy_s else 0.0
        self.recognizer = ScriptedRecognizer(
            still(PARAMS.hold_s + PARAMS.measure_s + 0.5, CENTER, width=HAND)
        )
        self.source = ScriptedFrameSource(image=images.normal())
        self.source.open()
        self.robot = RobotSide(DEMO, log=quiet())
        # 识别中断判定在主线程(spec「识别中断」),所以这里也照正式程序摆一个
        self.watch = VisionLost(DEMO.run.vision_lost_after_s, clock=self._clock)
        self.lost = False
        self.loop = VisionLoop(
            self.source,
            self.recognizer,
            DEMO,
            frame_size=(1280, 720),
            robot=self.robot,
            log=quiet(),
        )
        # 进冷却时状态逻辑清一次判定记录,正式程序里接的就是这一条
        self.rig.on_clear = self.loop.clear_records

    def align(self) -> None:
        for _ in range(round((PARAMS.hold_s + PARAMS.measure_s + 1.0) * FPS)):
            self.loop.step()
            if self.loop.alignment is not None:
                return
        raise AssertionError("编好的结果没能让它对上")

    def start(self) -> None:
        """对准 → 接上机械臂 → 启动 → 一路到空闲。"""
        self.align()
        self.robot.attach(self.rig.machine, self.rig.signal)
        self.rig.machine.start()
        assert self.until(State.IDLE), "启动之后没走到空闲"

    def play(self, script: list[FrameResult]) -> None:
        """换上这一段画面,从头放。"""
        self.recognizer.restart(script)

    def step(self) -> None:
        """一帧:识别线程走一步,主线程跟一轮。

        **顺序照正式程序**:识别线程先出这一帧(它用的是上一轮告诉它的状态),主线程
        再 tick 一次把事件收掉,最后才把新状态告诉识别线程。所以触发成立的那一帧送的
        还是检测区域,下一帧才换成整幅画面(spec「送哪块画面」)。
        """
        update = self.loop.step()
        # 主线程:判识别中断,要停机就排一件(和 `App._pump` / `App._on_lost` 一样)
        self.lost = self.watch.update(update)
        if self.lost:
            event = self.watch.halt_event(self.rig.machine.state)
            if event is not None:
                self.rig.machine.submit(event)
        self.rig.machine.tick()
        self.loop.set_state(self.rig.machine.state)
        self.rig.clock.advance(STEP_S)
        if self._sleep_s:
            time.sleep(self._sleep_s)

    def _clock(self) -> float:
        """识别中断按假时钟数,和机器人一侧同一个刻度。"""
        return self.rig.clock()

    def until(self, want: State, frames: int = LIMIT) -> bool:
        for _ in range(frames):
            self.step()
            if self.rig.machine.state is want:
                return True
        return False

    def close(self) -> None:
        self.rig.close()


@pytest.fixture
def demo():
    d = Demo()
    d.start()
    yield d
    d.close()


@pytest.fixture
def busy_demo():
    """每段运动要走 0.3 秒的那一档。

    叫停要落在**动作走到一半**的时候才有意义;假机械臂默认一问就说走完了,执行中
    只撑得住一两帧,停止手势的连续 3 帧还没数完就已经进冷却了。
    """
    d = Demo(busy_s=0.3)
    d.start()
    yield d
    d.close()


# ---- 六个触发手势各触发一次 --------------------------------------------


ACTION_TRIGGERS = [t for t in DEMO.triggers if t.action != HOME_ACTION]
HOME_TRIGGERS = [t for t in DEMO.triggers if t.action == HOME_ACTION]


@pytest.mark.parametrize("trigger", ACTION_TRIGGERS, ids=lambda t: f"{t.kind}:{t.key}")
def test_六个触发手势各走完一个预设动作(demo, trigger):
    demo.play(gesture_for(trigger) + nothing(2.0))
    assert demo.until(State.RUNNING), f"{trigger.kind} {trigger.key} 没触发"
    assert demo.rig.machine.view.action == DEMO.action(trigger.action).display
    # 走完 → 冷却 → 空闲
    assert demo.until(State.IDLE), "走完之后没回空闲"
    assert demo.rig.arm.moves, "机械臂一步没动"


@pytest.mark.parametrize("trigger", HOME_TRIGGERS, ids=lambda t: f"{t.kind}:{t.key}")
def test_回待命手势从停靠点回到待命点(demo, trigger):
    demo.play(fist_move("left") + nothing(2.0))
    assert demo.until(State.RUNNING)
    assert demo.until(State.IDLE)
    demo.play(gesture_for(trigger) + nothing(2.0))
    assert demo.until(State.RETURNING), f"{trigger.key} 没让它回待命"
    assert demo.until(State.IDLE)
    assert demo.rig.machine.view.step == "AT STANDBY"


def test_六个手势一口气走完(demo):
    # 一个接一个,中间不重建:冷却、清判定记录、送法切回来这几处接得对不对,
    # 只有连着跑才看得出来。回待命手势放在最后:先往左挪一步,它才有地方可回
    for trigger in ACTION_TRIGGERS:
        demo.play(gesture_for(trigger) + nothing(2.0))
        assert demo.until(State.RUNNING), f"{trigger.kind} {trigger.key} 没触发"
        assert demo.rig.machine.view.action == DEMO.action(trigger.action).display
        assert demo.until(State.IDLE), f"{trigger.key} 走完之后没回空闲"
    for trigger in HOME_TRIGGERS:
        demo.play(fist_move("left") + nothing(2.0))
        assert demo.until(State.RUNNING)
        assert demo.until(State.IDLE)
        demo.play(gesture_for(trigger) + nothing(2.0))
        assert demo.until(State.RETURNING), f"{trigger.key} 没让它回待命"
        assert demo.until(State.IDLE)


# ---- 执行中叫停 → 回待命 → 冷却 → 空闲 --------------------------------


def test_执行中用停止手势叫停走倒计时回待命冷却空闲(busy_demo):
    busy_demo.play(fist_move("left") + nothing(1.0))
    assert busy_demo.until(State.RUNNING)
    stopped = busy_demo.rig.arm.stopped

    busy_demo.play(palm(0.5) + nothing(2.0))
    assert busy_demo.until(State.RETURNING), "停止手势没让它进回待命"
    assert busy_demo.rig.arm.stopped > stopped, "机械臂没被停住"
    assert "STOP-GESTURE" in busy_demo.rig.machine.view.reason
    assert busy_demo.until(State.IDLE), "回待命之后没回空闲"


def test_回待命途中再叫停就进已停止回车再回待命(busy_demo):
    busy_demo.play(fist_move("right") + nothing(1.0))
    assert busy_demo.until(State.RUNNING)
    # 第一次叫停:执行中 → 回待命。手放下再举起来才算第二次
    busy_demo.play(palm(0.4) + nothing(0.4) + palm(0.4) + nothing(3.0))
    assert busy_demo.until(State.HALTED), "回待命途中再叫停没进已停止"

    busy_demo.rig.machine.submit(Event.confirm("key:enter"))
    assert busy_demo.until(State.IDLE), "按回车之后没回待命再回空闲"


# ---- 送哪块画面跟着状态走 ----------------------------------------------


def test_一次动作里送法只切两次(demo):
    # 空闲送检测区域 → 执行中整幅画面 → 回空闲再切回来。已停止时不认手势,
    # 送哪块都一样,所以跟着整幅画面走,一次运动只切两次(spec「送哪块画面」)
    demo.recognizer.sent.clear()
    demo.play(fist_move("up") + nothing(2.0))
    assert demo.until(State.RUNNING)
    assert demo.until(State.IDLE)
    switches = 0
    for before, after in zip(demo.recognizer.sent, demo.recognizer.sent[1:]):
        if before != after:
            switches += 1
    assert switches == 2, f"切了 {switches} 次"


def test_执行中送的是整幅画面(demo):
    demo.play(fist_move("down") + nothing(2.0))
    assert demo.until(State.RUNNING)
    demo.recognizer.sent.clear()
    demo.step()
    assert demo.recognizer.sent == [None]


# ---- 识别中断:摄像头隐私开关关上(验收第 3 条) -----------------------


def test_执行中摄像头被挡住就进已停止(busy_demo):
    """把摄像头画面换成隐私开关关着时那张占位图,整条链路要走到已停止。

    **「画面均匀」是真算出来的**(`ScriptedRecognizer(real_flat=True)` 里跑的是真
    识别器那个 `UniformWatch`),不是编一个 `flat=True` 塞进去——那样等于把要验的
    那一步跳过了。占位图本身照现场那张来(`tests/images.py`,2026-09-17 握拳平移
    原型开头 27 秒就是它,整段录制作废)。
    """
    busy_demo.recognizer.real_flat = True
    busy_demo.source.image = images.normal()
    busy_demo.play(fist_move("left") + nothing(6.0))
    assert busy_demo.until(State.RUNNING)

    # 隐私开关关上:画面变成占位图,手还是「看不见」
    busy_demo.source.image = images.placeholder()
    assert busy_demo.until(State.HALTED, frames=round(4.0 * FPS)), "挡住摄像头没进已停止"
    assert busy_demo.rig.machine.view.reason == VISION_LOST


def test_空闲时摄像头被挡住只报错不惊动机械臂(demo):
    demo.recognizer.real_flat = True
    demo.source.image = images.placeholder()
    before = demo.rig.arm.stopped
    moves = len(demo.rig.arm.moves)
    lost = False
    for _ in range(round(3.0 * FPS)):
        demo.step()
        if demo.lost:
            lost = True
    assert lost, "挡住摄像头没判成识别中断"
    assert demo.rig.machine.state is State.IDLE  # 机械臂本来就停着,不惊动它
    assert demo.rig.arm.stopped == before
    assert len(demo.rig.arm.moves) == moves


def test_画面回来就不再报识别中断(demo):
    demo.recognizer.real_flat = True
    demo.source.image = images.placeholder()
    for _ in range(round(2.0 * FPS)):
        demo.step()
    assert demo.lost
    demo.source.image = images.normal()
    for _ in range(round(1.0 * FPS)):
        demo.step()
    assert not demo.lost

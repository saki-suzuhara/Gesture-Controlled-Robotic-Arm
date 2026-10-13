"""整条链路在 RoboDK 仿真里走一遍,JAKA 和 Nova 5 各一次(第 04 票验收第 2 条)。

和 `tests/test_vision_demo.py` 是同一条链路、同一批手势画面,只换一样:**机械臂是
RoboDK 仿真里的真模型**,不是记账用的假机械臂。所以连接、按配置写入、可达性检查、
真的求逆解走直线,全都过一遍——除了「人真的站在摄像头前做手势」以外,票里验收第 2 条
的每一段都在。

**要开 RoboDK 才能跑**:没开就整份跳过,不算失败(和 `tests/test_robodk_arm.py`
同一个规矩)。跑之前请自己打开 RoboDK;这里不替你启动它,免得在后台开出一个进程来。

**跑得慢**:仿真按 1 倍速走(ADR-0011),六个预设动作加倒计时、冷却,一台大约两三
分钟。不动用户的东西:机械臂跑完放回原来那组关节角。
"""

from __future__ import annotations

import time

import pytest

from robot_side.arm import joint_drift
from robot_side.config import HOME_ACTION, load_arm, load_demo
from robot_side.robodk_arm import JAKA, NOVA5, RoboDKArm
from robot_side.startup import prepare
from robot_side.state import Event, Machine, State
from robot_side.station import STATION_FILE
from tests.robodk_guard import needs_robodk
from robot_side.stop import StopSignal
from tests.test_vision_demo import (
    CENTER,
    HAND,
    PARAMS,
    fist_move,
    gesture_for,
    palm,
    quiet,
)
from tests.vision_fakes import FPS, ScriptedFrameSource, ScriptedRecognizer, nothing, still
from vision_side.loop import RobotSide, VisionLoop

# 没开 RoboDK、或者有人连着真机(机械臂驱动在跑)就跳过,见 tests/robodk_guard.py
pytestmark = needs_robodk

DEMO = load_demo()
STEP_S = 1.0 / FPS
"""每帧歇多久。**按真时间走**:机械臂是真在 RoboDK 里动的,倒计时和冷却也是真的秒。"""

LIMIT_S = 90.0
"""一段最多等多久。draw_square 六段直线在仿真里走十几秒,余量给足。"""

ARMS = [("jaka", JAKA), ("nova5", NOVA5)]


class Station:
    """接好的一整套:识别线程 + 状态逻辑 + RoboDK 仿真里的机械臂。"""

    def __init__(self, config_name: str, robot_name: str) -> None:
        self.arm_config = load_arm(config_name)
        self.arm = RoboDKArm(STATION_FILE, robot_name)
        self.arm.connect()
        self.before = self.arm.joints()  # 跑完放回去,不动用户摆的姿势

        prepared = prepare(self.arm, self.arm_config, DEMO)
        assert prepared.ok, f"{config_name} 可达性检查不通过:{prepared.report.describe()}"
        self.report = prepared.report

        self.recognizer = ScriptedRecognizer(
            still(PARAMS.hold_s + PARAMS.measure_s + 0.5, CENTER, width=HAND)
        )
        self.source = ScriptedFrameSource()
        self.source.open()
        self.robot = RobotSide(DEMO, log=quiet())
        self.loop = VisionLoop(
            self.source,
            self.recognizer,
            DEMO,
            frame_size=(1280, 720),
            robot=self.robot,
            log=quiet(),
        )
        self.signal = StopSignal(self.arm)
        self.machine = Machine(
            self.arm,
            self.signal,
            plans=prepared.by_action,
            standby_point=prepared.standby_point,
            arm_config=self.arm_config,
            demo=DEMO,
            log=quiet(),
            on_clear_records=self.loop.clear_records,
        )
        self.robot.attach(self.machine, self.signal)

    # ---- 推着它走 ------------------------------------------------------

    def align(self) -> None:
        for _ in range(round((PARAMS.hold_s + PARAMS.measure_s + 1.0) * FPS)):
            self.loop.step()
            if self.loop.alignment is not None:
                return
        raise AssertionError("编好的结果没能让它对上")

    def start(self) -> None:
        self.align()
        self.machine.start()
        assert self.until(State.IDLE), "启动之后没走到空闲"

    def play(self, script) -> None:
        self.recognizer.restart(script)

    def step(self) -> None:
        """一帧。顺序和正式程序一样:识别线程先出这一帧,主线程再 tick、再传状态。"""
        self.loop.step()
        self.machine.tick()
        self.loop.set_state(self.machine.state)
        time.sleep(STEP_S)

    def until(self, want: State, limit_s: float = LIMIT_S) -> bool:
        deadline = time.monotonic() + limit_s
        while time.monotonic() < deadline:
            self.step()
            if self.machine.state is want:
                return True
        return False

    def close(self) -> None:
        self.machine.close()
        try:
            self.arm.set_joints(self.before)  # 放回跑之前那组关节角
        finally:
            self.arm.disconnect()
        self.source.close()


@pytest.fixture(params=ARMS, ids=[name for name, _ in ARMS])
def station(request):
    if not STATION_FILE.is_file():
        pytest.skip(f"没有工作站文件 {STATION_FILE};先跑建站脚本")
    config_name, robot_name = request.param
    made = Station(config_name, robot_name)
    try:
        made.start()
        yield made
    finally:
        made.close()


# ---- 验收第 2 条 -------------------------------------------------------


def test_连上写配置查可达性都过(station):
    # 启动顺序的第 5 步:连机械臂 → 按配置写入 → 可达性检查 → 回待命 → 空闲
    assert station.report.ok
    assert station.machine.state is State.IDLE
    assert station.arm.error() is None


def test_六个触发手势在仿真里各走完一个预设动作(station):
    for trigger in DEMO.triggers:
        if trigger.action == HOME_ACTION:
            continue  # 回待命手势不走预设动作,放到最后单独做
        station.play(gesture_for(trigger) + nothing(2.0))
        assert station.until(State.RUNNING), f"{trigger.kind} {trigger.key} 没触发"
        assert station.machine.view.action == DEMO.action(trigger.action).display
        assert station.until(State.IDLE), f"{trigger.key} 走完之后没回空闲"
    # 回待命手势:先往左挪一步,它才有地方可回
    for trigger in DEMO.triggers:
        if trigger.action != HOME_ACTION:
            continue
        station.play(fist_move("left") + nothing(2.0))
        assert station.until(State.RUNNING)
        assert station.until(State.IDLE)
        station.play(gesture_for(trigger) + nothing(2.0))
        assert station.until(State.RETURNING), f"{trigger.key} 没让它回待命"
        assert station.until(State.IDLE)
    assert station.arm.error() is None


def test_执行中叫停回待命冷却空闲再叫停已停止回车(station):
    station.play(fist_move("left") + nothing(2.0))
    assert station.until(State.RUNNING)
    stopped = station.arm.stopped if hasattr(station.arm, "stopped") else None

    # 执行中张开手掌:识别线程就地叫停,不经过主线程(ADR-0012)
    station.play(palm(0.5) + nothing(3.0))
    assert station.until(State.RETURNING), "停止手势没让它进回待命"
    assert "STOP-GESTURE" in station.machine.view.reason
    assert stopped is None or station.arm.stopped > stopped

    # 回待命途中再叫停 → 已停止 → 回车 → 回待命 → 冷却 → 空闲
    station.play(palm(0.4) + nothing(0.4) + palm(0.4) + nothing(4.0))
    assert station.until(State.HALTED), "回待命途中再叫停没进已停止"
    station.machine.submit(Event.confirm("key:enter"))
    assert station.until(State.IDLE), "按回车之后没回待命再回空闲"


def test_每一次叫停都真的停住了机械臂(station):
    station.play(fist_move("right") + nothing(2.0))
    assert station.until(State.RUNNING)
    station.play(palm(0.5) + nothing(3.0))
    assert station.until(State.RETURNING)
    # 叫停是直达机械臂的:记录上写着立标记到停住用了多久
    assert station.signal.records, "一次叫停都没记下"
    for record in station.signal.records:
        assert record.error is None, record
        assert record.stopped_at - record.flag_at < 0.5, record
    assert station.until(State.IDLE)


def test_做完停在那儿按H回到待命位姿(station):
    # 2026-09-23 起动作做完不回待命:左两步停在 300 mm 外,按 H 才回去
    for _ in range(2):
        station.play(fist_move("left") + nothing(2.0))
        assert station.until(State.RUNNING), "握拳左移没触发"
        assert station.until(State.IDLE), "左移走完之后没回空闲"
    assert station.machine.view.step == "AT X -300 MM"
    assert joint_drift(station.arm.joints(), station.arm_config.standby_joints_deg) > 1.0
    station.machine.submit(Event.home("key:h"))
    assert station.until(State.RETURNING)
    assert station.until(State.IDLE), "按 H 之后没回待命再回空闲"
    assert joint_drift(station.arm.joints(), station.arm_config.standby_joints_deg) <= 1.0
    assert station.machine.view.step == "AT STANDBY"

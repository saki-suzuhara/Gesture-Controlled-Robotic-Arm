"""叫停这一条的规矩:任意线程可调、重复调没有副作用(用户故事 37、ADR-0012)。

叫停不走事件队列——谁先发现要停,谁就直接调机械臂接口的「停止」。所以这里测的是
`StopSignal` 和机械臂接口的「停止」本身,以及叫停正好落在两段之间时下一段发不发。
"""

from __future__ import annotations

import threading

from robodk.robomath import transl

from robot_side.executor import run_plan
from robot_side.motion import JointSegment, LinearSegment
from robot_side.state import State
from robot_side.stop import StopSignal
from tests.conftest import Rig
from tests.fakes import RecordingArm

STANDBY = (0.0,) * 6


def plan(n: int = 4):
    segments = [
        LinearSegment(label=f"路点 {i + 1}", pose=transl(100.0 * i, 0.0, 0.0), speed=50.0)
        for i in range(n - 1)
    ]
    segments.append(JointSegment(label="回待命位姿", joints=STANDBY, speed=10.0))
    return segments


# ---- 重复调 ------------------------------------------------------------


def test_停止重复调没有副作用():
    arm = RecordingArm(STANDBY, busy_s=10.0)
    arm.move_joints([1.0] * 6)
    assert arm.is_moving()
    for _ in range(5):
        arm.stop()
    assert not arm.is_moving()
    assert arm.stopped == 5  # 每次都真的调到了机械臂,只是调几次结果都一样


def test_叫停重复来也只是多记一条():
    arm = RecordingArm(STANDBY)
    signal = StopSignal(arm)
    for _ in range(3):
        signal.raise_stop("space")
    assert signal.raised
    assert len(signal.records) == 3  # 连按三下,三条都在
    assert arm.stopped == 3


def test_放下标记不会把记下的叫停抹掉():
    """状态逻辑按「记了几条」认叫停,不按标记——标记会被我们自己放下。"""
    arm = RecordingArm(STANDBY)
    signal = StopSignal(arm)
    signal.raise_stop("space")
    signal.clear()
    assert not signal.raised
    assert len(signal.records) == 1


# ---- 从别的线程调 ------------------------------------------------------


def test_停止从别的线程调也停得住():
    arm = RecordingArm(STANDBY, busy_s=10.0)
    arm.move_joints([1.0] * 6)
    done = threading.Event()

    def stopper() -> None:
        arm.stop()
        done.set()

    thread = threading.Thread(target=stopper, name="stopper")
    thread.start()
    thread.join(2.0)
    assert done.is_set()
    assert not arm.is_moving()


def test_几个线程同时叫停也不会丢也不会炸():
    arm = RecordingArm(STANDBY, busy_s=10.0)
    signal = StopSignal(arm)
    arm.move_joints([1.0] * 6)
    ready = threading.Barrier(4)
    errors: list[str] = []

    def press(name: str) -> None:
        try:
            ready.wait(2.0)
            signal.raise_stop(name)
        except Exception as exc:  # 记下来,别让线程默默吞掉
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [
        threading.Thread(target=press, args=(f"source-{i}",), name=f"press-{i}")
        for i in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2.0)

    assert errors == []
    assert not arm.is_moving()
    assert len(signal.records) == 4  # 四个线程各记一条,一条都没丢
    assert {r.source for r in signal.records} == {f"source-{i}" for i in range(4)}


# ---- 正好落在两段之间 --------------------------------------------------


def test_叫停落在两段之间就不再发下一段():
    """机械臂本来就不动,「停止」没什么可停的,全靠「已叫停」标记把下一段拦住。"""
    class StopBetweenSegments(RecordingArm):
        def is_moving(self) -> bool:
            signal.raise_stop("space")  # 这一段刚走完、下一段还没发
            return False

    arm = StopBetweenSegments(STANDBY)
    signal = StopSignal(arm)
    sent = run_plan(arm, signal, plan(4))
    assert sent == 1
    assert len(arm.moves) == 1


def test_叫停落在两段之间时状态逻辑当作被叫停而不是走完():
    r = Rig(busy_s=0.05)
    try:
        r.to_idle()
        r.trigger("draw_square")  # 5 个路点 + 1 段回待命位姿
        r.machine.tick()
        assert r.machine.state is State.RUNNING
        r.signal.raise_stop("space")
        r.pump(lambda: r.machine.state is State.RETURNING, what="RETURNING")
        sent = len(r.arm.moves)
        assert sent < 6  # 剩下的段没发出去
        # 停在半路:摆成离待命位姿还有一段,这样回待命真的要走那两段
        r.arm.joints_now = [v + 30.0 for v in r.arm_config.standby_joints_deg]
        r.until(State.IDLE)
        # 之后只多了回待命那两段(抬起、回待命位姿,都是关节运动),动作剩下的段没被补上
        assert [kind for kind, _ in r.arm.moves[sent:]] == ["joints", "joints"]
    finally:
        r.close()

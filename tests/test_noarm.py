"""空实现(NO ARM):每段固定等一会儿,途中能被叫停打断,可达性检查一律通过。

不开 RoboDK,也不连任何机械臂——这个模块存在的意义就是这个。
"""

from __future__ import annotations

import threading
import time

from robodk.robomath import transl

from robot_side.config import load_arm
from robot_side.executor import run_plan
from robot_side.motion import JointSegment, LinearSegment
from robot_side.noarm import NoArm
from robot_side.stop import StopSignal

FAST = 0.05
"""测试里把每段等的时间改小,免得一个用例就跑掉几秒。"""


def linear(x: float) -> LinearSegment:
    return LinearSegment(label=f"x={x}", pose=transl(x, 0.0, 0.0), speed=50.0)


def test_每段等到时间才算走完():
    arm = NoArm(segment_s=FAST)
    arm.move_linear(transl(10.0, 0.0, 0.0))
    assert arm.is_moving()
    time.sleep(FAST * 3)
    assert not arm.is_moving()


def test_叫停立刻打断这一段():
    arm = NoArm(segment_s=10.0)
    arm.move_joints([1.0] * 6)
    assert arm.is_moving()
    arm.stop()
    assert not arm.is_moving()


def test_叫停可以从别的线程调也可以重复调():
    arm = NoArm(segment_s=10.0)
    arm.move_joints([1.0] * 6)
    threading.Thread(target=arm.stop).start()
    threading.Thread(target=arm.stop).start()
    deadline = time.monotonic() + 1.0
    while arm.is_moving() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not arm.is_moving()
    arm.stop()  # 再调一次,没有副作用
    assert not arm.is_moving()


def test_一整段计划能走完():
    arm = NoArm(segment_s=FAST)
    plan = [linear(10.0), linear(20.0), JointSegment(label="回", joints=(0.0,) * 6, speed=10.0)]
    t0 = time.monotonic()
    sent = run_plan(arm, StopSignal(arm), plan, timeout_s=5.0)
    assert sent == 3
    # 每段都真的等了一会儿,不是一发就返回
    assert time.monotonic() - t0 >= FAST * 3
    assert [kind for kind, _ in arm.sent] == ["linear", "linear", "joints"]


def test_走到一半叫停就不再发下一段():
    arm = NoArm(segment_s=0.5)
    signal = StopSignal(arm)
    plan = [linear(10.0), linear(20.0), linear(30.0)]
    threading.Timer(0.1, lambda: signal.raise_stop("test")).start()
    t0 = time.monotonic()
    sent = run_plan(arm, signal, plan, timeout_s=5.0)
    assert sent == 1
    # 第一段被打断,没等满 0.5 秒
    assert time.monotonic() - t0 < 0.4


def test_可达性检查一律通过():
    arm = NoArm()
    report = arm.reach_check({"a": [linear(10.0), linear(20.0)]}, [0.0] * 6)
    assert report.ok
    assert report.checked_waypoints == 2


def test_按自带的配置定每段等多久():
    arm = NoArm()
    config = load_arm("noarm")
    assert config.driver == "none"
    arm.apply_config(config)
    assert arm.segment_s == config.noarm_segment_s == 1.0


def test_真机那两份配置不写这一节也读得出来():
    assert load_arm("jaka").noarm_segment_s == 1.0
    assert load_arm("nova5").noarm_segment_s == 1.0

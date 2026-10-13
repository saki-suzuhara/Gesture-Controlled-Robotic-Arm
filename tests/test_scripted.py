"""编好的事件序列能驱动同一套状态逻辑;提示音、日志和叫停用时也在这里测。

不开 RoboDK、不开摄像头、不弹窗口。跑整条链路的两个用例用真时钟:事件要落在某一段
运动中间,靠假时钟摆不出来。所以把倒计时、冷却和每段运动的时间都按同一个比例调小。
"""

from __future__ import annotations

import dataclasses
import time

from robot_side import runlog
from robot_side.arm import CollisionError
from robot_side.beeps import Beeper
from robot_side.scripted import FULL, SCRIPTS, ScriptPlayer, Step
from robot_side.state import State
from robot_side.stop import SpaceKeyWatcher, StopSignal
from tests.conftest import Rig
from tests.fakes import Fault, RecordingArm

SCALE = 0.1
"""整条链路按这个比例缩时间跑。

倒计时、冷却、每段运动和脚本里的秒数**按同一个比例**缩,先后关系才和真跑一遍一样:
叫停仍然落在某一段运动中间、第二次叫停仍然落在倒计时里。
"""


def fast_rig() -> Rig:
    """跑整条链路用的测试台:真时钟,各项时间都按 `SCALE` 缩小。"""
    return Rig(
        countdown_s=3.0 * SCALE,  # 演示配置里是 3 秒
        cooldown_s=2.0 * SCALE,  # 演示配置里是 2 秒
        busy_s=1.0 * SCALE,  # 空实现每段 1 秒
    )


def scaled(steps) -> list[Step]:
    return [dataclasses.replace(s, delay_s=s.delay_s * SCALE) for s in steps]


def play(rig: Rig, steps=FULL, *, limit_s: float = 20.0) -> list[State]:
    """把脚本跑完,返回一路经过的状态。"""
    player = ScriptPlayer(scaled(steps), rig.machine, rig.signal, clock=rig.clock)
    seen: list[State] = [rig.machine.state]
    rig.machine.start()
    deadline = time.monotonic() + limit_s
    while not rig.machine.finished:
        rig.machine.tick()
        player.pump()
        if rig.machine.state is not seen[-1]:
            seen.append(rig.machine.state)
        if time.monotonic() > deadline:
            raise AssertionError(
                f"脚本跑不完,停在 {rig.machine.state.name};经过 {[s.name for s in seen]}"
            )
        time.sleep(0.002)
    return seen


def test_编好的事件序列能走完两条链路():
    rig = fast_rig()
    try:
        seen = play(rig)
    finally:
        rig.close()
    # 启动 → 回待命 → 冷却 → 空闲,再:触发 → 执行中 → 叫停 → 回待命
    assert seen[:6] == [
        State.IDLE,
        State.RETURNING,
        State.COOLDOWN,
        State.IDLE,
        State.RUNNING,
        State.RETURNING,
    ]
    # 回待命途中再叫停 → 已停止 → 回车 → 回待命 → 冷却 → 空闲
    assert State.HALTED in seen
    halted = seen.index(State.HALTED)
    assert seen[halted + 1 :] == [State.RETURNING, State.COOLDOWN, State.IDLE]
    assert rig.machine.finished


def test_编好的事件序列能把每个预设动作跑一遍():
    rig = fast_rig()
    try:
        seen = play(rig, SCRIPTS["actions"])
        assert seen.count(State.RUNNING) == len(rig.demo.actions)
        assert State.HALTED not in seen
        # 每个动作各自的路点;画方块、点头、鞠躬再多一段回出发点(做完不回待命位姿)
        want = sum(len(a.waypoints) + (a.end == "back") for a in rig.demo.actions)
        assert len(rig.arm.moves) == want
    finally:
        rig.close()


def test_脚本里的叫停走的是直达机械臂那条路():
    rig = fast_rig()
    try:
        play(rig)
        assert [r for r in rig.signal.records if r.source == "script"]
        assert rig.arm.stopped >= 3  # 脚本叫停两次,加上进已停止时那一次
    finally:
        rig.close()


def test_没等到那个状态就不往下走(rig):
    player = ScriptPlayer(
        [Step("trigger", action="nod", wait_for=State.HALTED)],
        rig.machine,
        rig.signal,
        clock=rig.clock,
    )
    rig.to_idle()
    for _ in range(20):
        player.pump()
        rig.machine.tick()
        rig.clock.advance(0.5)
    assert not player.done
    assert rig.machine.state is State.IDLE
    assert rig.arm.moves == []


# ---- 提示音 ------------------------------------------------------------


def test_提示音不阻塞主线程():
    played: list[tuple[int, int]] = []

    def slow(freq: int, ms: int) -> None:
        time.sleep(0.2)  # winsound.Beep 要等声音放完才返回
        played.append((freq, ms))

    beeps = Beeper(play=slow)
    t0 = time.monotonic()
    beeps.halted()  # 三声,真放要 0.6 秒
    beeps.countdown()
    spent = time.monotonic() - t0
    assert spent < 0.05, f"把主线程卡了 {spent:.3f} 秒"
    beeps.close(timeout_s=3.0)
    assert len(played) == 4  # 三声已停止 + 一声倒计时


def test_放不出声也不把程序带倒():
    def broken(freq: int, ms: int) -> None:
        raise RuntimeError("没有声卡")

    beeps = Beeper(play=broken, block=True)
    beeps.halted()
    beeps.countdown()
    beeps.close()


# ---- 叫停用时 ----------------------------------------------------------


def test_主线程卡住时按空格机械臂照样在半秒内停住(monkeypatch):
    """拖住预览窗口不放的那种情况:主线程卡死,叫停照样走得通。

    ADR-0012 那条路:按键线程自己发现空格按下、自己立标记、自己调机械臂接口的
    「停止」,一步都不经过主线程。这里把「空格按下」做成假的,主线程真的睡死过去。
    """
    import robot_side.stop as stop_module

    arm = RecordingArm(busy_s=10.0)
    signal = StopSignal(arm)
    down = False
    monkeypatch.setattr(stop_module, "key_is_down", lambda vk: down)

    with SpaceKeyWatcher(lambda: signal.raise_stop("space"), poll_s=0.01):
        time.sleep(0.05)  # 让按键线程先跑起来
        arm.move_joints([0.0] * 6)  # 机械臂正在走一段长的
        assert arm.is_moving()
        pressed_at = time.monotonic()
        down = True
        time.sleep(0.5)  # 主线程这 0.5 秒什么都做不了,和被拖住的窗口一样

    assert arm.stopped >= 1
    assert arm.stopped_at is not None
    spent = arm.stopped_at - pressed_at
    assert spent < 0.5, f"按下空格到机械臂停住用了 {spent:.3f} 秒"
    assert not arm.is_moving()


# ---- 日志文件 ----------------------------------------------------------


def test_日志同时进终端和本地文件(capsys):
    log, path = runlog.start("pytest")
    try:
        log.info("状态 IDLE → RUNNING:测试")
    finally:
        runlog.close(log)
    assert path.parent.name == "logs"
    assert path.parent.parent.name == "data"  # 整个 data/ 被 git 忽略
    written = path.read_text(encoding="utf-8")
    assert "状态 IDLE → RUNNING:测试" in written
    assert "状态 IDLE → RUNNING:测试" in capsys.readouterr().out
    path.unlink()


# ---- 停在已停止时爬出来 ------------------------------------------------


def drive(rig: Rig, player: ScriptPlayer, *, on_tick=None, limit_s: float = 20.0) -> None:
    """推着脚本跑到它自己收摊为止。"""
    rig.machine.start()
    deadline = time.monotonic() + limit_s
    while not rig.machine.finished:
        rig.machine.tick()
        player.pump()
        if on_tick is not None:
            on_tick()
        if time.monotonic() > deadline:
            raise AssertionError(
                f"脚本没跑完,停在 {rig.machine.state.name};日志:{rig.records[-4:]}"
            )
        time.sleep(0.002)


def test_半路被叫停进已停止也能自己按回车接着跑():
    # 照 2026-09-20 那次:外面按到的空格连叫两次停,脚本就在已停止里干等
    rig = fast_rig()
    player = ScriptPlayer(
        scaled(SCRIPTS["actions"]), rig.machine, rig.signal, clock=rig.clock, log=rig.log
    )
    done = {"halted": False}

    def meddle() -> None:
        if done["halted"] or rig.machine.state is not State.RUNNING:
            return
        rig.signal.raise_stop("space")
        rig.machine.tick()  # → 回待命
        rig.signal.raise_stop("space")  # 回待命途中再叫停 → 已停止
        done["halted"] = True

    try:
        drive(rig, player, on_tick=meddle)
        assert done["halted"]
        assert player.done
        assert any("自动按一次回车" in line for line in rig.records)
        # 爬出来之后把剩下的动作跑完了,不是就此收摊
        assert any("触发 nod" in line for line in rig.records)
    finally:
        rig.close()


def test_一直进已停止就收摊退出不在那儿空转():
    # 每一段运动都报错:按几次回车都没用,现场真有问题
    rig = Rig(
        countdown_s=0.3,
        cooldown_s=0.2,
        busy_s=0.1,
        faults=[Fault(CollisionError("碰撞保护"))],
    )
    player = ScriptPlayer(
        scaled(SCRIPTS["actions"]),
        rig.machine,
        rig.signal,
        clock=rig.clock,
        log=rig.log,
        max_recoveries=2,
    )
    try:
        drive(rig, player)
        assert player.done
        assert rig.machine.finished
        nudges = [line for line in rig.records if "自动按一次回车" in line]
        assert len(nudges) == 2  # 按满就不再按了
        assert any("收摊退出" in line for line in rig.records)
    finally:
        rig.close()


def test_不想让它替你按回车就把次数设成零():
    rig = Rig(
        countdown_s=0.3,
        cooldown_s=0.2,
        busy_s=0.1,
        faults=[Fault(CollisionError("碰撞保护"))],
    )
    player = ScriptPlayer(
        scaled(SCRIPTS["actions"]),
        rig.machine,
        rig.signal,
        clock=rig.clock,
        log=rig.log,
        max_recoveries=0,
    )
    try:
        drive(rig, player)
        assert not any("自动按一次回车" in line for line in rig.records)
        assert any("收摊退出" in line for line in rig.records)
    finally:
        rig.close()


def test_本来就在等已停止的那一步不受影响():
    # `full` 里有一步就是「等到已停止,再按回车」——别让自动按回车把它抢了
    rig = fast_rig()
    player = ScriptPlayer(
        scaled(FULL), rig.machine, rig.signal, clock=rig.clock, log=rig.log
    )
    try:
        drive(rig, player)
        assert not any("自动按一次回车" in line for line in rig.records)
        assert any("脚本 第 6 步:confirm" in line for line in rig.records)
    finally:
        rig.close()



# ---- 连着真机时不替人按回车(2026-09-23 真机上脚本跟人对着干) --------------


def test_连着真机时人叫停以后脚本不替人按回车():
    rig = fast_rig()
    try:
        player = ScriptPlayer(
            [Step("quit", wait_for=State.IDLE)],
            rig.machine,
            rig.signal,
            log=rig.log,
            confirm_halts=False,
        )
        rig.machine.start()
        rig.pump(lambda: rig.machine.state is State.RETURNING, what="RETURNING")
        rig.signal.raise_stop("space")  # 回待命途中叫停 → 已停止
        rig.pump(lambda: rig.machine.state is State.HALTED, what="HALTED")
        for _ in range(50):
            player.pump()
            rig.machine.tick()
        assert rig.machine.state is State.HALTED  # 还停着,等人
        assert rig.arm.cleared == 0  # 没替人按回车
        assert sum("不替你按回车" in line for line in rig.records) == 1  # 只说一次
    finally:
        rig.close()

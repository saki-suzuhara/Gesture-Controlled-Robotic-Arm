"""连真机这一侧:启动开关、驱动连接、上电使能检查、掉线判断。

全程不连真机、不开 RoboDK:驱动那一侧由一个假的替上(`FakePort`),端口探测由测试
自己给。真机上要实测的四项(断连保险、叫停用时和滑行、急停报错、拖窗口按空格)
不在这里,要另开 session 和用户一起做(票 05)。
"""

from __future__ import annotations

import dataclasses

import pytest
from robodk import robolink

from robot_side.arm import ArmError
from robot_side.config import Worktable, load_arm
from robot_side.guard import GuardRefused, PowerState, check_powered
from robot_side.robodk_arm import JAKA, RoboDKArm
from robot_side.real import (
    CONNECTED,
    LOST,
    STUBBORN_READS,
    UNSURE,
    LinkMonitor,
    LinkRead,
    RealArmRefused,
    bring_up_real,
    check_real_allowed,
    choose_arm,
    connect_real,
    executed_rpy,
    link_verdict,
    network_probe_is_honest,
    no_probe,
    read_power_state,
)

READY = robolink.ROBOTCOM_READY
WORKING = robolink.ROBOTCOM_WORKING
DISCONNECTED = robolink.ROBOTCOM_DISCONNECTED
UNKNOWN = robolink.ROBOTCOM_UNKNOWN


def read(code: int, reachable: bool | None = True, message: str = "") -> LinkRead:
    return LinkRead(code=code, message=message, controller_reachable=reachable)


# ---- 默认不连真机:要连必须加开关 + 明确指定机械臂配置 --------------------


def test_不加开关时用默认的机械臂配置():
    assert choose_arm("", real=False, default="noarm") == "noarm"


def test_加了开关却没明确指定机械臂配置就拒绝启动():
    with pytest.raises(RealArmRefused) as caught:
        choose_arm("", real=True, default="jaka")
    assert "--arm" in str(caught.value)
    assert "jaka" not in str(caught.value).split("--arm")[0]  # 不偷偷用默认值


def test_加了开关并明确指定了就用指定的那一份():
    assert choose_arm("jaka", real=True, default="noarm") == "jaka"


def test_空实现那份配置不许连真机():
    with pytest.raises(RealArmRefused) as caught:
        check_real_allowed(load_arm("noarm"))
    assert "none" in str(caught.value)


def test_没填控制柜_ip_就不许连真机():
    config = load_arm("jaka")
    assert config.ip is None, "jaka.toml 出厂时 ip 应当是空的"
    with pytest.raises(RealArmRefused) as caught:
        check_real_allowed(config)
    assert "ip" in str(caught.value)


def test_填了_ip_的_jaka_配置可以连真机():
    check_real_allowed(dataclasses.replace(load_arm("jaka"), ip="10.0.0.2"))


def test_查不了上电使能的机械臂不许连真机():
    """Nova 5 怎么接还没定,也就没有查上电使能的路;查不了就不连。"""
    config = dataclasses.replace(load_arm("nova5"), ip="10.0.0.3")
    with pytest.raises(RealArmRefused) as caught:
        check_real_allowed(config)
    assert "上电使能" in str(caught.value)


# ---- 掉线判断:不只看一个返回值 ------------------------------------------


def test_驱动说_ready_就是连着():
    assert link_verdict([read(READY)]) == CONNECTED


def test_两路都说坏就判掉线():
    assert link_verdict([read(DISCONNECTED, reachable=False)]) == LOST


def test_只有驱动说坏时先不下结论():
    """驱动报 DISCONNECTED,但控制柜的命令端口还通着——只有一个返回值,不算数。"""
    assert link_verdict([read(DISCONNECTED, reachable=True)]) == UNSURE


def test_只有一路说坏但连着几次都不好也判掉线():
    reads = [read(DISCONNECTED, reachable=True)] * STUBBORN_READS
    assert link_verdict(reads) == LOST
    assert link_verdict(reads[:-1]) == UNSURE


def test_前面说不准最后一次说坏不算掉线():
    """`[说不准, 说不准, 说坏]` 判的仍然是最后那一个返回值,不认(第 05 票第 5 条)。"""
    reads = [
        read(UNKNOWN, reachable=True, message="Sending command..."),
        read(UNKNOWN, reachable=True, message="Sending command..."),
        read(DISCONNECTED, reachable=True),
    ]
    assert link_verdict(reads) == UNSURE


def test_中间只要有一次_ready_就重新算起():
    reads = [
        read(DISCONNECTED, reachable=True),
        read(READY),
        read(DISCONNECTED, reachable=True),
    ]
    assert link_verdict(reads) == UNSURE


def test_断开之后那句_sending_command_不算掉线的证据():
    """`notes/jaka-first-connection-results.md`:`Disconnect()` 之后报的是
    `ROBOTCOM_UNKNOWN 'Sending command...'`,不是 DISCONNECTED。只凭它不能判掉线。"""
    reads = [read(UNKNOWN, reachable=True, message="Sending command...")] * 5
    assert link_verdict(reads) == UNSURE


def test_连接途中的_working_不算掉线():
    assert link_verdict([read(WORKING)] * 5) == UNSURE


def test_没读到就不下结论():
    assert link_verdict([]) == UNSURE


def test_没探第二路时照样判得出掉线():
    """这台笔记本上网络探针说假话,所以默认不探;只剩驱动一路时靠「连着几次都不好」。"""
    reads = [LinkRead(code=DISCONNECTED, message="Disconnected")] * STUBBORN_READS
    assert all(r.controller_reachable is None for r in reads)
    assert link_verdict(reads) == LOST


def test_不探的探针返回没探():
    assert no_probe("10.0.0.2") is None


def test_探针说假话时认得出来():
    """对着一个不该有人应答的地址还说「通」的探针,不算数。"""
    assert not network_probe_is_honest(lambda ip: True)
    assert network_probe_is_honest(lambda ip: False)
    assert not network_probe_is_honest(lambda ip: None)


# ---- 驱动连接:连上才算,连不上不往下走 ----------------------------------


class FakePort:
    """替代 RoboDK 那一侧:记下被调了什么,按安排报状态。"""

    def __init__(self, states, *, reachable=True, joints=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
        self.states = list(states)
        self.reachable = reachable
        self.connects: list[tuple[str, int]] = []
        self.disconnects = 0
        self.stops = 0
        self.joints_now = list(joints)
        self.staged: list[float] | None = None
        self.staged_moves = 0
        """热身用到的那几件事:放目标、读回来、发出去。"""

    def connect_driver(self, ip: str, port: int) -> None:
        self.connects.append((ip, port))

    def driver_state(self) -> tuple[int, str]:
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def disconnect_driver(self) -> None:
        self.disconnects += 1

    def stop(self) -> None:
        self.stops += 1

    # ---- 热身要用的 ----
    def joints(self) -> list[float]:
        return list(self.joints_now)

    def stage_joint_target(self, joints):
        self.staged = [float(v) for v in joints]
        return list(self.staged)

    def move_joints_staged(self) -> None:
        self.staged_moves += 1

    def is_moving(self) -> bool:
        return False


def reachable_always(ip: str) -> bool:
    return True


def unreachable_always(ip: str) -> bool:
    return False


def test_连上驱动之后才返回():
    port = FakePort([(WORKING, "Working..."), (READY, "Ready")])
    connect_real(port, "10.0.0.2", 10000, reachable=reachable_always, sleep=lambda _: None)
    assert port.connects == [("10.0.0.2", 10000)]


def test_一直连不上就拒绝并且把驱动断开():
    port = FakePort([(WORKING, "Working...")])
    with pytest.raises(RealArmRefused) as caught:
        connect_real(
            port,
            "10.0.0.2",
            10000,
            timeout_s=0.05,
            reachable=reachable_always,
            sleep=lambda _: None,
        )
    assert "连不上" in str(caught.value)
    assert port.disconnects == 1, "连不上也要把驱动这一侧收干净"


def test_判定掉线时立刻停下不等超时():
    port = FakePort([(DISCONNECTED, "Disconnected")])
    with pytest.raises(RealArmRefused):
        connect_real(
            port,
            "10.0.0.2",
            10000,
            timeout_s=60.0,  # 等满就说明没走掉线那条路
            reachable=unreachable_always,
            sleep=lambda _: None,
        )
    assert port.disconnects == 1


# ---- 上电使能:只检查、不执行 --------------------------------------------


def ok_runner(payload: str):
    return lambda cmd: (0, payload, "")


def test_读到上电使能都有():
    state = read_power_state(
        "10.0.0.2", runner=ok_runner('{"ok": true, "powered": 1, "enabled": 1}')
    )
    assert state.ready


def test_读到没使能():
    state = read_power_state(
        "10.0.0.2", runner=ok_runner('{"ok": true, "powered": 1, "enabled": 0}')
    )
    assert not state.ready
    assert state.powered is True and state.enabled is False


def test_小脚本报错时按读不出来处理():
    state = read_power_state("10.0.0.2", runner=lambda cmd: (1, "", "jkrc login 失败"))
    assert state.powered is None and state.enabled is None
    assert "jkrc login 失败" in state.detail
    with pytest.raises(GuardRefused):
        check_powered(state)


def test_小脚本吐的不是_json_也按读不出来处理():
    state = read_power_state("10.0.0.2", runner=ok_runner("Traceback (most recent"))
    assert state.powered is None and state.enabled is None


def test_小脚本说自己没读成时按读不出来处理():
    state = read_power_state(
        "10.0.0.2", runner=ok_runner('{"ok": false, "error": "状态位读不出来"}')
    )
    assert state.powered is None
    assert "状态位读不出来" in state.detail


def test_读小脚本时自己抛了也按读不出来处理():
    def blows_up(cmd):
        raise OSError("找不到 RoboDK 自带的 Python")

    state = read_power_state("10.0.0.2", runner=blows_up)
    assert state.powered is None
    assert "找不到 RoboDK 自带的 Python" in state.detail


# ---- 掉线的盯梢:判定掉线就挂报错 + 叫停 ---------------------------------


class Watched:
    def __init__(self):
        self.lost: list[str] = []

    def __call__(self, why: str) -> None:
        self.lost.append(why)


def test_盯梢连着时什么也不做():
    port = FakePort([(READY, "Ready")])
    seen = Watched()
    monitor = LinkMonitor(port, "10.0.0.2", on_lost=seen, reachable=reachable_always)
    for _ in range(5):
        assert monitor.poll() == CONNECTED
    assert seen.lost == []


def test_盯梢判定掉线就报一次():
    port = FakePort([(DISCONNECTED, "Disconnected")])
    seen = Watched()
    monitor = LinkMonitor(port, "10.0.0.2", on_lost=seen, reachable=unreachable_always)
    assert monitor.poll() == LOST
    assert len(seen.lost) == 1
    monitor.poll()  # 再看一眼也不重复报:已经报过的事没必要再响一次
    assert len(seen.lost) == 1


def test_盯梢说不准时不报():
    port = FakePort([(UNKNOWN, "Sending command...")])
    seen = Watched()
    monitor = LinkMonitor(port, "10.0.0.2", on_lost=seen, reachable=reachable_always)
    for _ in range(STUBBORN_READS + 2):
        assert monitor.poll() == UNSURE
    assert seen.lost == []


def test_盯梢读状态时抛错也当作一路说坏():
    class Angry(FakePort):
        def driver_state(self):
            raise RuntimeError("Communication problems")

    port = Angry([(READY, "Ready")])
    seen = Watched()
    monitor = LinkMonitor(port, "10.0.0.2", on_lost=seen, reachable=unreachable_always)
    assert monitor.poll() == LOST
    assert "Communication problems" in seen.lost[0]


# ---- 按顺序接上真机:连驱动 → 查上电使能 --------------------------------


def jaka_with_ip():
    return dataclasses.replace(load_arm("jaka"), ip="10.0.0.2")


def test_连上并且上电使能都有才算接上():
    port = FakePort([(READY, "Ready")])
    bring_up_real(
        port, jaka_with_ip(), read_power=lambda: PowerState(powered=True, enabled=True)
    )
    assert port.connects == [("10.0.0.2", 10000)]


def test_接上之后会热身一次而且那一条指令零行程():
    """连上之后第一条指令要等约 21 秒才执行(2026-09-22 真机实测),先付掉。"""
    port = FakePort([(READY, "Ready")])
    bring_up_real(
        port, jaka_with_ip(), read_power=lambda: PowerState(powered=True, enabled=True)
    )
    assert port.staged_moves == 1, "该发一条热身指令"
    assert port.staged == port.joints_now, "热身发的必须是「回到你现在所在的地方」"


def test_热身那一条一轴都不准动():
    """读回来的目标要是和当前位置不一样,当场拒发——不能让机械臂真动起来。"""

    class Sneaky(FakePort):
        def stage_joint_target(self, joints):
            out = list(joints)
            out[3] += 5.0  # 指令里的目标被改掉了
            self.staged = out
            return out

    port = Sneaky([(READY, "Ready")])
    with pytest.raises(GuardRefused):
        bring_up_real(
            port, jaka_with_ip(), read_power=lambda: PowerState(True, True)
        )
    assert port.staged_moves == 0, "护栏没过就不该发出去"


def test_没使能时连热身都不发():
    port = FakePort([(READY, "Ready")])
    with pytest.raises(GuardRefused):
        bring_up_real(
            port, jaka_with_ip(), read_power=lambda: PowerState(True, False)
        )
    assert port.staged_moves == 0


def test_没使能就停在这一步且一步不动():
    port = FakePort([(READY, "Ready")])
    with pytest.raises(GuardRefused) as caught:
        bring_up_real(
            port,
            jaka_with_ip(),
            read_power=lambda: PowerState(powered=True, enabled=False),
        )
    assert "示教器" in str(caught.value)
    assert port.stops == 0, "没使能时连停都不用停——本来就一条指令都没发"


def test_读不出上电使能也停在这一步():
    port = FakePort([(READY, "Ready")])
    with pytest.raises(GuardRefused):
        bring_up_real(
            port,
            jaka_with_ip(),
            read_power=lambda: PowerState(None, None, detail="读不到"),
        )


def test_连不上驱动就根本不去查上电使能():
    port = FakePort([(DISCONNECTED, "Disconnected")])
    asked = []

    def read_power():
        asked.append(1)
        return PowerState(powered=True, enabled=True)

    with pytest.raises(RealArmRefused):
        bring_up_real(port, jaka_with_ip(), read_power=read_power)
    assert asked == [], "连都没连上,不该去问上电使能"


def test_不许连真机的配置连驱动都不碰():
    port = FakePort([(READY, "Ready")])
    with pytest.raises(RealArmRefused):
        bring_up_real(
            port, load_arm("noarm"), read_power=lambda: PowerState(True, True)
        )
    assert port.connects == []


# ---- 连着真机时不许做的两件事(第 05 票第 4 条) ------------------------
#
# 这两条卡在 RoboDK 实现自己身上,不用开 RoboDK 就测得了:判断在碰 RoboDK 之前。


def half_connected_to_real():
    """一个「以为自己连着真机」的 RoboDK 实现。没有 RoboDK,也没有机械臂。

    **这里伸手改了私有字段。** 走正门(`connect_driver`)要开 RoboDK、还要一台真控制柜,
    而这两条要测的恰恰是「连着真机时什么都别做」——一开真机就没法测了。
    `tests/conftest.py` 的 `wait_for_job` 也是同样的理由伸过一次手。
    """
    from robot_side.station import STATION_FILE

    arm = RoboDKArm(STATION_FILE, JAKA)
    arm._real_ip = "10.0.0.2"
    return arm


def test_连着真机时不写配置():
    """写配置要调 setPoseFrame / setPoseTool,而 JAKA 驱动里有 set_tool_id /
    set_user_frame_id——连着写可能改掉控制柜的工具 0 和 0 号坐标系(用户故事 29)。"""
    with pytest.raises(ArmError) as caught:
        half_connected_to_real().apply_config(load_arm("jaka"))
    assert "工具 0" in str(caught.value)


def test_连着真机时一次都不设速度():
    """真机的速度和加速度由人在示教器上调,程序不设(ADR-0011)。

    没连真机的话,这一下会因为「还没连接」而抛错;连着真机时它直接返回,
    所以「不抛」就说明它压根没往下走。
    """
    half_connected_to_real().set_speed(linear_mms=50.0, joints_degs=10.0)


# ---- 工作台坐标系写进去之前不许连真机(启动顺序) ------------------------


def test_填了_ip_也要工作台坐标系是标过的():
    """角度还是出厂的 0 也照连:仿真里两台都用 0,标定是另一支脚本的事。"""
    config = dataclasses.replace(
        load_arm("jaka"), ip="10.0.0.2", worktable=Worktable(angle_deg=0.0)
    )
    check_real_allowed(config)  # 不抛


# ---- 驱动那条已知的姿态镜像 ----------------------------------------------


def test_驱动会把_ry_和_rz_取反():
    """2026-09-22 真机实测:`(RX, RY, RZ)` 执行成 `(RX, −RY, −RZ)`。"""
    assert executed_rpy((-179.48, 0.42, -30.23)) == (-179.48, -0.42, 30.23)


def test_ry_和_rz_都是零时镜像等于没翻():
    """Home 就定在这个不动点上,六个预设动作全程停在这儿。"""
    assert executed_rpy((-179.34, 0.0, 0.0)) == (-179.34, 0.0, 0.0)


# ---- 真机上每段等多久 ------------------------------------------------------


def test_没开程序设速度时每段固定等那一个数():
    from robot_side.real import REAL_SEGMENT_TIMEOUT_S, real_segment_timeout

    config = dataclasses.replace(load_arm("jaka"), program_sets_speed=False)
    assert real_segment_timeout(config) == REAL_SEGMENT_TIMEOUT_S


def test_开了程序设速度就按倍率下限放慢估算():
    from robot_side.executor import SegmentTimeout
    from robot_side.real import MIN_OVERRIDE, REAL_SEGMENT_TIMEOUT_S, real_segment_timeout

    config = dataclasses.replace(load_arm("jaka"), program_sets_speed=True)
    assert real_segment_timeout(config) == SegmentTimeout(
        slowdown=1.0 / MIN_OVERRIDE, floor_s=REAL_SEGMENT_TIMEOUT_S
    )

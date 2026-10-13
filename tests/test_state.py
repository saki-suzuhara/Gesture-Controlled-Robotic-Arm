"""状态逻辑:喂进事件,看状态怎么变、对机械臂接口调了什么(第 03 票)。

测试台在 `tests/conftest.py`:假机械臂 + 真配置 + 假时钟,不开 RoboDK、不开摄像头、
不弹窗口。
"""

from __future__ import annotations

from robot_side.arm import ArmError, MotionError, joint_drift
from robot_side.state import Event, State
from tests.conftest import Rig
from tests.fakes import Fault, RecordingArm

# ---- 状态齐不齐 --------------------------------------------------------


def test_五个状态都在():
    assert {s.name for s in State} == {
        "IDLE",
        "COOLDOWN",
        "RUNNING",
        "RETURNING",
        "HALTED",
    }


# ---- 启动 --------------------------------------------------------------


def test_启动先走回待命再冷却再空闲(rig):
    rig.machine.start()
    assert rig.machine.state is State.RETURNING
    rig.until(State.COOLDOWN)
    rig.until(State.IDLE)


def test_启动时已经在待命位姿附近就只倒计时不移动(rig):
    rig.machine.start()
    rig.until(State.IDLE)
    assert rig.arm.moves == []


def test_启动时不在待命位姿就抬起再慢速回去():
    r = Rig()
    r.arm.joints_now = [v + 30.0 for v in r.arm_config.standby_joints_deg]
    r.machine.start()
    r.until(State.IDLE)
    kinds = [kind for kind, _ in r.arm.moves]
    assert kinds == ["joints", "joints"]  # 先抬起,再回待命位姿;两段都是关节运动
    assert joint_drift(r.arm.joints(), r.arm_config.standby_joints_deg) == 0.0
    r.close()


def test_倒计时每秒响一声(rig):
    rig.machine.start()
    rig.until(State.IDLE)
    assert len(rig.beeps) == 3  # 倒计时 3 秒 → 3 声


def test_倒计时期间界面显示还剩几秒(rig):
    rig.machine.start()
    assert rig.machine.view.headline == "RETURNING IN 3"
    rig.clock.advance(1.0)
    rig.machine.tick()
    assert rig.machine.view.headline == "RETURNING IN 2"


# ---- 触发和执行 --------------------------------------------------------


def test_空闲加触发走完一个预设动作再回空闲(rig):
    rig.to_idle()
    rig.trigger("point_left")
    rig.machine.tick()
    assert rig.machine.state is State.RUNNING
    rig.until(State.COOLDOWN)
    rig.until(State.IDLE)
    # 一个路点一段直线,做完停在那儿,不回待命位姿(2026-09-23 起)
    assert [kind for kind, _ in rig.arm.moves] == ["linear"]
    assert rig.machine.view.step == "AT X -150 MM"


def test_六个预设动作都能用调试按键触发(rig):
    rig.to_idle()
    for action in rig.demo.actions:
        before = len(rig.arm.moves)
        rig.trigger(action.name)
        rig.until(State.RUNNING)
        rig.until(State.IDLE)
        sent = len(rig.arm.moves) - before
        # 画方块、点头做完多一段直线回出发点;左右上下停在最后一个路点上
        back = 1 if action.end == "back" else 0
        assert sent == len(action.waypoints) + back, action.name


def test_执行中界面显示动作名和第几段(rig):
    rig.to_idle()
    rig.trigger("draw_square")
    rig.machine.tick()
    view = rig.machine.view
    assert view.action == "DRAW SQUARE"
    assert view.step.startswith("STEP ")
    assert view.step.endswith("/ 6")  # 5 个路点 + 回到出发点


def test_执行中除叫停外的事件一律忽略(rig):
    rig.to_idle()
    rig.trigger("draw_square")
    rig.machine.tick()
    assert rig.machine.state is State.RUNNING
    rig.trigger("nod")  # 执行中再触发:丢掉,不打断也不排队
    rig.machine.submit(Event.confirm("key:enter"))
    rig.until(State.IDLE)
    assert len(rig.arm.moves) == 6  # 只走了 draw_square,没有 nod 跟在后面
    rig.clock.advance(5.0)
    rig.machine.tick()
    assert rig.machine.state is State.IDLE
    assert len(rig.arm.moves) == 6


def test_执行预设动作前不在待命位姿附近就进已停止(rig):
    rig.to_idle()
    rig.arm.joints_now = [v + 5.0 for v in rig.arm_config.standby_joints_deg]
    rig.trigger("point_left")
    rig.machine.tick()
    assert rig.machine.state is State.HALTED
    assert rig.machine.view.reason == "NOT AT STANDBY"
    assert rig.arm.moves == []  # 核对不过就一步不动


def test_差一点点还在容差里就照常执行(rig):
    rig.to_idle()
    rig.arm.joints_now = [v + 0.5 for v in rig.arm_config.standby_joints_deg]
    rig.trigger("point_left")
    rig.machine.tick()
    assert rig.machine.state is State.RUNNING


# ---- 叫停 --------------------------------------------------------------


def test_执行中叫停走倒计时抬起回待命冷却空闲(rig):
    rig.to_idle()
    rig.trigger("draw_square")
    rig.machine.tick()
    # 假机械臂走直线不改关节角;照真机那样,动起来就离开了待命位姿
    rig.arm.joints_now = [v + 5.0 for v in rig.arm_config.standby_joints_deg]
    rig.signal.raise_stop("space")
    rig.machine.tick()
    assert rig.machine.state is State.RETURNING
    assert rig.arm.stopped >= 1
    rig.until(State.COOLDOWN)
    rig.until(State.IDLE)
    # 叫停之后最后两段是回待命:先抬起,再回待命位姿(2026-09-23 起都是关节运动)
    assert [kind for kind, _ in rig.arm.moves][-2:] == ["joints", "joints"]


def test_叫停之后不再发下一段():
    # 每段走 0.2 秒,好让叫停落在某一段中间,而不是等整套走完
    r = Rig(busy_s=0.2)
    r.to_idle()
    r.trigger("draw_square")  # 5 个路点 + 1 段回到出发点
    r.machine.tick()
    r.signal.raise_stop("space")
    r.pump(lambda: r.machine.state is State.RETURNING, what="RETURNING")
    assert len(r.arm.moves) < 6
    r.until(State.IDLE)
    r.close()


def test_回待命途中再叫停就进已停止(rig):
    rig.to_idle()
    rig.trigger("point_left")
    rig.machine.tick()
    rig.signal.raise_stop("space")
    rig.machine.tick()
    assert rig.machine.state is State.RETURNING
    rig.signal.raise_stop("space")
    rig.machine.tick()
    assert rig.machine.state is State.HALTED


def test_进已停止连响三声(rig):
    rig.to_idle()
    rig.trigger("point_left")
    rig.machine.tick()
    rig.signal.raise_stop("space")
    rig.machine.tick()
    rig.beeps.clear()  # 倒计时那几声不算
    rig.signal.raise_stop("space")
    rig.machine.tick()
    assert rig.machine.state is State.HALTED
    assert len(rig.beeps) == 3


def test_已停止按回车回待命再回空闲(rig):
    rig.to_idle()
    rig.trigger("point_left")
    rig.machine.tick()
    rig.signal.raise_stop("space")
    rig.machine.tick()
    rig.signal.raise_stop("space")
    rig.machine.tick()
    assert rig.machine.state is State.HALTED
    rig.machine.submit(Event.confirm("key:enter"))
    rig.machine.tick()
    assert rig.machine.state is State.RETURNING
    rig.until(State.IDLE)


def test_回车只在已停止时有效(rig):
    rig.to_idle()
    rig.machine.submit(Event.confirm("key:enter"))
    rig.machine.tick()
    assert rig.machine.state is State.IDLE
    assert rig.arm.moves == []


def test_空闲冷却已停止时叫停不起作用(rig):
    rig.to_idle()
    rig.signal.raise_stop("space")
    rig.machine.tick()
    assert rig.machine.state is State.IDLE
    # 标记被放下了,下一个动作不会被它拦住
    assert not rig.signal.raised
    rig.trigger("point_left")
    rig.until(State.RUNNING)
    rig.until(State.IDLE)
    assert len(rig.arm.moves) == 1


# ---- 冷却 --------------------------------------------------------------


def test_冷却期间不接受触发冷却结束后才接受(rig):
    rig.machine.start()
    rig.until(State.COOLDOWN)
    rig.trigger("point_left")
    rig.machine.tick()
    assert rig.machine.state is State.COOLDOWN
    assert rig.arm.moves == []
    rig.until(State.IDLE)
    rig.trigger("point_left")
    rig.machine.tick()
    assert rig.machine.state is State.RUNNING


def test_进冷却时先清空判定记录(rig):
    assert rig.cleared == 0
    rig.machine.start()
    rig.until(State.COOLDOWN)
    assert rig.cleared == 1
    rig.until(State.IDLE)
    rig.trigger("point_left")
    rig.until(State.COOLDOWN)
    assert rig.cleared == 2


# ---- 机械臂报错 --------------------------------------------------------

# 三种报错各自那条路在 `tests/test_faults.py`;这里只管「执行线程自己出的错」。


class BuggyArm(RecordingArm):
    """下发运动时抛一个**不是机械臂接口的**错,照的是执行线程里的 bug。

    机械臂身上什么也没挂(真的故障才挂),所以这个错只能由执行线程带出来。
    """

    def move_linear(self, pose):
        raise ValueError("执行线程里的 bug")


def test_执行线程自己出错就当作机械臂一侧报错():
    r = Rig(arm=BuggyArm((0.0,) * 6))
    try:
        r.to_idle()
        r.trigger("point_left")
        r.pump(lambda: r.machine.state is State.HALTED, what="HALTED")
        assert r.machine.view.reason == "ARM ERROR"
        assert "ValueError" in r.machine.view.detail
        # 认不出的错也清得掉:机械臂身上没挂东西,回车照样能恢复
        r.machine.submit(Event.confirm("key:enter"))
        r.machine.tick()
        assert r.machine.state is State.RETURNING
    finally:
        r.close()


# ---- 日志 --------------------------------------------------------------


def test_日志记下状态切换每段运动和叫停来源(rig):
    rig.to_idle()
    rig.trigger("point_left")
    rig.machine.tick()
    rig.signal.raise_stop("space")
    rig.machine.tick()
    rig.until(State.IDLE)
    text = "\n".join(rig.records)
    assert "IDLE → RUNNING" in text
    assert "RUNNING → RETURNING" in text
    assert "RETURNING → COOLDOWN" in text
    assert "COOLDOWN → IDLE" in text
    assert "point_left 路点 1" in text  # 每段运动
    assert "space" in text  # 叫停的来源


# ---- 退出 --------------------------------------------------------------


def test_按退出就结束(rig):
    rig.to_idle()
    assert not rig.machine.finished
    rig.machine.submit(Event.quit("key:q"))
    rig.machine.tick()
    assert rig.machine.finished


# ---- 执行线程没收好尾 --------------------------------------------------


class StuckArm(RecordingArm):
    """一直说自己还在动,执行线程就卡在「等这一段走完」里。

    照的是控制器不回「到位」那种情况:`SEGMENT_TIMEOUT_S` 到之前,程序这边只知道
    它还在动。`stuck = False` 可以把执行线程放走。
    """

    stuck = True

    def is_moving(self) -> bool:
        return self.stuck


def test_执行线程不肯收尾就停下等人不再起一个():
    arm = StuckArm((0.0,) * 6)
    r = Rig(arm=arm, job_join_s=0.1)
    try:
        # 摆成启动时不在待命位姿,这样要走回待命;第一段发出去就卡住了
        arm.joints_now = [v + 30.0 for v in r.arm_config.standby_joints_deg]
        r.machine.start()
        r.pump(lambda: len(arm.moves) >= 1, what="发出第一段")
        # 两次叫停 → 已停止。机械臂卡着,叫停照样管用
        r.signal.raise_stop("space")
        r.machine.tick()
        r.signal.raise_stop("space")
        r.pump(lambda: r.machine.state is State.HALTED, what="HALTED")

        sent = len(arm.moves)
        r.machine.submit(Event.confirm("key:enter"))  # 按回车想恢复
        r.machine.tick()
        assert r.machine.state is State.RETURNING
        # 倒计时走完该动了,但上一个执行线程还活着:停下等人,绝不再起一个
        r.pump(lambda: r.machine.state is State.HALTED, what="再次 HALTED")
        assert "EXEC STUCK" in r.machine.view.reason
        r.clock.advance(5.0)
        r.machine.tick()
        assert len(arm.moves) == sent  # 一段都没有多发出去
    finally:
        arm.stuck = False  # 放走那个执行线程,免得它挂到测试跑完
        r.close()


# ---- 叫停正好落在放下标记那一下 ----------------------------------------


class StopWhileCheckingArm(RecordingArm):
    """第一次被问关节角时顺手叫一次停。

    问关节角发生在「核对在不在待命位姿」那一步,也就是放下「已叫停」标记之前一点点;
    这正是那条极窄的缝。
    """

    def __init__(self, *args, signal=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.signal = signal
        self.armed = False

    def joints(self) -> list[float]:
        if self.armed:
            self.armed = False
            assert self.signal is not None
            self.signal.raise_stop("space")
        return super().joints()


def test_叫停落在放下标记那一下也不会被漏掉():
    arm = StopWhileCheckingArm((0.0,) * 6)
    r = Rig(arm=arm)
    try:
        arm.signal = r.signal
        r.to_idle()
        arm.armed = True
        r.trigger("draw_square")
        r.machine.tick()  # 这一轮里叫停发生在「放下标记」之前一点点
        # 下一轮就该看见它,而不是当作没发生过
        r.machine.tick()
        assert r.machine.state is State.RETURNING
        r.until(State.IDLE)
    finally:
        r.close()


# ---- 叫停和报错撞在一起 ------------------------------------------------


def test_叫停和报错撞在一起时报错说了算():
    """两件事撞在同一轮里:报错比叫停重,结果是已停止,不是回待命。

    叫停先被看见(它已经发生了,机械臂已经在停了),但同一轮里紧接着就问出机械臂身上
    挂着报错——带着故障不能再命令它动,哪怕只是回待命。
    """
    r = Rig(faults=[Fault(MotionError("运动错误"), at_move=1)])  # 第二段运动报错
    try:
        r.to_idle()
        r.trigger("draw_square")
        r.machine.tick()  # 起执行线程:第一段发出去,第二段报错
        r.wait_for_job()  # 等它报完错,但**不推状态逻辑**
        r.signal.raise_stop("space")  # 状态逻辑还没看到那个错,叫停先到了
        moved = len(r.arm.moves)
        r.machine.tick()
        assert r.machine.state is State.HALTED
        assert r.machine.view.reason == "MOTION ERROR"
        assert any("叫停 来自 space" in line for line in r.records)  # 叫停也没被漏掉
        r.clock.advance(5.0)
        r.machine.tick()
        assert len(r.arm.moves) == moved  # 带着故障没有再动过
        # 报错只报一次:按回车就能恢复,不用按两次
        r.machine.submit(Event.confirm("key:enter"))
        r.machine.tick()
        assert r.machine.state is State.RETURNING
        assert r.arm.cleared == 1
    finally:
        r.close()


# ---- 主线程问机械臂时报错 ----------------------------------------------


class BadJointsArm(RecordingArm):
    """一问关节角就报错,照的是 RoboDK 中途被关掉那种情况。"""

    def joints(self) -> list[float]:
        raise ArmError("RoboDK 断开了")


def test_主线程问机械臂报错就进已停止而不是把主循环带倒():
    r = Rig(arm=BadJointsArm((0.0,) * 6))
    try:
        r.machine.start()
        r.pump(lambda: r.machine.state is State.HALTED, what="HALTED")
        assert "ARM ERROR" in r.machine.view.reason
        assert r.arm.moves == []
        r.machine.tick()  # 再推一轮也不抛,主循环还活着
    finally:
        r.close()


# ---- 外面报上来的故障(第 04 票:识别中断走这一条) --------------------


def test_执行中收到停机就进已停止(rig):
    rig.to_idle()
    rig.trigger()
    rig.until(State.RUNNING)
    rig.machine.submit(Event.halt("VISION LOST", "vision"))
    rig.until(State.HALTED)
    assert rig.machine.view.reason == "VISION LOST"


def test_回待命途中收到停机就进已停止(rig):
    rig.machine.start()
    assert rig.machine.state is State.RETURNING
    rig.machine.submit(Event.halt("VISION LOST", "vision"))
    rig.until(State.HALTED)


def test_收到停机时机械臂当场停住(rig):
    # 进已停止本来就会立「已叫停」标记、调机械臂的停止(第 03 票),这里核对它确实发生了
    rig.to_idle()
    rig.trigger()
    rig.until(State.RUNNING)
    before = rig.arm.stopped
    rig.machine.submit(Event.halt("VISION LOST", "vision"))
    rig.until(State.HALTED)
    assert rig.arm.stopped > before


def test_停机也要能按回车爬出来(rig):
    rig.to_idle()
    rig.trigger()
    rig.until(State.RUNNING)
    rig.machine.submit(Event.halt("VISION LOST", "vision"))
    rig.until(State.HALTED)
    rig.machine.submit(Event.confirm("key:enter"))
    rig.until(State.IDLE)


# ---- 做完不回待命:停靠点(2026-09-23 用户定) -------------------------


def sent_positions(rig, since=0):
    return [pos for kind, pos in rig.arm.moves[since:] if kind == "linear"]


def test_做完停在那儿下一个动作从那儿算起(rig):
    rig.to_idle()
    rig.trigger("point_left")
    rig.until(State.RUNNING)
    rig.until(State.IDLE)
    rig.trigger("raise", "key:up")
    rig.until(State.RUNNING)
    rig.until(State.IDLE)
    # 先往左一步,再从那儿往上一步:停在左上,没有回过待命
    assert sent_positions(rig) == [(-150.0, -400.0, 550.0), (-150.0, -400.0, 650.0)]
    assert "joints" not in [kind for kind, _ in rig.arm.moves]
    assert rig.machine.view.step == "AT X -150 Z +100 MM"


def test_空闲时画面上写停在哪儿(rig):
    rig.to_idle()
    assert rig.machine.view.step == "AT STANDBY"


def test_画方块在停靠点上原地做完回原处(rig):
    rig.to_idle()
    rig.trigger("point_right", "key:right")
    rig.until(State.RUNNING)
    rig.until(State.IDLE)
    before = len(rig.arm.moves)
    rig.trigger("draw_square", "key:p")
    rig.until(State.RUNNING)
    rig.until(State.IDLE)
    square = sent_positions(rig, before)
    assert square[0] == (100.0, -400.0, 500.0)  # 方块左下角,以右边那一步为中心
    assert square[-1] == (150.0, -400.0, 550.0)  # 做完回到出发的地方
    assert rig.machine.view.step == "AT X +150 MM"


def test_做完要出活动范围就这一次不动(rig):
    rig.to_idle()
    for _ in range(2):  # 右两步,正好到边上(300 mm)
        rig.trigger("point_right", "key:right")
        rig.until(State.RUNNING)
        rig.until(State.IDLE)
    moves, beeps = len(rig.arm.moves), len(rig.beeps)
    rig.trigger("point_right", "key:right")
    rig.machine.tick()
    assert rig.machine.state is State.IDLE  # 不切状态
    assert len(rig.arm.moves) == moves  # 一步不动
    assert len(rig.beeps) == beeps  # 也不响:没认它
    assert rig.machine.view.reason == "OUT OF RANGE: POINT RIGHT"
    # 往回走照样可以,提示也就清掉了
    rig.trigger("point_left")
    rig.until(State.RUNNING)
    assert rig.machine.view.reason == ""
    rig.until(State.IDLE)
    assert rig.machine.view.step == "AT X +150 MM"


def test_从别的停靠点出发第一次先查可达性查过的记住(rig):
    rig.to_idle()
    startup_checks = len(rig.arm.reach_checks)
    rig.trigger("point_left")  # 从待命点出发:启动时查过,不再查
    rig.until(State.RUNNING)
    rig.until(State.IDLE)
    assert len(rig.arm.reach_checks) == startup_checks
    for _ in range(2):  # 从左边那一步出发做点头:第一次查,第二次不查
        rig.trigger("nod", "key:l")
        rig.until(State.RUNNING)
        rig.until(State.IDLE)
    assert rig.arm.reach_checks[startup_checks:] == [("nod",)]


def test_从停靠点出发够不着就不动(rig):
    rig.to_idle()
    rig.trigger("point_left")
    rig.until(State.RUNNING)
    rig.until(State.IDLE)
    rig.arm.unreachable.add("draw_square")
    moves = len(rig.arm.moves)
    rig.trigger("draw_square", "key:p")
    rig.until(State.RUNNING)
    rig.until(State.COOLDOWN)
    assert len(rig.arm.moves) == moves  # 一段都没发
    assert rig.machine.view.reason == "CAN'T REACH: DRAW SQUARE"
    rig.until(State.IDLE)
    assert rig.machine.view.step == "AT X -150 MM"  # 还在原来那儿
    # 再做一次:之前查过够不着,当场就不动,不再查
    checks = len(rig.arm.reach_checks)
    rig.trigger("draw_square", "key:p")
    rig.machine.tick()
    assert rig.machine.state is State.IDLE
    assert len(rig.arm.reach_checks) == checks
    assert rig.machine.view.reason == "CAN'T REACH: DRAW SQUARE"


def test_停靠点上被人挪过就进已停止(rig):
    rig.to_idle()
    rig.trigger("point_left")
    rig.until(State.RUNNING)
    rig.until(State.IDLE)
    rig.arm.joints_now = [v + 5.0 for v in rig.arm.joints_now]
    moves = len(rig.arm.moves)
    rig.trigger("raise", "key:up")
    rig.machine.tick()
    assert rig.machine.state is State.HALTED
    assert rig.machine.view.reason == "NOT WHERE IT STOPPED"
    assert len(rig.arm.moves) == moves


# ---- 按 H 回待命 -------------------------------------------------------


def test_按H回待命走倒计时抬起再慢速回(rig):
    rig.to_idle()
    rig.trigger("point_left")
    rig.until(State.RUNNING)
    rig.until(State.IDLE)
    moves = len(rig.arm.moves)
    rig.machine.submit(Event.home("key:h"))
    rig.machine.tick()
    assert rig.machine.state is State.RETURNING
    assert rig.machine.view.headline == "RETURNING IN 3"
    rig.until(State.COOLDOWN)
    rig.until(State.IDLE)
    kinds = [kind for kind, _ in rig.arm.moves[moves:]]
    assert kinds == ["joints", "joints"]  # 先抬起,再回待命位姿;两段都是关节运动
    assert rig.machine.view.step == "AT STANDBY"
    # 回来以后再往左,又是从待命点算
    rig.trigger("point_left")
    rig.until(State.RUNNING)
    rig.until(State.IDLE)
    assert sent_positions(rig)[-1] == (-150.0, -400.0, 550.0)


def test_冷却时按H也回待命(rig):
    rig.to_idle()
    rig.trigger("point_left")
    rig.until(State.RUNNING)
    rig.until(State.COOLDOWN)
    rig.machine.submit(Event.home("key:h"))
    rig.machine.tick()
    assert rig.machine.state is State.RETURNING


def test_已经在待命点上按H不动(rig):
    rig.to_idle()
    rig.machine.submit(Event.home("key:h"))
    rig.machine.tick()
    assert rig.machine.state is State.IDLE
    assert rig.arm.moves == []


def test_回待命手势和按H是同一件事(rig):
    """rock 手势对应的是保留名 home:空闲时停在别处就回待命,和按 H 一样。"""
    from robot_side.config import HOME_ACTION

    rig.to_idle()
    rig.trigger("point_left")
    rig.until(State.RUNNING)
    rig.until(State.IDLE)
    rig.machine.submit(Event.trigger(HOME_ACTION, "gesture:pose_hold:rock"))
    rig.machine.tick()
    assert rig.machine.state is State.RETURNING
    rig.until(State.IDLE)
    assert rig.machine.view.step == "AT STANDBY"


def test_已经在待命点上做回待命手势不动(rig):
    from robot_side.config import HOME_ACTION

    rig.to_idle()
    rig.machine.submit(Event.trigger(HOME_ACTION, "gesture:pose_hold:rock"))
    rig.machine.tick()
    assert rig.machine.state is State.IDLE
    assert rig.arm.moves == []


def test_执行中按H不起作用要回待命就叫停():
    r = Rig(busy_s=0.2)  # 每段走 0.2 秒,按 H 时还在执行中
    try:
        r.to_idle()
        r.trigger("draw_square")
        r.machine.tick()
        assert r.machine.state is State.RUNNING
        r.machine.submit(Event.home("key:h"))
        r.machine.tick()
        assert r.machine.state is State.RUNNING
        assert any("只有空闲、冷却时才有效" in line for line in r.records)
    finally:
        r.close()



# ---- 连真机:活动范围事先查好,运行中不查 --------------------------------


LEFT = (-150.0, 0.0, 0.0)


def test_事先查好的组合直接走运行中一次都不查():
    r = Rig(check_ahead=False, reachable={(LEFT, "nod"): True})
    try:
        r.to_idle()
        checks = len(r.arm.reach_checks)
        r.trigger("point_left")
        r.until(State.RUNNING)
        r.until(State.IDLE)
        r.trigger("nod", "key:l")
        r.until(State.RUNNING)
        r.until(State.IDLE)
        assert len(r.arm.reach_checks) == checks  # 一次都没查
        assert r.machine.view.step == "AT X -150 MM"
    finally:
        r.close()


def test_运行中不许查时没查过的就当够不着():
    r = Rig(check_ahead=False, reachable={(LEFT, "draw_square"): False})
    try:
        r.to_idle()
        r.trigger("point_left")
        r.until(State.RUNNING)
        r.until(State.IDLE)
        moves, checks = len(r.arm.moves), len(r.arm.reach_checks)
        for action in ("draw_square", "nod"):  # 查过不过的、根本没查过的
            r.trigger(action, "key:p")
            r.machine.tick()
            assert r.machine.state is State.IDLE
            assert r.machine.view.reason.startswith("CAN'T REACH")
        assert len(r.arm.moves) == moves
        assert len(r.arm.reach_checks) == checks
    finally:
        r.close()



# ---- 回待命的抬起:关节运动,加护栏(2026-09-23 真机那一次之后) ------------


def test_回待命抬起到求出来的那组关节角():
    r = Rig()
    try:
        r.arm.joints_now = [v + 30.0 for v in r.arm_config.standby_joints_deg]
        lift = [v + 31.0 for v in r.arm_config.standby_joints_deg]  # 各轴只转 1°
        r.arm.ik_answer = lift
        r.machine.start()
        r.until(State.IDLE)
        assert r.arm.moves[0] == ("joints", tuple(lift))
        assert r.arm.moves[1] == ("joints", tuple(r.arm_config.standby_joints_deg))
        # 出发时的关节角记进日志,事后查得到从哪儿回来的
        assert any("回待命 从关节角" in line for line in r.records)
    finally:
        r.close()


def test_抬起求不出关节角就一步不动停下等人():
    r = Rig()
    try:
        r.arm.joints_now = [v + 30.0 for v in r.arm_config.standby_joints_deg]
        r.arm.ik_answer = None
        r.machine.start()
        r.until(State.HALTED)
        assert r.machine.view.reason == "CAN'T LIFT"
        assert "示教器" in r.machine.view.detail
        assert r.arm.moves == []
    finally:
        r.close()


def test_抬起要甩臂就一步不动停下等人():
    # 逆解换了一支构型:照着关节运动走过去,机械臂会大幅甩一下
    r = Rig()
    try:
        here = [v + 30.0 for v in r.arm_config.standby_joints_deg]
        r.arm.joints_now = list(here)
        r.arm.ik_answer = [here[0] + 90.0, *here[1:]]
        r.machine.start()
        r.until(State.HALTED)
        assert r.machine.view.reason == "CAN'T LIFT"
        assert r.arm.moves == []
    finally:
        r.close()

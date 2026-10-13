"""RoboDK 实现把 API 抛的错翻成机械臂一侧的报错。

**要开 RoboDK 才能跑**:没开就整份跳过,不算失败(spec「测试」)。别的测试都不需要
RoboDK——报错路径由假机械臂覆盖(`tests/test_faults.py`)。

跑之前请自己打开 RoboDK 5.9;这里不替你启动它,免得测试在后台开出一个进程来。
不动用户的东西:工作站、仿真速度、碰撞检查开关都由 `RoboDKArm.disconnect()` 恢复,
机械臂也放回跑之前的那组关节角。
"""

from __future__ import annotations

import pytest
from robodk.robomath import transl

from robot_side.arm import ArmError, MotionError
from robot_side.robodk_arm import JAKA, RoboDKArm
from robot_side.station import STATION_FILE
from tests.robodk_guard import needs_robodk

# 没开 RoboDK、或者有人连着真机(机械臂驱动在跑)就跳过,见 tests/robodk_guard.py
pytestmark = needs_robodk

UNREACHABLE = transl(9000.0, 0.0, 0.0)
"""远在臂展之外的一个点。求不出逆解,RoboDK 会当场报错,机械臂一步不动。"""


@pytest.fixture
def arm():
    if not STATION_FILE.is_file():
        pytest.skip(f"没有工作站文件 {STATION_FILE};先跑建站脚本")
    connected = RoboDKArm(STATION_FILE, JAKA)
    connected.connect()
    before = connected.joints()
    try:
        yield connected
    finally:
        connected.set_joints(before)  # 放回跑之前的那组关节角
        connected.disconnect()


def test_连上之后身上没挂报错(arm):
    assert arm.error() is None
    arm.clear_error()  # 没有可清的,也不该抛
    assert arm.error() is None


def test_够不着的一段翻成运动错误并挂起来(arm):
    before = arm.joints()
    with pytest.raises(ArmError) as caught:
        arm.move_linear(UNREACHABLE)
    assert isinstance(caught.value, MotionError)
    assert arm.error() is caught.value  # 挂住了,问得出来
    assert not arm.is_moving()
    assert arm.joints() == before  # 报错的那一段一步没动

    arm.clear_error()
    assert arm.error() is None
    # 清完还能接着用:同一台机械臂照常答得上话
    assert len(arm.joints()) == 6


def test_断开之后再下发就报没连接(arm):
    arm.disconnect()
    with pytest.raises(ArmError):
        arm.move_linear(transl(0.0, 0.0, 0.0))
    arm.connect()  # 连回来,好让 fixture 收尾


# ---- 速度:按尖端换算、上限、真机上发不发、驱动换没换 ----------------------


def test_只转_j6_尖端几乎不动就不换算(arm):
    from robot_side.config import load_arm

    config = load_arm("jaka")
    arm.apply_config(config)
    here = list(config.standby_joints_deg)
    turned = here[:5] + [here[5] + 10.0]
    assert arm.joint_speed_for(here, turned, 50.0) is None


def test_点头那种按尖端走多快换算出关节速度(arm):
    from robot_side.config import load_arm
    from robot_side.motion import joint_path_mm

    config = load_arm("jaka")
    arm.apply_config(config)
    here = list(config.standby_joints_deg)
    nodded = here[:3] + [here[3] + 10.0] + here[4:]
    degs = arm.joint_speed_for(here, nodded, 50.0)
    path = joint_path_mm(arm.flange_pose, here, nodded, config.pointer_offset_mm)
    assert 0.0 < degs <= config.max_joint_degs
    assert 10.0 / degs == pytest.approx(path / 50.0)  # 转完 10° 的时间 = 尖端走完的时间


def test_发出去的速度不超过配置里的上限(arm, monkeypatch):
    from robot_side.config import load_arm

    config = load_arm("jaka")
    arm.apply_config(config)
    sent = []
    monkeypatch.setattr(arm._robot, "setSpeed", lambda **kw: sent.append(kw))
    arm.set_speed(joints_degs=1000.0)
    arm.set_speed(linear_mms=5000.0)
    assert sent == [
        {"speed_linear": -1, "speed_joints": config.max_joint_degs},
        {"speed_linear": config.max_linear_mms, "speed_joints": -1},
    ]


def test_连着真机时没换上包过的驱动就不发速度(arm, monkeypatch):
    sent = []
    monkeypatch.setattr(arm._robot, "setSpeed", lambda **kw: sent.append(kw))
    monkeypatch.setattr(arm, "_real_ip", "192.0.2.1")  # 只是假装连着:不真连
    arm.set_speed(linear_mms=20.0)
    assert sent == []
    monkeypatch.setattr(arm, "_driver_before", "JAKADriver")  # 换上了
    arm.set_speed(linear_mms=20.0)
    assert sent == [{"speed_linear": 20.0, "speed_joints": -1}]


def test_驱动路径换成包过的核对过再改回原来的(arm):
    from robot_side.robodk_arm import KEEP_SPEED_DRIVER

    robot = arm._robot
    before = robot.setParam("PathDriver", "")
    arm._use_keep_speed_driver(robot)
    assert robot.setParam("PathDriver", "") == KEEP_SPEED_DRIVER.as_posix()
    arm._restore_driver(robot)
    assert robot.setParam("PathDriver", "") == before

"""拿 RoboDK 里的 JAKA 模型核对 9-22 那一次手算的待命位姿。

9-22 用户在示教器上摆的 J6 是 −52.043;手算出来要转 −30.233° 到 −82.276,基坐标系姿态
落在 `RX=−179.335 RY=0.091 RZ=0.000`(`config/arms/jaka.toml` 待命位姿上面那段)。
`plan_standby` 从同一组关节角出发,应当算出同一个数。

**要开 RoboDK 才能跑**:没开就整份跳过,不算失败(和 `tests/test_robodk_arm.py` 同一个
规矩)。只算正解,机械臂模型一步不动。
"""

from __future__ import annotations

import pytest

from robot_side.config import load_arm, load_demo
from robot_side.motion import orientation_pose
from robot_side.robodk_arm import JAKA, RoboDKArm
from robot_side.standby import plan_standby
from robot_side.station import STATION_FILE
from tests.robodk_guard import needs_robodk

# 没开 RoboDK、或者有人连着真机(机械臂驱动在跑)就跳过,见 tests/robodk_guard.py
pytestmark = needs_robodk

PICKED_2026_09_22 = (187.723, 90.782, 98.739, 80.480, -89.329, -52.043)


@pytest.fixture
def jaka():
    if not STATION_FILE.is_file():
        pytest.skip(f"没有工作站文件 {STATION_FILE};先跑建站脚本")
    arm = RoboDKArm(STATION_FILE, JAKA)
    arm.connect()
    try:
        arm.apply_config(load_arm("jaka"))
        yield arm
    finally:
        arm.disconnect()


def test_算出来的就是_9_22_手算的那一组(jaka):
    result = plan_standby(
        PICKED_2026_09_22,
        flange_in_worktable=jaka.flange_pose,
        flange_in_base=jaka.flange_pose_in_base,
        default_orientation=orientation_pose(load_demo().default_orientation_deg),
        fixed_point=True,
    )
    assert result.ok, result.problems
    assert result.joints[:5] == PICKED_2026_09_22[:5]
    assert result.joints[5] == pytest.approx(-82.276, abs=0.01)
    assert result.j6_turn_deg == pytest.approx(-30.233, abs=0.01)
    assert result.moved_mm < 0.01
    rx, ry, rz = result.base_rpy_after
    assert abs(rx) == pytest.approx(179.335, abs=0.01)
    assert ry == pytest.approx(0.091, abs=0.01)
    assert rz == pytest.approx(0.0, abs=0.01)
    assert result.joints == pytest.approx(load_arm("jaka").standby_joints_deg, abs=0.01)

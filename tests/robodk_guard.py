"""要开 RoboDK 的那几份测试,跑之前先看 RoboDK 能不能拿来用。

两种情况不跑,都跳过、不算失败:

- RoboDK 没开;
- **有机械臂驱动在跑**——多半有人经 RoboDK 连着真机。这时测试连同一个 RoboDK、往工作站里
  那台机械臂写配置,指令会顺着驱动发到控制柜:2026-09-24 用户连着真机做手势,自动测试的
  `apply_config` 经驱动发了两次 `SETTOOL`,把控制柜的当前工具切到了 10 号。
  查不了有没有驱动(`arm_driver_processes` 返回 None)也按「可能有」处理。
"""

from __future__ import annotations

import pytest

from robot_side.robodk_arm import arm_driver_processes, robodk_is_running


def why_not_robodk() -> str | None:
    """不能跑的原因;能跑返回 None。"""
    if not robodk_is_running():
        return "RoboDK 没开:这几条要开 RoboDK 才能跑,跳过不算失败"
    drivers = arm_driver_processes()
    if drivers is None:
        return "查不了有没有机械臂驱动在跑,按可能有人连着真机算,RoboDK 测试不跑"
    if drivers:
        return (
            f"有机械臂驱动在跑(进程 {drivers}),多半有人连着真机;"
            "RoboDK 测试会往连着的驱动发指令,不跑"
        )
    return None


WHY_NOT = why_not_robodk()

needs_robodk = pytest.mark.skipif(WHY_NOT is not None, reason=WHY_NOT or "")
"""三份 RoboDK 测试的 `pytestmark`。"""

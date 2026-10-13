"""核对速度的小脚本:从包过的驱动那份日志里挑出这一次的几行。不开 RoboDK。"""

from __future__ import annotations

from datetime import datetime

from robot_side.scripts.speed_check import driver_log_lines

LOG = """\
21:48:45.924 启动:RoboDK Driver for JAKAZu v1.1.2,速度初值同原驱动 直线 100 mm/s、关节 100 °/s(再乘控制柜倍率)
21:49:21.541 指令 'CONNECT 127.0.0.1 10000 6'
22:10:03.100 启动:RoboDK Driver for JAKAZu v1.1.2,速度初值同原驱动 直线 100 mm/s、关节 100 °/s(再乘控制柜倍率)
22:10:30.200 SPEED 收到 '-1.000 20.000 -1.000 -1.000',现在 直线 100 mm/s、关节 20 °/s(再乘控制柜倍率)
22:10:30.300 MOVJ 按 直线 100 mm/s、关节 20 °/s(再乘控制柜倍率)
"""


def test_只要这一次开始以后的(tmp_path):
    (tmp_path / "jaka-driver-20260924.log").write_text(LOG, encoding="utf-8")
    lines = driver_log_lines(datetime(2026, 9, 24, 22, 10, 0), tmp_path)
    assert len(lines) == 3
    assert lines[0].startswith("22:10:03.100 启动")
    assert "MOVJ 按" in lines[-1]


def test_没有这一天的日志就是空的(tmp_path):
    assert driver_log_lines(datetime(2026, 9, 24, 22, 10, 0), tmp_path) == []

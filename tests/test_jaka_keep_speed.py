"""包过的 JAKA 驱动:程序设的速度要真的传到 `joint_move` / `linear_move`。

跑的是**真的**原驱动(`C:/RoboDK/api/Robot/v310/JakaDriver.pyc`),在 RoboDK 的嵌入式
Python 3.10 里,控制柜换成只记账的假的(`tests/jaka_driver_harness.py`)。没装 RoboDK
就整份跳过。不连任何东西。

第一条测试钉住的是诊断本身:原驱动收下 SPEED,下一行 MOVJ 又重置回 100。哪天 RoboDK
更新了驱动、修掉了这一条,它会失败——那时候这一层包装就该重新想想还要不要。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

EMBEDDED_PYTHON = Path("C:/RoboDK/Python-Embedded/python.exe")
DRIVER = Path("C:/RoboDK/api/Robot/v310/JakaDriver.pyc")
HARNESS = Path(__file__).with_name("jaka_driver_harness.py")

pytestmark = pytest.mark.skipif(
    not (EMBEDDED_PYTHON.is_file() and DRIVER.is_file()),
    reason="没装 RoboDK 或没有它的 JAKA 驱动,跳过不算失败",
)


@pytest.fixture(scope="module")
def fed(tmp_path_factory):
    logs = tmp_path_factory.mktemp("driver-logs")
    done = subprocess.run(
        [str(EMBEDDED_PYTHON), str(HARNESS), str(logs)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    last = done.stdout.strip().splitlines()[-1]
    return json.loads(last), logs


def test_原驱动收下速度又重置回_100(fed):
    out, _ = fed
    assert out["stock"] == [["MOVJ", pytest.approx(100.0)], ["MOVL", pytest.approx(100.0)]]


def test_包过以后程序设的速度传到了(fed):
    out, _ = fed
    assert out["kept"] == [
        ["MOVJ", pytest.approx(5.0)],
        ["MOVL", pytest.approx(20.0)],
        ["MOVJ", pytest.approx(5.0)],
    ]


def test_只准比原驱动慢(fed):
    out, _ = fed
    assert out["capped"] == [["MOVJ", pytest.approx(100.0)], ["MOVL", pytest.approx(100.0)]]


def test_负一是不改(fed):
    out, _ = fed
    assert out["unchanged"] == [["MOVJ", pytest.approx(5.0)], ["MOVL", pytest.approx(20.0)]]
    assert out["partial"] == [["MOVJ", pytest.approx(5.0)], ["MOVL", pytest.approx(30.0)]]


def test_字节码对不上就不装(fed):
    out, _ = fed
    assert out["mismatch"] != "installed"
    assert "不启动" in out["mismatch"]


def test_每行指令和生效的速度记进日志(fed):
    _, logs = fed
    (log,) = logs.glob("jaka-driver-*.log")
    text = log.read_text(encoding="utf-8")
    assert "SPEED 收到 '20 5 100 30'" in text
    assert "MOVJ 按 直线 20 mm/s、关节 5 °/s" in text

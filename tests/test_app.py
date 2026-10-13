"""正式程序的入口:参数和「什么都没对时怎么退出」(第 04 票)。

真正跑起来要摄像头和人站在前面,所以这里只验不用摄像头就能验的几件:默认不连会动的
那两档、配置不对时一次都不碰机械臂、几种退出码分得开。整条链路见
`tests/test_vision_demo.py`,画面见 `tests/test_overlay.py`。
"""

from __future__ import annotations

import time

import pytest

from robot_side import preview
from vision_side import app


def test_不给arm就是空实现():
    # 会动的那两档要明写是哪一台,和机器人一侧的规矩一致
    assert app.which_arm(app.parse_args([])) == "noarm"
    assert app.which_arm(app.parse_args(["--arm", "jaka"])) == "jaka"


def test_默认不连真机():
    assert not app.parse_args([]).real
    assert app.parse_args(["--arm", "jaka", "--real"]).real


def test_不给camera就照演示配置():
    # 默认留空,连摄像头时才去问**这一次用的那份**演示配置(--demo 换过就是换过的那份)
    assert app.parse_args([]).camera is None
    assert app.parse_args(["--camera", "2"]).camera == 2


def test_默认不开调试按键():
    # 正式演示时方向键别被误按
    assert not app.parse_args([]).debug
    assert app.parse_args(["--debug"]).debug


def test_配置不对就退出一次都不碰机械臂(capsys):
    assert app.main(["--arm", "没有这一台"]) == app.EXIT_CONFIG
    said = capsys.readouterr().err
    assert "什么也没连" in said


def test_几种退出码分得开():
    # 屏幕上说清是哪一步不对,退出码也要分得开(票:「说清是哪一步不对」)
    codes = [
        app.EXIT_OK,
        app.EXIT_VISION,
        app.EXIT_CONFIG,
        app.EXIT_MODELS,
        app.EXIT_REACH,
        app.EXIT_REAL,
    ]
    assert len(set(codes)) == len(codes)


def test_对准期间那一行写明还没连机械臂():
    # 前四步一次都不碰机械臂,这件事要让人在屏幕上看得见
    assert app.NOT_CONNECTED.isascii()
    assert "NOT CONNECTED" in app.NOT_CONNECTED


# ---- 预览窗口怎么摆 ----------------------------------------------------


def _arm(name: str):
    from robot_side.config import arm_config_path, load_arm_config

    return load_arm_config(arm_config_path(name))


def test_走RoboDK默认并排好看得见仿真():
    # 2026-09-23:全屏置顶把 RoboDK 整个挡住了
    assert app.parse_args([]).window is None
    assert app.layout_for(None, _arm("jaka")) == preview.SIDE
    assert app.layout_for(None, _arm("nova5")) == preview.SIDE


def test_空实现默认整屏():
    assert app.layout_for(None, _arm("noarm")) == preview.FULL


def test_给了window就照给的():
    args = app.parse_args(["--arm", "jaka", "--window", "full"])
    assert app.layout_for(args.window, _arm("jaka")) == preview.FULL
    args = app.parse_args(["--window", "side"])
    assert app.layout_for(args.window, _arm("noarm")) == preview.SIDE


# ---- 连机械臂那几秒画面照样刷 ------------------------------------------


def test_连机械臂的时候画面照样刷():
    # 原来连 RoboDK 那 6 秒在主线程上做,预览窗口一动不动,看着像卡死在对准那一步
    pumps = []
    result = app.while_pumping(
        lambda: time.sleep(0.2) or "plans", lambda: pumps.append(1), every_s=0.01
    )
    assert result == "plans"
    assert len(pumps) >= 5


def test_连机械臂出的错原样抛回主线程():
    def work():
        raise RuntimeError("RoboDK 打不开")

    with pytest.raises(RuntimeError, match="RoboDK 打不开"):
        app.while_pumping(work, lambda: None, every_s=0.01)


def test_连得再快也至少刷一次画面():
    pumps = []
    assert app.while_pumping(lambda: 1, lambda: pumps.append(1), every_s=0.01) == 1
    assert pumps



# ---- 连真机(机器人一侧第 05 票接进正式程序) -----------------------------


def test_连真机不写arm就拒绝什么也不连(capsys):
    # 连错台比不连更糟,而默认值恰恰最容易连错(机器人一侧用户故事 2)
    assert app.main(["--real"]) == app.EXIT_CONFIG
    said = capsys.readouterr().err
    assert "什么也没连" in said and "--arm" in said


def test_连真机配置里没填IP就拒绝(capsys):
    # 仓库里那份 jaka.toml 的 ip 是空的:IP 不进仓库,只填在副本里
    assert app.main(["--arm", "jaka", "--real"]) == app.EXIT_CONFIG
    assert "IP" in capsys.readouterr().err


def test_空实现连不了真机(capsys):
    assert app.main(["--arm", "noarm", "--real"]) == app.EXIT_CONFIG
    assert "robodk" in capsys.readouterr().err


def test_连真机默认整屏看的是真机():
    jaka = _arm("jaka")
    assert app.layout_for(None, jaka, real=True) == preview.FULL
    assert app.layout_for(None, jaka, real=False) == preview.SIDE
    assert app.layout_for("side", jaka, real=True) == preview.SIDE

"""预览窗口里那几件不用真开窗口就能验的事:按键说明和前台检查。

真开窗口会抢屏幕(全屏置顶),所以跑得起来的那一条只做最小的一次:开、画一帧、关。
"""

from __future__ import annotations

import pytest

from robot_side import preview
from robot_side.config import load_demo
from robot_side.state import State, View


def _font_px(label) -> int:
    """这一处字的字号(像素)。窗口里用的是起了名的字体,窗口大小一变跟着改。"""
    import tkinter.font as tkfont

    font = tkfont.nametofont(str(label.cget("font")), root=label)
    return -int(font.cget("size"))


@pytest.fixture
def demo():
    return load_demo()


# ---- 按键说明 ----------------------------------------------------------


def test_按键说明一律英文(demo):
    text = preview.key_help(demo, debug=True)
    assert text.isascii(), text


def test_空格那一条写明到哪儿都管用(demo):
    # 窗口不在前台也叫得停(ADR-0012),要让人一眼看到
    assert "SPACE = STOP (WORKS ANYWHERE)" in preview.key_help(demo, debug=False)


def test_不加调试开关就不列调试按键(demo):
    assert "DEBUG KEYS" not in preview.key_help(demo, debug=False)
    assert "DEBUG KEYS" in preview.key_help(demo, debug=True)


def test_调试按键按演示配置里的对应关系列出来(demo):
    text = preview.key_help(demo, debug=True)
    for trigger in demo.triggers:
        name = preview.DEBUG_KEY_NAMES[(trigger.kind, trigger.key)]
        assert f"{name} = {demo.display_for(trigger.action)}" in text


def test_每个触发手势都有一个调试按键(demo):
    for trigger in demo.triggers:
        assert (trigger.kind, trigger.key) in preview.DEBUG_KEY_NAMES
    # 绑定的键和列出来的名字是同一批
    assert set(preview.DEBUG_KEYS.values()) == set(preview.DEBUG_KEY_NAMES)
    # 启动日志和按键说明列的是同一份,每个触发手势一条(日志原来写死了「方向键、P、L」)
    pairs = preview.debug_key_pairs(demo)
    assert len(pairs) == len(demo.triggers)
    assert "F = MOVE AWAY" in pairs and "B = COME CLOSER" in pairs


def test_红字写明空格仍然能叫停():
    assert "SPACE STILL STOPS" in preview.NOT_FOREGROUND


# ---- 前台检查 ----------------------------------------------------------


class _FakeUser32:
    def __init__(self, mine: int, foreground: int) -> None:
        self.mine = mine
        self.foreground = foreground

    def GetAncestor(self, hwnd: int, flag: int) -> int:
        return self.mine

    def GetForegroundWindow(self) -> int:
        return self.foreground


class _FakeWindow:
    def winfo_id(self) -> int:
        return 1


def test_在前台时不提示(monkeypatch):
    monkeypatch.setattr(preview, "_user32", _FakeUser32(mine=7, foreground=7))
    assert preview._is_foreground(_FakeWindow())


def test_别的窗口在前台时算不在前台(monkeypatch):
    monkeypatch.setattr(preview, "_user32", _FakeUser32(mine=7, foreground=9))
    assert not preview._is_foreground(_FakeWindow())


def test_找不到自己的窗口也算不在前台(monkeypatch):
    # 宁可多提示一次(`docs/grill-decisions.md` 第二节)
    monkeypatch.setattr(preview, "_user32", _FakeUser32(mine=0, foreground=0))
    assert not preview._is_foreground(_FakeWindow())


# ---- 真开一次窗口 ------------------------------------------------------


def test_窗口开得起来画得出一帧(demo):
    tk = pytest.importorskip("tkinter")
    try:
        window = preview.PreviewWindow(demo, on_confirm=print, on_quit=print)
    except tk.TclError as exc:  # 没有桌面就跳过,不算失败
        pytest.skip(f"开不了窗口:{exc}")
    try:
        # 字高按屏幕的每厘米像素数算,至少 1 cm(1–1.5 m 外看得清)
        per_cm = window._root.winfo_fpixels("1c")
        assert per_cm > 0
        # Tk 的字号写成负数就是「多少像素高」,每一处写字的地方都得有 1 cm:
        # 顶栏那几行、按键说明的键帽和说明、红字
        labels = window._labels()
        assert window._key_labels, "按键说明一条也没排出来"
        for label in labels:
            assert _font_px(label) >= per_cm * preview.MIN_CM, label.cget("text")
        window.update(
            View(State.HALTED, "HALTED", reason="STOPPED WHILE RETURNING (SPACE)")
        )
        assert window._headline.cget("text") == "HALTED"
        assert window._headline.cget("fg") == preview.COLORS[State.HALTED]
    finally:
        window.close()
    window.update(View(State.IDLE, "IDLE"))  # 关掉之后再画也不报错


# ---- 中间那块画面(视觉一侧第 04 票填上的) ---------------------------


def test_接了摄像头才列出按R重新对准(demo):
    assert "R = REALIGN" not in preview.key_help(demo, debug=False)
    assert "R = REALIGN" in preview.key_help(demo, debug=False, realign=True)


def test_按键说明加上R之后还是一律英文(demo):
    text = preview.key_help(demo, debug=True, realign=True)
    assert text.isascii(), text


def test_中间那块放得进一幅画面(demo):
    tk = pytest.importorskip("tkinter")
    np = pytest.importorskip("numpy")
    try:
        window = preview.PreviewWindow(
            demo, on_confirm=print, on_quit=print, on_realign=print
        )
    except tk.TclError as exc:  # 没有桌面就跳过,不算失败
        pytest.skip(f"开不了窗口:{exc}")
    try:
        # 不给画面就不动中间那块:机器人一侧自己跑时那里一直是留白
        window.update(View(State.IDLE, "IDLE"))
        assert window._photo is None
        # 中间那块有多大要先问过再画:画面缩到放得下由画它的人做
        width, height = window.panel_size
        assert width > 1 and height > 1
        image = np.zeros((48, 64, 3), np.uint8)
        image[:, :, 2] = 255  # BGR 的红
        window.update(View(State.IDLE, "IDLE"), image)
        assert window._photo is not None
        assert (window._photo.width(), window._photo.height()) == (64, 48)
        # 放进去的是红的:换通道那一步反了的话这里会是蓝的
        assert window._photo.get(0, 0)[:3] == (255, 0, 0)
    finally:
        window.close()


# ---- 已停止时怎么出来 --------------------------------------------------


def test_已停止时写明按回车恢复(demo):
    # 2026-09-23 用户被停住以后对着摄像头又做了半分钟手势,不知道要按回车
    tk = pytest.importorskip("tkinter")
    try:
        window = preview.PreviewWindow(demo, on_confirm=print, on_quit=print)
    except tk.TclError as exc:  # 没有桌面就跳过,不算失败
        pytest.skip(f"开不了窗口:{exc}")
    try:
        window.update(View(State.HALTED, "HALTED", reason="STOPPED BY SPACE"))
        assert window._step.cget("text") == preview.RESUME_HINT
        assert "ENTER" in preview.RESUME_HINT and preview.RESUME_HINT.isascii()
        # 出了已停止就不再显示
        window.update(View(State.RETURNING, "RETURNING IN 3"))
        assert window._step.cget("text") == ""
        window.update(View(State.RUNNING, "RUNNING", step="STEP 2 / 6"))
        assert window._step.cget("text") == "STEP 2 / 6"
    finally:
        window.close()


# ---- 按键说明排不下时怎么折 --------------------------------------------


def _chars(text: str) -> float:
    return float(len(text))  # 每个字一像素宽,好算


def test_放得下就不折行(demo):
    text = preview.key_help(demo, debug=False, realign=True)
    assert preview.fit_lines(text, 1000, _chars) == text


def test_放不下时只在两条之间折行(demo):
    text = preview.key_help(demo, debug=True, realign=True)
    items = [item for line in text.split("\n") for item in line.split(preview.GAP)]
    fitted = preview.fit_lines(text, 40, _chars)
    rows = fitted.split("\n")
    assert len(rows) > len(text.split("\n"))
    # 每一条都原样留在某一行里,没有哪一条被从中间断开
    assert [item for row in rows for item in row.split(preview.GAP)] == items
    for row in rows:
        assert len(row) <= 40 or preview.GAP not in row


def test_一条本身就放不下时单占一行():
    fitted = preview.fit_lines(f"A{preview.GAP}{'B' * 50}{preview.GAP}C", 10, _chars)
    assert fitted.split("\n") == ["A", "B" * 50, "C"]


# ---- 并排(仿真时看得见 RoboDK) ----------------------------------------


def test_没有的那一档开不起来(demo):
    with pytest.raises(ValueError):
        preview.PreviewWindow(demo, on_confirm=print, on_quit=print, layout="tiny")


def test_并排时只占左半屏字跟着缩(demo):
    tk = pytest.importorskip("tkinter")
    try:
        window = preview.PreviewWindow(
            demo, on_confirm=print, on_quit=print, layout=preview.SIDE
        )
    except tk.TclError as exc:  # 没有桌面就跳过,不算失败
        pytest.skip(f"开不了窗口:{exc}")
    try:
        window.update(View(State.IDLE, "IDLE"))
        root = window._root
        screen = root.winfo_screenwidth()
        assert not root.attributes("-fullscreen")
        assert root.attributes("-topmost")
        # 连边框占工作区左边一半,右边那一半留给 RoboDK
        assert root.winfo_x() <= 0 + 1
        assert root.winfo_width() <= screen * preview.SIDE_SHARE
        assert root.winfo_width() > screen * preview.SIDE_SHARE * 0.9
        # 字号和画中间那块用的每厘米像素数一起缩,整屏时的比例不变
        per_cm = root.winfo_fpixels("1c")
        assert window.px_per_cm < per_cm
        assert _font_px(window._warning) < per_cm * preview.MIN_CM
        # 中间那块照样有地方放画面
        width, height = window.panel_size
        assert width > 1 and height > 1
    finally:
        window.close()


def test_整屏时不去挪别的窗口(demo):
    tk = pytest.importorskip("tkinter")
    try:
        window = preview.PreviewWindow(demo, on_confirm=print, on_quit=print)
    except tk.TclError as exc:  # 没有桌面就跳过,不算失败
        pytest.skip(f"开不了窗口:{exc}")
    try:
        assert not window.make_room("RoboDK.exe")
    finally:
        window.close()


def test_找不到那个程序的窗口就不挪(demo):
    tk = pytest.importorskip("tkinter")
    try:
        window = preview.PreviewWindow(
            demo, on_confirm=print, on_quit=print, layout=preview.SIDE
        )
    except tk.TclError as exc:  # 没有桌面就跳过,不算失败
        pytest.skip(f"开不了窗口:{exc}")
    try:
        assert not window.make_room("no-such-program-here.exe")
    finally:
        window.close()


def test_接了状态逻辑才列出按H回待命(demo):
    # 2026-09-23 起动作做完不回待命,要回就按 H
    assert "H = HOME" not in preview.key_help(demo, debug=False)
    text = preview.key_help(demo, debug=False, home=True, realign=True)
    assert "H = HOME" in text and text.isascii()


# ---- 图例放哪儿、调试按键写在图例里(2026-10-10) ------------------------


class _FakeLegend:
    """只记账的图例:窗口问它要哪一种、给多大地方。"""

    def __init__(self, *, shows_keys: bool) -> None:
        self.shows_keys = shows_keys
        self.asked: list[tuple[str, int]] = []
        self.scales: list[float] = []

    def rescale(self, px_per_cm: float) -> None:
        self.scales.append(px_per_cm)

    def legend_image(self, height: int):
        import numpy as np

        self.asked.append(("side", height))
        return np.zeros((height, 300, 3), np.uint8)

    def legend_strip(self, width: int):
        import numpy as np

        self.asked.append(("below", width))
        return np.zeros((120, width, 3), np.uint8)


def _open(demo, **kwargs):
    tk = pytest.importorskip("tkinter")
    try:
        return preview.PreviewWindow(demo, on_confirm=print, on_quit=print, **kwargs)
    except tk.TclError as exc:  # 没有桌面就跳过,不算失败
        pytest.skip(f"开不了窗口:{exc}")


def test_图例写了调试按键底下就不再列一遍(demo):
    window = _open(demo, on_trigger=lambda *_: None)
    try:
        assert "DEBUG KEYS" in window._key_help
        window.set_legend(_FakeLegend(shows_keys=True))
        assert "DEBUG KEYS" not in window._key_help
        assert "SPACE = STOP (WORKS ANYWHERE)" in window._key_help
    finally:
        window.close()


def test_图例没写调试按键就照旧列在底下(demo):
    window = _open(demo, on_trigger=lambda *_: None)
    try:
        window.set_legend(_FakeLegend(shows_keys=False))
        assert "DEBUG KEYS" in window._key_help
    finally:
        window.close()


def test_整屏时图例在右侧整列(demo):
    window = _open(demo)
    try:
        legend = _FakeLegend(shows_keys=False)
        window.set_legend(legend)
        assert {kind for kind, _ in legend.asked} == {"side"}
        assert window._side_width == 300
        # 画之前先按窗口的字高缩放过
        assert legend.scales and legend.scales[-1] == pytest.approx(window.px_per_cm)
    finally:
        window.close()


def test_并排时图例在画面下面占满宽(demo):
    # 半屏宽的窗口里图例放右边要占掉一半,画面只剩不到 400 px 宽
    window = _open(demo, layout=preview.SIDE)
    try:
        legend = _FakeLegend(shows_keys=False)
        window.set_legend(legend)
        kind, width = legend.asked[-1]
        assert kind == "below" and width == window._width
        assert window._side_width == 0  # 左边那几行不用给右侧那一列让地方
        window.update(View(State.IDLE, "IDLE"))
        panel_width, _ = window.panel_size
        assert panel_width >= window._width - 2
    finally:
        window.close()


# ---- 跟着窗口大小排版、F11 全屏(2026-10-10) -----------------------------


def test_按键说明里有F11全屏(demo):
    assert "F11 = FULL SCREEN" in preview.key_help(demo, debug=False)


def _settle(window) -> None:
    """等窗口大小变完、重新排版的那一下跑完。"""
    import time

    for _ in range(10):
        window._root.update()
        time.sleep(preview.RELAYOUT_MS / 1000 / 4)
    window._relayout()


def test_并排时按F11全屏图例挪到右侧字回到1厘米(demo):
    window = _open(demo, layout=preview.SIDE)
    try:
        legend = _FakeLegend(shows_keys=False)
        window.set_legend(legend)
        half = window.px_per_cm
        assert legend.asked[-1][0] == "below"
        window._toggle_fullscreen()
        _settle(window)
        assert window._root.attributes("-fullscreen")
        assert legend.asked[-1][0] == "side", "全屏是横着的:图例该在右侧整列"
        per_cm = window._root.winfo_fpixels("1c")
        assert window.px_per_cm == pytest.approx(per_cm)
        assert window.px_per_cm > half
        assert _font_px(window._warning) >= per_cm * preview.MIN_CM
        # 再按一次回到左半屏,图例回到画面下面
        window._toggle_fullscreen()
        _settle(window)
        assert not window._root.attributes("-fullscreen")
        assert legend.asked[-1][0] == "below"
        assert window.px_per_cm == pytest.approx(half, rel=0.05)
    finally:
        window.close()


def test_红字和按键说明换着显示时画面不跳(demo):
    window = _open(demo)
    try:
        window.update(View(State.IDLE, "IDLE"))
        window._show_warning(False)
        window._root.update()
        keys = window.panel_size
        window._show_warning(True)
        window._root.update()
        assert window.panel_size == keys
    finally:
        window.close()


def test_顶栏放得下就左右摆放不下就挪到下面(demo):
    window = _open(demo)
    try:
        window.update(View(State.RUNNING, "RUNNING", action="POINT LEFT", step="STEP 1 / 1"))
        assert window._header_state[0], "放得下:大字在左、那两行在右边"
        window.update(
            View(State.HALTED, "HALTED", reason="ARM ERROR", detail="VERY LONG ERROR " * 12)
        )
        assert not window._header_state[0], "放不下:那两行挪到大字下面"
        # 报错原文折了行,但哪一段都没切掉:折行宽度不超过一行能放的宽度
        assert int(window._detail.cget("wraplength")) <= window._width
    finally:
        window.close()

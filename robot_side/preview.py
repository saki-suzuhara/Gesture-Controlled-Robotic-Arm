"""预览窗口的骨架:全屏、置顶,大字显示状态、倒计时、执行进度和按键说明。

中间那块放摄像头画面和识别结果,由视觉一侧画好了递进来(第 04 票填上的留白);
机器人一侧自己跑时不给画面,那里还是留白。

几条定下来的规矩:

- **文字一律英文**(`CONTEXT.md`「预览窗口」),和以后视觉部分的界面一致;
- **字有个下限**:站在 1–1.5 m 外看得清。原来至少约 1 cm(`docs/grill-decisions.md`
  第一节第 3 组),2026-10-10 用户嫌字大、要画面大些,改成 0.7 cm(`MIN_CM`)。字号按
  屏幕的每厘米像素数算,换台电脑也还是这么高;
- **不在前台时用红字提示**,并注明空格仍然能叫停——窗口收不到回车、R、q / Esc,
  但空格走的是按键线程,不受影响(ADR-0012);
- **窗口卡住不影响叫停**:这里只画画面,不碰机械臂。拖着标题栏不放时,这边的
  `update()` 半天回不来,而机械臂照样在 0.5 秒内停住。

**长相**(2026-10-10 用户要界面更美观、更现代,简约大气,全屏时要有适应全屏的界面):
深色。顶栏横贯整个窗口,状态大字在左,「动作 · 执行进度」「为什么停 · 报错原文」靠右,
底下一条色带跟着状态变色;下面左边是画面、最底下一行按键说明(不在前台时换成红字),
手势图例在横着的窗口里放右侧整列、竖着的窗口里放画面下面。窗口大小一变(全屏、最大化、
F11)就重新排版。字一律用 Bahnschrift,中间那块和图例(`vision_side.overlay`)用同一款字、
同一套颜色。

**并排(`SIDE`)是例外**:仿真时要看得见 RoboDK 里的机械臂,全屏把它整个挡住了
(2026-09-23 用户实跑)。这时预览窗口只占屏幕左边一半、仍然置顶,右边那一半留给
RoboDK(`make_room`);所有字号跟着窗口宽度等比缩小,字高下限在这一档不成立。

用 Tk(Python 自带),不加依赖。主循环自己调 `update()`,不用 `mainloop()`——状态
逻辑要在同一个线程里推。
"""

from __future__ import annotations

import ctypes
import math
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path
from typing import Protocol

import numpy as np

from robot_side.config import DemoConfig
from robot_side.state import State, View

#: 一套深色配色。中间那块(`vision_side.overlay`)用的是同一套,上下连成一片。
BACKGROUND = "#0b0d12"
SURFACE = "#151922"
"""顶栏、右侧图例那一列的底色,比底色亮一档。"""
FOREGROUND = "#e8ecf3"
DIM = "#8c94a6"
MUTED = "#5b6375"
WARNING = "#ff5c5c"
WARNING_BG = "#3a1a1f"
"""不在前台那一行红字底下垫的暗红。"""

FONT = "Bahnschrift"
"""字体:Windows 10 起自带,窄而规整,同样高的字一行放得下更多;中间那块用的也是它。
没有这款字的机器上 Tk 会自己换一款,照样跑。"""

#: 每个状态用什么颜色。已停止用红,回待命用黄,好在余光里分得开。
COLORS = {
    State.IDLE: "#34d399",
    State.COOLDOWN: DIM,
    State.RUNNING: "#60a5fa",
    State.RETURNING: "#fbbf24",
    State.HALTED: WARNING,
}

SEPARATOR = "·"
"""同一行里两段字中间的圆点(`POINT LEFT · STEP 2 / 6`)。"""

MIN_CM = 0.7
"""最小的一档字高(厘米):为什么停、按键说明、红字这几行。

原来是 1 cm(`docs/grill-decisions.md`:1–1.5 m 外也看得清)。2026-10-10 用户嫌字多、字大,
要字小一些、摄像头画面占比大一些,改成 0.7 cm;中间那块(`vision_side.overlay.MIN_CM`)同一档。
"""

HEADLINE_CM = 1.8
"""状态大字多高。原来 4 cm,2026-09-23 为了给图例和画面腾地方缩到 2.5 cm,2026-10-10
用户要字小一些又缩到 1.8 cm:状态用颜色分得开,余光里照样看得出。"""

MID_CM = 0.9
"""「动作 · 执行进度」那一行多高。"""

NOT_FOREGROUND = "KEYS INACTIVE - CLICK THIS WINDOW (SPACE STILL STOPS)"
"""窗口不在前台时的红字。空格仍然能叫停,所以特意写明(ADR-0012)。"""

RESUME_HINT = "PRESS ENTER TO RESUME"
"""已停止时执行进度那一行换成这一句。

已停止只有这一条路出来:走到笔记本前按回车(用户故事 17)。原来只写在最底下那行
按键说明里,2026-09-23 用户被停住以后对着摄像头又做了半分钟手势,不知道要按回车。
"""

FULL = "full"
"""整屏:全屏、置顶。演示时用这一档(`CONTEXT.md`「预览窗口」)。"""
SIDE = "side"
"""并排:占屏幕左边一半、置顶,右边留给 RoboDK。仿真时用这一档。"""
LAYOUTS = (FULL, SIDE)

SIDE_SHARE = 0.5
"""并排时预览窗口占工作区宽度的几成(连边框)。"""

GAP = "   "
"""按键说明里两条之间的空白。排不下换行时只在这里断,不把一条从中间断开。"""

KEY_SEP = " = "
"""按键说明里一条的写法:`SPACE = STOP (WORKS ANYWHERE)`。画的时候按它拆成键帽和说明。"""

#: 调试按键(只在加了调试开关时有效):方向键对应握拳平移,F / B 是往前推、敲两下,
#: P 比 V,L 竖拇指,O 是 OK,K 是 rock(回待命)。
DEBUG_KEYS: dict[str, tuple[str, str]] = {
    "Left": ("fist_move", "left"),
    "Right": ("fist_move", "right"),
    "Up": ("fist_move", "up"),
    "Down": ("fist_move", "down"),
    "f": ("fist_move", "forward"),
    "b": ("fist_move", "back"),
    "p": ("pose_hold", "peace"),
    "l": ("pose_hold", "like"),
    "o": ("pose_hold", "ok"),
    "k": ("pose_hold", "rock"),
}

DEBUG_KEY_NAMES = {
    ("fist_move", "left"): "LEFT",
    ("fist_move", "right"): "RIGHT",
    ("fist_move", "up"): "UP",
    ("fist_move", "down"): "DOWN",
    ("fist_move", "forward"): "F",
    ("fist_move", "back"): "B",
    ("pose_hold", "peace"): "P",
    ("pose_hold", "like"): "L",
    ("pose_hold", "ok"): "O",
    ("pose_hold", "rock"): "K",
}

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
GA_ROOT = 2
GW_OWNER = 4
SPI_GETWORKAREA = 0x0030
SW_SHOWNOACTIVATE = 4
SWP_NOACTIVATE = 0x0010
HWND_TOP = 0
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

DWMWA_USE_IMMERSIVE_DARK_MODE = 20
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_FRAMECHANGED = 0x0020

_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
_kernel32.OpenProcess.restype = wintypes.HANDLE  # 默认按 int 收,64 位下句柄会被截断


def key_help(
    demo: DemoConfig, *, debug: bool, realign: bool = False, home: bool = False
) -> str:
    """窗口下方的按键说明。一律英文(`CONTEXT.md`「预览窗口」)。

    空格那一条特意写明「到哪儿都管用」:它由按键线程全局地查,窗口不在前台也叫得停
    (ADR-0012)。调试按键只在加了调试开关时列出来,而且**按对应关系从演示配置里
    查**——改了哪个手势对哪个动作,这里跟着变,不用改代码。

    `realign` 是接了摄像头那一档:空闲时按 R 重新对准(视觉一侧第 04 票)。
    `home` 是按 H 回待命:动作做完不再自己回去(2026-09-23 用户定)。
    F11 在全屏和原来那样之间切(2026-10-10:仿真时也要能看全屏的界面)。
    """
    first = ["SPACE = STOP (WORKS ANYWHERE)", "ENTER = CONFIRM WHEN HALTED"]
    if home:
        first.append("H = HOME")
    if realign:
        first.append("R = REALIGN")
    first.append("F11 = FULL SCREEN")
    first.append("Q / ESC = QUIT")
    lines = [GAP.join(first)]
    if not debug:
        return lines[0]
    pairs = debug_key_pairs(demo)
    if pairs:
        lines.append(GAP.join(["DEBUG KEYS:", *pairs]))
    return "\n".join(lines)


def debug_key_pairs(demo: DemoConfig) -> list[str]:
    """每个调试按键对应哪个动作(`F = MOVE AWAY`),按演示配置查。

    窗口下方的按键说明和启动时那行日志用的是同一份,不各写各的——日志里原来写死了
    「方向键、P、L」,加了 O、K、F、B 以后就没跟上。
    """
    return [
        f"{DEBUG_KEY_NAMES[(t.kind, t.key)]} = {demo.display_for(t.action)}"
        for t in demo.triggers
        if (t.kind, t.key) in DEBUG_KEY_NAMES
    ]


def fit_lines(text: str, width: float, measure: Callable[[str], float]) -> str:
    """把按键说明排进 `width` 像素宽:一条一条往一行里放,放不下才换行。

    只在 `GAP` 处断,**不把一条从中间断开**——Tk 自己的 `wraplength` 会断在任意空格上,
    排出「R =」和「REALIGN」各在一行。一条本身就比 `width` 宽时只好让它单占一行。

    全屏时也用得上:1600 px 宽的屏幕上那一行按键说明要 1890 px,原来两头各被切掉一截。
    """
    rows: list[str] = []
    for line in text.split("\n"):
        row = ""
        for item in line.split(GAP):
            wider = f"{row}{GAP}{item}" if row else item
            if row and measure(wider) > width:
                rows.append(row)
                row = item
            else:
                row = wider
        rows.append(row)
    return "\n".join(rows)


def _is_foreground(window: tk.Misc) -> bool:
    """这个窗口在前台吗。找不到自己的窗口时也算不在前台——宁可多提示一次。"""
    try:
        mine = _user32.GetAncestor(window.winfo_id(), GA_ROOT)
    except Exception:
        return False
    if not mine:
        return False
    return _user32.GetForegroundWindow() == mine


def _work_area() -> tuple[int, int, int, int] | None:
    """屏幕去掉任务栏那一块:`(left, top, width, height)`。问不到就是 `None`。"""
    rect = wintypes.RECT()
    if not _user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
        return None
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def _exe_of(hwnd: int) -> str:
    """这个窗口是哪个程序的(可执行文件名,比如 `RoboDK.exe`)。问不到就是空串。"""
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    handle = _kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, wintypes.DWORD(pid.value)
    )
    if not handle:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buffer))
        if not _kernel32.QueryFullProcessImageNameW(
            wintypes.HANDLE(handle), 0, buffer, ctypes.byref(size)
        ):
            return ""
        return Path(buffer.value).name
    finally:
        _kernel32.CloseHandle(wintypes.HANDLE(handle))


def _main_window_of(exe_name: str) -> int | None:
    """`exe_name` 那个程序的主窗口:看得见、没有主人、带标题的第一个顶层窗口。"""
    found: list[int] = []

    def visit(hwnd: int, _param: int) -> bool:
        if (
            _user32.IsWindowVisible(wintypes.HWND(hwnd))
            and not _user32.GetWindow(wintypes.HWND(hwnd), GW_OWNER)
            and _user32.GetWindowTextLengthW(wintypes.HWND(hwnd)) > 0
            and _exe_of(hwnd).lower() == exe_name.lower()
        ):
            found.append(hwnd)
            return False  # 找到就停
        return True

    _user32.EnumWindows(_WNDENUMPROC(visit), 0)
    return found[0] if found else None


class Legend(Protocol):
    """手势图例,由视觉一侧画(`vision_side.overlay.Painter`)。

    预览窗口只决定放哪儿、给多大地方:横着的窗口(全屏)放右侧整列(给高度),竖着的
    窗口(并排)放画面下面一条(给宽度)。窗口大小一变,先按新的字高 `rescale` 再重画。
    """

    @property
    def shows_keys(self) -> bool:
        """调试按键是不是已经写在图例里了。"""
        ...

    def rescale(self, px_per_cm: float) -> None: ...

    def legend_image(self, height: int) -> np.ndarray: ...

    def legend_strip(self, width: int) -> np.ndarray: ...


def _dark_title_bar(window: tk.Misc) -> None:
    """标题栏也用深色,和窗口里面连成一片。只有并排时有标题栏(整屏时没有)。

    Windows 10 2004 起才认这个开关;更老的系统上什么也不发生,照样跑。
    """
    try:
        hwnd = wintypes.HWND(_user32.GetAncestor(window.winfo_id(), GA_ROOT))
        on = ctypes.c_int(1)
        ctypes.WinDLL("dwmapi").DwmSetWindowAttribute(
            hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(on), ctypes.sizeof(on)
        )
        # 标题栏要重画一次才换颜色
        _user32.SetWindowPos(
            hwnd, None, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        )
    except (OSError, tk.TclError):
        pass


PORTRAIT = 1.15
"""窗口宽不到高的这么多倍,就算竖着的:图例放画面下面。并排时约 784×913,全屏 1600×1000。"""

RELAYOUT_MS = 120
"""窗口大小变了以后等多久再重新排版。拖着边框改大小时一路都在变,等它停下来再排。"""


class PreviewWindow:
    """预览窗口。主循环每轮调一次 `update(view)`。

    回车、q / Esc 和调试按键都在这里收,收到就排进状态逻辑的事件队列(`on_event`)。
    **空格不在这里收**:它由按键线程全局地查(ADR-0012),窗口卡住或不在前台照样叫停。

    **跟着窗口大小排版**(2026-10-10 用户要全屏时有适应全屏的界面):横着的窗口(全屏、
    并排时按 F11 或最大化)图例放右侧整列,字按定下的高度;竖着的窗口(并排)图例放画面下面,
    字跟着窗口宽度缩。大小一变就重新排一遍(`_relayout`)。
    """

    def __init__(
        self,
        demo: DemoConfig,
        *,
        on_confirm: Callable[[str], None],
        on_quit: Callable[[str], None],
        on_trigger: Callable[[str, str], None] | None = None,
        on_realign: Callable[[str], None] | None = None,
        on_home: Callable[[str], None] | None = None,
        title: str = "ROBOT SIDE",
        layout: str = FULL,
    ) -> None:
        if layout not in LAYOUTS:
            raise ValueError(f"预览窗口没有 {layout!r} 这一档;有的是 {LAYOUTS}")
        self._demo = demo
        self._on_trigger = on_trigger
        self._on_realign = on_realign
        self._has_home = on_home is not None
        self._layout = layout
        self._closed = False
        self._photo: tk.PhotoImage | None = None
        """现在挂在中间那块上的图。**得留着这个名字**:Tk 只记着图的名字,
        Python 这边一撒手,图就被回收,中间那块当场变空白。"""
        self._side_photo: tk.PhotoImage | None = None
        self._legend: Legend | None = None
        self._legend_at: tuple[bool, int] | None = None
        """图例上一次按什么画的:(竖着?, 给的宽或高)。没变就不重画。"""
        self._side_width = 0
        """右侧那一列(图例)占多宽。还没放图例、或者图例放在画面下面时是 0。"""
        self._portrait = layout == SIDE
        self._laid_out: tuple[int, int] = (0, 0)
        self._pending: str | None = None
        self._header_state: tuple | None = None
        self._warning_on: bool | None = None

        root = tk.Tk()
        root.title(title)
        root.configure(bg=BACKGROUND)
        root.attributes("-topmost", True)
        root.protocol("WM_DELETE_WINDOW", lambda: on_quit("key:close"))
        self._root = root
        self._screen = root.winfo_screenwidth()
        if layout == SIDE:
            self._width, self._height = self._dock_left()
            _dark_title_bar(root)
        else:
            root.attributes("-fullscreen", True)
            self._width, self._height = self._screen, root.winfo_screenheight()
        self._scale = min(1.0, self._width / self._screen)
        """字号跟着窗口宽度缩的比例。全屏是 1;并排时约 0.7(见 `_relayout`,开窗口时先按这个
        粗算,马上就按窗口的样子重算)。"""

        # 字号按这块屏幕的每厘米像素数算,换台电脑也还是这么高(`_size_fonts`)。用起名的
        # 字体:窗口大小一变,改一处所有字跟着变
        self._big = tkfont.Font(root=root, family=FONT, weight="bold")
        self._mid = tkfont.Font(root=root, family=FONT, weight="bold")
        self._small = tkfont.Font(root=root, family=FONT, weight="bold")
        self._small_light = tkfont.Font(root=root, family=FONT, weight="normal")
        self._size_fonts()

        # 顶栏横贯整个窗口:状态大字在左,「动作 · 执行进度」「为什么停 · 报错原文」在右边
        # 靠右对齐。原来五行各占一行、空着也占地方,把画面挤得很小;大字和那两行放成
        # 左右两边,顶栏只有大字那么高(2026-10-10 用户要界面简约大气)。底下一条色带跟着
        # 状态变色,余光里就分得出现在是什么状态
        self._header = tk.Frame(root, bg=SURFACE)
        self._header.pack(side="top", fill="x")
        self._header.columnconfigure(1, weight=1)
        self._headline = self._label(self._header, self._big, FOREGROUND, bg=SURFACE)
        self._info = tk.Frame(self._header, bg=SURFACE)
        self._what = _Row(self._info, self._mid, self._mid, bg=SURFACE)
        self._why = _Row(self._info, self._small, self._small_light, bg=SURFACE)
        self._action, self._step = self._what.first, self._what.second
        self._reason, self._detail = self._why.first, self._why.second
        self._accent = tk.Frame(root, bg=COLORS[State.IDLE], height=3)
        self._accent.pack(side="top", fill="x")

        # 下面这块:左边一栏是画面和按键说明,图例横着时在右边、竖着时在画面下面
        self._body = tk.Frame(root, bg=BACKGROUND)
        self._body.pack(side="top", expand=True, fill="both")
        self._column = tk.Frame(self._body, bg=BACKGROUND)
        self._column.pack(side="left", expand=True, fill="both")
        # 按键说明在中间那块**之前**排:窗口高度不够时先保住它,让画面那块让(原来画面
        # 先排,上面一行一折行,底下那行按键说明就被挤掉半截)。窗口不在前台时同一块地方
        # 换成红字——这时回车、R、q 本来就收不到;高度不变,画面不跟着跳
        self._footer = tk.Frame(self._column, bg=BACKGROUND)
        self._footer.pack(side="bottom", fill="x")
        self._footer.pack_propagate(False)
        self._keys = tk.Frame(self._footer, bg=BACKGROUND)
        self._warning = self._label(
            self._footer, self._small, WARNING, bg=WARNING_BG, text=NOT_FOREGROUND
        )
        self._warning.configure(anchor="w")
        self._key_labels: list[tk.Label] = []
        self._key_help = key_help(
            demo,
            debug=on_trigger is not None,
            realign=on_realign is not None,
            home=on_home is not None,
        )
        # 中间这块放摄像头画面和识别结果(视觉一侧第 04 票填上的;不接摄像头时就是留白)。
        # 不跟着里面那幅图撑大(`pack_propagate(False)`):它有多大由窗口剩下多少决定,
        # 画面再按它的大小来画(`panel_size`)
        self._middle = tk.Frame(self._column, bg=BACKGROUND)
        self._middle.pack_propagate(False)
        self._middle.pack(expand=True, fill="both")
        self._image = tk.Label(self._middle, bg=BACKGROUND, borderwidth=0)
        self._image.pack(expand=True)
        # 图例:在 `_column` 之后建,竖着时排进 `_column` 里也压得住它(Tk 后建的在上面)
        self._side = tk.Label(self._body, bg=SURFACE, borderwidth=0)

        root.bind("<Configure>", self._on_configure)
        root.bind("<F11>", lambda _e: self._toggle_fullscreen())
        root.bind("<Return>", lambda _e: on_confirm("key:enter"))
        for key in ("q", "Q"):
            root.bind(f"<{key}>", lambda _e: on_quit("key:q"))
        root.bind("<Escape>", lambda _e: on_quit("key:esc"))
        if on_realign is not None:
            for key in ("r", "R"):
                root.bind(f"<{key}>", lambda _e: on_realign("key:r"))
        if on_home is not None:
            for key in ("h", "H"):
                root.bind(f"<{key}>", lambda _e: on_home("key:h"))
        if on_trigger is not None:
            for key, (kind, name) in DEBUG_KEYS.items():
                root.bind(
                    f"<{key}>",
                    lambda _e, k=kind, n=name: self._debug_trigger(k, n),
                )
        self._relayout()
        self._show_warning(False)
        root.focus_force()

    # ---- 主循环用的 ----------------------------------------------------

    @property
    def px_per_cm(self) -> float:
        """这块屏幕每厘米多少像素。画中间那块的人按它算字高,和这里几行字同一个算法。

        窗口比屏幕窄时按宽度等比缩过:并排时窗口只有半屏宽,字还按全屏的高度画的话,
        「RETURNING IN 3」一行就放不下,图例也会把画面挤没。
        """
        return float(self._root.winfo_fpixels("1c")) * self._scale

    def make_room(self, exe_name: str) -> bool:
        """并排时把 `exe_name` 那个程序的主窗口摆到预览窗口右边,占满剩下那块。

        **只挪、不激活**:键盘焦点留在预览窗口上,回车、R、q / Esc 照样收得到。
        最小化、最大化着的先还原。整屏时什么也不做;找不到那个窗口返回 False。
        """
        if self._layout != SIDE or self._closed:
            return False
        area = _work_area()
        hwnd = _main_window_of(exe_name)
        if area is None or hwnd is None:
            return False
        left, top, width, height = area
        mine = wintypes.RECT()
        _user32.GetWindowRect(
            wintypes.HWND(_user32.GetAncestor(self._root.winfo_id(), GA_ROOT)),
            ctypes.byref(mine),
        )
        x = max(left, mine.right)
        if _user32.IsIconic(wintypes.HWND(hwnd)) or _user32.IsZoomed(wintypes.HWND(hwnd)):
            _user32.ShowWindow(wintypes.HWND(hwnd), SW_SHOWNOACTIVATE)
        moved = _user32.SetWindowPos(
            wintypes.HWND(hwnd),
            wintypes.HWND(HWND_TOP),
            x,
            top,
            left + width - x,
            height,
            SWP_NOACTIVATE,
        )
        # 刚由程序启动的 RoboDK 会自己抢到前台。试着要回来;Windows 不让的话,
        # 红字(NOT_FOREGROUND)会提示点一下这个窗口
        try:
            self._root.focus_force()
        except tk.TclError:
            pass
        return bool(moved)

    @property
    def panel_size(self) -> tuple[int, int]:
        """中间那块现在多大。画面缩到放得下,由画它的人按这个尺寸做。

        窗口还没排好版时 `winfo_width` 报 1(第一次 `update()` 之前都是这样)。这时先
        催它排一次版再问——不然第一帧要么白丢,要么按整屏画出来把下面几行字挤出屏幕。
        """
        size = self._size()
        if size is not None:
            return size
        try:
            self._root.update_idletasks()
        except tk.TclError:
            self._closed = True
            return 0, 0
        return self._size() or (0, 0)

    @property
    def legend_height(self) -> int:
        """右侧那一列有多高:顶栏以下整个那么高。排好版之前先催它排一次。"""
        try:
            self._root.update_idletasks()
        except tk.TclError:
            self._closed = True
            return 0
        height = self._body.winfo_height()
        if height <= 1:  # 还没露出来:按窗口高减去顶栏算
            height = self._height - self._header.winfo_reqheight() - 3
        return max(1, height)

    def set_legend(self, legend: Legend) -> None:
        """把手势图例放上去。放哪儿、多大由窗口定(`_place_legend`),以后窗口大小一变跟着重画。

        调试按键已经写在图例里的话(每行最右边的小键帽),底下的按键说明就不再列一遍——
        原来那一串在并排时要占 5 行,把画面挤得很小。
        """
        self._legend = legend
        if legend.shows_keys:
            self._key_help = key_help(
                self._demo,
                debug=False,
                realign=self._on_realign is not None,
                home=self._has_home,
            )
        self._relayout(force=True)

    def _size(self) -> tuple[int, int] | None:
        width = self._middle.winfo_width()
        height = self._middle.winfo_height()
        return (width, height) if width > 1 and height > 1 else None

    def update(self, view: View, image: np.ndarray | None = None) -> None:
        """刷新一次画面。窗口已经关掉时什么也不做。

        `image` 是要放进中间那块的 BGR 画面(视觉一侧画好的);不给就不动中间那块
        ——不接摄像头时(机器人一侧自己跑)它一直是留白。
        """
        if self._closed:
            return
        if image is not None:
            self._show(image)
        color = COLORS.get(view.state, FOREGROUND)
        self._headline.configure(text=view.headline, fg=color)
        self._accent.configure(bg=color)
        if view.state is State.HALTED and not view.step:
            what, what_fg = (view.action, RESUME_HINT), color
        else:
            what, what_fg = (view.action, view.step), DIM
        self._arrange_header(view.headline, what, (view.reason, view.detail))
        self._what.show(*what, second_fg=what_fg)
        self._why.show(view.reason, view.detail, first_fg=color, second_fg=DIM)
        self._show_warning(not _is_foreground(self._root))
        try:
            self._root.update()
        except tk.TclError:
            self._closed = True  # 窗口被关掉了
            return
        # 顶栏变高变矮(比如报错原文折成两行)时,右侧那一列跟着重画
        if self._legend is not None and not self._portrait:
            self._place_legend()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._pending is not None:
                self._root.after_cancel(self._pending)
            self._root.destroy()
        except tk.TclError:
            pass

    def __enter__(self) -> PreviewWindow:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- 排版 ----------------------------------------------------------

    def _size_fonts(self) -> None:
        """按现在的每厘米像素数定字号。Tk 的字号写成负数就是「多少像素高」;一律往上
        取整,免得截掉小数之后比定下的高度矮一丝。

        几档多高见 `HEADLINE_CM`、`MID_CM`、`MIN_CM`(2026-10-10 用户要字小一些,都缩过)。
        """
        cm = self.px_per_cm
        self._big.configure(size=-math.ceil(cm * HEADLINE_CM))
        self._mid.configure(size=-math.ceil(cm * MID_CM))
        self._small.configure(size=-math.ceil(cm * MIN_CM))
        self._small_light.configure(size=-math.ceil(cm * MIN_CM))
        self._pad = max(2, round(cm * 0.5))
        """顶栏、画面、按键说明左右都留这么宽:半厘米,上下几块左边对齐成一条线。"""

    def _on_configure(self, event: tk.Event) -> None:
        """窗口大小变了:等它停下来再重新排版。里面每个小部件的大小变化也会走到这儿,不管。"""
        if event.widget is not self._root or self._closed:
            return
        if (event.width, event.height) == self._laid_out:
            return
        if self._pending is not None:
            self._root.after_cancel(self._pending)
        self._pending = self._root.after(RELAYOUT_MS, self._relayout)

    def _relayout(self, *, force: bool = False) -> None:
        """按窗口现在的大小重新排一遍:字号、图例放哪儿、按键说明怎么折行。"""
        self._pending = None
        if self._closed:
            return
        try:
            width, height = self._root.winfo_width(), self._root.winfo_height()
        except tk.TclError:
            return
        if width <= 1 or height <= 1:  # 还没露出来:按开窗口时摆的大小算
            width, height = self._width, self._height
        if (width, height) == self._laid_out and not force:
            return
        self._laid_out = (width, height)
        self._width, self._height = width, height
        self._portrait = width < height * PORTRAIT
        share = width / self._screen
        # 竖着的窗口(并排)里画面被宽度卡住,字再小也换不来更大的画面,只多出上下两块空白;
        # 所以字只按窗口宽度等比缩原来的 1 cm,不跟着 2026-10-10 那次一起缩到 0.7 cm
        self._scale = min(1.0, share / MIN_CM if self._portrait else share)
        self._size_fonts()
        self._accent.configure(height=max(3, round(self.px_per_cm * 0.12)))
        self._header_state = None  # 留白跟着字号变了,顶栏重新摆
        self._warning.configure(
            padx=self._pad, pady=round(self.px_per_cm * 0.2), wraplength=self._wrap_px
        )
        self._legend_at = None
        if self._legend is not None:
            self._place_legend()
        self._layout_keys()

    def _place_legend(self) -> None:
        """图例:横着的窗口放右侧整列(顶栏以下那么高),竖着的放画面下面(整个那么宽)。

        整屏时右侧整列放手势图例,从上到下有整个窗口那么高。图例原来画在摄像头画面那块
        里,只有中间一条那么高,2026-09-23 用户的屏幕上六行只露出五行(用户选了这个改法)。
        并排时窗口只有半屏宽,图例放右边要占掉一半、画面只剩不到 400 px 宽,所以放到
        画面下面(2026-10-10 用户看了截图要改)。
        """
        legend = self._legend
        assert legend is not None
        at = (self._portrait, self._width if self._portrait else self.legend_height)
        if at == self._legend_at:
            return
        self._legend_at = at
        legend.rescale(self.px_per_cm)
        portrait, size = at
        image = legend.legend_strip(size) if portrait else legend.legend_image(size)
        photo = self._photo_of(image)
        if photo is None:
            return
        self._side_photo = photo
        self._side.configure(image=photo)
        self._side.pack_forget()
        if portrait:
            # 排在按键说明上面、画面下面:按键说明照旧贴着最底下
            self._side.pack(
                in_=self._column, side="bottom", fill="x", before=self._middle
            )
            self._side_width = 0
        else:
            self._side.pack(side="right", fill="y", before=self._column)
            self._side_width = int(image.shape[1])
        self._warning.configure(wraplength=self._wrap_px)

    def _arrange_header(
        self, headline: str, what: tuple[str, str], why: tuple[str, str]
    ) -> None:
        """顶栏怎么摆:放得下就大字在左、那两行在右边靠右;放不下(窗口窄、报错原文长)
        就那两行挪到大字下面、靠左。哪一行空着就不占地方。只在要变的时候重新摆。"""
        room = self._width - 2 * self._pad
        gap = 2 * self._pad
        need = max(self._what.natural(*what), self._why.natural(*why))
        # 和 `_Row.show` 里一样留 16 px 余量,免得摆在右边了那一行自己又折成两行
        beside = need == 0 or self._big.measure(headline) + gap + need + 16 <= room
        rows = tuple(row for row, texts in ((self._what, what), (self._why, why)) if any(texts))
        state = (beside, rows, self._pad)
        width = room - self._big.measure(headline) - gap if beside else room
        self._what.width = self._why.width = max(1, width)
        if state == self._header_state:
            return
        self._header_state = state
        top = round(self.px_per_cm * 0.25)
        bottom = round(self.px_per_cm * 0.35)
        if beside:
            self._headline.grid(
                row=0, column=0, sticky="w", padx=(self._pad, 0), pady=(top, bottom)
            )
            self._info.grid(
                row=0, column=1, columnspan=1, sticky="e",
                padx=(gap, self._pad), pady=(top, bottom),
            )
        else:
            self._headline.grid(
                row=0, column=0, columnspan=2, sticky="w", padx=self._pad, pady=(top, 0)
            )
            self._info.grid(
                row=1, column=0, columnspan=2, sticky="w", padx=self._pad, pady=(0, bottom)
            )
        for row in (self._what, self._why):
            row.frame.pack_forget()
        for row in rows:
            row.frame.pack(side="top", anchor="e" if beside else "w")
            row.anchor = "e" if beside else "w"

    def _show_warning(self, on: bool) -> None:
        """底下那一块:在前台时是按键说明,不在前台时换成红字。只在要变的时候换。"""
        if on == self._warning_on:
            return
        self._warning_on = on
        if on:
            self._keys.pack_forget()
            self._warning.pack(expand=True, fill="both")
        else:
            self._warning.pack_forget()
            self._keys.pack(side="top", fill="x", padx=self._pad, pady=self._key_pady)

    @property
    def _key_pady(self) -> tuple[int, int]:
        return round(self.px_per_cm * 0.2), round(self.px_per_cm * 0.35)

    # ---- 内部 ----------------------------------------------------------

    @property
    def _wrap_px(self) -> int:
        """画面那一栏里一行字最宽多少像素,再宽就折行。两边各留 `_pad`。"""
        return max(1, self._width - self._side_width - 2 * self._pad)

    def _label(
        self,
        parent: tk.Misc,
        font: tkfont.Font,
        color: str,
        *,
        bg: str = BACKGROUND,
        text: str = "",
    ) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            font=font,
            fg=color,
            bg=bg,
            borderwidth=0,
            padx=0,
            pady=0,
            justify="left",
        )

    def _labels(self) -> list[tk.Label]:
        """窗口里所有写字的地方。测试按它核对每一处的字号都不低于下限(`MIN_CM`)。"""
        return [
            self._headline,
            *self._what.labels,
            *self._why.labels,
            self._warning,
            *self._key_labels,
        ]

    def _layout_keys(self) -> None:
        """按键说明:键名白色粗体、说明灰色,一条一条往后排,`SPACE STOP (WORKS ANYWHERE)   ENTER ...`。

        不画键帽底色:字和留白就分得开,干净些(2026-10-10 用户要简约)。一行放不下就
        折行,**只在两条之间折**(`fit_lines`),量宽度时键名和说明分开量。窗口大小变了、
        图例换了地方,要重排一次(`_relayout`)。这一块的高度按按键说明和红字里高的那个
        定死,两样换着显示时画面不跟着跳。
        """
        for child in self._keys.winfo_children():
            child.destroy()
        self._key_labels = []
        space = max(1, round(self.px_per_cm * 0.25))
        gap = max(2, round(self.px_per_cm * 0.8))

        def measure(row: str) -> float:
            return sum(self._item_width(item, space) + gap for item in row.split(GAP)) - gap

        rows = fit_lines(self._key_help, self._wrap_px, measure).split("\n")
        for row in rows:
            line = tk.Frame(self._keys, bg=BACKGROUND)
            line.pack(side="top", anchor="w")
            for index, item in enumerate(row.split(GAP)):
                key, _, desc = item.partition(KEY_SEP)
                # 「DEBUG KEYS:」这样的小标题没有说明,整条用淡一档的颜色
                color = FOREGROUND if desc else MUTED
                name = self._label(line, self._small, color, text=key)
                name.pack(side="left", padx=(0 if index == 0 else gap, 0))
                self._key_labels.append(name)
                if desc:
                    text = self._label(line, self._small_light, DIM, text=desc)
                    text.pack(side="left", padx=(space, 0))
                    self._key_labels.append(text)
        if self._warning_on is False:
            self._keys.pack_configure(padx=self._pad, pady=self._key_pady)
        self._root.update_idletasks()
        keys = self._keys.winfo_reqheight() + sum(self._key_pady)
        self._footer.configure(height=max(keys, self._warning.winfo_reqheight()))

    def _item_width(self, item: str, space: int) -> float:
        """按键说明里一条画出来多宽:键名、空一点、说明。"""
        key, _, desc = item.partition(KEY_SEP)
        if not desc:
            return self._small.measure(key)
        return self._small.measure(key) + space + self._small_light.measure(desc)

    def _toggle_fullscreen(self) -> None:
        """F11:全屏和原来那样来回切。并排时切回来又摆回左半屏(右边是 RoboDK)。

        切完窗口大小变了,`<Configure>` 会带着重新排版:全屏时图例挪到右侧整列。
        """
        root = self._root
        full = not bool(root.attributes("-fullscreen"))
        root.attributes("-fullscreen", full)
        if not full:
            if self._layout == SIDE:
                self._dock_left()
            else:
                root.state("zoomed")
        try:
            root.focus_force()
        except tk.TclError:
            pass

    def _dock_left(self) -> tuple[int, int]:
        """并排:把窗口摆在工作区左边、占 `SIDE_SHARE` 那么宽、上下占满。返回里面多宽、多高。

        Tk 的 `geometry` 管的是里面那块,标题栏和边框另算;这两样多厚要窗口真露出来
        以后才量得到,所以先摆一次、量完再摆一次。问不到工作区就按整个屏幕算。
        """
        root = self._root
        left, top, width, height = _work_area() or (
            0,
            0,
            root.winfo_screenwidth(),
            root.winfo_screenheight(),
        )
        outer = round(width * SIDE_SHARE)
        root.geometry(f"{outer}x{height}+{left}+{top}")
        root.update()
        side = max(0, root.winfo_rootx() - root.winfo_x())  # 左右和底下的边框
        title = max(0, root.winfo_rooty() - root.winfo_y())  # 标题栏连上边框
        inner = max(1, outer - 2 * side)
        tall = max(1, height - title - side)
        root.geometry(f"{inner}x{tall}+{left}+{top}")
        return inner, tall

    def _show(self, image: np.ndarray) -> None:
        """把一幅 BGR 画面放进中间那块。**照原样放**,缩放由画它的人按 `panel_size` 做。

        走 Tk 自带的 PPM:拼一个 `P6 宽 高 255 ` 的头,后面直接跟 RGB 的原始字节。
        这样不用装 Pillow(装包前先问用户,spec「分支与提交」),1280×720 一帧约
        10 ms,而画面本来就只有 30 fps,够用。

        换通道用 numpy 的反向切片,不用 `cv2`:这一份是机器人一侧的,不该为了显示
        一幅图就把整个 OpenCV 拖进来。
        """
        photo = self._photo_of(image)
        if photo is None:
            return
        self._photo = photo
        self._image.configure(image=photo)

    def _photo_of(self, image: np.ndarray) -> tk.PhotoImage | None:
        """BGR 画面 → Tk 的图(PPM,见 `_show`)。窗口已经关掉时返回 None。"""
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        rows, cols = rgb.shape[:2]
        try:
            return tk.PhotoImage(
                master=self._root,
                data=b"P6 %d %d 255 " % (cols, rows) + rgb.tobytes(),
            )
        except tk.TclError:
            self._closed = True  # 窗口被关掉了
            return None

    def _debug_trigger(self, kind: str, key: str) -> None:
        """调试按键按的是**触发手势**,不是预设动作:对应关系还是查演示配置。

        这样按键走的就是以后手势走的同一条路,对应关系改了也不用改代码。
        """
        if self._on_trigger is None:
            return
        action = self._demo.action_for_trigger(kind, key)
        if action is None:
            return
        self._on_trigger(action, f"key:{key}")


class _Row:
    """顶栏里并排的两段字,中间一个圆点:`POINT LEFT · STEP 2 / 6`。

    并排放不下就上下排,不切掉——报错原文这类长句可能很长。哪一段空着就不画圆点。
    两段的字都不低于下限(用的是窗口那几个起了名的字体,`PreviewWindow._size_fonts`)。
    一行最宽多少(`width`)、靠哪边(`anchor`)由窗口摆顶栏时定(`_arrange_header`)。

    **不记着预览窗口**:窗口记着它、它又记着窗口,就成了一个环,窗口关掉以后要等垃圾
    回收才放掉;回收要是碰巧在别的线程里做,Tk 的字体、窗口在那个线程里析构,Tcl 当场
    崩掉(2026-10-10 测试里撞上过)。
    """

    def __init__(
        self, parent: tk.Misc, font: tkfont.Font, second_font: tkfont.Font, *, bg: str
    ) -> None:
        self.frame = tk.Frame(parent, bg=bg)
        self._fonts = (font, font, second_font)
        self.first, self.dot, self.second = (
            tk.Label(
                self.frame, text="", font=f, fg=color, bg=bg, borderwidth=0, padx=0, pady=0
            )
            for f, color in zip(self._fonts, (FOREGROUND, MUTED, DIM), strict=True)
        )
        self.labels = (self.first, self.dot, self.second)
        self.width = 1
        self.anchor = "w"
        self._packed: tuple[bool, str] | None = None

    def natural(self, first: str, second: str) -> int:
        """两段并排写成一行要多宽。都空着是 0。"""
        dot = _dot(first, second)
        font_first, font_dot, font_second = self._fonts
        return font_first.measure(first) + font_dot.measure(dot) + font_second.measure(second)

    def show(
        self,
        first: str,
        second: str,
        *,
        first_fg: str | None = None,
        second_fg: str | None = None,
    ) -> None:
        dot = _dot(first, second)
        # 留一点余量:Tk 量出来的宽和真画出来的差几个像素,刚好卡在边上时最后一个字
        # 会被折到下一行,孤零零一个
        stacked = bool(dot) and self.natural(first, second) > self.width - 16
        packed = (stacked, self.anchor)
        if packed != self._packed:
            for label in self.labels:
                label.pack_forget()
            if stacked:
                self.first.pack(side="top", anchor=self.anchor)
                self.second.pack(side="top", anchor=self.anchor)
            else:
                for label in self.labels:
                    label.pack(side="left", anchor="s")
            self._packed = packed
        justify = "right" if self.anchor == "e" else "left"
        self.first.configure(text=first, wraplength=self.width, justify=justify)
        if first_fg is not None:
            self.first.configure(fg=first_fg)
        self.dot.configure(text="" if stacked else dot)
        # 并排时已经量过放得下,不再让它自己折(0 是不折)
        self.second.configure(
            text=second, wraplength=self.width if stacked or not dot else 0, justify=justify
        )
        if second_fg is not None:
            self.second.configure(fg=second_fg)


def _dot(first: str, second: str) -> str:
    return f"  {SEPARATOR}  " if first and second else ""

"""画面叠加:把摄像头画面和识别结果画成预览窗口中间那块(spec「预览窗口(补画面)」)。

画出来的是**和中间那块一样大**的一幅图,不是摄像头原来那么大。这么定有两个好处:

- **字高好算。**画布上一个像素就是屏幕上一个像素,字高的下限(`MIN_CM`)直接按屏幕的
  每厘米像素数算(和骨架里那几行字同一个算法),不用再跟着缩放倒推。
- **手势图例有地方放。**录制工具里图例画在这块的右边(`legend=True`),画面缩进
  左边那一条里;正式程序把图例另放在预览窗口右侧整列(`legend_image`)。

画面本身已经水平翻转过(取帧那一层翻的,`vision_side.frames`),所以这里照原样画出来
就是镜像显示——人抬右手,画面右边的手跟着动。

文字一律英文(`CONTEXT.md`「预览窗口」)。

**长相**(2026-10-10 用户要界面更美观、更现代):和骨架(`robot_side.preview`)用同一套
深色配色、同一款字(Bahnschrift),画面是圆角卡片,手框画成四个角标,姿势标签、READY
这些写在彩色小标签上,提示和横幅是半透明的卡片。颜色的意思没变:绿是认出来了、灰是
没确认、红是叫停、黄是「来早了 / 起点区」。
"""

from __future__ import annotations

import functools
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from robot_side.config import HOME_ACTION, STOP_GESTURE_LABELS, DemoConfig
from vision_side.align import TOO_CLOSE_HINT
from vision_side.loop import VISION_LOST, AlignView, Ignored, VisionUpdate
from vision_side.prompts import GO, PREP, PromptView
from vision_side.recognize import FrameResult, Region
from vision_side.triggers import DIRECTIONS, FIST_MOVE, POSE_HOLD

Color = tuple[int, int, int]
"""一个颜色,BGR 三个分量(`cv2` 的次序)。"""

FONT = cv2.FONT_HERSHEY_SIMPLEX
"""`cv2` 老接口的字。两个查看脚本还在用(它们从这儿 import);预览窗口里的字一律走 `FACE`。"""

FONT_FILE = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "bahnschrift.ttf"
"""预览窗口里的字:Bahnschrift,Windows 10 起自带,和骨架那几行字同一款。

窄、笔画规整,同样高的字一行放得下更多;又是可变字重的,粗细一个文件就够。
OpenCV 5 自己就画得了 TrueType 字,不用加依赖。没有这个文件时退回 OpenCV 自带的 `sans`。
"""

REGULAR, SEMIBOLD, BOLD = 400, 600, 700
"""字重。可变字体才分得出来;退回 `sans` 时一律一样粗。"""

#: 一套颜色(BGR)。和骨架那几行字(`robot_side.preview`)是同一套,中间那块和上下连成一片。
BACKGROUND = (18, 13, 11)
"""窗口底色 `#0b0d12`。"""
SURFACE = (34, 25, 21)
"""卡片、图例那一列的底色 `#151922`,比底色亮一档。"""
LINE = (64, 49, 42)
"""分隔线 `#2a3140`。"""
WHITE = (243, 236, 232)
GREY = (166, 148, 140)
MUTED = (117, 99, 91)
"""比灰还淡一档 `#5b6375`:图例里的小箭头这类只起连接作用的东西。"""
GREEN = (153, 211, 52)
YELLOW = (36, 191, 251)
RED = (92, 92, 255)
BLUE = (250, 165, 96)
"""图例里预设动作那一栏 `#60a5fa`。和骨架里执行中(RUNNING)同一个颜色。"""
INK = BACKGROUND
"""彩色小标签上的字:深色,压在亮底上最清楚。"""

CARD_ALPHA = 0.85
"""压在画面上的卡片有多不透明。画面背景是什么颜色不由我们定——浅色墙上灰字看不见,
所以字底下一律垫一层深色;留一点透明,看得出底下还是画面。"""

MIN_CM = 0.7
"""基准字高(厘米),也是最小的一档:手框上的姿势标签、图例、按键说明都是这么高
(和 `robot_side.preview.MIN_CM` 同一档)。

原来是 1 cm(`docs/grill-decisions.md`:1–1.5 m 外也看得清)。2026-10-10 用户看了改版后的
全屏界面,嫌字多、字大,要字小一些、摄像头画面占比大一些,改成 0.7 cm。图例跟着字等比
变窄,全屏时画面从约 830 px 宽放大到约 1050 px。
"""

CAP_SHARE = 0.74
"""「字高」里大写字母占几成。

字高说的是字号——一行字连上伸下伸有多高,和骨架里 Tk 的字号(`-像素数`)同一个意思;
大写字母只占其中约 3/4。这个数是照改版前量的:原来 1 cm(37.8 px)的字,大写字母
28 px 高。照这个数画,改版前后字一样大;要是让大写字母本身有 1 cm,整体放大三成,
图例宽到 974 px,把画面挤小了。
"""

LEGEND_DIR = Path("data") / "legend"
"""手势图例的示范小图放这儿。`data/` 被 git 忽略:**只存本地,不提交、不分发**
(spec「分支与提交」)。没有图就只显示文字,照样跑得起来。"""

LEGEND_SOURCE = "POSE IMAGES: HAGRID"
"""图例小图的出处,常驻屏幕一角(票里要求标出出处)。

短到能和别的字一样高(不低于 `MIN_CM`)。完整那一串在 `vision_side/scripts/legend_images.py`
和 PR 正文里:`github.com/hukenovs/hagrid`,CC BY-SA 4.0。
"""

#: 姿势保持那几个手势**怎么做**,附在 `CONTEXT.md` 定下的英文名后面。
#: 光写英文名不够:2026-09-21 站着走一遍时,`HOLD PEACE` 那一件卡了 10.6 秒,因为屏幕上
#: 没说 peace 是哪个手势;光写怎么做又和已定的英文名对不上,所以两样都写。
POSE_HOWTO = {
    "peace": "V SIGN",
    "like": "THUMBS UP",
    "ok": "OK SIGN",
    "rock": "INDEX+PINKY",
}

#: 握拳平移里前后那两个**怎么做**,和姿势保持一样括注在英文名后面:`FIST MOVE BACK`
#: 光看名字猜不出是敲两下。认出来的那一下也写这几个字(画面里画不出前后的箭头)。
FIST_HOWTO = {
    "forward": "PUSH",
    "back": "KNOCK x2",
}

STOP_LINES = ("OPEN PALM = STOP", "SPACE = STOP")
"""图例最后两行:停止手势和空格。两条都是叫停,挨着写;分两行是为了图例别太宽。"""
STOP_ROW = "    ".join(STOP_LINES)
"""同一句写成一行(日志、测试里用)。"""


# ---- 字 ----------------------------------------------------------------


def _load_face() -> cv2.FontFace:
    face = cv2.FontFace(str(FONT_FILE))
    if not face.getName():  # 没有这个文件(不是 Windows,或者太老的 Windows)
        face = cv2.FontFace("sans")
    return face


FACE = _load_face()


def _cap_ratio() -> float:
    """字号和大写字母高度之比。画一个 H 量出来:每款字不一样,不写死。"""
    probe = np.zeros((200, 200), np.uint8)
    cv2.putText(probe, "H", (20, 160), 255, FACE, 100, SEMIBOLD)
    rows = np.flatnonzero(probe.max(axis=1) > 127)
    return (rows[-1] - rows[0] + 1) / 100.0 if rows.size else 0.7


CAP_RATIO = _cap_ratio()


def font_size(px: float) -> int:
    """字高 `px`(见 `CAP_SHARE`)折成 `cv2` 的字号。

    `cv2` 的字号既不是字高也不是字母的高度,所以先量一次比例(`CAP_RATIO`)再折算。
    量大写字母:屏幕上的字一律大写(手势名、动作名、状态),它的高度就是看上去有多大。
    字号只收整数,往上取整,免得比定下的高度矮一丝。
    """
    return max(1, math.ceil(max(1.0, px) * CAP_SHARE / CAP_RATIO))


def cap_px(px: float) -> float:
    """字高 `px` 的字,大写字母有多高。排版(居中、留边)按它算。"""
    return px * CAP_SHARE


@functools.lru_cache(maxsize=512)
def text_width(text: str, px: float, weight: int = SEMIBOLD) -> int:
    """这一行字画出来有多宽(像素)。"""
    _x, _y, width, _h = cv2.getTextSize((0, 0), text, (0, 0), FACE, font_size(px), weight)
    return int(width)


def _put(
    canvas: np.ndarray,
    text: str,
    x: float,
    baseline: float,
    color: Color,
    px: float,
    weight: int = SEMIBOLD,
) -> None:
    """在 `(x, baseline)` 写一行字:`x` 是左边,`baseline` 是基线。"""
    cv2.putText(canvas, text, (round(x), round(baseline)), color, FACE, font_size(px), weight)


def chip_height(px: float) -> int:
    """一个小标签多高:大写字母上下各留出大半个字母。"""
    return round(cap_px(px) * 2.1)


@dataclass(frozen=True)
class Place:
    """摄像头画面在画布上的位置:缩了多少、左上角落在哪儿。"""

    scale: float
    x: int
    y: int

    def at(self, point: tuple[float, float]) -> tuple[int, int]:
        """画面坐标 → 画布坐标。"""
        px, py = point
        return round(px * self.scale + self.x), round(py * self.scale + self.y)

    def box(self, region: Region) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.at((region.x, region.y)), self.at((region.right, region.bottom))


def load_legend_images(
    poses: list[str], *, folder: Path = LEGEND_DIR, log: logging.Logger | None = None
) -> dict[str, np.ndarray]:
    """读手势图例要用的示范小图。没有就少几张,不报错。

    图只存本地、不提交(spec「分支与提交」),所以别人拉下代码是没有的;那时图例
    只显示文字,程序照跑。裁图见 `vision_side/scripts/legend_images.py`。
    """
    log = log if log is not None else logging.getLogger("vision_side.overlay")
    found: dict[str, np.ndarray] = {}
    for pose in poses:
        path = folder / f"{pose}.png"
        if not path.exists():
            continue
        # 路径里有中文时 `cv2.imread` 读不出来,所以走 `fromfile`(和 `save_snap` 一样)
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            found[pose] = image
    if not found:
        log.info("图例 %s 下没有示范小图,图例只显示文字", folder)
    else:
        log.info("图例 读到 %d 张示范小图(只存本地,不提交)", len(found))
    return found


class Painter:
    """把一帧画成中间那块。一个线程用(主线程)。

    `px_per_cm` 是这块屏幕每厘米多少像素,从预览窗口问来(`winfo_fpixels("1c")`);
    字高都按它算,换台电脑也还是那么高。`keys` 是调试按键(触发手势 → 键名,
    `robot_side.preview.DEBUG_KEY_NAMES`),给了就画进图例,底下的按键说明不再列一遍。
    """

    def __init__(
        self,
        demo: DemoConfig,
        *,
        px_per_cm: float,
        images: dict[str, np.ndarray] | None = None,
        keys: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self._demo = demo
        self._px_cm = max(8.0, px_per_cm)
        self._images = images if images is not None else {}
        self._rows = _legend_rows(demo)
        self._lines = _legend_lines(demo, keys)
        if not self._images:
            # 出处是给示范小图标的;一张图也没有时那一行只是多出来的字(2026-10-10 用户嫌字多)
            self._lines = [line for line in self._lines if line.style != SOURCE]
        self._view = (0, 0, 1, 1)
        """这一帧画面在画布上占的那块 `(x1, y1, x2, y2)`。居中的提示、横幅都按它摆,
        不按整块画布——右边可能是图例,上下可能是空出来的边。"""
        self._legend_cache: tuple[int, np.ndarray] | None = None
        self._grid: _Grid | None = None

    # ---- 外面看得到的 --------------------------------------------------

    def rescale(self, px_per_cm: float) -> None:
        """窗口变大变小(并排 ↔ 全屏)时,字高、图例跟着变。量过的、画过的图例都作废。

        预览窗口重新排版时调(`PreviewWindow.set_legend` 放进去的就是这个画图的人)。
        """
        px = max(8.0, px_per_cm)
        if px == self._px_cm:
            return
        self._px_cm = px
        self._grid = None
        self._legend_cache = None

    def text_px(self, factor: float = 1.0) -> float:
        """这一档字画多高(像素):基准字高(`MIN_CM`)的 `factor` 倍,**一律不低于基准**。

        「文字一律英文,字高有个下限」是票里写死的一条(`CONTEXT.md`「预览窗口」;下限
        原来 1 cm,2026-10-10 用户要字小一些,改成 `MIN_CM`)。写成一个口子、每处都从这儿
        要尺寸,就不会有哪一处偷偷画小了——手框上的姿势标签和图例底下的出处原来都比下限
        矮,是复查时才发现的。
        """
        return self._px_cm * MIN_CM * max(1.0, factor)

    def paint(
        self,
        size: tuple[int, int],
        update: VisionUpdate | None,
        *,
        lost: bool = False,
        prompt: PromptView | None = None,
        legend: bool = True,
    ) -> np.ndarray:
        """画一帧。`update` 是 `None`(还没有画面)时只画图例和一行提示。

        `prompt` 是录制工具那一档的屏幕提示(取景检查和录制都用它);正式程序不给。
        `legend=False` 时不画图例、也不给它留地方:图例另画在预览窗口右侧整列
        (`legend_image`),正式程序这么用。
        """
        width, height = size
        canvas = np.full((max(1, height), max(1, width), 3), BACKGROUND, np.uint8)
        legend_w = self._legend_width() if legend else 0
        self._view = (0, 0, max(1, width - legend_w), max(1, height))
        if update is None:
            mid = (self._view[1] + self._view[3]) // 2
            top = mid - chip_height(self.text_px(1.2)) // 2
            self._centered(canvas, "WAITING FOR CAMERA", top, GREY)
            if legend:
                self._legend(canvas)
            return canvas

        place = self._place(update.image.shape[:2], width - legend_w, height)
        self._blit(canvas, update.image, place)
        if update.alignment is not None:
            self._regions(canvas, place, update)
        self._hands(canvas, place, update.result)
        if update.align is not None:
            self._align(canvas, update.align)
        if update.judge is not None:
            self._judge(canvas, place, update)
        self._corner(canvas, update)
        if update.ignored is not None:
            self._ignored(canvas, update.ignored)
        if prompt is not None:
            self._prompt(canvas, place, prompt)
        if lost:
            self._banner(canvas, VISION_LOST, RED)
        if legend:
            self._legend(canvas)
        return canvas

    def legend_width(self) -> int:
        """图例要占多宽(像素)。预览窗口按它给右侧那一列留地方。"""
        return self._legend_width()

    def legend_image(self, height: int) -> np.ndarray:
        """预览窗口右侧整列的图例:宽 `legend_width()`、高 `height`。

        图例放在摄像头画面那块里时,只有中间那一条那么高——2026-09-23 用户的屏幕上六行
        只露出五行,NOD 和 STOP 两行被切掉了。整列从上到下有整个窗口那么高;还放不下就
        先收紧行距,字高不变。
        """
        height = max(1, height)
        if self._legend_cache is not None and self._legend_cache[0] == height:
            return self._legend_cache[1]
        column = np.full((height, self._legend_width(), 3), SURFACE, np.uint8)
        self._draw_legend(column)
        self._legend_cache = (height, column)
        return column

    # ---- 摆画面 --------------------------------------------------------

    def _margin(self, width: int, height: int) -> int:
        """画面四周留多宽的边:半厘米,和顶栏、按键说明的左边留白一样宽,上下几块左边
        对齐成一条线。窗口小的时候跟着缩,别把画面挤没了。"""
        return max(0, min(round(self._px_cm * 0.5), width // 20, height // 20))

    def _place(self, shape: tuple[int, int], width: int, height: int) -> Place:
        """画面缩到放得下,在给它的那块里**居中**,四周留一圈窄边(卡片的样子)。

        录制工具里给它的那块是图例左边那一条(`width` 已经减掉了图例),所以画面还是
        落在图例左边,不压在图例上。
        """
        h, w = shape
        margin = self._margin(width, height)
        scale = max(1e-3, min((max(1, width) - 2 * margin) / w, (height - 2 * margin) / h))
        return Place(
            scale=scale,
            x=max(0, round((width - w * scale) / 2)),
            y=max(0, round((height - h * scale) / 2)),
        )

    def _blit(self, canvas: np.ndarray, image: np.ndarray, place: Place) -> None:
        piece = _scaled(image, place.scale)
        _paste(canvas, piece, place.x, place.y)
        x2 = min(canvas.shape[1], place.x + piece.shape[1])
        y2 = min(canvas.shape[0], place.y + piece.shape[0])
        self._view = (place.x, place.y, x2, y2)
        _round_corners(canvas, self._view, self._radius(), BACKGROUND)

    def _radius(self) -> int:
        """卡片的圆角多大。"""
        return max(4, round(self._px_cm * 0.3))

    # ---- 画面上的那几样 ------------------------------------------------

    def _regions(self, canvas: np.ndarray, place: Place, update: VisionUpdate) -> None:
        """检测区域和起点区的边框。**送整幅画面时检测区域变灰**(ADR-0008)。"""
        alignment = update.alignment
        assert alignment is not None
        full = update.sent is None
        radius = max(3, round(self._px_cm * 0.2))
        (x1, y1), (x2, y2) = place.box(alignment.detection_region)
        _round_outline(canvas, x1, y1, x2, y2, radius, GREY if full else GREEN, 2)
        (x1, y1), (x2, y2) = place.box(alignment.start_zone)
        _round_outline(canvas, x1, y1, x2, y2, radius, YELLOW, 2)

    def _hands(self, canvas: np.ndarray, place: Place, result: FrameResult) -> None:
        """每只手的手框和姿势标签。已确认的绿框,没确认的灰框。

        会被认成停止手势的那几个标签用红的:一眼看得出这一下为什么停了。手框画成四个
        角标加一圈细线,姿势标签写在同色的小标签上,贴着手框上沿。
        """
        px = self.text_px()  # 手框上的姿势标签也不低于基准字高:站在 1–1.5 m 外得看得清
        tag = chip_height(px)
        for hand in result.hands:
            x1, y1, x2, y2 = hand.box
            if hand.label in STOP_GESTURE_LABELS:
                color = RED
            else:
                color = GREEN if hand.confirmed else GREY
            top_left, bottom_right = place.at((x1, y1)), place.at((x2, y2))
            _brackets(canvas, top_left, bottom_right, color)
            # 上沿放不下(手在画面顶上)就贴进框里
            above = top_left[1] - tag - 4
            y = above if above >= self._view[1] else top_left[1] + 4
            _chip(
                canvas, hand.label.upper(), top_left[0], y, px,
                fg=INK, bg=color, weight=BOLD, radius=round(cap_px(px) * 0.45),
            )

    def _align(self, canvas: np.ndarray, align: AlignView) -> None:
        """对准阶段:提示怎么站、怎么举拳,停稳进度,站太近。

        提示和停稳进度条放在画面下方同一张卡片里;站太近另起一个红标签,压在卡片上面。
        这几样都画在画布上固定的位置,不跟着画面里的坐标走,所以不用 `Place`。
        """
        x1, _y1, x2, y2 = self._view
        px = self._fit(align.hint, self.text_px(1.2), SEMIBOLD, 0.8)
        cap = round(cap_px(px))
        pad = round(cap * 0.8)
        bar = max(6, round(self._px_cm * 0.3))
        width = max(text_width(align.hint, px) + 2 * pad, round((x2 - x1) * 0.5))
        height = pad + cap + pad + bar + pad
        bottom = y2 - round(self._px_cm * 0.5)
        left = (x1 + x2 - width) // 2
        top = bottom - height
        _round_rect(
            canvas, left, top, left + width, bottom, self._radius(), SURFACE, alpha=CARD_ALPHA
        )
        hint_x = (x1 + x2 - text_width(align.hint, px)) / 2
        _put(canvas, align.hint, hint_x, top + pad + cap, WHITE, px)
        self._bar(canvas, left + pad, top + 2 * pad + cap, width - 2 * pad, bar, align.progress)
        if align.too_close:
            close = self.text_px(1.2)
            _chip(
                canvas, TOO_CLOSE_HINT, (x1 + x2) // 2, top - chip_height(close) - pad, close,
                fg=INK, bg=RED, weight=BOLD, align="center",
            )

    def _judge(self, canvas: np.ndarray, place: Place, update: VisionUpdate) -> None:
        """READY、方向箭头、姿势保持的进度圈。"""
        judge = update.judge
        alignment = update.alignment
        assert judge is not None and alignment is not None
        zone = alignment.start_zone
        px = self.text_px(1.4)

        if judge.ready:
            # 锁定的手这一帧没被认出来:READY 变暗,但锁定不解除(ADR-0010)。
            # 骑在起点区下沿的正中:上沿那边是拳头和它的姿势标签,放上面会叠在一起
            (at_x, _), (to_x, at_y) = place.box(zone)
            top = min(at_y - chip_height(px) // 2, self._view[3] - chip_height(px))
            _chip(
                canvas, "READY", (at_x + to_x) // 2, top, px,
                fg=INK, bg=GREY if judge.ready_dim else GREEN, weight=BOLD, align="center",
            )
        if judge.direction is not None:
            self._arrow(canvas, place, update, judge.direction, judge.direction_track)
        if judge.hold_progress > 0.0:
            self._ring(canvas, place, zone, judge.hold_progress)
        if judge.holding is not None:
            at_x, at_y = place.at(zone.center)
            radius = self._ring_radius(place, zone)
            _chip(
                canvas, f"HOLD {judge.holding.upper()}", at_x, at_y + radius + 10,
                self.text_px(), fg=INK, bg=GREEN, weight=BOLD, align="center",
            )

    def _arrow(
        self,
        canvas: np.ndarray,
        place: Place,
        update: VisionUpdate,
        direction: str,
        track_id: int,
    ) -> None:
        """方向箭头画在**做这一下那只手的手框旁边**(票里要求)。

        按跟踪编号找那只手,**不拿「画面里最大的手」顶替**:触发那一下锁定已经解除了
        (ADR-0010),最大的手可能已经换成旁人的了,箭头会画到别人手上。找不到就退回
        画在起点区中间——认成了哪个方向照样看得见。

        往前推、敲两下在画面里没有方向可画,就在同一个位置写 `PUSH` / `KNOCK x2`。
        """
        assert update.alignment is not None
        hand = next(
            (h for h in update.result.hands if h.track_id == track_id), None
        )
        start = hand.center if hand is not None else update.alignment.start_zone.center
        if direction not in DIRECTIONS:
            px = self.text_px(1.4)
            at_x, at_y = place.at(start)
            _chip(
                canvas, FIST_HOWTO.get(direction, direction.upper()), at_x,
                at_y - chip_height(px) // 2, px, fg=INK, bg=GREEN, weight=BOLD, align="center",
            )
            return
        ux, uy = DIRECTIONS[direction]
        reach = update.alignment.hand_width * 2.0
        _bold_arrow(
            canvas, place.at(start), place.at((start[0] + ux * reach, start[1] + uy * reach)),
            GREEN, 6,
        )

    def _ring_radius(self, place: Place, zone: Region) -> int:
        return max(12, round(zone.width * place.scale * 0.22))

    def _ring(
        self, canvas: np.ndarray, place: Place, zone: Region, progress: float
    ) -> None:
        """姿势保持的进度圈。半径按起点区算,换个手宽也还是套在区里。

        底下先画一整圈暗的轨道,看得出离满圈还差多少。
        """
        radius = self._ring_radius(place, zone)
        center = place.at(zone.center)
        cv2.circle(canvas, center, radius, BACKGROUND, 10, cv2.LINE_AA)
        cv2.circle(canvas, center, radius, LINE, 6, cv2.LINE_AA)
        sweep = 360.0 * min(1.0, progress)
        cv2.ellipse(
            canvas, center, (radius, radius), -90.0, 0.0, sweep, GREEN, 6, cv2.LINE_AA
        )
        # 两头补成圆的
        for angle in (-90.0, -90.0 + sweep):
            end = (
                round(center[0] + radius * math.cos(math.radians(angle))),
                round(center[1] + radius * math.sin(math.radians(angle))),
            )
            cv2.circle(canvas, end, 3, GREEN, -1, cv2.LINE_AA)

    def _bar(
        self, canvas: np.ndarray, left: int, top: int, width: int, height: int, progress: float
    ) -> None:
        """对准的停稳进度条。举着不动要举 2 秒,不画个东西的话不知道还要多久。

        两头是圆的:底下一条暗的轨道,上面是走了多少。
        """
        radius = height // 2
        _round_rect(canvas, left, top, left + width, top + height, radius, LINE)
        filled = round(width * min(1.0, max(0.0, progress)))
        if filled > 0:
            _round_rect(
                canvas, left, top, left + max(filled, height), top + height, radius, YELLOW
            )

    def _corner(self, canvas: np.ndarray, update: VisionUpdate) -> None:
        """角上那个小标签:送的是整幅画面还是检测区域(ADR-0008)。

        前面的圆点和检测区域边框同色:送检测区域时绿,送整幅画面时灰(边框也是灰的)。
        """
        px = self.text_px()
        region = update.sent is not None
        text = "DETECTION REGION" if region else "FULL FRAME"
        margin = round(self._px_cm * 0.3)
        _chip(
            canvas, text, self._view[0] + margin, self._view[1] + margin, px,
            fg=GREY if region else WHITE, bg=SURFACE, alpha=CARD_ALPHA,
            dot=GREEN if region else GREY,
        )

    def _ignored(self, canvas: np.ndarray, ignored: Ignored) -> None:
        """「这一下看见了,但现在不受理」。

        不报的话,人做完手势什么也没发生,只能再做一次——2026-09-22 站着验收时 17 次
        手势里有 5 次落在冷却上。用黄的不用红的:这不是出错,只是来早了。

        放在角上那个小标签下面一行、居中,两个不会叠在一起。
        """
        text = (
            f"{gesture_name(ignored.kind, ignored.key)} - "
            f"NOT NOW ({ignored.state.value})"
        )
        margin = round(self._px_cm * 0.3)
        top = self._view[1] + 2 * margin + chip_height(self.text_px())
        self._centered(canvas, text, top, YELLOW, border=YELLOW)

    # ---- 录制工具的屏幕提示 --------------------------------------------

    def _prompt(self, canvas: np.ndarray, place: Place, prompt: PromptView) -> None:
        """录制时的提示:大字、说明、示范图、动作箭头、剩多少秒。

        **「做!」段有加粗框**(spec「录制工具」第 4 条):录的时候人站在 1–1.5 m 外、
        眼睛多半在自己手上,余光里要分得出现在是准备还是该做了,所以不光换字,
        整块画面描一圈粗的。
        """
        px = self.text_px(1.6)
        color = GREEN if prompt.going else YELLOW
        x1, y1, x2, y2 = self._view
        if prompt.going:
            # 整块画面描一圈:余光里就能看出「该做了」
            _round_outline(canvas, x1 + 3, y1 + 3, x2 - 4, y2 - 4, self._radius(), color, 6)

        # 大字从角上那个小标签下面一行开始,两个不叠在一起
        margin = round(self._px_cm * 0.3)
        bottom = self._centered(
            canvas, prompt.headline, y1 + 2 * margin + chip_height(self.text_px()), color,
            px=px, weight=BOLD,
        )
        self._centered(canvas, prompt.text, bottom + round(self._px_cm * 0.2), WHITE)
        # 剩多少秒和整张计划走了几成:一眼知道还要站多久
        self._centered(
            canvas,
            f"{prompt.left_s:0.0f}S    {prompt.progress * 100:0.0f}% DONE",
            y2 - margin - chip_height(self.text_px(1.2)),
            GREY,
        )
        # **箭头只在准备段和「做!」段画,而且两段长得不一样**(见 `_demo_arrow`);
        # 歇一下那段撤掉——那时不该再有任何「照这个做」的信号
        if prompt.arrow is not None and prompt.seg in (PREP, GO):
            self._demo_arrow(canvas, place, prompt.arrow, live=prompt.going)
        if prompt.demo is not None:
            self._demo_image(canvas, prompt.demo, big=prompt.big)

    def _demo_arrow(
        self,
        canvas: np.ndarray,
        place: Place,
        arrow: tuple[tuple[float, float], tuple[float, float]],
        *,
        live: bool,
    ) -> None:
        """示意图里的箭头。**起点和终点画在真实动作的位置**(spec「录制工具」第 1 条)。

        起点是起点区中心,终点是朝那个方向移过判定要求的那么多个手宽之后的位置——
        照着箭头做出来的那一下,正好够触发。画个空心圈标出起点,不然只看箭头分不清
        手该从哪儿出发。

        **准备段画细的灰虚线,「做!」段才画粗的绿实线。** 两段画成一样的话,那根箭头
        本身就是「现在照这个做」的信号,人看见就做了——2026-09-23 站着录第一遍时,
        9 次误触发里有 6 次是**正确的方向、只是早做了 0.26–1.6 秒**,正是这么来的。
        加粗框和高音区分不过一根一模一样的大箭头,所以箭头自己也得跟着变。
        """
        start, end = arrow
        at_start, at_end = place.at(start), place.at(end)
        color = GREEN if live else GREY
        ring = max(6, round(self._px_cm * 0.4))
        cv2.circle(canvas, at_start, ring, color, 3 if live else 2, cv2.LINE_AA)
        if live:
            _bold_arrow(canvas, at_start, at_end, color, 6, tip=0.25)
            return
        # 准备段:虚线加一个小箭头,只说「等下往这边」,不说「现在做」
        _dashed_line(canvas, at_start, at_end, color, 2)
        head = (
            round(at_start[0] + (at_end[0] - at_start[0]) * 0.75),
            round(at_start[1] + (at_end[1] - at_start[1]) * 0.75),
        )
        cv2.arrowedLine(canvas, head, at_end, color, 2, cv2.LINE_AA, tipLength=0.5)

    def _demo_image(self, canvas: np.ndarray, image: np.ndarray, *, big: bool) -> None:
        """示范姿势图。**准备段画大的**,「做!」段缩小让出画面(spec 第 4 条)。

        放在画面的左下角:右边是图例,中间是人和起点区,左下角本来就空着。
        """
        # 示范图按厘米定大小,不跟着字高走:字改小了,示范图照旧看得清手形
        want = round(self._px_cm * (6.0 if big else 3.0))
        h, w = image.shape[:2]
        piece = _scaled(image, min(want / max(1, w), want / max(1, h)))
        pad = round(self._px_cm * 0.2)
        x = self._view[0] + round(self._px_cm * 0.6)
        y = self._view[3] - piece.shape[0] - round(self._px_cm * 2.4)
        _round_rect(
            canvas, x - pad, y - pad, x + piece.shape[1] + pad, y + piece.shape[0] + pad,
            self._radius(), SURFACE, alpha=CARD_ALPHA,
        )
        _paste(canvas, piece, x, max(0, y))

    def _banner(self, canvas: np.ndarray, text: str, color: Color) -> None:
        """一块横幅。识别中断用它——看不见停止手势这件事必须一眼看到。

        整块画面先压暗,正中一张带红边的卡片,前面一个圆圈里的叹号。
        """
        x1, y1, x2, y2 = self._view
        _round_rect(canvas, x1, y1, x2, y2, 0, BACKGROUND, alpha=0.45)
        px = self.text_px(2.0)
        cap = cap_px(px)
        pad = round(cap * 0.8)
        icon = round(cap * 1.5)
        width = icon + pad + text_width(text, px, BOLD) + 2 * pad
        height = round(cap * 2.6)
        left = (x1 + x2 - width) // 2
        top = (y1 + y2 - height) // 2
        _round_rect(
            canvas, left, top, left + width, top + height, self._radius(), SURFACE, alpha=0.95
        )
        _round_outline(canvas, left, top, left + width, top + height, self._radius(), color, 3)
        center = (left + pad + icon // 2, top + height // 2)
        cv2.circle(canvas, center, icon // 2, color, -1, cv2.LINE_AA)
        bang = text_width("!", px, BOLD)
        _put(canvas, "!", center[0] - bang / 2, center[1] + cap / 2, INK, px, BOLD)
        _put(canvas, text, left + pad + icon + pad, top + (height + cap) / 2, color, px, BOLD)

    # ---- 手势图例 ------------------------------------------------------

    @property
    def shows_keys(self) -> bool:
        """调试按键是不是已经写在图例里了(每行最右边的小键帽)。

        写了的话,预览窗口底下的按键说明就不再列一遍调试按键(`PreviewWindow.set_legend`)。
        """
        return any(line.key for line in self._lines)

    def _legend_width(self) -> int:
        """图例要占多宽。

        **每一行都量**,包括小标题、停止手势那两行和出处那一行:手势名是从演示配置里来的,
        写死或者只量对照表那几行,换个动作名就会被画面右边切掉(2026-09-22 存图时
        「SPACE = STOP」正是这么被切掉半截的)。
        """
        if self._grid is None:
            self._grid = self._grid_for(self._lines)
        return self._grid.width

    def _thumb_px(self) -> int:
        """示范小图画多大。和一行字一样高的两倍,看得清手形。"""
        return round(self.text_px(2.0))

    def _grid_for(
        self, lines: list[_Line], width: int | None = None, *, tight: bool = False
    ) -> _Grid:
        """这几行排成一张表:手势名(连怎么做)、预设动作、调试键,各从哪儿开始、多宽。

        动作那一栏上下对齐,一眼扫下来就是一张表;名字和动作之间只靠留白分开,不画箭头。
        `width` 给了就是这一栏分到多宽(小标题后面的细线、叫停那块红底拉到这么宽),
        不给就按内容算。`tight` 是留白收紧一档:画面下面那一条两栏并排放不下时用。
        """
        px = self.text_px()
        cap = cap_px(px)
        pad = round(px * (0.45 if tight else 0.7))
        gap = round(px * (0.3 if tight else 0.45))
        thumb = self._thumb_px() if self._images else 0
        names = actions = heads = 0
        keyed = False
        for line in lines:
            if line.style in (ROW, STOP):
                names = max(names, _name_width(line, px, gap))
                actions = max(actions, text_width(line.action or "", px, SEMIBOLD))
                keyed = keyed or line.key is not None
            elif line.style == HEAD:
                heads = max(heads, text_width(line.name, px, BOLD) + 4 * gap)
            else:
                heads = max(heads, text_width(line.name, px, REGULAR))
        indent = round(px * INDENT_CM)
        name_x = pad + indent + (thumb + gap if thumb else 0)
        action_x = name_x + names + 2 * gap
        key_w = round(cap * 1.9) if keyed else 0
        key_x = action_x + actions + 2 * gap
        natural = max((key_x + key_w) if keyed else (action_x + actions), pad + heads) + pad
        return _Grid(
            px=px, pad=pad, gap=gap, thumb=thumb, indent=indent, name_x=name_x,
            action_x=action_x, key_x=key_x, key_w=key_w,
            width=natural if width is None else max(natural, width),
        )

    def _legend(self, canvas: np.ndarray) -> None:
        """录制工具里的图例:画在这块画布的右边,上下占满。"""
        column = self.legend_image(canvas.shape[0])
        _paste(canvas, column, canvas.shape[1] - column.shape[1], 0)

    def legend_strip(self, width: int) -> np.ndarray:
        """画面**下面**那一条图例:宽 `width`,FIST MOVE 一栏、HOLD 和叫停一栏,左右并排。

        并排(仿真)时预览窗口只有半屏宽,图例放右边的话要占掉一半,画面只剩不到 400 px
        宽、上下还空着一大块(2026-10-10 用户看了截图要改)。窗口又窄又高,图例挪到画面
        下面分两栏排,画面就能用满整个宽。
        """
        width = max(1, width)
        groups = _columns(self._lines)
        # 每栏按内容要多宽分,剩下的平分;两栏加起来比窗口宽就先把留白收紧一档
        tight = sum(self._grid_for(group).width for group in groups) > width
        natural = [self._grid_for(group, tight=tight).width for group in groups]
        extra = max(0, width - sum(natural)) // len(groups)
        grids = [
            self._grid_for(group, size + extra, tight=tight)
            for group, size in zip(groups, natural, strict=True)
        ]
        px = self.text_px()
        step, head, section = round(px * 1.55), round(px * 1.5), round(px * 0.6)
        top = bottom = round(px * 0.6)
        tall = max(self._stack(group, step, head, section)[1] for group in groups)
        strip = np.full((top + tall + bottom, width, 3), SURFACE, np.uint8)
        cv2.line(strip, (0, 0), (width - 1, 0), LINE, 1)
        x0 = 0
        for group, grid in zip(groups, grids, strict=True):
            self._draw_lines(strip, group, grid, x0, top, step, head, section)
            x0 += grid.width
        return strip

    def _draw_legend(self, column: np.ndarray) -> None:
        """「触发手势 → 预设动作」对照表,附示范小图和出处。

        按 FIST MOVE / HOLD 分组,小标题后面拖一条细线;动作那一栏用执行中的蓝色,
        回待命用回待命的黄色;叫停那两行垫一块淡淡的红底,和别的分开。行距按这一列的
        高度收紧,放不下也不缩字。
        """
        grid = self._grid_for(self._lines)
        px = grid.px
        height = column.shape[0]
        for factor in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
            step = max(round(px * 1.75 * factor), grid.thumb + round(px * 0.2))
            head = round(px * 1.6 * factor)
            section = round(px * 0.8 * factor)
            top = round(px * 0.9 * factor)
            if top + self._stack(self._lines, step, head, section)[1] <= height:
                break
        # 左边一条细线,和画面那块分开
        cv2.line(column, (0, 0), (0, height - 1), LINE, 1)
        self._draw_lines(column, self._lines, grid, 0, top, step, head, section)

    def _stack(
        self, lines: list[_Line], step: int, head: int, section: int
    ) -> tuple[list[tuple[_Line, int]], int]:
        """这几行从上往下各落在哪儿(相对上沿),和一共多高。

        小标题之间、叫停那块前面各空 `section`;出处跟在最后,前面空半个。
        """
        placed: list[tuple[_Line, int]] = []
        y = 0
        stopped = False
        for line in lines:
            if line.style == HEAD:
                if placed:
                    y += section
                placed.append((line, y))
                y += head
                continue
            if line.style == STOP and not stopped:
                y += section
                stopped = True
            elif line.style == SOURCE:
                y += section // 2
            placed.append((line, y))
            y += step
        return placed, y

    def _draw_lines(
        self,
        canvas: np.ndarray,
        lines: list[_Line],
        g: _Grid,
        x0: int,
        top: int,
        step: int,
        head: int,
        section: int,
    ) -> None:
        """把一栏画到 `canvas` 上:左边从 `x0` 起,上沿在 `top`。"""
        px = g.px
        cap = cap_px(px)
        placed, _ = self._stack(lines, step, head, section)
        stops = [y for line, y in placed if line.style == STOP]
        if stops:
            # 叫停那块:先垫红底再写字
            _round_rect(
                canvas, x0 + g.pad, top + stops[0] - round(px * 0.15),
                x0 + g.width - g.pad, top + stops[-1] + step + round(px * 0.15),
                round(px * 0.4), RED, alpha=0.1,
            )
        for line, y in placed:
            y += top
            if line.style == HEAD:
                baseline = y + (head + cap) / 2
                _put(canvas, line.name, x0 + g.pad, baseline, GREY, px, BOLD)
                rule_x = x0 + g.pad + text_width(line.name, px, BOLD) + g.gap
                rule_y = round(baseline - cap / 2)
                if rule_x < x0 + g.width - g.pad:
                    cv2.line(canvas, (rule_x, rule_y), (x0 + g.width - g.pad, rule_y), LINE, 1)
            elif line.style in (ROW, STOP):
                self._legend_row(canvas, line, g, x0, y, step)
            else:
                _put(canvas, line.name, x0 + g.pad, y + (step + cap) / 2, GREY, px, REGULAR)

    def _legend_row(
        self, canvas: np.ndarray, line: _Line, g: _Grid, x0: int, top: int, step: int
    ) -> None:
        px = g.px
        cap = cap_px(px)
        baseline = top + (step + cap) / 2
        image = self._images.get(line.pose) if line.pose else None
        if image is not None and g.thumb:
            thumb_x = x0 + g.pad + g.indent
            self._thumb(canvas, image, thumb_x, top + (step - g.thumb) // 2, g.thumb)
        x = x0 + g.name_x
        _put(canvas, line.name, x, baseline, WHITE, px, SEMIBOLD)
        if line.howto:
            x += text_width(line.name, px, SEMIBOLD) + round(g.gap * 0.6)
            _put(canvas, line.howto, x, baseline, GREY, px, REGULAR)
        _put(canvas, line.action or "", x0 + g.action_x, baseline, line.tone, px, SEMIBOLD)
        if line.key is not None:
            size = g.key_w
            middle = round(baseline - cap / 2)
            _keycap(canvas, line.key, x0 + g.key_x, middle - size // 2, size, px)

    def _thumb(
        self, canvas: np.ndarray, image: np.ndarray, left: int, top: int, size: int
    ) -> None:
        h, w = image.shape[:2]
        piece = _scaled(image, min(size / w, size / h))
        _paste(canvas, piece, left, top)
        _round_corners(
            canvas,
            (left, top, left + piece.shape[1], top + piece.shape[0]),
            max(3, size // 8),
            SURFACE,
        )

    # ---- 小工具 --------------------------------------------------------

    def _fit(self, text: str, px: float, weight: int, pad: float) -> float:
        """这一行字(左右各留 `pad` 个大写字母高的边)比画面还宽时,缩到放得下的字高。

        **不小于基准字高**(`text_px()`):真放不下宁可两头被切,也不把字画小。
        """
        room = (self._view[2] - self._view[0]) - 2 * round(self._px_cm * 0.3)
        width = text_width(text, px, weight) + 2 * round(cap_px(px) * pad)
        if width <= room:
            return px
        return max(self.text_px(), px * room / width)

    def _centered(
        self,
        canvas: np.ndarray,
        text: str,
        top: int,
        color: Color,
        *,
        px: float | None = None,
        weight: int = SEMIBOLD,
        border: Color | None = None,
    ) -> int:
        """把一行字写在一张半透明的卡片上,放在**画面**的正中(左右方向),上沿在 `top`。

        按画面那块居中,不按整块画布:右边可能是图例。比画面还宽时先把字缩小,
        但不小于基准字高——`FIST MOVE LEFT - NOT NOW (COOLDOWN)` 在图例占了右侧整列以后,
        按 1.2 倍写就比画面宽,两头被切掉。返回卡片的下沿。
        """
        px = self._fit(text, self.text_px(1.2) if px is None else px, weight, 0.75)
        center = (self._view[0] + self._view[2]) // 2
        _x1, _y1, _x2, bottom = _chip(
            canvas, text, center, top, px, fg=color, bg=SURFACE, alpha=CARD_ALPHA,
            weight=weight, align="center", radius=self._radius(), border=border,
        )
        return bottom


def gesture_name(kind: str, key: str) -> str:
    """一个触发手势的英文名,照 `CONTEXT.md`「Language」:`FIST MOVE LEFT`、`HOLD PEACE`。

    图例、屏幕上的提示、日志说的都得是同一个名字,所以只有这一处拼它。
    """
    if kind == FIST_MOVE:
        return f"FIST MOVE {key.upper()}"
    return f"HOLD {key.upper()}"


def legend_poses(demo: DemoConfig) -> list[str]:
    """图例要哪几张示范小图。裁图脚本按这几个名字存,读的时候按同样的名字找。

    六个触发手势用到的姿势,外加停止手势那一张(它不是触发手势,但图例上要列)。
    """
    names = [pose for pose, _ in _legend_rows(demo)]
    names.append("palm")
    return list(dict.fromkeys(names))  # 去重,但保持先后


def _legend_rows(demo: DemoConfig) -> list[tuple[str, str]]:
    """图例每一行:用哪张示范小图,写什么字。对应关系**查演示配置**。

    手势的写法一律用 `CONTEXT.md`「Language」里定下的英文名(`FIST MOVE LEFT`、
    `HOLD PEACE`),姿势保持那几个再在后面括注怎么做——屏幕上说的和文档、日志里
    说的是同一个名字,才对得上。
    """
    rows = []
    for trigger in demo.triggers:
        action = demo.display_for(trigger.action)
        name = gesture_name(trigger.kind, trigger.key)
        if trigger.kind == FIST_MOVE:
            howto = FIST_HOWTO.get(trigger.key)
            if howto is not None:
                name += f" ({howto})"
            rows.append(("fist", f"{name} = {action}"))
        elif trigger.kind == POSE_HOLD:
            howto = POSE_HOWTO.get(trigger.key)
            if howto is not None:
                name += f" ({howto})"
            rows.append((trigger.key, f"{name} = {action}"))
    return rows


HEAD, ROW, STOP, SOURCE = "head", "row", "stop", "source"
"""图例里一行是哪一种:小标题、手势那一行、叫停、出处。"""

INDENT_CM = 0.5
"""手势那几行比小标题往里缩多少(厘米)。"""


@dataclass(frozen=True)
class _Line:
    """图例里的一行。"""

    style: str
    name: str
    """小标题、手势名(去掉小标题那一截:`LEFT`、`PEACE`)、出处。"""
    pose: str | None = None
    """用哪张示范小图。"""
    howto: str | None = None
    """怎么做(`V SIGN`、`KNOCK x2`),灰字跟在名字后面。"""
    action: str | None = None
    """对应的预设动作(`POINT LEFT`)、HOME、STOP。"""
    tone: Color = BLUE
    """动作那一栏的颜色。"""
    key: str | None = None
    """调试时按哪个键等于做这个手势(`F`、`LEFT`),画成最右边的小键帽。不调试时没有。"""


@dataclass(frozen=True)
class _Grid:
    """图例的几栏从哪儿开始、整列多宽(像素)。"""

    px: float
    pad: int
    gap: int
    thumb: int
    indent: int
    name_x: int
    action_x: int
    key_x: int
    key_w: int
    """调试键那个小键帽多宽。没有调试键时是 0。"""
    width: int


def _name_width(line: _Line, px: float, gap: int) -> int:
    width = text_width(line.name, px, SEMIBOLD)
    if line.howto:
        width += round(gap * 0.6) + text_width(line.howto, px, REGULAR)
    return width


def _legend_lines(
    demo: DemoConfig, keys: dict[tuple[str, str], str] | None = None
) -> list[_Line]:
    """图例画出来的一行行。

    同一类手势归到一个小标题下:`FIST MOVE` 下面是 `LEFT → POINT LEFT`……,`HOLD` 下面
    是 `PEACE V SIGN → DRAW SQUARE`……。名字是从 `gesture_name` 拼出来的整名里把开头的
    类名挪到小标题上得来的,所以连起来读还是 `CONTEXT.md` 定下的那个英文名(FIST MOVE
    LEFT、HOLD PEACE)。每行都带着类名的话图例太宽:2026-09-23 挪到右侧整列后宽到
    1172 px,把左边的大字挤得折行。

    怎么做原来写在括号里,现在用灰字跟在名字后面;动作那一栏单独对齐。`keys` 是调试按键
    (触发手势 → 键名),给了就在每行最右边画上。
    """
    keys = keys if keys is not None else {}
    lines: list[_Line] = []
    heading = None
    for trigger in demo.triggers:
        prefix = gesture_name(trigger.kind, "").strip()
        if prefix != heading:
            lines.append(_Line(HEAD, prefix))
            heading = prefix
        name = gesture_name(trigger.kind, trigger.key)[len(prefix):].strip()
        if trigger.kind == FIST_MOVE:
            pose, howto = "fist", FIST_HOWTO.get(trigger.key)
        else:
            pose, howto = trigger.key, POSE_HOWTO.get(trigger.key)
        lines.append(
            _Line(
                ROW, name, pose=pose, howto=howto,
                action=demo.display_for(trigger.action),
                tone=YELLOW if trigger.action == HOME_ACTION else BLUE,
                key=keys.get((trigger.kind, trigger.key)),
            )
        )
    for text in STOP_LINES:
        name, action = text.split(" = ")
        pose = "palm" if name == "OPEN PALM" else None
        lines.append(_Line(STOP, name, pose=pose, action=action, tone=RED))
    lines.append(_Line(SOURCE, LEGEND_SOURCE))
    return lines


def _columns(lines: list[_Line]) -> list[list[_Line]]:
    """画面下面那一条图例分两栏:第一个小标题(FIST MOVE)那组一栏,剩下的一栏。

    只有一组手势时就一栏。
    """
    heads = [i for i, line in enumerate(lines) if line.style == HEAD]
    if len(heads) < 2:
        return [lines]
    return [lines[: heads[1]], lines[heads[1] :]]


# ---- 画图的小工具 ------------------------------------------------------


def _scaled(image: np.ndarray, scale: float) -> np.ndarray:
    """缩成这个倍数。

    缩小一律用 `INTER_AREA`:画面里的手框只有几十像素,别的插值会糊成一团。
    """
    h, w = image.shape[:2]
    return cv2.resize(
        image,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )


def _paste(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
    """把一幅图贴到画布上,**超出画布的部分裁掉**,不报错。

    窗口小到放不下时(比如中间那块被上下几行字挤扁),宁可少画一块,也不该把主循环
    带倒——预览窗口卡住不该影响别的(`robot_side.preview` 上的说明)。
    """
    x, y = max(0, x), max(0, y)
    rows = min(canvas.shape[0] - y, image.shape[0])
    cols = min(canvas.shape[1] - x, image.shape[1])
    if rows > 0 and cols > 0:
        canvas[y : y + rows, x : x + cols] = image[:rows, :cols]


def _blend(roi: np.ndarray, mask: np.ndarray, color: Color, alpha: float) -> None:
    """按 `mask`(0–255)把 `color` 盖到 `roi` 上。`mask` 满、`alpha` 为 1 的地方正好是这个颜色。"""
    a = mask.astype(np.float32)[..., None] * (alpha / 255.0)
    mixed = roi.astype(np.float32) * (1.0 - a) + np.array(color, np.float32) * a
    roi[:] = (mixed + 0.5).astype(np.uint8)


def _round_rect(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    radius: int,
    color: Color,
    *,
    alpha: float = 1.0,
) -> None:
    """一块填满的圆角矩形,可以半透明。圆角抗锯齿,直边不用。超出画布的部分裁掉。"""
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return
    w, h = x2 - x1, y2 - y1
    r = max(0, min(radius, w // 2, h // 2))
    if r == 0:
        # 没有圆角(比如整块画面压暗):直接混,不做遮罩。整块画面逐像素算要十几毫秒
        roi = canvas[cy1:cy2, cx1:cx2]
        cv2.addWeighted(roi, 1.0 - alpha, np.full_like(roi, color), alpha, 0.0, dst=roi)
        return
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
    if r > 0:
        for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
            cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    mask = mask[cy1 - y1 : cy2 - y1, cx1 - x1 : cx2 - x1]
    _blend(canvas[cy1:cy2, cx1:cx2], mask, color, alpha)


def _round_outline(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    radius: int,
    color: Color,
    thickness: int,
) -> None:
    """圆角矩形的边框。直边是实打实的颜色(测试按像素核对边框颜色),圆角抗锯齿。"""
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.line(canvas, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(canvas, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(canvas, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(canvas, (x2, y1 + r), (x2, y2 - r), color, thickness)
    if r > 0:
        for center, angle in (
            ((x1 + r, y1 + r), 180),
            ((x2 - r, y1 + r), 270),
            ((x2 - r, y2 - r), 0),
            ((x1 + r, y2 - r), 90),
        ):
            cv2.ellipse(canvas, center, (r, r), angle, 0, 90, color, thickness, cv2.LINE_AA)


@functools.lru_cache(maxsize=16)
def _corner_mask(radius: int) -> np.ndarray:
    """左上角那一小块里「圆外」的部分(0–255,抗锯齿)。四个角翻一翻都用它。"""
    yy, xx = np.mgrid[0:radius, 0:radius].astype(np.float32) + 0.5
    distance = np.hypot(radius - xx, radius - yy)
    return (np.clip(distance - radius + 0.5, 0.0, 1.0) * 255).astype(np.uint8)


def _round_corners(
    canvas: np.ndarray, rect: tuple[int, int, int, int], radius: int, color: Color
) -> None:
    """把 `rect` 这块的四个角削圆:角外那一点涂成底色 `color`。"""
    x1, y1, x2, y2 = rect
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r <= 0:
        return
    mask = _corner_mask(r)
    for flip_x, flip_y in ((False, False), (True, False), (False, True), (True, True)):
        piece = mask[:, ::-1] if flip_x else mask
        piece = piece[::-1, :] if flip_y else piece
        left = x2 - r if flip_x else x1
        top = y2 - r if flip_y else y1
        if left < 0 or top < 0 or left + r > canvas.shape[1] or top + r > canvas.shape[0]:
            continue
        _blend(canvas[top : top + r, left : left + r], piece, color, 1.0)


def _chip(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    px: float,
    *,
    fg: Color,
    bg: Color,
    weight: int = SEMIBOLD,
    alpha: float = 1.0,
    align: str = "left",
    radius: int | None = None,
    border: Color | None = None,
    dot: Color | None = None,
) -> tuple[int, int, int, int]:
    """一个带底色的小标签。`y` 是上沿;`x` 是左边,`align="center"` 时是中线。

    `radius` 不给就是两头全圆的胶囊。`dot` 给了就在字前面画一个小圆点。
    返回占的那一块 `(x1, y1, x2, y2)`。
    """
    cap = cap_px(px)
    pad = round(cap * 0.75)
    height = chip_height(px)
    dot_w = round(cap * 1.2) if dot is not None else 0
    width = text_width(text, px, weight) + 2 * pad + dot_w
    if align == "center":
        x -= width // 2
    elif align == "right":
        x -= width
    r = height // 2 if radius is None else radius
    _round_rect(canvas, x, y, x + width, y + height, r, bg, alpha=alpha)
    if border is not None:
        _round_outline(canvas, x, y, x + width, y + height, r, border, 2)
    if dot is not None:
        cv2.circle(
            canvas, (x + pad + round(cap * 0.35), y + height // 2), max(3, round(cap * 0.3)),
            dot, -1, cv2.LINE_AA,
        )
    _put(canvas, text, x + pad + dot_w, y + (height + cap) / 2, fg, px, weight)
    return x, y, x + width, y + height


def _brackets(
    canvas: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], color: Color
) -> None:
    """手框:一圈细线,四个角再描粗。"""
    (x1, y1), (x2, y2) = top_left, bottom_right
    cv2.rectangle(canvas, top_left, bottom_right, color, 1)
    arm = max(6, round(min(x2 - x1, y2 - y1) * 0.28))
    for (cx, cy), (dx, dy) in (
        ((x1, y1), (1, 1)),
        ((x2, y1), (-1, 1)),
        ((x1, y2), (1, -1)),
        ((x2, y2), (-1, -1)),
    ):
        cv2.line(canvas, (cx, cy), (cx + dx * arm, cy), color, 4)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * arm), color, 4)


def _bold_arrow(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: Color,
    thickness: int,
    *,
    tip: float = 0.3,
) -> None:
    """一根粗箭头,外面描一圈深色边:浅色背景上也看得见。"""
    cv2.arrowedLine(canvas, start, end, BACKGROUND, thickness + 5, cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(canvas, start, end, color, thickness, cv2.LINE_AA, tipLength=tip)


ARROW_KEYS = {"LEFT": (-1, 0), "RIGHT": (1, 0), "UP": (0, -1), "DOWN": (0, 1)}
"""方向键在小键帽上画成箭头,不写字:`RIGHT` 四个字母塞不进一个方块。"""

KEYCAP = (55, 42, 35)
"""小键帽的底色 `#232a37`,比图例那一列的底色亮一档。"""


def _keycap(canvas: np.ndarray, label: str, x: int, y: int, size: int, px: float) -> None:
    """图例里一个调试键的小键帽:圆角方块,里面写字母,方向键画箭头。"""
    _round_rect(canvas, x, y, x + size, y + size, max(3, size // 5), KEYCAP)
    cx, cy = x + size // 2, y + size // 2
    direction = ARROW_KEYS.get(label)
    if direction is None:
        width = text_width(label, px, SEMIBOLD)
        _put(canvas, label, cx - width / 2, cy + cap_px(px) / 2, GREY, px, SEMIBOLD)
        return
    dx, dy = direction
    half = max(4, round(size * 0.26))
    tail = (cx - dx * half, cy - dy * half)
    tip = (cx + dx * half, cy + dy * half)
    cv2.arrowedLine(canvas, tail, tip, GREY, 2, cv2.LINE_AA, tipLength=0.55)


def _dashed_line(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: Color,
    thickness: int,
    *,
    dash: int = 14,
) -> None:
    """一条虚线。`cv2` 没有虚线,按长度切成一段一段画。

    准备段的箭头用它:虚线在余光里一眼就和实线分得开,而实线那根是「现在做」。
    """
    (x1, y1), (x2, y2) = start, end
    span = math.hypot(x2 - x1, y2 - y1)
    if span < 1:
        return
    steps = max(1, int(span / max(1, dash)))
    for i in range(0, steps, 2):
        a = i / steps
        b = min(1.0, (i + 1) / steps)
        cv2.line(
            canvas,
            (round(x1 + (x2 - x1) * a), round(y1 + (y2 - y1) * a)),
            (round(x1 + (x2 - x1) * b), round(y1 + (y2 - y1) * b)),
            color,
            thickness,
        )

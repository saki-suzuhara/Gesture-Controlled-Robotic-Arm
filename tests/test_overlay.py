"""画面叠加:画出来的那一幅对不对(第 04 票)。

不开窗口、不开摄像头。看的是**画布多大、字是不是英文、该变灰的变没变灰**这类
一眼能核对的事;具体每一笔画在哪儿不断言——那是要人看的,靠 `--snap` 存图去看。
"""

from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from robot_side.config import load_demo
from vision_side import overlay
from vision_side.align import Alignment
from vision_side.loop import AlignView, JudgeView, VisionUpdate
from vision_side.recognize import FrameResult, HandResult, Region

DEMO = load_demo()
PX_PER_CM = 38.0
"""本机 96 DPI 时每厘米多少像素。字高都按它算。"""

PANEL = (1920, 620)
"""全屏时中间那块大致这么大:又宽又扁,所以图例放得进右边那条。"""

ALIGNMENT = Alignment(
    detection_region=Region(240, 60, 800, 600),
    start_zone=Region(520, 240, 240, 240),
    hand_width=80.0,
    full_frame_width=88.0,
    too_close=False,
    t=1.0,
)


def painter(images=None) -> overlay.Painter:
    return overlay.Painter(DEMO, px_per_cm=PX_PER_CM, images=images)


def frame(width: int = 1280, height: int = 720) -> np.ndarray:
    return np.full((height, width, 3), 60, np.uint8)


def update(
    *,
    ignored=None,
    sent: Region | None = ALIGNMENT.detection_region,
    hands: tuple[HandResult, ...] = (),
    alignment: Alignment | None = ALIGNMENT,
    align: AlignView | None = None,
    judge: JudgeView | None = None,
    flat: bool = False,
) -> VisionUpdate:
    return VisionUpdate(
        t=1.0,
        image=frame(),
        result=FrameResult(t=1.0, hands=hands, flat=flat, region=sent),
        sent=sent,
        alignment=alignment,
        align=align,
        judge=judge,
        trigger=None,
        ignored=ignored,
        stop=False,
    )


def judging(**kwargs) -> JudgeView:
    base = dict(
        ready=False,
        ready_dim=False,
        direction=None,
        direction_track=-1,
        hold_progress=0.0,
        holding=None,
    )
    base.update(kwargs)
    return JudgeView(**base)


# ---- 画布 --------------------------------------------------------------


def test_画出来的正好是中间那块那么大():
    got = painter().paint(PANEL, update())
    assert (got.shape[1], got.shape[0]) == PANEL


def test_还没有画面时也画得出来():
    # 启动那几步里预览窗口已经开着了,这时还没有第一帧
    got = painter().paint(PANEL, None)
    assert (got.shape[1], got.shape[0]) == PANEL


def test_窄窗口也画得出来不越界():
    for size in ((640, 360), (320, 900), (100, 100)):
        got = painter().paint(size, update())
        assert (got.shape[1], got.shape[0]) == size, size


def test_画面在左边右边那条留给图例():
    # 画面缩进图例左边那一条里:图例那一列不该被画面盖住
    p = painter()
    got = p.paint(PANEL, update())
    width = p._legend_width()
    assert 0 < width < PANEL[0]
    place = p._place((720, 1280), PANEL[0] - width, PANEL[1])
    assert place.x + round(1280 * place.scale) <= PANEL[0] - width
    # 画面正中是画面(帧是灰的 60),图例那一列是图例的底色
    middle = place.at((640, 360))
    assert got[middle[1], middle[0]].tolist() == [60, 60, 60]
    assert got[PANEL[1] // 2, PANEL[0] - 3].tolist() == list(overlay.SURFACE)


# ---- 文字一律英文 ------------------------------------------------------


def test_图例上的字一律英文():
    for _pose, text in overlay._legend_rows(DEMO):
        assert text.isascii(), text
    assert overlay.STOP_ROW.isascii()
    assert overlay.LEGEND_SOURCE.isascii()


def test_姿势怎么做写成人能照做的话():
    # 光写标签名不够:站着走一遍时 HOLD PEACE 那一件卡了 10.6 秒
    rows = dict(overlay._legend_rows(DEMO))
    assert "V SIGN" in rows["peace"]
    assert "THUMBS UP" in rows["like"]


def test_图例列全每个触发手势加停止手势():
    rows = overlay._legend_rows(DEMO)
    assert len(rows) == len(DEMO.triggers)
    # 对应的预设动作按演示配置查,不在代码里写死;回待命手势写 HOME
    for trigger in DEMO.triggers:
        display = DEMO.display_for(trigger.action)
        assert any(display in text for _pose, text in rows), display
    assert "STOP" in overlay.STOP_ROW
    assert "SPACE" in overlay.STOP_ROW


def test_图例标出示范图的出处():
    assert "HAGRID" in overlay.LEGEND_SOURCE


# ---- 送哪块画面 --------------------------------------------------------


def test_送整幅画面时角上写FULL_FRAME():
    # 只核对这一档和另一档画出来的不一样:字本身画在哪儿靠存图看
    full = painter().paint(PANEL, update(sent=None))
    region = painter().paint(PANEL, update(sent=ALIGNMENT.detection_region))
    assert not np.array_equal(full, region)


def test_送整幅画面时检测区域的边框变灰():
    full = painter().paint(PANEL, update(sent=None))
    region = painter().paint(PANEL, update(sent=ALIGNMENT.detection_region))
    # 边框上取一点:检测区域左边那条
    place = painter()._place((720, 1280), PANEL[0] - painter()._legend_width(), PANEL[1])
    x, y = place.at((ALIGNMENT.detection_region.x, ALIGNMENT.detection_region.y + 200))
    assert tuple(full[y, x]) == overlay.GREY
    assert tuple(region[y, x]) == overlay.GREEN


# ---- 手框和判定 --------------------------------------------------------


def test_停止类标签的手框画成红的():
    # 一眼看得出这一下为什么停了
    hand = HandResult(box=(100, 100, 180, 180), label="palm", track_id=1, confirmed=True)
    got = painter().paint(PANEL, update(hands=(hand,)))
    assert (got == np.array(overlay.RED, np.uint8)).all(axis=2).any()


def test_没确认的手框画成灰的():
    hand = HandResult(box=(100, 100, 180, 180), label="fist", track_id=1, confirmed=False)
    got = painter().paint(PANEL, update(hands=(hand,)))
    confirmed = painter().paint(
        PANEL,
        update(hands=(HandResult(box=(100, 100, 180, 180), label="fist", track_id=1, confirmed=True),)),
    )
    assert not np.array_equal(got, confirmed)


def test_READY变暗和不变暗画出来不一样():
    bright = painter().paint(PANEL, update(judge=judging(ready=True)))
    dim = painter().paint(PANEL, update(judge=judging(ready=True, ready_dim=True)))
    assert not np.array_equal(bright, dim)


def test_认出方向就画箭头():
    plain = painter().paint(PANEL, update(judge=judging()))
    arrow = painter().paint(PANEL, update(judge=judging(direction="left")))
    assert not np.array_equal(plain, arrow)


def test_姿势保持画进度圈():
    plain = painter().paint(PANEL, update(judge=judging()))
    ring = painter().paint(PANEL, update(judge=judging(hold_progress=0.6)))
    assert not np.array_equal(plain, ring)


def test_对准阶段画提示和进度条():
    got = painter().paint(
        PANEL, update(alignment=None, align=AlignView("HOLD YOUR FIST STILL", 0.5, False))
    )
    close = painter().paint(
        PANEL, update(alignment=None, align=AlignView("HOLD YOUR FIST STILL", 0.5, True))
    )
    assert not np.array_equal(got, close)  # TOO CLOSE 那一行多出来了


# ---- 识别中断 ----------------------------------------------------------


def test_识别中断画一条横幅():
    plain = painter().paint(PANEL, update())
    lost = painter().paint(PANEL, update(), lost=True)
    assert not np.array_equal(plain, lost)
    assert (lost == np.array(overlay.RED, np.uint8)).all(axis=2).any()


# ---- 字高 --------------------------------------------------------------


def test_字高至少一厘米():
    # 字高 1 cm 说的是字号,大写字母占其中 CAP_SHARE 那么多(和骨架里 Tk 的字号一个意思)。
    # 真画一个大写 H 量墨迹有多高:字号和字母高度之比每款字不一样
    import cv2

    probe = np.zeros((200, 200), np.uint8)
    size = overlay.font_size(PX_PER_CM * overlay.MIN_CM)
    cv2.putText(probe, "H", (20, 150), 255, overlay.FACE, size, overlay.SEMIBOLD)
    rows = np.flatnonzero(probe.max(axis=1) > 127)
    assert rows[-1] - rows[0] + 1 >= overlay.cap_px(PX_PER_CM * overlay.MIN_CM) - 1


def test_画面和骨架的字高下限是同一档():
    # 2026-10-10 用户要字小一些:1 cm → 0.7 cm。两边一起改,不然画面里的字和上下几行不一样大
    from robot_side import preview

    assert overlay.MIN_CM == preview.MIN_CM


def test_有示范小图才写出处(tmp_path):
    # 出处是给示范小图标的;一张图也没有时那一行只是多出来的字
    import cv2

    assert all(line.style != overlay.SOURCE for line in painter()._lines)
    cv2.imwrite(str(tmp_path / "fist.png"), np.full((64, 64, 3), 200, np.uint8))
    images = overlay.load_legend_images(["fist"], folder=tmp_path)
    with_images = overlay.Painter(DEMO, px_per_cm=PX_PER_CM, images=images)
    assert any(line.name == overlay.LEGEND_SOURCE for line in with_images._lines)


def test_屏幕报的每厘米像素数离谱时也不会把字画没():
    small = overlay.Painter(DEMO, px_per_cm=0.0)
    got = small.paint(PANEL, update())
    assert (got.shape[1], got.shape[0]) == PANEL


# ---- 示范小图 ----------------------------------------------------------


def test_没有示范小图时图例只显示文字(tmp_path):
    images = overlay.load_legend_images(["fist", "peace"], folder=tmp_path)
    assert images == {}
    got = overlay.Painter(DEMO, px_per_cm=PX_PER_CM, images=images).paint(PANEL, update())
    assert (got.shape[1], got.shape[0]) == PANEL


def test_有示范小图就画进图例(tmp_path):
    import cv2

    cv2.imwrite(str(tmp_path / "fist.png"), np.full((64, 64, 3), 200, np.uint8))
    images = overlay.load_legend_images(["fist", "peace"], folder=tmp_path)
    assert set(images) == {"fist"}
    with_image = overlay.Painter(DEMO, px_per_cm=PX_PER_CM, images=images).paint(
        PANEL, update()
    )
    without = painter().paint(PANEL, update())
    assert not np.array_equal(with_image, without)


def test_示范小图读不出来时不报错(tmp_path):
    (tmp_path / "fist.png").write_bytes("这不是一张图".encode())
    assert overlay.load_legend_images(["fist"], folder=tmp_path) == {}


@pytest.mark.parametrize("pose", ["fist", "peace", "like"])
def test_图例要用的示范小图各有一个名字(pose):
    # 裁图脚本按这几个名字存,读的时候按同样的名字找
    names = {p for p, _ in overlay._legend_rows(DEMO)}
    assert pose in names


# ---- 裁图脚本 ----------------------------------------------------------


def grid_source() -> np.ndarray:
    """照着 HaGRID 那张图的版式造一张:每一格涂成一个认得出的颜色。

    格子的位置按脚本里那套比例算,所以这几条验的是**取哪一格**的算术,
    不是那几个比例量得准不准——比例是 2026-09-22 在浏览器里把框叠回原图上核对的。
    """
    from vision_side.scripts import legend_images as li

    width, height = 2910, 1588
    image = np.zeros((height, width, 3), np.uint8)
    for col in range(6):
        for row in range(3):
            x = round((li.GRID_X0 + col * li.COL_STEP) * width)
            y = round((li.GRID_Y0 + row * li.ROW_STEP) * height)
            w, h = round(li.CELL_W * width), round(li.CELL_H * height)
            image[y : y + h, x : x + w] = (col * 40 + 10, row * 80 + 20, 200)
    return image


def test_裁出来的正好是那一格():
    from vision_side.scripts import legend_images as li

    image = grid_source()
    for pose, (col, row) in li.CELLS.items():
        piece = li.crop(image, col, row)
        assert piece.size > 0, pose
        # 一格之内颜色是一致的:裁偏了会掺进旁边那格或者白底
        middle = piece[piece.shape[0] // 2, piece.shape[1] // 2]
        assert tuple(int(v) for v in middle) == (col * 40 + 10, row * 80 + 20, 200), pose


def test_四个姿势各在一格上():
    from vision_side.scripts import legend_images as li

    # 拳头和张开手掌是同一列、上下两行;比 V 在张开手掌右边一格
    assert li.CELLS["fist"] == (2, 0)
    assert li.CELLS["palm"] == (2, 1)
    assert li.CELLS["peace"] == (3, 1)
    assert li.CELLS["like"] == (4, 0)


def test_图例要的每一张裁图脚本都量过():
    from vision_side.scripts import legend_images as li

    for pose in overlay.legend_poses(DEMO):
        assert pose in li.CELLS, pose


def test_裁出来的缩到存的那个大小():
    from vision_side.scripts import legend_images as li

    small = li.fit(np.zeros((800, 400, 3), np.uint8))
    assert max(small.shape[:2]) == li.OUT_PX


# ---- 字高的下限(复查时发现两处偷偷画小了) ---------------------------


def test_每一档字高都不低于下限():
    # 「文字一律英文,字高至少约 1 cm」是票里写死的一条。手框上的姿势标签和图例
    # 底下的出处原来都比 1 cm 矮,所以尺寸统一从 `text_px` 出,这里把口子守住
    p = painter()
    for factor in (0.0, 0.5, 0.7, 1.0, 1.2, 1.4, 2.0):
        assert p.text_px(factor) >= PX_PER_CM * overlay.MIN_CM, factor


def test_画的时候不再直接拿每厘米像素数算字高():
    # 挡住「又有人写了 self._px_cm * 0.7」:画字一律走 text_px
    import inspect

    source = inspect.getsource(overlay.Painter)
    for line in source.splitlines():
        writes = any(call in line for call in ("font_size(", "_put(", "_chip(", "text_width("))
        if writes and "self._px_cm" in line:
            raise AssertionError(f"这一行绕过了 text_px:{line.strip()}")


def test_图例上的手势用的是已定的英文名():
    # CONTEXT.md「Language」定下 FIST MOVE LEFT / HOLD PEACE;屏幕上说的和文档、
    # 日志里说的要是同一个名字
    rows = dict(overlay._legend_rows(DEMO))
    assert rows["fist"].startswith("FIST MOVE ")
    assert rows["peace"].startswith("HOLD PEACE")
    assert rows["like"].startswith("HOLD LIKE")
    # 怎么做括注在后面,两样都写
    assert "(V SIGN)" in rows["peace"]
    assert "(THUMBS UP)" in rows["like"]


def test_送检测区域时角上写的是已定的英文名():
    # CONTEXT.md 里那一块叫 DETECTION REGION
    p = painter()
    canvas = p.paint(PANEL, update())
    assert canvas is not None
    # 只核对常量本身:字画在哪儿靠存图看
    import inspect

    assert "DETECTION REGION" in inspect.getsource(overlay.Painter._corner)


def test_箭头画在做这一下那只手旁边():
    # 触发那一下锁定已经解除,拿「最大的手」会画到旁人手上
    mine = HandResult(box=(600, 300, 680, 380), label="fist", track_id=3, confirmed=True)
    bigger = HandResult(box=(100, 100, 400, 400), label="palm", track_id=9, confirmed=True)
    on_mine = painter().paint(
        PANEL,
        update(hands=(mine, bigger), judge=judging(direction="left", direction_track=3)),
    )
    on_other = painter().paint(
        PANEL,
        update(hands=(mine, bigger), judge=judging(direction="left", direction_track=9)),
    )
    assert not np.array_equal(on_mine, on_other)


# ---- 认出来但没受理时报的那一行 ----------------------------------------


def test_报出来的是手势的英文名和挡下它的状态():
    from robot_side.state import State
    from vision_side.loop import Ignored

    plain = painter().paint(PANEL, update())
    said = painter().paint(
        PANEL, update(ignored=Ignored("fist_move", "left", State.COOLDOWN))
    )
    assert not np.array_equal(plain, said)


def test_手势英文名照CONTEXT定的拼():
    assert overlay.gesture_name("fist_move", "left") == "FIST MOVE LEFT"
    assert overlay.gesture_name("pose_hold", "peace") == "HOLD PEACE"


def test_图例和那一行用的是同一处拼名字():
    # 屏幕上、文档里、日志里说的得是同一个名字。四个握拳平移共用 fist 那张小图,
    # 所以按行比,不按小图的名字比
    texts = [text for _pose, text in overlay._legend_rows(DEMO)]
    for trigger in DEMO.triggers:
        name = overlay.gesture_name(trigger.kind, trigger.key)
        assert any(text.startswith(name) for text in texts), name


def test_那一行用黄的不用红的():
    # 这不是出错,只是来早了
    from robot_side.state import State
    from vision_side.loop import Ignored

    said = painter().paint(
        PANEL, update(ignored=Ignored("pose_hold", "like", State.RUNNING))
    )
    assert (said == np.array(overlay.YELLOW, np.uint8)).all(axis=2).any()


# ---- 录制时的提示 ------------------------------------------------------


def prompt(seg: str):
    """一条握拳平移的提示,箭头从起点区中心朝右画 1.5 个手宽。"""
    from vision_side.prompts import PromptView

    cx, cy = ALIGNMENT.start_zone.center
    reach = ALIGNMENT.hand_width * DEMO.vision.move_hand_widths
    return PromptView(
        headline="FIST MOVE RIGHT (1/3)",
        text="GO - MOVE YOUR FIST TO YOUR RIGHT",
        seg=seg,
        left_s=1.0,
        progress=0.5,
        arrow=((cx, cy), (cx + reach, cy)),
    )


def painted(seg: str):
    return painter().paint(PANEL, update(), prompt=prompt(seg))


def test_准备段和做那一段的箭头长得不一样():
    """两段画成一样的话,那根箭头本身就是「现在照这个做」的信号,人看见就做了。

    2026-09-23 站着录第一遍时,9 次误触发里有 6 次是**正确的方向、只是早做了
    0.26–1.6 秒**,正是这么来的:准备段的箭头和「做!」段的一模一样。
    """
    from vision_side.prompts import GO, PREP

    assert not np.array_equal(painted(PREP), painted(GO))


def test_做那一段的箭头是绿的准备段不是():
    from vision_side.prompts import GO, PREP

    green = (painted(GO) == np.array(overlay.GREEN, np.uint8)).all(axis=2)
    ready = (painted(PREP) == np.array(overlay.GREEN, np.uint8)).all(axis=2)
    assert green.sum() > ready.sum(), "「做!」那一段该比准备段绿得多"


def test_歇一下那段不画箭头():
    """那时不该再有任何「照这个做」的信号。"""
    from vision_side.prompts import REST

    with_arrow = painted(REST)
    no_arrow = painter().paint(
        PANEL, update(), prompt=dataclasses.replace(prompt(REST), arrow=None)
    )
    assert np.array_equal(with_arrow, no_arrow)


def test_准备段的箭头是虚的():
    """虚线在余光里一眼就和实线分得开,而实线那根是「现在做」。"""
    from vision_side.prompts import GO, PREP

    grey = np.array(overlay.GREY, np.uint8)
    ready = (painted(PREP) == grey).all(axis=2)
    assert ready.any(), "准备段该有灰色的箭头"
    # 虚线:沿着箭头那一行,灰点是断开的
    row = ready[ready.any(axis=1).argmax()]
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0])))))
    assert len(runs) > 2, "画成实线了,虚线该断成好几段"


# ---- 调试按键画进图例、画面下面那一条(2026-10-10) ----------------------


def keyed() -> overlay.Painter:
    from robot_side.preview import DEBUG_KEY_NAMES

    return overlay.Painter(DEMO, px_per_cm=PX_PER_CM, keys=DEBUG_KEY_NAMES)


def test_给了调试按键才画进图例():
    # 画进图例的话,底下的按键说明就不再列一遍(原来并排时要占 5 行)
    assert not painter().shows_keys
    assert keyed().shows_keys
    assert keyed().legend_width() > painter().legend_width()
    lines = [line for line in keyed()._lines if line.style == overlay.ROW]
    assert all(line.key for line in lines), "每个触发手势都该有一个调试键"


def test_方向键在键帽上画成箭头():
    from robot_side.preview import DEBUG_KEY_NAMES

    arrows = {name for name in DEBUG_KEY_NAMES.values() if len(name) > 1}
    assert arrows <= set(overlay.ARROW_KEYS), arrows


@pytest.mark.parametrize("debug", [False, True])
def test_画面下面那一条两栏都画了也不越界(debug):
    # 并排时窗口半屏宽(约 784 px,每厘米约 18.5 px):两栏都要放得下,右边不被切掉
    from robot_side.preview import DEBUG_KEY_NAMES

    keys = DEBUG_KEY_NAMES if debug else None
    p = overlay.Painter(DEMO, px_per_cm=PX_PER_CM / 2, keys=keys)
    strip = p.legend_strip(784)
    assert strip.shape[1] == 784 and strip.shape[0] > 0
    ink = (strip != np.array(overlay.SURFACE, np.uint8)).any(axis=2)
    ink[0, :] = False  # 上沿那条分隔线
    assert ink[:, :392].any() and ink[:, 392:].any(), "两栏都该有字"
    assert not ink[:, -4:].any(), "最右边贴边了:有一栏被切掉"

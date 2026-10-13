"""启动对准:从识别结果那个缝进,不开摄像头、不跑模型。

喂编好的逐帧结果,看对准结果对不对——检测区域和起点区放在哪儿、多大,手宽量的是
哪一个。参数用真的演示配置(和 `conftest.py` 一样),所以时长一律按配置里的数算,
改了配置这些测试跟着变,不用改数字。

**送检测区域时区外的手看不见**:真识别器只把送进去的那块交给检测器,这里照着来,
不然「切到检测区域以后再量手宽」就测不出真假。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from robot_side.config import load_demo
from vision_side.align import HOLD_HINT, SEEK_HINT, Aligner
from vision_side.recognize import FrameResult, HandResult, Region

FULL = (1280, 720)
FPS = 30.0

MIDDLE = (640.0, 360.0)
"""画面正中。举在哪里都行,所以多数测试随便挑一个位置。"""

FAR_WIDTH = 86.0
"""1.2 m 处整幅画面里的拳头手框宽(2026-09-17 原型实测的参考值)。"""

NEAR_WIDTH = 76.0
"""同一只拳头在检测区域里量到的宽。整幅画面里量到的大 1–16%,这里取中间。"""

PARAMS = load_demo().vision


def make(*, params=PARAMS, on_switch=None, size=FULL) -> Aligner:
    return Aligner(size, params, on_switch=on_switch)


@dataclass(frozen=True)
class Fist:
    """这一帧真实的拳头:在哪儿、整幅画面里多宽、检测区域里多宽、什么标签。

    `seen()` 给出识别器送某一块画面时会报上来的那只手——这就是两处宽度分得开的地方。
    """

    center: tuple[float, float] = MIDDLE
    full_width: float = FAR_WIDTH
    region_width: float = NEAR_WIDTH
    label: str = "fist"
    track_id: int = 1
    confirmed: bool = True

    def seen(self, region: Region | None) -> HandResult | None:
        width = self.full_width
        if region is not None:
            cx, cy = self.center
            if not (region.x <= cx <= region.right and region.y <= cy <= region.bottom):
                return None  # 区外的手,检测器根本看不到
            width = self.region_width
        cx, cy = self.center
        half = width / 2
        return HandResult(
            box=(round(cx - half), round(cy - half), round(cx + half), round(cy + half)),
            label=self.label,
            track_id=self.track_id,
            confirmed=self.confirmed,
        )


def steady(seconds: float, **kwargs) -> list[Fist | None]:
    """一段举着不动的拳头。`None` 那几项是这一帧没认出手。"""
    return [Fist(**kwargs)] * round(seconds * FPS)


def drive(aligner: Aligner, script, *, start_t: float = 0.0):
    """把编好的一串帧从识别结果那个缝喂进去,对上了就返回对准结果。

    每一帧送哪块画面由对准自己说了算(`aligner.send`),和真程序里一样。
    """
    for i, item in enumerate(script):
        t = start_t + i / FPS
        region = aligner.send
        hands = ()
        if item is not None:
            hand = item.seen(region)
            if hand is not None:
                hands = (hand,)
        got = aligner.update(FrameResult(t=t, hands=hands, region=region))
        if got is not None:
            return got
    return None


def align_at(center=MIDDLE, *, extra: float = 0.5, aligner=None, **kwargs):
    """对准一次,返回对准结果。喂够时长再多喂一点,免得差在取整上。"""
    aligner = aligner if aligner is not None else make()
    seconds = PARAMS.hold_s + PARAMS.measure_s + extra
    return drive(aligner, steady(seconds, center=center, **kwargs))


def center_of(region: Region) -> tuple[float, float]:
    return region.center


# ---- 停稳 2 秒就放下区域 -----------------------------------------------


def test_拳头停稳两秒就放下两个区域():
    got = align_at()
    assert got is not None
    # 「停稳约 2 秒」:总时长还在 2 秒这一档
    assert got.t == pytest.approx(PARAMS.hold_s + PARAMS.measure_s, abs=0.1)
    assert (
        got.detection_region.width,
        got.detection_region.height,
    ) == PARAMS.detection_region_px
    assert center_of(got.detection_region) == pytest.approx(MIDDLE, abs=1.0)
    assert center_of(got.start_zone) == pytest.approx(MIDDLE, abs=1.0)


def test_还没停够就不算():
    aligner = make()
    short = steady(PARAMS.hold_s + PARAMS.measure_s - 0.3)
    assert drive(aligner, short) is None


def test_举在哪里都行():
    left_up = align_at((360.0, 200.0))
    right_down = align_at((900.0, 480.0))
    assert center_of(left_up.start_zone) == pytest.approx((360.0, 200.0), abs=1.0)
    assert center_of(right_down.start_zone) == pytest.approx((900.0, 480.0), abs=1.0)


def test_一直没有拳头就一直对不上():
    aligner = make()
    assert drive(aligner, [None] * 200) is None
    assert aligner.send is None  # 还在整幅画面里找


def test_张开的手不算拳头():
    assert align_at(label="palm") is None


# ---- 中途动了就不算 ----------------------------------------------------


def test_中途移动超过半个手宽就不算():
    aligner = make()
    moved_by = FAR_WIDTH * PARAMS.still_hand_widths + 10
    script = steady(PARAMS.hold_s - 0.2) + steady(
        PARAMS.measure_s + 0.2, center=(MIDDLE[0] + moved_by, MIDDLE[1])
    )
    assert drive(aligner, script) is None


def test_动了以后从新位置重新对得上():
    aligner = make()
    moved_to = (MIDDLE[0] + FAR_WIDTH * PARAMS.still_hand_widths + 10, MIDDLE[1])
    script = steady(PARAMS.hold_s - 0.2) + steady(
        PARAMS.hold_s + PARAMS.measure_s + 0.5, center=moved_to
    )
    got = drive(aligner, script)
    assert got is not None
    assert center_of(got.start_zone) == pytest.approx(moved_to, abs=2.0)


def test_小幅晃动不打断():
    aligner = make()
    # 来回晃:离停稳那一刻的位置最远也没到半个手宽
    wobble = NEAR_WIDTH * PARAMS.still_hand_widths / 2 - 4
    script = []
    for i in range(round((PARAMS.hold_s + PARAMS.measure_s + 0.5) * FPS)):
        dx = wobble if i % 2 else -wobble
        script.append(Fist(center=(MIDDLE[0] + dx, MIDDLE[1])))
    assert drive(aligner, script) is not None


def test_切到检测区域以后动了也重来():
    aligner = make()
    moved_to = (MIDDLE[0] + NEAR_WIDTH * PARAMS.still_hand_widths + 10, MIDDLE[1])
    script = steady(PARAMS.hold_s + 0.2) + steady(PARAMS.measure_s, center=moved_to)
    assert drive(aligner, script) is None
    assert aligner.send is None  # 退回整幅画面重新找


# ---- 八成帧要是握拳类 --------------------------------------------------


def test_偶尔漏一帧还算数():
    aligner = make()
    script = steady(PARAMS.hold_s + PARAMS.measure_s + 0.5)
    for i in range(0, len(script), 10):  # 一成的帧没认出手
        script[i] = None
    assert drive(aligner, script) is not None


def test_一半的帧没认出手就不算():
    aligner = make()
    script = steady(PARAMS.hold_s + PARAMS.measure_s + 0.5)
    for i in range(0, len(script), 2):
        script[i] = None
    assert drive(aligner, script) is None


def until_switch(aligner: Aligner) -> float:
    """一直喂拳头,直到切去检测区域那一帧;返回再下一帧的时间戳。

    切过去以后的头几帧是「跟踪记录刚清掉」那一段,几条规矩都卡在这里,所以要正好
    停在那一刻,不能靠喂满多少秒去凑。
    """
    t = 0.0
    while aligner.send is None:
        assert t < 5.0, "一直没切到检测区域"
        hand = Fist().seen(None)
        aligner.update(FrameResult(t=t, hands=(hand,), region=None))
        t += 1 / FPS
    return t


def test_切过去那几帧没认出手不算在量手宽里():
    # 上游的跟踪记录刚清过,同一只手要连上几帧才重新算「已确认」。这几帧要是算进
    # 「八成帧」里,帧率一掉就会把好好举着的手判成不算,所以从手重新出现那一帧起算。
    aligner = make()
    t = until_switch(aligner)
    blind = [None] * round(PARAMS.measure_s * 0.5 * FPS)
    got = drive(aligner, blind + steady(PARAMS.measure_s + 0.5), start_t=t)
    assert got is not None
    assert center_of(got.start_zone) == pytest.approx(MIDDLE, abs=1.0)


def test_切过去以后一直没再认出手就重来():
    aligner = make()
    t = until_switch(aligner)
    blind = [None] * round((PARAMS.measure_s + 0.3) * FPS)
    assert drive(aligner, blind, start_t=t) is None
    assert aligner.send is None  # 退回整幅画面
    assert aligner.hint == SEEK_HINT  # 提示也跳回「举起拳头」


def test_等手重新出现那几帧提示不跳回去():
    aligner = make()
    t = until_switch(aligner)
    before = aligner.progress
    drive(aligner, [None] * 3, start_t=t)
    assert aligner.hint == HOLD_HINT  # 还在「别动」这一档
    assert aligner.progress >= before


# ---- 检测区域超出画面就贴边 --------------------------------------------


def test_检测区域超出左上角就贴边():
    got = align_at((60.0, 50.0))
    assert (got.detection_region.x, got.detection_region.y) == (0, 0)
    assert (
        got.detection_region.width,
        got.detection_region.height,
    ) == PARAMS.detection_region_px


def test_检测区域超出右下角就贴边():
    got = align_at((1200.0, 660.0))
    width, height = PARAMS.detection_region_px
    assert (got.detection_region.x, got.detection_region.y) == (
        FULL[0] - width,
        FULL[1] - height,
    )
    assert (got.detection_region.width, got.detection_region.height) == (width, height)


def test_贴边以后起点区还在拳头那儿():
    got = align_at((60.0, 50.0))
    # 检测区域贴了边,起点区仍以拳头为中心;它比检测区域小得多,还在里面
    assert got.start_zone.x >= got.detection_region.x
    assert got.start_zone.y >= got.detection_region.y
    assert got.start_zone.right <= got.detection_region.right


# ---- 手宽 --------------------------------------------------------------


def test_手宽量的是检测区域里的那个():
    got = align_at(full_width=96.0, region_width=84.0)
    assert got.hand_width == pytest.approx(84.0, abs=1.0)
    assert got.full_frame_width == pytest.approx(96.0, abs=1.0)


def test_手宽取检测区域里的中位数():
    aligner = make()
    widths = [70.0, 76.0, 82.0]
    script = [
        Fist(region_width=widths[i % 3])
        for i in range(round((PARAMS.hold_s + PARAMS.measure_s + 0.5) * FPS))
    ]
    got = drive(aligner, script)
    assert got.hand_width == pytest.approx(76.0, abs=1.0)


def test_起点区边长是三个手宽():
    got = align_at(region_width=80.0)
    side = PARAMS.start_zone_hand_widths * got.hand_width
    assert got.start_zone.width == pytest.approx(side, abs=2.0)
    assert got.start_zone.height == got.start_zone.width


def test_对上以后手宽不再变():
    aligner = make()
    got = align_at(aligner=aligner)
    # 手一下子大了一圈(走近了),对准结果不跟着变:锁定的解除依赖手宽固定
    again = drive(aligner, steady(1.0, region_width=200.0), start_t=10.0)
    assert again is got
    assert again.hand_width == got.hand_width
    assert aligner.done is got


# ---- 站太近 ------------------------------------------------------------


def test_站太近照样对得上只是提示():
    got = align_at(full_width=PARAMS.too_close_px + 20, region_width=110.0)
    assert got is not None
    assert got.too_close
    assert got.full_frame_width == pytest.approx(PARAMS.too_close_px + 20, abs=1.0)


def test_站在操作位上不提示太近():
    got = align_at()
    assert not got.too_close


def test_还没对上就已经提示太近():
    aligner = make()
    # 只喂到一半:提示得在对上之前就出现,不然退一步的话来不及
    drive(aligner, steady(PARAMS.hold_s - 0.2, full_width=PARAMS.too_close_px + 20))
    assert aligner.too_close


def test_太近的门槛按整幅画面里的宽算():
    # 检测区域里量到的比门槛还宽,但整幅画面里没超,就不算太近
    got = align_at(full_width=FAR_WIDTH, region_width=PARAMS.too_close_px + 20)
    assert not got.too_close


# ---- 换送法时清跟踪记录 ------------------------------------------------


def test_切到检测区域时清一次跟踪记录():
    switched: list[Region | None] = []
    aligner = make(on_switch=switched.append)
    got = align_at(aligner=aligner)
    assert switched == [got.detection_region]


def test_退回整幅画面时也清一次():
    switched: list[Region | None] = []
    aligner = make(on_switch=switched.append)
    moved_to = (MIDDLE[0] + NEAR_WIDTH * PARAMS.still_hand_widths + 10, MIDDLE[1])
    script = steady(PARAMS.hold_s + 0.2) + steady(PARAMS.measure_s, center=moved_to)
    drive(aligner, script)
    assert len(switched) == 2
    assert switched[-1] is None


def test_没动过送法就不清():
    switched: list[Region | None] = []
    aligner = make(on_switch=switched.append)
    drive(aligner, [None] * 60)
    assert switched == []


# ---- 屏幕上的提示 ------------------------------------------------------


def test_没看见拳头时提示怎么站怎么举():
    aligner = make()
    drive(aligner, [None] * 10)
    assert aligner.hint == SEEK_HINT
    assert "FIST" in SEEK_HINT  # 怎么举拳写在这一行里


def test_停稳的时候提示别动():
    aligner = make()
    drive(aligner, steady(0.3))
    assert aligner.hint == HOLD_HINT
    assert aligner.progress > 0.0


def test_对上以后进度满格():
    aligner = make()
    align_at(aligner=aligner)
    assert aligner.progress == 1.0


# ---- 重新对准 ----------------------------------------------------------


def test_重新对准从头来():
    aligner = make()
    got = align_at(aligner=aligner)
    aligner.reset()
    assert aligner.done is None
    assert aligner.send is None
    assert aligner.progress == 0.0
    again = align_at((900.0, 480.0), aligner=aligner)
    assert again is not got
    assert center_of(again.start_zone) == pytest.approx((900.0, 480.0), abs=1.0)


def test_重新对准也会清跟踪记录():
    switched: list[Region | None] = []
    aligner = make(on_switch=switched.append)
    align_at(aligner=aligner)
    aligner.reset()
    assert switched[-1] is None


# ---- 写进日志的那一行 --------------------------------------------------


def test_对准结果写得出一行日志():
    got = align_at()
    line = got.describe()
    for piece in ("DETECTION", "START ZONE", "HAND WIDTH", "FULL FRAME"):
        assert piece in line
    assert str(got.detection_region.x) in line
    assert f"{got.hand_width:.1f}" in line


def test_太近的时候日志里也写明():
    got = align_at(full_width=PARAMS.too_close_px + 20, region_width=110.0)
    assert "TOO CLOSE" in got.describe()


# ---- 两处宽度差多少 ----------------------------------------------------


def test_两处宽度差多少算得出来():
    got = align_at(full_width=100.0, region_width=90.0)
    # 整幅画面里量到的比检测区域里大 1–16%;差多少每次都不一样,不能折算,只记下来
    assert got.width_gap_percent == pytest.approx(11.1, abs=0.5)


# ---- 配置 --------------------------------------------------------------


def test_配置里的初值就是定下来的那几个():
    assert PARAMS.detection_region_px == (800, 600)
    assert PARAMS.start_zone_hand_widths == 3.0
    assert PARAMS.still_hand_widths == 0.5
    assert PARAMS.too_close_px == pytest.approx(107.0)
    assert PARAMS.hold_s + PARAMS.measure_s == pytest.approx(2.0, abs=0.2)


def test_参数改得动():
    params = replace(PARAMS, start_zone_hand_widths=5.0)
    aligner = Aligner(FULL, params)
    got = drive(
        aligner, steady(params.hold_s + params.measure_s + 0.5, region_width=80.0)
    )
    assert got.start_zone.width == pytest.approx(5.0 * got.hand_width, abs=2.0)


# ---- 跑一遍的那个脚本画得出来 ------------------------------------------
#
# 画错了(坐标是小数、区域超出画面)要在这里就撞出来,不能等站到摄像头前面才发现。


def screen(aligner, result=None):
    """按对准现在的样子画一帧,返回画好的图。"""
    import numpy as np

    from vision_side.scripts import align as script

    image = np.zeros((FULL[1], FULL[0], 3), np.uint8)
    if result is None:
        item = Fist()
        hand = item.seen(aligner.send)
        result = FrameResult(
            t=0.0, hands=() if hand is None else (hand,), region=aligner.send
        )
    script.draw(image, result, aligner, fps=29.5)
    return image


def test_还在找拳头的时候画得出来():
    aligner = make()
    drive(aligner, [None] * 5)
    assert screen(aligner, FrameResult(t=0.0)).any()


def test_停稳和量手宽的时候都画得出来():
    aligner = make()
    drive(aligner, steady(PARAMS.hold_s - 0.2))
    assert screen(aligner).any()
    drive(aligner, steady(0.4), start_t=PARAMS.hold_s)
    assert aligner.send is not None
    assert screen(aligner).any()


def test_对上以后两个边框都画得出来():
    aligner = make()
    align_at(aligner=aligner)
    assert screen(aligner).any()


def test_贴着边的区域也画得出来():
    aligner = make()
    align_at((60.0, 50.0), aligner=aligner)
    assert screen(aligner).any()


def test_站太近那一行也画得出来():
    aligner = make()
    align_at(aligner=aligner, full_width=PARAMS.too_close_px + 20, region_width=110.0)
    assert aligner.too_close
    assert screen(aligner).any()


# ---- 不够就延长,以及几处不该把人赶回去从头来的 ------------------------


def test_量手宽那一段漏检多就延长着量():
    # 手好好举着、只是切过去以后漏检了一阵:再给一倍时间接着量,不该赶回去重停 2 秒
    switched: list[Region | None] = []
    aligner = make(on_switch=switched.append)
    t = until_switch(aligner)
    script = [Fist()] + [None] * round(0.3 * FPS) + steady(2.0)
    got = drive(aligner, script, start_t=t)
    assert got is not None
    assert got.t - t > PARAMS.measure_s  # 确实延长了
    # 只切过去那一次:中间没退回整幅画面从头停,人不用重举一遍
    assert switched == [got.detection_region]


def test_延到头还不够才算这一次不算():
    aligner = make()
    t = until_switch(aligner)
    # 一帧有一帧没有,怎么延都到不了八成
    flaky = [Fist() if i % 2 else None for i in range(round(2.0 * FPS))]
    # `measure_s` 刚到点时还不能退:漏检一阵的人会白举一次
    just_over = round(PARAMS.measure_s * 1.1 * FPS)
    assert drive(aligner, flaky[:just_over], start_t=t) is None
    assert aligner.send is not None  # 还在检测区域里延长着量
    assert drive(aligner, flaky[just_over:], start_t=t + just_over / FPS) is None
    assert aligner.send is None  # 延满了才退回整幅画面


def test_站太近那一行不会一晃就灭():
    # 站太近是「人站在哪儿」的事,不是「这一次对准」的事
    aligner = make()
    too_wide = PARAMS.too_close_px + 20
    drive(aligner, steady(PARAMS.hold_s - 0.2, full_width=too_wide))
    assert aligner.too_close
    far = (MIDDLE[0] + too_wide, MIDDLE[1])
    drive(aligner, steady(0.2, center=far, full_width=too_wide), start_t=PARAMS.hold_s)
    assert aligner.send is None  # 动得太多,这一次不算
    assert aligner.too_close  # 但提示还亮着:人还没退那一步


def test_手框退化成零宽也不会把识别线程带倒():
    # 手框偶尔会退化成 0 宽;边长算成 0 的话 Region 会抛异常
    aligner = make()
    got = align_at(aligner=aligner, region_width=0.0)
    assert got is not None
    assert got.start_zone.width >= 1
    assert got.hand_width == 0.0  # 离谱的手宽一看日志就知道

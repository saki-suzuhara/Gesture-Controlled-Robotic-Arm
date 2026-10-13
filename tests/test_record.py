"""录制工具:提示走得对不对、什么时候才开始写、示范图的箭头画在哪儿(第 05 票)。

不开摄像头、不弹窗口、不跑模型:画面和识别结果都是编好的(`tests/vision_fakes.py`),
预览窗口换成一个只记账的假窗口。

最后一条是整条跑一遍:对准 → 按提示录 → 落盘 → 出表。跑得慢一点(画面按真实的
30 fps 出),因为要验的正是「显示和读帧识别分在不同线程,提示还能按时切过去」。
"""

from __future__ import annotations


import time

import numpy as np
import pytest


from robot_side.config import FIST_MOVE, load_demo
from vision_side.align import Alignment
from vision_side.frames import SCRIPTED_STEP_S, Frame
from vision_side.prompts import (
    GO,
    PREP,
    REST,
    Plan,
    Rep,
    Segment,
    Timetable,
)
from vision_side.record import Recorder, Schedule, demo_for
from vision_side.recognize import FrameResult, Region
from vision_side.recordings import DETECTION_REGION, Meta, Run, RunWriter
from tests.vision_fakes import ScriptedFrameSource, ScriptedRecognizer, hand

DEMO = load_demo()

ALIGNMENT = Alignment(
    detection_region=Region(240, 60, 800, 600),
    start_zone=Region(520, 240, 240, 240),
    hand_width=80.0,
    full_frame_width=88.0,
    too_close=False,
    t=1.5,
)


def tiny_plan() -> Plan:
    """一条提示,三段各几分之一秒。整条跑一遍时用它,不然要等 8 秒。"""
    rep = Rep(
        index=0,
        kind=FIST_MOVE,
        key="left",
        expect="left",
        label="FIST MOVE LEFT (1/1)",
        segments=(
            Segment(PREP, 0.3, "GET READY - MAKE A FIST, HOLD IT STILL"),
            Segment(GO, 0.3, "GO - MOVE YOUR FIST TO YOUR LEFT"),
            Segment(REST, 0.2, "REST - LOWER YOUR HAND"),
        ),
    )
    return Plan("tiny", (rep,))


class Beeps:
    """记下响过哪几声。真 `Beeper` 在没有 winsound 的机器上是静音的,数不出来。"""

    def __init__(self) -> None:
        self.sounds: list[str] = []

    def go(self) -> None:
        self.sounds.append("go")

    def aligned(self) -> None:
        self.sounds.append("aligned")


# ---- 提示走到哪儿了 ----------------------------------------------------


def test_开头留一段让人走回操作位():
    schedule = Schedule(tiny_plan(), lead_s=1.0)
    view = schedule.update(0.0)
    assert view is not None and view.seg == ""
    assert view.left_s == pytest.approx(1.0)
    assert len(schedule.timetable) == 0, "还没开始提示,时间表上不该有行"


def test_按计划一段一段往前走():
    schedule = Schedule(tiny_plan(), lead_s=0.0)
    schedule.start(0.0)
    assert schedule.update(0.1).seg == PREP
    assert schedule.update(0.4).seg == GO
    assert schedule.update(0.7).seg == REST
    assert schedule.update(0.9) is None, "走完了就该收工"
    assert schedule.done


def test_时间表记的是画面上真正的时刻():
    """显示卡一下,提示就该跟着晚,而时间表记下的必须是真发生的那一刻。"""
    schedule = Schedule(tiny_plan(), lead_s=0.0)
    schedule.start(100.0)
    schedule.update(100.0)
    schedule.update(100.45)  # 卡了一下,到这一刻才看见「做!」
    slots = schedule.timetable.slots
    assert slots[0].seg == PREP and slots[0].t0 == 100.0
    assert slots[1].seg == GO and slots[1].t0 == 100.45


def test_做那一段开头响一声高音():
    beeps = Beeps()
    schedule = Schedule(tiny_plan(), lead_s=0.0)
    schedule.start(0.0)
    for t in (0.1, 0.2, 0.4, 0.5, 0.55):
        schedule.update(t, beeps)
    assert beeps.sounds == ["go"], "同一段只该响一次"


def test_整整跨过一段也要在时间表上留一行():
    """主循环卡得比一段还久时会跨过去。跨过的要是「做!」那一段,这一条提示就一行
    `go` 都没有,窗口只好退回拿准备段的开头当「做!」——准备段里冒出来的那一次会被
    算成**认出**,识别率凭空变高。"""
    schedule = Schedule(tiny_plan(), lead_s=0.0)
    schedule.start(0.0)
    schedule.update(0.0)  # 准备段
    schedule.update(0.65)  # 卡了 0.65 秒:整个「做!」段跨过去了
    segs = [s.seg for s in schedule.timetable.slots]
    assert GO in segs, "跨过去的那一段也要留一行"
    window = schedule.timetable.reps()[0]
    assert window.go > window.t0, "「做!」不该退回准备段的开头"


def test_走完以后时间表能划出窗口来():
    schedule = Schedule(tiny_plan(), lead_s=0.0)
    schedule.start(0.0)
    for t in (0.0, 0.31, 0.61, 0.85):
        schedule.update(t)
    window = schedule.timetable.reps()[0]
    assert window.t0 == 0.0
    assert window.go == pytest.approx(0.31), "「做!」从准备段之后算起"


# ---- 什么时候才开始写 --------------------------------------------------


def writer(tmp_path) -> RunWriter:
    return RunWriter(
        tmp_path / "run",
        Meta(
            plan="tiny",
            started="2026-09-23T10:00:00",
            frame_size=(1280, 720),
            send=DETECTION_REGION,
        ),
        video=False,
    )


def feed(recorder: Recorder, n: int, *, t0: float = 0.0) -> None:
    image = np.zeros((4, 4, 3), np.uint8)
    for i in range(n):
        recorder.recognize(Frame(image, t0 + i * SCRIPTED_STEP_S, i), ALIGNMENT.detection_region)


def test_没开录之前一帧都不写(tmp_path):
    """启动检查和启动对准都在录制开始之前,那几百帧不该进数据。"""
    inner = ScriptedRecognizer([FrameResult(t=0.0, hands=(hand((600, 360)),))])
    recorder = Recorder(inner, writer(tmp_path))
    feed(recorder, 10)
    assert recorder.written == 0


def test_开录以后每帧都写(tmp_path):
    inner = ScriptedRecognizer([FrameResult(t=0.0, hands=(hand((600, 360)),))])
    out = writer(tmp_path)
    recorder = Recorder(inner, out)
    feed(recorder, 5)  # 对准那一段,不写
    recorder.arm()
    feed(recorder, 4, t0=10.0)
    out.finish(Timetable())
    records = Run(tmp_path / "run").frames()
    assert recorder.written == 4
    assert len(records) == 4


def test_时间戳从录的第一帧算零(tmp_path):
    """提示时间表和逐帧日志要同一个起点,回放才对得上。"""
    inner = ScriptedRecognizer([FrameResult(t=0.0, hands=())])
    out = writer(tmp_path)
    recorder = Recorder(inner, out)
    recorder.arm()
    feed(recorder, 3, t0=123.0)
    out.finish(Timetable())
    # 逐帧日志里的时间戳存到小数点后 4 位(整 30 fps 下够分得开每一帧)
    assert [r.t for r in Run(tmp_path / "run").frames()] == [
        pytest.approx(0.0),
        pytest.approx(SCRIPTED_STEP_S, abs=1e-4),
        pytest.approx(2 * SCRIPTED_STEP_S, abs=1e-4),
    ]


def test_没接上要写的那一份时开录也不写():
    """取景检查那一档:一帧也不会写下来(第 40 条 user story)。"""
    recorder = Recorder(ScriptedRecognizer([FrameResult(t=0.0)]), None)
    recorder.arm()
    feed(recorder, 5)
    assert not recorder.armed and recorder.written == 0


def test_结果照样往上交(tmp_path):
    """套在识别器外面这一层不该改结果——录的时候走的就是正式程序那一条路。"""
    want = FrameResult(t=0.0, hands=(hand((600, 360), label="peace"),))
    recorder = Recorder(ScriptedRecognizer([want]), writer(tmp_path))
    got = recorder.recognize(Frame(np.zeros((4, 4, 3), np.uint8), 0.0, 0), None)
    assert got.hands == want.hands


# ---- 示范图和箭头 ------------------------------------------------------


def images() -> dict:
    return {name: np.zeros((20, 20, 3), np.uint8) for name in ("fist", "palm", "peace")}


def test_箭头的起点是起点区中心():
    """判定要求停住段落在起点区里,所以手本来就该从那儿出发(spec 第 1 条)。"""
    rep = tiny_plan().reps[0]
    _image, arrow = demo_for(rep, ALIGNMENT, DEMO, images())
    assert arrow is not None
    start, _end = arrow
    assert start == ALIGNMENT.start_zone.center


def test_箭头的终点正好是判定要求的那么远():
    """照着箭头做出来的那一下,正好够触发,不多不少。"""
    rep = tiny_plan().reps[0]
    _image, arrow = demo_for(rep, ALIGNMENT, DEMO, images())
    (sx, sy), (ex, ey) = arrow
    want = ALIGNMENT.hand_width * DEMO.vision.move_hand_widths
    assert (sx - ex) == pytest.approx(want), "向左就该正好朝画面左边移这么远"
    assert ey == pytest.approx(sy), "只朝一个方向,另一轴不该动"


def test_四个方向的箭头各朝各的():
    ends = {}
    for direction in ("left", "right", "up", "down"):
        rep = Rep(0, FIST_MOVE, direction, direction, "", ())
        _image, arrow = demo_for(rep, ALIGNMENT, DEMO, images())
        ends[direction] = arrow[1]
    cx, cy = ALIGNMENT.start_zone.center
    assert ends["left"][0] < cx < ends["right"][0]
    assert ends["up"][1] < cy < ends["down"][1]


def test_姿势保持没有位移就不画箭头():
    rep = Rep(0, "pose_hold", "peace", "peace", "", ())
    image, arrow = demo_for(rep, ALIGNMENT, DEMO, images())
    assert arrow is None
    assert image is not None, "姿势还是要有示范图"


def test_叫停那一组配的是张开手掌那张():
    pictures = images()
    pictures["palm"][:] = 7  # 标记一下,好认出拿到的是不是这一张
    rep = Rep(0, "stop_gesture", "palm", "stop", "", ())
    image, arrow = demo_for(rep, ALIGNMENT, DEMO, pictures)
    assert image is pictures["palm"]
    assert arrow is None, "叫停没有位移,不该画箭头"


def test_没有示范图时照样跑得起来():
    """图只存本地、不提交,别人拉下代码是没有的(`vision_side.overlay`)。"""
    rep = tiny_plan().reps[0]
    image, arrow = demo_for(rep, ALIGNMENT, DEMO, {})
    assert image is None
    assert arrow is not None, "没有图,箭头照画"


# ---- 整条跑一遍 --------------------------------------------------------


class FakeWindow:
    """只记账的预览窗口。不弹窗,所以能在自动测试里跑。"""

    px_per_cm = 38.0
    panel_size = (960, 420)
    legend_height = 1000

    def __init__(self) -> None:
        self.views: list = []
        self.images: list = []
        self.legend = None
        self.closed = False

    def set_legend(self, legend) -> None:
        # 和真的预览窗口一样:右侧整列,给高度让图例自己画
        self.legend = legend.legend_image(self.legend_height)

    def update(self, view, image=None) -> None:
        self.views.append(view)
        if image is not None:
            self.images.append(image)

    def close(self) -> None:
        self.closed = True


class PacedFrames(ScriptedFrameSource):
    """按差不多真实的速度出画面。

    整条跑一遍时要的正是「显示和读帧识别在两条线程上,提示还能按时切过去」;
    画面一下子全出来的话,主线程一次就跳过好几段,验不到这件事。`speed` 是比真实
    快几倍——测试不必真等 30 fps。
    """

    def __init__(self, *, speed: float = 4.0, image=None) -> None:
        super().__init__(image=image)
        self._nap = SCRIPTED_STEP_S / speed

    def _read(self):
        time.sleep(self._nap)
        return super()._read()


def quick_demo():
    """把对准那两段时长压小,不然一条测试要等 2 秒对准。"""
    import dataclasses

    vision = dataclasses.replace(DEMO.vision, hold_s=0.2, measure_s=0.2)
    return dataclasses.replace(DEMO, vision=vision)


def session(tmp_path, *, plan, source=None, recognizer=None, demo=None, **kwargs):
    import logging

    from vision_side.record import RecordSession

    log = logging.getLogger(f"test.record.{id(tmp_path)}")
    log.handlers = []
    log.propagate = False
    demo = demo if demo is not None else quick_demo()
    if recognizer is None:
        # 一只拳头,举在画面正中不动:对准过得去,录的时候也不会触发什么
        recognizer = ScriptedRecognizer(
            [FrameResult(t=0.0, hands=(hand((640, 360), width=80.0),))]
        )
    return RecordSession(
        source if source is not None else PacedFrames(),
        recognizer,
        demo,
        log,
        plan=plan,
        folder=tmp_path,
        video=False,
        table_s=0.0,
        window=FakeWindow(),
        beeps=Beeps(),
        **kwargs,
    )



def test_录一遍落盘出表(tmp_path):
    """对准 → 按提示录 → 视频以外的三样都落盘 → 立刻出表(spec 第 3、6、7 条)。"""
    run = session(tmp_path, plan=tiny_plan())
    assert run.run_session() == 0
    assert run.run is not None

    got = run.run
    assert got.complete(), "逐帧日志、提示时间表、meta 三样都该在"
    records = got.frames()
    assert records, "一帧都没写下来"
    assert got.timetable().slots, "提示时间表是空的"

    # **录制前也做启动对准,把区域记进日志**(第 3 条)
    meta = got.meta()
    assert meta.alignment_source == "recorded"
    alignment = got.alignment()
    assert alignment.hand_width > 0
    assert meta.send == DETECTION_REGION, "空闲送检测区域,和正式程序一样"

    # 录完立刻出表:用的是刚写下去的那份日志,所以屏幕上看到的就是往后要比的基准
    assert (got.path / "replay-log.json").exists()



def test_录下来的每一帧都带间隔(tmp_path):
    """上一轮录制循环卡过 6 次;这一次卡不卡,靠的就是这一列(spec 最后一条)。"""
    run = session(tmp_path, plan=tiny_plan())
    assert run.run_session() == 0
    records = run.run.frames()
    assert records[0].interval_ms is None
    assert all(r.interval_ms is not None for r in records[1:])
    assert run.run.meta().health, "循环卡没卡要写进 meta"



def test_时间表和逐帧日志同一条时间轴(tmp_path):
    """回放拿时间表当标准答案,按帧的时间戳去里面查这一刻该做什么。

    两边差一截——比如时间表从摄像头开机算、日志从录的第一帧算 0——每一条的窗口都会
    错位,算出来的数全不作数。启动对准占掉的那一两秒尤其容易漏掉。
    """
    run = session(tmp_path, plan=tiny_plan())
    assert run.run_session() == 0
    records = run.run.frames()
    slots = run.run.timetable().slots
    assert records[0].t == pytest.approx(0.0), "逐帧日志从录的第一帧算 0"
    assert slots[0].t0 >= 0.0, "时间表不该从负数开始"
    first, last = records[0].t, records[-1].t
    assert first <= slots[0].t0 <= last, "第一段提示得落在录下来的那一段里"
    assert first <= slots[-1].t1 <= last + 0.5, "最后一段提示也得落在录下来的那一段里"



def test_遇到摄像头占位图拦住开始(tmp_path):
    """隐私开关关着时不让白录一遍(第 41 条 user story)。"""
    from vision_side.frames import FrameError

    from tests import images as pictures

    source = PacedFrames(image=pictures.placeholder())
    recognizer = ScriptedRecognizer([FrameResult(t=0.0)], real_flat=True)
    run = session(tmp_path, plan=tiny_plan(), source=source, recognizer=recognizer)
    with pytest.raises(FrameError) as caught:
        run.run_session()
    assert caught.value.reason == "CAMERA BLANK"
    assert run.run is None
    assert not list(tmp_path.iterdir()), "一个文件夹都不该建出来"



def test_取景检查一帧都不写(tmp_path):
    """只显示画面和识别结果,不录像(第 40 条 user story)。"""
    run = session(tmp_path, plan=None, seconds=1.0)
    assert run.run_session() == 0
    assert run.run is None
    assert not list(tmp_path.iterdir()), "取景检查不该留下任何文件"


def test_图例放在右侧整列不压在画面里(tmp_path):
    # 和正式程序一样:画在画面那块里只有中间一条那么高,十几行放不下
    run = session(tmp_path, plan=None, seconds=1.0)
    assert run.run_session() == 0
    window = run._window
    assert window.legend is not None
    assert window.legend.shape[0] == FakeWindow.legend_height
    # 画面那块里不再画图例:右边那条不是图例的底色
    from vision_side.overlay import SURFACE

    assert window.images
    assert all(image[210, -3].tolist() != list(SURFACE) for image in window.images)

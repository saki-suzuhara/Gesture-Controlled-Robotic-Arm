"""录下来的那一份:写出去、读回来,以及逐帧日志那一路回放(第 05 票)。

不开摄像头、不跑模型、不写视频。看的是**存下去读回来一模一样**——回放测试要求
「同一段数据、同样的代码,结果应当逐条一致」,那就得先保证数据本身一进一出不走样。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from robot_side.config import load_demo
from vision_side.align import Alignment
from vision_side.frames import Frame
from vision_side.prompts import Timetable
from vision_side.recognize import FrameResult, HandResult, Region
from vision_side.recordings import (
    DETECTION_REGION,
    FrameRecord,
    LogFrames,
    LogRecognizer,
    Meta,
    Run,
    RunWriter,
    alignment_from_json,
    alignment_to_json,
    find_runs,
    read_frames,
    run_dir,
)

DEMO = load_demo()

ALIGNMENT = Alignment(
    detection_region=Region(240, 60, 800, 600),
    start_zone=Region(520, 240, 240, 240),
    hand_width=80.0,
    full_frame_width=88.0,
    too_close=False,
    t=1.5,
)


def hand(track_id: int = 1, label: str = "fist") -> HandResult:
    return HandResult(box=(100, 120, 180, 200), label=label, track_id=track_id, confirmed=True)


def meta(**kwargs) -> Meta:
    base = dict(
        plan="dirs",
        started="2026-09-23T10:00:00",
        frame_size=(1280, 720),
        send=DETECTION_REGION,
        alignment=alignment_to_json(ALIGNMENT),
    )
    base.update(kwargs)
    return Meta(**base)


def write_run(path, frames: int = 5, *, table: Timetable | None = None) -> Run:
    """写一份能回放的录制出来。不写视频(这里不测编码)。"""
    writer = RunWriter(path, meta(), video=False)
    for i in range(frames):
        image = np.zeros((4, 4, 3), np.uint8)
        writer.write(
            Frame(image, i / 30.0, i),
            FrameResult(
                t=i / 30.0,
                hands=(hand(),),
                flat=False,
                region=ALIGNMENT.detection_region,
            ),
        )
    writer.finish(table if table is not None else Timetable())
    return Run(path)


# ---- 一进一出 ----------------------------------------------------------


def test_逐帧日志存下去读回来一模一样(tmp_path):
    run = write_run(tmp_path / "run")
    records = run.frames()
    assert len(records) == 5
    assert records[0].hands == (hand(),)
    assert records[0].region == ALIGNMENT.detection_region
    assert [r.index for r in records] == [0, 1, 2, 3, 4]


def test_姿势标签存的是名字不是上游的下标(tmp_path):
    """上游 `targets` 的次序一改,存下标的老数据就全错位了,而且错得不声不响。"""
    run = write_run(tmp_path / "run", frames=1)
    row = json.loads((run.path / "frames.jsonl").read_text(encoding="utf-8").strip())
    assert row["hands"][0][1] == "fist"


def test_记下每帧的间隔(tmp_path):
    """上一轮录制循环卡过 6 次 0.3–1.4 秒,这一票要能看出还卡不卡(spec 最后一条)。"""
    run = write_run(tmp_path / "run")
    records = run.frames()
    assert records[0].interval_ms is None, "第一帧没有上一帧,间隔该是空的"
    assert records[1].interval_ms == pytest.approx(1000.0 / 30.0, abs=0.2)


def test_对准结果存下去读回来一模一样(tmp_path):
    run = write_run(tmp_path / "run")
    again = run.alignment()
    assert again == ALIGNMENT


def test_meta里记着录的时候送的是哪块画面(tmp_path):
    run = write_run(tmp_path / "run")
    assert run.meta().send == DETECTION_REGION


def test_写了一半的最后一行不会把整份带倒(tmp_path):
    """录到一半断电时,最后一行可能只写了半截。前面的照用。"""
    run = write_run(tmp_path / "run")
    path = run.path / "frames.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"i":5,"t":0.1', encoding="utf-8")
    assert len(read_frames(path)) == 5


def test_少了文件的那一份不算能回放(tmp_path):
    run = write_run(tmp_path / "run")
    (run.path / "timetable.json").unlink()
    assert not run.complete()
    assert find_runs(tmp_path) == []


def test_本机没有这一层时是空的不报错(tmp_path):
    """数据不在版本库里,本机没有是常事——**测试据此跳过,不算失败**(spec)。"""
    assert find_runs(tmp_path / "没有这一层") == []


def test_文件夹按时间和计划起名():
    from datetime import datetime

    path = run_dir("dirs", folder=tmp_root(), now=datetime(2026, 9, 23, 10, 30, 0))
    assert path.name == "20260923-103000-dirs"


def tmp_root():
    from pathlib import Path

    return Path("data") / "recordings"


# ---- 逐帧日志那一路回放 ------------------------------------------------


def records(n: int = 4) -> list[FrameRecord]:
    return [
        FrameRecord(index=i, t=i / 30.0, hands=(hand(),), region=Region(10, 10, 100, 100))
        for i in range(n)
    ]


def test_逐帧日志当一路画面放():
    source = LogFrames(records(3))
    source.open()
    got = [source.next_frame() for _ in range(3)]
    assert [f.t for f in got] == [0.0, pytest.approx(1 / 30), pytest.approx(2 / 30)]
    assert not source.has_next()
    assert source.next_frame() is None


def test_日志回放交出来的就是录的时候那一份():
    rows = records(2)
    recognizer = LogRecognizer(rows)
    source = LogFrames(rows)
    source.open()
    frame = source.next_frame()
    got = recognizer.recognize(frame, rows[0].region)
    assert got.hands == rows[0].hands
    assert got.region == rows[0].region
    assert recognizer.mismatch == 0


def test_要的那块画面和录的时候对不上会记一笔():
    """那一帧的手框当初就是在录的时候那块画面上出的,换一块去套等于凭空编数。"""
    rows = records(1)
    recognizer = LogRecognizer(rows)
    source = LogFrames(rows)
    source.open()
    got = recognizer.recognize(source.next_frame(), None)
    assert recognizer.mismatch == 1
    assert got.region == rows[0].region, "照录的时候那一份算,不按要的那块"


def test_送哪块画面逐帧照录的时候来():
    """正式程序认出触发手势以后不等主线程就切整幅画面约 1 秒,录的时候走的也是那一套。

    所以一次录制里送的画面是会变的,拿 `meta.send` 一个值套整段等于拿另一块画面的
    结果去喂判定(第 05 票复查)。
    """
    from vision_side.replay import RegionTrack

    rows = [
        FrameRecord(index=0, t=0.0, region=Region(10, 10, 100, 100)),
        FrameRecord(index=1, t=1.0, region=None),  # 触发之后切整幅画面
        FrameRecord(index=2, t=2.0, region=Region(10, 10, 100, 100)),
    ]
    track = RegionTrack(rows)
    assert track.at(0.0) == Region(10, 10, 100, 100)
    assert track.at(1.0) is None
    assert track.at(1.5) is None, "时间戳对不上就沿用上一帧那块"
    assert track.at(2.0) == Region(10, 10, 100, 100)


def test_日志回放不清跟踪记录():
    """跟踪编号是录的时候定的,这里清不了,也不该清。"""
    recognizer = LogRecognizer(records(1))
    recognizer.clear_tracks()  # 不报错就对了


def test_中间少一帧也接得上():
    """视频那一路偶尔会少一帧,按时间找得回来。"""
    rows = records(4)
    recognizer = LogRecognizer(rows)
    image = np.zeros((1, 1, 3), np.uint8)
    assert recognizer.recognize(Frame(image, rows[0].t, 0), rows[0].region).hands
    got = recognizer.recognize(Frame(image, rows[2].t, 2), rows[2].region)
    assert got.t == pytest.approx(rows[2].t)


# ---- meta ---------------------------------------------------------------


def test_meta存下去读回来一模一样():
    one = meta(video_t0=11.6, alignment_source="derived", note="转过来的")
    again = Meta.from_json(one.to_json())
    assert again == one
    assert alignment_from_json(again.alignment) == ALIGNMENT


def test_没记对准结果的那一份回放时说清楚(tmp_path):
    path = tmp_path / "run"
    writer = RunWriter(path, meta(alignment=None), video=False)
    writer.finish(Timetable())
    with pytest.raises(Exception, match="没记下启动对准"):
        Run(path).alignment()

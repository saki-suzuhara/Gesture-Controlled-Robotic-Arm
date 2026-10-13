"""识别器本身:三处修补还在,喂一小段录像出得来手框和标签。

**没有模型或者没有录像时跳过,不算失败**(spec「Testing Decisions」)。录像放在
`data/recordings/smoke/video.mp4`——约 3 秒、画面里有一只拳头,从 2026-09-17 握拳平移
原型那批里截的;`data/` 被 git 忽略,所以这一条在别人的机器上默认跳过。

一句说明白它测不到什么:**线程自旋那一条这里验不出来**。自旋饿住的是摄像头读帧线程,
喂录像时根本没有那条线程。这里只能盯住「修补还在源码里」和「每帧的识别时间还在 30 fps
的预算内」,真的帧率要用 `vision_side.scripts.view` 对着摄像头看(第 01 票最后一条)。
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest

from vision_side.dynamic_gestures.utils.enums import targets
from vision_side.frames import ScriptedFrames, VideoFrames
from vision_side.gestures import GestureRecognizer, ModelsMissing
from vision_side.recognize import UNKNOWN_LABEL, Region

from tests import images

CLIP = Path("data/recordings/smoke/video.mp4")

FRAME_BUDGET_MS = 1000.0 / 30.0
"""每帧的识别时间预算:摄像头是 30 fps,超了就跟不上取帧。本机实测 p50 5–10 ms。"""

VENDOR = Path("vision_side/dynamic_gestures")


# ---- 三处修补还在(不要模型)------------------------------------------


def test_修补一还在_不再把数组转成标量():
    # numpy >= 2.4 起 `w / float(h)` 会崩(h 是形状 (1,) 的数组)
    source = (VENDOR / "ocsort/kalmanfilter.py").read_text(encoding="utf-8")
    assert "r = w / h" in source
    # 注释里还留着上游原来那一行,所以只看代码行
    code = [ln.strip() for ln in source.splitlines() if not ln.strip().startswith("#")]
    assert not [ln for ln in code if "w / float(h)" in ln]


def test_修补二还在_关掉线程自旋():
    source = (VENDOR / "onnx_models.py").read_text(encoding="utf-8")
    assert 'add_session_config_entry("session.intra_op.allow_spinning", "0")' in source
    # 上游在 HandDetection 里又建了一遍 session,把这份设置丢掉了;那一遍已经删掉
    assert source.count("ort.InferenceSession(") == 1


def test_修补三还在_空的_provider_options_不传():
    source = (VENDOR / "onnx_models.py").read_text(encoding="utf-8")
    assert 'extra = {"provider_options": prov_opts} if prov_opts else {}' in source


def test_上游的许可声明跟着代码一起在():
    assert (VENDOR / "license/LICENSE-2.0.txt").exists()


# ---- 喂录像(要模型、要录像)------------------------------------------


@pytest.fixture(scope="module")
def recognizer():
    try:
        return GestureRecognizer()
    except ModelsMissing as exc:
        pytest.skip(f"{exc}")


@pytest.fixture(scope="module")
def clip():
    if not CLIP.exists():
        pytest.skip(f"没有 {CLIP},跳过喂录像那几条")
    return CLIP


@pytest.fixture(scope="module")
def replay(recognizer, clip):
    """把整段录像喂一遍,记下每帧的结果和识别耗时。"""
    results, costs = [], []
    with VideoFrames(clip) as source:
        while source.has_next():
            frame = source.next_frame()
            if frame is None:
                break
            t0 = time.perf_counter()
            results.append(recognizer.recognize(frame))
            costs.append((time.perf_counter() - t0) * 1000.0)
    recognizer.clear_tracks()
    return results, costs


def test_录像里认得出手(replay):
    results, _ = replay
    hands = [h for r in results for h in r.hands]
    assert hands, "整段录像一只手都没认出来"
    assert any(h.confirmed for h in hands)


def test_每只手都带标签手框和跟踪编号(replay):
    results, _ = replay
    for result in results:
        for hand in result.hands:
            assert hand.label in targets or hand.label == UNKNOWN_LABEL
            x1, y1, x2, y2 = hand.box
            assert 0 <= x1 < x2 <= 1280
            assert 0 <= y1 < y2 <= 720
            assert hand.track_id > 0


def test_一只手在一段里是同一个跟踪编号(replay):
    results, _ = replay
    ids = [h.track_id for r in results for h in r.hands if h.confirmed]
    # 录像里只有一只手,偶尔跟丢重开也不该冒出一串编号
    assert len(set(ids)) <= 3, sorted(set(ids))


def test_每帧的时间戳照录像里的位置往前走(replay):
    results, _ = replay
    times = [r.t for r in results]
    assert times == sorted(times)
    assert times[0] == 0.0


def test_识别时间在三十帧每秒的预算里(replay):
    _, costs = replay
    # 第一帧带着模型预热(本机约 330 ms),不算它
    rest = costs[1:]
    assert statistics.median(rest) < FRAME_BUDGET_MS


def test_送检测区域时手框还是整幅画面的坐标(recognizer, clip):
    region = Region(440, 60, 800, 600)
    recognizer.clear_tracks()
    seen = 0
    with VideoFrames(clip) as source:
        while source.has_next():
            frame = source.next_frame()
            if frame is None:
                break
            for hand in recognizer.recognize(frame, region).hands:
                x1, y1, x2, y2 = hand.box
                # 少加那一次偏移的话,手框会落在画面左上角、掉出检测区域
                assert region.x <= x1 < x2 <= region.right
                assert region.y <= y1 < y2 <= region.bottom
                seen += 1
    recognizer.clear_tracks()
    assert seen, "送检测区域时一只手都没认出来"


def test_清掉跟踪记录之后是新的编号(recognizer, clip):
    def ids_of(frames_at_most: int) -> set[int]:
        got = set()
        with VideoFrames(clip) as source:
            for _ in range(frames_at_most):
                if not source.has_next():
                    break
                frame = source.next_frame()
                if frame is None:
                    break
                got |= {h.track_id for h in recognizer.recognize(frame).hands}
        return got

    recognizer.clear_tracks()
    before = ids_of(40)
    recognizer.clear_tracks()
    after = ids_of(40)
    recognizer.clear_tracks()
    assert before and after
    assert min(after) > max(before)


def test_识别器把均匀画面报上来(recognizer):
    # 「画面均匀」那一位由识别器算,这里确认它真的报上来了(规则本身见 test_flat.py)
    got = []
    with ScriptedFrames(images.run("gray", 30)) as source:
        while True:
            frame = source.next_frame()
            if frame is None:
                break
            got.append(recognizer.recognize(frame).flat)
    recognizer.clear_tracks()
    assert not got[0]  # 第一帧只是「这一帧灰」
    assert got[-1]  # 连着一秒之后算

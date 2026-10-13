"""录下来的一次录制:存在哪儿、里面有什么、怎么读回来(spec「录制工具」)。

一次录制是 `data/recordings/<时间>-<计划>/` 下的一个文件夹:

    video.mp4       画面。**存的是翻转之前的原始画面**(见 `RunWriter._write_video`)
    frames.jsonl    逐帧日志:手框、姿势标签、跟踪编号、时间,外加**每帧的间隔**
    timetable.json  提示时间表(`vision_side.prompts.Timetable`)
    meta.json       启动对准的结果、判定参数、版本号、循环卡没卡

**`data/` 整个被 git 忽略:录像里有用户本人,只存本地,不提交、不分发**
(spec「分支与提交」)。

这一层同时是回放测试的**入口**:两种输入各有一套取帧 / 识别结果的实现,所以回放
和正式程序走的是同一个缝、同一套判定(spec「两个缝的形状」)。

- **视频**(`VideoFrames` + 真识别器):换模型、换分辨率、改预处理之后用,慢但真。
- **逐帧日志**(`LogFrames` + `LogRecognizer`):只改判定规则时用,快得多——不载模型、
  不解码画面。**送哪块画面照录的时候原样来**:那一帧的识别结果本来就是在那块画面上
  出的,拿另一块去套等于凭空编数。

几条定下来的规矩:

- **姿势标签存名字,不存上游的下标**:上游 `targets` 的次序一改,存下标的老数据就
  全错位了,而且错得不声不响。
- **时间戳从录制循环的第一帧算 0**,和提示时间表同一个起点;视频那一路也是(取帧
  那一层拿的是画面在视频里的位置,`vision_side.frames.VideoFrames`)。
- **每帧的间隔单独记一列**(`iv`,毫秒):上一轮录制循环卡过 6 次 0.3–1.4 秒,这一票
  要确认分了线程之后还卡不卡,靠的就是它(spec「录制工具」最后一条)。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from robot_side.config import ROOT
from vision_side.align import Alignment
from vision_side.frames import Frame, FrameError, FrameSource
from vision_side.prompts import Timetable
from vision_side.recognize import FrameResult, HandResult, Recognizer, Region

RECORDINGS_DIR = ROOT / "data" / "recordings"
"""录下来的放这里。`data/` 整个被 git 忽略(录像里有用户本人)。"""

VIDEO_NAME = "video.mp4"
FRAMES_NAME = "frames.jsonl"
TIMETABLE_NAME = "timetable.json"
META_NAME = "meta.json"

VIDEO_CODEC = "mp4v"
"""写视频用的编码。和 2026-09-17 原型同一个,本机不用另装东西。"""

FULL_FRAME = "full"
DETECTION_REGION = "region"
"""录的时候送哪块画面。空闲送检测区域,执行中送整幅画面(ADR-0008);录叫停那一组
时机械臂本来就该在动,所以那一组按整幅画面录。"""


def run_dir(plan: str, *, folder: Path = RECORDINGS_DIR, now: datetime | None = None) -> Path:
    """这一次录制放哪个文件夹。名字和日志同一个写法,好对着看。"""
    stamp = (now if now is not None else datetime.now()).strftime("%Y%m%d-%H%M%S")
    return folder / f"{stamp}-{plan}"


# ---- 逐帧日志里的一行 --------------------------------------------------


@dataclass(frozen=True)
class FrameRecord:
    """逐帧日志里的一行。

    `hands` 是这一帧**真的看见了**的手(和 `FrameResult.hands` 一样,跟丢的不算),
    手框一律是**整幅画面的坐标**。`region` 是这一帧送进检测器的那块,`None` 是整幅
    画面。`interval_ms` 是离上一帧多久——查卡顿就看它。`video_index` 是这一帧在
    `video.mp4` 里的第几帧,没录像时是 `None`。
    """

    index: int
    t: float
    hands: tuple[HandResult, ...] = ()
    flat: bool = False
    region: Region | None = None
    interval_ms: float | None = None
    video_index: int | None = None

    def to_json(self) -> dict:
        return {
            "i": self.index,
            "t": round(self.t, 4),
            "iv": None if self.interval_ms is None else round(self.interval_ms, 1),
            "vi": self.video_index,
            "flat": self.flat,
            "region": None
            if self.region is None
            else [self.region.x, self.region.y, self.region.width, self.region.height],
            # 一只手一行:编号、标签、手框、确认没有。存标签的**名字**,不存下标
            "hands": [
                [h.track_id, h.label, *h.box, int(h.confirmed)] for h in self.hands
            ],
        }

    @staticmethod
    def from_json(raw: dict) -> FrameRecord:
        region = raw.get("region")
        return FrameRecord(
            index=int(raw["i"]),
            t=float(raw["t"]),
            hands=tuple(
                HandResult(
                    box=(int(h[2]), int(h[3]), int(h[4]), int(h[5])),
                    label=str(h[1]),
                    track_id=int(h[0]),
                    confirmed=bool(h[6]),
                )
                for h in raw.get("hands", ())
            ),
            flat=bool(raw.get("flat", False)),
            region=None if region is None else Region(*(int(v) for v in region)),
            interval_ms=None if raw.get("iv") is None else float(raw["iv"]),
            video_index=None if raw.get("vi") is None else int(raw["vi"]),
        )

    def to_result(self) -> FrameResult:
        """还原成识别结果那个缝上的东西。"""
        return FrameResult(t=self.t, hands=self.hands, flat=self.flat, region=self.region)


def read_frames(path: Path) -> list[FrameRecord]:
    """读一份逐帧日志。空行跳过(录到一半断电时最后一行可能只写了半截)。"""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(FrameRecord.from_json(json.loads(line)))
        except (ValueError, KeyError, IndexError, TypeError):
            continue  # 最后一行写了半截:丢掉它,前面的照用
    return out


# ---- 一次录制 ----------------------------------------------------------


@dataclass
class Meta:
    """`meta.json` 里存的东西。

    `alignment` 是录制前那一次启动对准的结果:**回放按它重跑**,数据和正式程序的画面
    范围才一致(spec「录制工具」第 3 条)。`alignment_source` 分两档——`recorded` 是
    录的时候真对过一次,`derived` 是从老数据倒推出来的(2026-09-17 那一批录的时候
    还没有启动对准这一步,见 `vision_side.scripts.import_runs`)。
    """

    plan: str
    started: str
    frame_size: tuple[int, int]
    send: str = FULL_FRAME
    video_t0: float = 0.0
    """`video.mp4` 第 0 帧对应逐帧日志里的哪一刻。

    自己录的一律是 0(录像和日志同时开始)。转过来的老数据不是:那几轮都是**对准
    做完才开始录像**,所以视频的 0 秒对应日志里的十几秒。回放视频时要把它加回去,
    不然算出来的时刻和提示时间表差着一截,逐条全对不上。
    """
    alignment: dict | None = None
    alignment_source: str = "recorded"
    vision: dict = field(default_factory=dict)
    """录的时候用的那一份 `[vision]` 判定参数。事后要知道这批数是拿什么参数录的。"""
    versions: dict = field(default_factory=dict)
    health: dict = field(default_factory=dict)
    """循环卡没卡(`vision_side.loop.LoopHealth` 的数)。"""
    note: str = ""

    def to_json(self) -> dict:
        return {
            "plan": self.plan,
            "started": self.started,
            "frame_size": list(self.frame_size),
            "send": self.send,
            "video_t0": round(self.video_t0, 3),
            "alignment": self.alignment,
            "alignment_source": self.alignment_source,
            "vision": self.vision,
            "versions": self.versions,
            "health": self.health,
            "note": self.note,
        }

    @staticmethod
    def from_json(raw: dict) -> Meta:
        size = raw.get("frame_size") or [1280, 720]
        return Meta(
            plan=str(raw.get("plan", "")),
            started=str(raw.get("started", "")),
            frame_size=(int(size[0]), int(size[1])),
            send=str(raw.get("send", FULL_FRAME)),
            video_t0=float(raw.get("video_t0", 0.0)),
            alignment=raw.get("alignment"),
            alignment_source=str(raw.get("alignment_source", "recorded")),
            vision=raw.get("vision") or {},
            versions=raw.get("versions") or {},
            health=raw.get("health") or {},
            note=str(raw.get("note", "")),
        )


def alignment_to_json(alignment: Alignment) -> dict:
    return {
        "detection_region": list(_region_json(alignment.detection_region)),
        "start_zone": list(_region_json(alignment.start_zone)),
        "hand_width": round(alignment.hand_width, 2),
        "full_frame_width": round(alignment.full_frame_width, 2),
        "too_close": alignment.too_close,
        "t": round(alignment.t, 3),
    }


def alignment_from_json(raw: dict) -> Alignment:
    return Alignment(
        detection_region=Region(*(int(v) for v in raw["detection_region"])),
        start_zone=Region(*(int(v) for v in raw["start_zone"])),
        hand_width=float(raw["hand_width"]),
        full_frame_width=float(raw["full_frame_width"]),
        too_close=bool(raw.get("too_close", False)),
        t=float(raw.get("t", 0.0)),
    )


def _region_json(region: Region) -> tuple[int, int, int, int]:
    return region.x, region.y, region.width, region.height


class Run:
    """磁盘上的一次录制。读的一侧。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def video(self) -> Path:
        return self.path / VIDEO_NAME

    @property
    def has_video(self) -> bool:
        return self.video.exists()

    def meta(self) -> Meta:
        path = self.path / META_NAME
        if not path.exists():
            raise FrameError(f"{self.path.name} 里没有 {META_NAME}", reason="RUN INCOMPLETE")
        return Meta.from_json(json.loads(path.read_text(encoding="utf-8")))

    def timetable(self) -> Timetable:
        return Timetable.load(self.path / TIMETABLE_NAME)

    def frames(self) -> list[FrameRecord]:
        return read_frames(self.path / FRAMES_NAME)

    def alignment(self) -> Alignment:
        """录制前那一次启动对准的结果。回放按它重跑。"""
        meta = self.meta()
        if meta.alignment is None:
            raise FrameError(
                f"{self.path.name} 没记下启动对准的结果,回放没有检测区域可用",
                reason="NO ALIGNMENT",
            )
        return alignment_from_json(meta.alignment)

    def complete(self) -> bool:
        """这一份够不够回放。视频可以没有(只改判定规则时用不着)。"""
        return all(
            (self.path / n).exists() for n in (FRAMES_NAME, TIMETABLE_NAME, META_NAME)
        )


def find_runs(folder: Path = RECORDINGS_DIR) -> list[Run]:
    """本机有哪些录制,按名字排。没有这一层就是空的——**测试据此跳过,不算失败**
    (spec「回放测试」最后一条)。"""
    if not folder.exists():
        return []
    return [Run(p) for p in sorted(folder.iterdir()) if p.is_dir() and Run(p).complete()]


# ---- 写的一侧 ----------------------------------------------------------


class RunWriter:
    """把一次录制写到磁盘上。

    **写视频的活在这里做,不在显示那条线程上**:`cv2.VideoWriter.write` 是同步的,
    1280×720 一帧几毫秒,搁在显示那边会把提示的切换时刻带偏,而提示时间表正是回放
    的标准答案。

    **写的是识别线程,收尾的是主线程**,所以两边用一把锁隔开,而且**关掉以后再来的
    帧直接丢掉**:主线程说「录完了」的那一刻,识别线程手上可能正好有一帧还没写完,
    没有这一道就会在已经关掉的文件上写,把整次录制带倒(一次录十分钟,不能栽在
    最后这一帧上)。

    视频可以不录(`video=False`):只改判定规则时用不上,不录省一大半磁盘。
    """

    def __init__(
        self,
        path: Path,
        meta: Meta,
        *,
        video: bool = True,
        fps: float = 30.0,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._meta = meta
        self._want_video = video
        self._fps = fps
        self._writer = None
        self._frames = (self.path / FRAMES_NAME).open("w", encoding="utf-8")
        self._count = 0
        self._last_t: float | None = None
        self._lock = threading.Lock()
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._count

    @property
    def closed(self) -> bool:
        return self._closed

    def write(self, frame: Frame, result: FrameResult) -> FrameRecord | None:
        """记下一帧。返回记下去的那一行;**已经收尾了就返回 `None`,什么也不写。**"""
        with self._lock:
            if self._closed:
                return None
            video_index = self._write_video(frame.image)
            interval = (
                None if self._last_t is None else (frame.t - self._last_t) * 1000.0
            )
            self._last_t = frame.t
            record = FrameRecord(
                index=self._count,
                t=frame.t,
                hands=result.hands,
                flat=result.flat,
                region=result.region,
                interval_ms=interval,
                video_index=video_index,
            )
            self._frames.write(
                json.dumps(record.to_json(), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            self._count += 1
            return record

    def finish(self, timetable: Timetable, *, health: dict | None = None) -> None:
        """收尾:提示时间表和 `meta.json` 落盘。主线程调。"""
        if health is not None:
            self._meta.health = health
        self.close()
        timetable.save(self.path / TIMETABLE_NAME)
        (self.path / META_NAME).write_text(
            json.dumps(self._meta.to_json(), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    def close(self) -> None:
        """关掉。之后再来的帧直接丢掉,不报错。关过了再关一次也不出错。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._frames is not None and not self._frames.closed:
                self._frames.close()
            if self._writer is not None:
                self._writer.release()
                self._writer = None

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _write_video(self, image: np.ndarray) -> int | None:
        """写一帧到 `video.mp4`。**存的是水平翻转之前的原始画面。**

        取帧那一层一进来就翻(`vision_side.frames.FrameSource.next_frame`),回放时
        也会翻——`VideoFrames` 和摄像头走的是同一条路。所以这里要存翻**之前**的那一幅:
        存翻过的,回放再翻一次就翻回去了,手框坐标和逐帧日志全对不上。为此多翻一次
        (1280×720 约 1 ms),换来的是「录像喂回去,识别看到的和当时一模一样」。

        2026-09-17 原型存的也是翻转之前的,所以那一批录像转过来直接copy 就能用。
        """
        if not self._want_video:
            return None
        import cv2

        if self._writer is None:
            height, width = image.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path / VIDEO_NAME),
                cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                self._fps,
                (width, height),
            )
            if not self._writer.isOpened():
                raise FrameError(
                    f"写不了 {self.path / VIDEO_NAME}(装了 {VIDEO_CODEC} 编码吗)",
                    reason="VIDEO NOT WRITABLE",
                )
        index = self._count
        self._writer.write(cv2.flip(image, 1))  # 翻回去:存的是原始画面(见上)
        return index


# ---- 逐帧日志这一路(回放用) ------------------------------------------


class LogFrames(FrameSource):
    """把逐帧日志当成一路画面。**没有像素**——只改判定规则时用不上。

    `next_frame()` 交出去的画面是一幅 1×1 的黑点:识别那一侧换成了
    `LogRecognizer`,谁也不会去看它。要看画面就走视频那一路。
    """

    def __init__(self, records: Sequence[FrameRecord]) -> None:
        super().__init__()
        self.records = list(records)
        self._pos = 0
        self._blank = np.zeros((1, 1, 3), np.uint8)

    def _open(self) -> None:
        self._pos = 0

    def _close(self) -> None:
        pass

    def _read(self) -> tuple[np.ndarray, float] | None:
        if self._pos >= len(self.records):
            return None
        record = self.records[self._pos]
        self._pos += 1
        return self._blank, record.t

    def has_next(self) -> bool:
        return self.is_open and self._pos < len(self.records)


class LogRecognizer(Recognizer):
    """从逐帧日志里读结果,不载模型、不解码画面(spec「回放测试」第 1 条)。

    **送哪块画面照录的时候原样来。**调这个方法的人照样按状态传 `region` 进来,但
    这一帧的手框当初就是在录的时候那块画面上出的,换一块去套等于凭空编数——所以
    传进来的那块只用来核对,对不上就记一笔(`mismatch`),由回放测试报出来。
    """

    def __init__(self, records: Sequence[FrameRecord]) -> None:
        self.records = list(records)
        self.mismatch = 0
        """要的那块画面和录的时候那块对不上的帧数。"""
        self._at = 0

    def recognize(self, frame: Frame, region: Region | None = None) -> FrameResult:
        record = self._next(frame.t)
        if record is None:
            return FrameResult(t=frame.t, region=region)
        if region != record.region:
            self.mismatch += 1
        return record.to_result()

    def clear_tracks(self) -> None:
        """日志里的跟踪编号是录的时候定的,这里清不了,也不该清。"""

    def _next(self, t: float) -> FrameRecord | None:
        """按次序往下取。时间戳对不上就按时间找——视频那一路偶尔会少一帧。"""
        if self._at < len(self.records):
            record = self.records[self._at]
            if abs(record.t - t) < 1e-6:
                self._at += 1
                return record
        for i in range(self._at, len(self.records)):
            if abs(self.records[i].t - t) < 1e-6:
                self._at = i + 1
                return self.records[i]
        return None

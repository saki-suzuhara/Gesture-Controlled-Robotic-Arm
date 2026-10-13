"""回放测试:拿录下来的数据重跑一遍判定,算出几个数,和上一次逐项比(spec「回放测试」)。

改了判定规则以后,光看下一次录像的感觉说不清是好了还是坏了——那些参数本来就是在同一
批数据上调出来的,偏乐观。所以:同一批数据、同一张提示时间表,跑出来的数逐条对比,
**只要有一项变差就报出来,不设容差**(同一段视频、同样的代码,结果应当逐条一致)。

两种输入,走的是同一套判定:

- **视频**:重新跑识别(`VideoFrames` + 真识别器)。换模型、换分辨率、改预处理之后用。
- **逐帧日志**:只喂结果,不跑模型(`LogFrames` + `LogRecognizer`)。只改判定规则时用,快。

计数口径(票里写死的四条,`tests/test_replay.py` 一条一条盯着):

1. 一条提示的窗口**从准备段算起**(操作者常在准备段就摆好了),但**判做成没做成
   只看「做!」那一截**;准备段里冒出来的算误触发。
2. **两次误停间隔超过 1 秒才算两次**——一次叫停会连着认到好几帧,那是同一次。
3. **同一种触发 3 秒内只计一次**。提示之间隔 8 秒,所以这一条并不会把真做成的那次
   和下一条的并到一起。
4. **移动组只看窗口里的第一次触发**,之后到下一条准备段之前的相当于执行中,不计。

这一层不碰摄像头、不碰磁盘:`score()` 吃的是时间表和一串认出来的东西,所以整条能
直接测。跑一次回放是 `run_replay()` 的事。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from robot_side.config import FIST_MOVE, POSE_HOLD, DemoConfig, VisionParams
from vision_side.frames import Frame, FrameSource
from vision_side.prompts import FREE, STOP_GESTURE, Timetable
from vision_side.recognize import Recognizer
from vision_side.recordings import (
    DETECTION_REGION,
    LogFrames,
    LogRecognizer,
    Run,
)
from vision_side.triggers import TriggerJudge

SAME_TRIGGER_S = 3.0
"""同一种触发隔这么久之内只计一次(spec「回放测试」计数口径)。"""

FALSE_STOP_GAP_S = 1.0
"""两次误停隔这么久以上才算两次。"""

STOP = "stop"
"""叫停在 `Fired.kind` 里的写法。它不是触发手势,单独一档。"""

#: 一条提示的结局。
HIT = "hit"
"""认出来了,而且是该认的那一个。"""
MISS = "miss"
"""窗口里一次都没认出来。"""
DIRECTION = "direction"
"""认成握拳平移了,但方向不对。"""
WRONG = "wrong"
"""认成另一类手势了(该做握拳平移,认出来一个姿势保持)。"""
STOP_HIT = "stop_hit"
STOP_MISS = "stop_miss"
"""叫停那一组:停住了 / 没停住。"""
NONE = "none"
"""自由活动那一组没有结局可言——它只贡献误触发和误停。"""


@dataclass(frozen=True)
class Fired:
    """回放时认出来的一件事:一个触发手势,或者一次叫停。"""

    t: float
    kind: str
    key: str

    def describe(self) -> str:
        if self.kind == STOP:
            return f"{self.t:7.2f}s  叫停"
        return f"{self.t:7.2f}s  {self.kind} {self.key}"


@dataclass(frozen=True)
class RepOutcome:
    """一条提示的结局,连同落在它身上的那几件。"""

    index: int
    kind: str
    key: str
    expect: str | None
    label: str
    verdict: str
    got: Fired | None
    """判这一次用的那一件(窗口里的第一次)。没有就是 `None`。"""
    prep: tuple[Fired, ...] = ()
    """准备段里冒出来的。算误触发。"""
    later: tuple[Fired, ...] = ()
    """窗口里第一次之后的。相当于执行中,不计。"""
    stray: tuple[Fired, ...] = ()
    """这一条的窗口里、不该有的那些(自由活动那一组的全部触发)。算误触发。"""
    latency: float | None = None
    """叫停那一组:从「做!」到真停住隔了多久(秒)。没停住就是 `None`。

    只看停没停住不够:`three2` 那件事的说法是「**晚了 0.45 秒**,没漏停」
    (spec「停止手势与叫停」),多认一个标签救回来的正是这段时间,次数上看不出来。
    """

    def line(self) -> str:
        """屏幕上和结果文件里的那一行。"""
        mark = {
            HIT: "认出",
            MISS: "漏认",
            DIRECTION: "方向错",
            WRONG: "认错",
            STOP_HIT: "停住",
            STOP_MISS: "漏停",
            NONE: "-",
        }[self.verdict]
        got = "" if self.got is None else f"  → {self.got.kind} {self.got.key}"
        extra = []
        if self.prep:
            extra.append(f"准备段 {len(self.prep)}")
        if self.later:
            extra.append(f"多余 {len(self.later)}")
        if self.stray:
            extra.append(f"误触发 {len(self.stray)}")
        tail = ("  " + "、".join(extra)) if extra else ""
        return f"#{self.index:3d} {self.label:28s} {mark:4s}{got}{tail}"


@dataclass(frozen=True)
class Change:
    """和上一次比,某一项变了。`worse` 是这个方向算不算变差。"""

    name: str
    before: float
    now: float
    worse: bool

    def line(self) -> str:
        arrow = "变差" if self.worse else "变好"
        return f"{self.name}:{_num(self.before)} → {_num(self.now)}({arrow})"


def _num(value: float) -> str:
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.3f}"


#: 每一项的名字,和「大了算好还是算坏」。逐项比就按这张表走。
#: 名字进结果文件,改了名就对不上上一次的结果,所以不要随手改。
HIGHER_IS_BETTER = {
    "认出": True,
    "识别率": True,
    "漏认": False,
    "方向判错": False,
    "认错": False,
    "误触发": False,
    "叫停成功": True,
    "叫停成功率": True,
    "漏停": False,
    "误叫停": False,
}


@dataclass
class Report:
    """一次回放的结果。逐项的数、逐条的结局,外加跑的时候的几笔。"""

    run: str
    mode: str
    """`log`(逐帧日志)还是 `video`(重跑识别)。"""
    reps: list[RepOutcome] = field(default_factory=list)
    fired: list[Fired] = field(default_factory=list)
    """去重之后的那一串。原样的在 `raw_fired` 里。"""
    raw_fired: list[Fired] = field(default_factory=list)
    stops: list[float] = field(default_factory=list)
    merged: int = 0
    """被「3 秒内只计一次」并掉了几件。"""
    frames: int = 0
    note: str = ""
    intervals: dict = field(default_factory=dict)
    """每帧的间隔:p50 / p99 / 最长 / 卡过几次(毫秒)。逐帧日志那一路才有。

    上一轮录制循环卡过 6 次 0.3–1.4 秒,不在读帧和识别,只剩显示和等按键
    (spec「录制工具」最后一条)。这一票把显示和读帧识别分到了两条线程,还卡不卡
    就看这一行。**不进逐项比**:它说的是录的时候机器忙不忙,和判定规则没关系。
    """

    # ---- 逐项的数 ------------------------------------------------------

    @property
    def gesture_reps(self) -> list[RepOutcome]:
        """真要做出一个手势的那些条。识别率的分母。

        不含叫停组、自由活动,也不含 `expect` 是空的那些——那几条要的是「什么也
        别认出来」(2026-09-17 原型的自然抬手组),拿它们当识别率的分母会把率算低。
        """
        return [
            r
            for r in self.reps
            if r.kind in (FIST_MOVE, POSE_HOLD) and r.expect is not None
        ]

    @property
    def stop_reps(self) -> list[RepOutcome]:
        return [r for r in self.reps if r.kind == STOP_GESTURE]

    @property
    def hits(self) -> int:
        return sum(r.verdict == HIT for r in self.reps)

    @property
    def misses(self) -> int:
        return sum(r.verdict == MISS for r in self.reps)

    @property
    def direction_errors(self) -> int:
        return sum(r.verdict == DIRECTION for r in self.reps)

    @property
    def wrong(self) -> int:
        return sum(r.verdict == WRONG for r in self.reps)

    @property
    def later(self) -> int:
        return sum(len(r.later) for r in self.reps)

    @property
    def false_triggers(self) -> int:
        """误触发:准备段里冒出来的,加上自由活动里的全部。"""
        return sum(len(r.prep) + len(r.stray) for r in self.reps)

    @property
    def stop_hits(self) -> int:
        return sum(r.verdict == STOP_HIT for r in self.reps)

    @property
    def stop_misses(self) -> int:
        return sum(r.verdict == STOP_MISS for r in self.reps)

    @property
    def false_stops(self) -> int:
        return self._false_stops

    @property
    def orphans(self) -> int:
        """认出来了、却不落在任何一条提示的窗口里的触发,有几件。

        开头那 8 秒(人从电脑前走回操作位)和最后一条做完之后都不在任何窗口里,而那
        两段正是容易冒出误触发的时候。**不进逐项比**(它不属于任何一条提示),但要在
        表上露一行——不然这几件就悄悄没了,看表的人以为一次误触发都没有。
        """
        return self._orphans

    @property
    def stop_latency(self) -> float | None:
        """停住得多快:从「做!」到真停住,取中位数。一次都没停住就是 `None`。

        **不进逐项比**:它是个连续量,同一段数据上小数位一模一样,拿「变差就报」
        那一条去卡会天天报。要看它的是 three2 那类取舍(`vision_side.scripts.three2`)。
        """
        spans = sorted(r.latency for r in self.reps if r.latency is not None)
        return spans[len(spans) // 2] if spans else None

    @property
    def rate(self) -> float:
        """识别率。没有要做手势的条时是 0。"""
        total = len(self.gesture_reps)
        return self.hits / total if total else 0.0

    @property
    def stop_rate(self) -> float:
        total = len(self.stop_reps)
        return self.stop_hits / total if total else 0.0

    def totals(self) -> dict[str, float]:
        """逐项比的时候比的就是这一张表。"""
        return {
            "认出": self.hits,
            "识别率": round(self.rate, 4),
            "漏认": self.misses,
            "方向判错": self.direction_errors,
            "认错": self.wrong,
            "误触发": self.false_triggers,
            "叫停成功": self.stop_hits,
            "叫停成功率": round(self.stop_rate, 4),
            "漏停": self.stop_misses,
            "误叫停": self.false_stops,
        }

    # ---- 和上一次比 ----------------------------------------------------

    def changes(self, before: Report) -> list[Change]:
        """逐项比,变了的都列出来(变好变差都算)。"""
        mine, theirs = _totals_of(self), _totals_of(before)
        out = []
        for name, now in mine.items():
            was = theirs.get(name)
            if was is None or float(was) == float(now):
                continue
            higher = HIGHER_IS_BETTER[name]
            worse = (now < was) if higher else (now > was)
            out.append(Change(name, float(was), float(now), worse))
        return out

    def compare(self, before: Report) -> list[Change]:
        """**只要有一项变差就报出来,不设容差。**返回变差的那几项。"""
        return [c for c in self.changes(before) if c.worse]

    def per_rep_changes(self, before: Report) -> list[Change]:
        """逐条比:哪一条的结局变了。

        总数一样、逐条不一样也要看得见——不然「一次漏认换成一次方向判错」这种
        会被总数盖住(同一段视频、同样的代码,结果应当**逐条**一致)。
        """
        was = {r.index: r.verdict for r in before.reps}
        out = []
        for rep in self.reps:
            old = was.get(rep.index)
            if old is not None and old != rep.verdict:
                # 好结局有两种:触发那几组是 `HIT`,叫停那一组是 `STOP_HIT`。
                # 只认 `HIT` 的话,一次「漏停变成停住」会被当成变差
                out.append(
                    Change(
                        f"#{rep.index} {rep.label}",
                        0.0,
                        0.0,
                        worse=rep.verdict not in (HIT, STOP_HIT),
                    )
                )
        return out

    # ---- 存取 ----------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "run": self.run,
            "mode": self.mode,
            "frames": self.frames,
            "merged": self.merged,
            "note": self.note,
            "totals": self.totals(),
            "reps": [
                {"i": r.index, "label": r.label, "verdict": r.verdict}
                for r in self.reps
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @staticmethod
    def load(path: Path) -> Report:
        """读回上一次的结果。逐项和逐条都读得回来,所以比得了。"""
        raw = json.loads(path.read_text(encoding="utf-8"))
        report = Report(run=str(raw.get("run", "")), mode=str(raw.get("mode", "")))
        report.frames = int(raw.get("frames", 0))
        report.merged = int(raw.get("merged", 0))
        report.note = str(raw.get("note", ""))
        report._loaded = raw.get("totals") or {}
        report.reps = [
            RepOutcome(
                index=int(r["i"]),
                kind="",
                key="",
                expect=None,
                label=str(r.get("label", "")),
                verdict=str(r["verdict"]),
                got=None,
            )
            for r in raw.get("reps", ())
        ]
        return report

    def table(self) -> str:
        """录完立刻在屏幕上出的那张表(spec「录制工具」第 6 条)。"""
        lines = [f"{self.run}({self.mode},{self.frames} 帧)"]
        lines += [r.line() for r in self.reps]
        lines.append("-" * 52)
        lines.append(
            "  ".join(f"{k} {_num(v)}" for k, v in self.totals().items())
        )
        if self.orphans:
            lines.append(
                f"(另有 {self.orphans} 件认出来了、不在任何一条提示的窗口里:"
                "开头走回操作位那几秒,或者最后一条做完之后)"
            )
        if self.intervals:
            lines.append(self.interval_line())
        if self.merged:
            lines.append(f"(并掉 {self.merged} 件:同一种触发 3 秒内只计一次)")
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)

    def interval_line(self) -> str:
        """每帧的间隔那一行。卡过几次是这一票要回答的(spec「录制工具」最后一条)。"""
        got = self.intervals
        if not got:
            return ""
        return (
            f"每帧间隔 p50 {got['p50']:.0f} ms、p99 {got['p99']:.0f} ms、"
            f"最长 {got['max']:.0f} ms;卡过 {got['slow']} 次"
            f"(超过 {SLOW_FRAME_MS / 1000:.1f} 秒算)"
        )

    # 读回来的那一份没有 `reps` 的明细,逐项的数直接用存下来的
    _loaded: dict | None = None

    def __post_init__(self) -> None:
        self._false_stops = 0
        self._orphans = 0


def _totals_of(report: Report) -> dict[str, float]:
    return report._loaded if report._loaded is not None else report.totals()


# ---- 算分 --------------------------------------------------------------


def dedupe(fired: Sequence[Fired], *, within_s: float = SAME_TRIGGER_S) -> tuple[list[Fired], int]:
    """同一种触发 `within_s` 秒内只计一次。返回留下的那些和并掉了几件。

    「同一种」指 `kind` 和 `key` 都一样。**隔的时间从上一件留下的那一件算起**:
    一串连着冒出来的(每次隔 1 秒、连着三次)是同一下动作带出来的,算一次。
    """
    kept: list[Fired] = []
    last: dict[tuple[str, str], float] = {}
    merged = 0
    for item in sorted(fired, key=lambda f: f.t):
        key = (item.kind, item.key)
        when = last.get(key)
        if when is not None and (item.t - when) < within_s:
            merged += 1
            last[key] = item.t  # 从最近这一件接着算,一串连着的并成一次
            continue
        kept.append(item)
        last[key] = item.t
    return kept, merged


def merge_stops(stops: Sequence[float], *, gap_s: float = FALSE_STOP_GAP_S) -> list[float]:
    """把挨得太近的叫停并成一次。**间隔超过 `gap_s` 才算两次**(spec)。"""
    out: list[float] = []
    for t in sorted(stops):
        if out and (t - out[-1]) <= gap_s:
            continue
        out.append(t)
    return out


def score(
    timetable: Timetable,
    fired: Sequence[Fired],
    stops: Sequence[float],
    *,
    run: str = "",
    mode: str = "",
    frames: int = 0,
    note: str = "",
) -> Report:
    """把认出来的东西按时间表分派,算出逐项的数(计数口径见模块说明)。"""
    kept, merged = dedupe(fired)
    report = Report(
        run=run, mode=mode, raw_fired=list(fired), fired=kept, merged=merged,
        frames=frames, note=note,
    )
    windows = timetable.reps()
    used_stops: set[float] = set()
    raw_stops = sorted(stops)
    """**认领该叫停的那几次用原始的,不用并过的。**

    票里那一条说的是「两次**误停**间隔超过 1 秒才算两次」——并是为了不把同一下动作
    数成好几次误停,和「这一次该停的停住没有」是两码事。先并再认领的话,操作者比
    提示早 0.1 秒抬手(13.9 秒一次、14.2 秒一次)会被并成 13.9 秒那一次,窗口
    [14.0, 17.0) 里就一次都没有了——同一个正确动作既算漏停、又算误停(第 05 票复查)。
    """

    for window in windows:
        mine = [f for f in kept if window.t0 <= f.t < window.t1]
        prep = [f for f in mine if f.t < window.go]
        inside = [f for f in mine if window.go <= f.t < window.end]

        if window.kind == STOP_GESTURE:
            inside_stops = [t for t in raw_stops if window.go <= t < window.end]
            hit = inside_stops[0] if inside_stops else None
            # 窗口里的几次都算这一次动作,一个都不再算误停
            used_stops.update(inside_stops)
            report.reps.append(
                RepOutcome(
                    index=window.index,
                    kind=window.kind,
                    key=window.key,
                    expect=window.expect,
                    label=window.label,
                    verdict=STOP_HIT if hit is not None else STOP_MISS,
                    got=None if hit is None else Fired(hit, STOP, "palm"),
                    # 叫停那一组的准备段说的是「放下手」,这时的触发也是误触发
                    prep=tuple(prep),
                    stray=tuple(inside),
                    latency=None if hit is None else hit - window.go,
                )
            )
            continue

        if window.kind == FREE or window.expect is None:
            # 什么都不该认出来的那几组:自由活动,以及「自然抬手」这类只看会不会
            # 误触发的提示(2026-09-17 原型的 `raise` 组,`expect` 是空的)。
            # 整段里认出来的全算误触发;一次都没有才是对的,所以不算漏认
            report.reps.append(
                RepOutcome(
                    index=window.index,
                    kind=window.kind,
                    key=window.key,
                    expect=None,
                    label=window.label,
                    verdict=NONE,
                    got=None,
                    stray=tuple(mine),
                )
            )
            continue

        first = inside[0] if inside else None
        report.reps.append(
            RepOutcome(
                index=window.index,
                kind=window.kind,
                key=window.key,
                expect=window.expect,
                label=window.label,
                verdict=_verdict(window.kind, window.expect, first),
                got=first,
                prep=tuple(prep),
                # 第一次之后的到下一条准备段之前都不计(相当于执行中)
                later=tuple(inside[1:]),
            )
        )

    # 剩下的才是误停,**这时候才并**:同一下动作连着认到好几次只算一次
    report._false_stops = len(
        merge_stops([t for t in raw_stops if t not in used_stops])
    )
    report._orphans = len([f for f in kept if not _inside_any(f.t, windows)])
    return report


def _inside_any(t: float, windows: Sequence) -> bool:
    return any(w.t0 <= t < w.t1 for w in windows)


def _verdict(kind: str, expect: str | None, first: Fired | None) -> str:
    if first is None:
        return MISS
    if first.kind == kind and first.key == expect:
        return HIT
    if kind == FIST_MOVE and first.kind == FIST_MOVE:
        return DIRECTION
    return WRONG


# ---- 跑一次回放 --------------------------------------------------------


class RegionTrack:
    """录的时候**每一帧**送的是哪块画面。回放照它逐帧还原。

    不能拿 `meta.send` 一个值套整段:正式程序认出触发手势以后,**不等主线程**就切到
    整幅画面约 1 秒(`vision_side.loop.AFTER_TRIGGER_FULL_FRAME_S`),录的时候走的
    又正是那一套,所以每做成一次手势,后面约 30 帧送的就是整幅画面。拿「检测区域」
    去套那 30 帧,等于拿另一块画面的结果去喂判定(第 05 票复查)。

    时间戳对不上就沿用上一帧那块:视频那一路偶尔会少一帧,不该为此换送法。
    """

    def __init__(self, records: Sequence) -> None:
        self._at = [(r.t, r.region) for r in records]
        self._i = 0

    def at(self, t: float) -> Region | None:
        """这一刻送的是哪块画面。`None` 是整幅画面。"""
        if not self._at:
            return None
        while self._i + 1 < len(self._at) and self._at[self._i + 1][0] <= t + 1e-9:
            self._i += 1
        return self._at[self._i][1]


def judge_frames(
    source: FrameSource,
    recognizer: Recognizer,
    run_info: Run,
    demo: DemoConfig,
    *,
    regions: RegionTrack | None = None,
    params: VisionParams | None = None,
    stop_labels: frozenset[str] | None = None,
    t_offset: float = 0.0,
    log: logging.Logger | None = None,
) -> tuple[list[Fired], list[float], int]:
    """把一路画面从头喂到尾,收下认出来的触发手势和叫停。

    **判定用的是正式程序那一套**(`TriggerJudge`),对准结果按录的时候记下的那一份
    (spec「录制工具」第 3 条),所以回放和正式程序的画面范围一致。

    `stop_labels` 换一套停止标签(算 three2 要不要加进去时用,spec「three2」);
    不给就照 `robot_side.config.STOP_GESTURE_LABELS`。

    `t_offset` 把这一路的时间戳挪到提示时间表那条轴上(转过来的老数据,录像比日志
    晚开始;见 `Meta.video_t0`)。
    """
    log = log if log is not None else logging.getLogger("vision_side.replay")
    alignment = run_info.alignment()
    meta = run_info.meta()
    params = params if params is not None else demo.vision
    poses = tuple(t.key for t in demo.triggers if t.kind == POSE_HOLD)
    judge = TriggerJudge(
        alignment,
        params,
        poses=poses,
        stop_frames=demo.run.stop_gesture_frames,
        stop_labels=stop_labels,
    )
    # 送哪块画面**逐帧照录的时候来**;没有逐帧记录时才退回 `meta.send` 那一个值
    fallback = alignment.detection_region if meta.send == DETECTION_REGION else None

    fired: list[Fired] = []
    stops: list[float] = []
    was_stopped = False
    count = 0
    with source:
        while True:
            frame = source.next_frame()
            if frame is None:
                if not source.has_next():
                    break
                continue
            if t_offset:
                frame = Frame(frame.image, frame.t + t_offset, frame.index)
            region = fallback if regions is None else regions.at(frame.t)
            result = recognizer.recognize(frame, region)
            count += 1
            got = judge.update(result)
            if got is not None:
                fired.append(Fired(frame.t, got.kind, got.key))
            # 和正式程序一样,只在**从没认到变成认到**的那一下记一次
            if judge.stop and not was_stopped:
                stops.append(frame.t)
            was_stopped = judge.stop
    log.info("回放 %s:%d 帧,触发 %d 次,叫停 %d 次", run_info.name, count, len(fired), len(stops))
    return fired, stops, count


def replay_log(
    run_info: Run,
    demo: DemoConfig,
    *,
    params: VisionParams | None = None,
    stop_labels: frozenset[str] | None = None,
    log: logging.Logger | None = None,
) -> Report:
    """从逐帧日志回放。不载模型、不解码画面——只改判定规则时用这一路。"""
    records = run_info.frames()
    recognizer = LogRecognizer(records)
    fired, stops, count = judge_frames(
        LogFrames(records), recognizer, run_info, demo,
        regions=RegionTrack(records),
        params=params, stop_labels=stop_labels, log=log,
    )
    note = ""
    if recognizer.mismatch:
        note = (
            f"注意:{recognizer.mismatch} 帧录的时候送的画面和这次要的不一样,"
            "按录的时候那一份算的"
        )
    report = score(
        run_info.timetable(), fired, stops,
        run=run_info.name, mode="log", frames=count, note=note,
    )
    report.intervals = frame_intervals(records)
    return report


SLOW_FRAME_MS = 300.0
"""一帧隔这么久以上就算卡了一下。

上一轮录制循环卡过 6 次 **0.3–1.4 秒**(spec「录制工具」最后一条),门槛就取那一档的
下限;比主循环那条(`vision_side.loop.SLOW_TICK_S`)同一个数,两边的「卡过几次」
才是同一个意思。
"""


def frame_intervals(records: Sequence) -> dict:
    """每帧的间隔:p50 / p99 / 最长 / 卡过几次(毫秒)。

    录的时候一帧一帧记下来的(`FrameRecord.interval_ms`),这里只是汇总。
    """
    spans = sorted(r.interval_ms for r in records if r.interval_ms is not None)
    if not spans:
        return {}
    return {
        "p50": round(spans[len(spans) // 2], 1),
        "p99": round(spans[min(len(spans) - 1, int(len(spans) * 0.99))], 1),
        "max": round(spans[-1], 1),
        "slow": sum(s >= SLOW_FRAME_MS for s in spans),
    }


def replay_video(
    run_info: Run,
    demo: DemoConfig,
    recognizer: Recognizer,
    *,
    params: VisionParams | None = None,
    stop_labels: frozenset[str] | None = None,
    log: logging.Logger | None = None,
) -> Report:
    """从视频回放:**重新跑识别**。换模型、换分辨率、改预处理之后用这一路。

    识别器由调用的人建好传进来(载两个 ONNX 模型要几百毫秒,一次回放多个录制时
    不该每份重建一次)。
    """
    from vision_side.frames import VideoFrames

    fired, stops, count = judge_frames(
        VideoFrames(run_info.video), recognizer, run_info, demo,
        # 视频这一路也照逐帧日志换送法,不然重跑出来的和当时看见的不是一块画面
        regions=RegionTrack(run_info.frames()),
        params=params, stop_labels=stop_labels,
        # 录像比日志晚开始的,把差的那一截加回去(转过来的老数据都这样)
        t_offset=run_info.meta().video_t0,
        log=log,
    )
    return score(
        run_info.timetable(), fired, stops,
        run=run_info.name, mode="video", frames=count,
    )


def result_path(run_info: Run, mode: str) -> Path:
    """上一次的结果存哪儿。**跟着数据走**(spec「回放测试」),所以就放在录制那个
    文件夹里:数据拷到别的机器上,比对的基准跟着一起过去。"""
    return run_info.path / f"replay-{mode}.json"

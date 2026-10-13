"""录制时的屏幕提示和提示时间表(spec「录制工具」)。

一次录制按一张**提示计划**走:一条一条提示操作者做什么,录完把每一段真正占了哪段
时间写成**提示时间表**存下来。回放测试拿这张表当标准答案——哪一段该出哪个触发手势、
哪一段本来就不该有,全看它。

这一层是纯数据:不碰摄像头、不碰画布,所以计划怎么排、窗口怎么划分都能直接测
(`tests/test_prompts.py`)。画在屏幕上是 `vision_side.overlay` 的事,录是
`vision_side.record` 的事。

几条定下来的规矩:

- **屏幕提示一律英文**(`CONTEXT.md`「预览窗口」),和正式程序的界面一致;录的人站在
  1–1.5 m 外,字高照样按 1 cm 起算。
- **每一条提示分准备段 / 「做!」段 / 歇一下**:准备段显示说明和大示范图,「做!」段
  有加粗框和高音。**一条提示的窗口从准备段算起**(`Timetable.reps`):操作者常在准备段
  就把手摆好了,那时冒出来的触发得算在这一条头上,不能算到上一条的尾巴里
  (2026-09-17 原型的 `analyze_fist.py` 也是这么分的,两边的数才比得了)。
- **时间表存的是真正占了哪段时间,不是计划里写的时长**:显示会卡,按计划的秒数倒推
  对不上真实画面。所以录的时候一段一段记下 `t0` / `t1`。
- **时间戳的起点和逐帧日志同一个**(录制循环的第一帧算 0),这样表和帧对得上。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from robot_side.config import FIST_MOVE, POSE_HOLD

PREP = "prep"
GO = "go"
REST = "rest"
"""一条提示里的三段。`prep` 说明要做什么,`go` 是「做!」,`rest` 放下手歇一下。"""

STOP_GESTURE = "stop_gesture"
"""叫停那一组的 `kind`。它不是触发手势,单独一档:算的是叫停成功率。"""

FREE = "free"
"""自由活动那一组的 `kind`。什么也不要求做,算的是误触发和误叫停。"""

PREP_S = 3.0
"""准备段多长。3 秒够看清说明和大示范图(spec「录制工具」)。"""

GO_S = 3.0
"""「做!」段多长。和 2026-09-17 原型同一档,这样两批数据的窗口一样宽。"""

REST_S = 2.0
"""歇一下多长。"""

LEAD_S = 8.0
"""按下开始到第一条提示之间留多久。人要从电脑前走回操作位(原型用的也是 8 秒)。"""

FREE_S = 160.0
"""自由活动一段多长。2026-09-22 站着录过 2 分 40 秒,这里照同一档,好逐项比。"""


@dataclass(frozen=True)
class Segment:
    """一条提示里的一段。"""

    name: str
    seconds: float
    text: str
    """屏幕上那一行。英文。"""


@dataclass(frozen=True)
class Rep:
    """一条提示:做一次某个手势。

    `kind` / `key` 和演示配置里 `[[triggers]]` 的写法一致(`fist_move` + 方向、
    `pose_hold` + 姿势),回放测试因此不用另做一套对照。`expect` 是这一条**应该**
    认出什么;自由活动那一组是 `None`——什么都不该认出来。
    """

    index: int
    kind: str
    key: str
    expect: str | None
    label: str
    """屏幕上的大字,比如 `FIST MOVE LEFT (3/10)`。"""
    segments: tuple[Segment, ...]

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.segments)


@dataclass(frozen=True)
class Plan:
    """一次录制要走的全部提示。"""

    name: str
    reps: tuple[Rep, ...]

    @property
    def seconds(self) -> float:
        """走完全部提示要多久,不含开头那段 `LEAD_S`。"""
        return sum(r.seconds for r in self.reps)

    def describe(self) -> str:
        return (
            f"计划 {self.name}:{len(self.reps)} 条提示,约 "
            f"{(self.seconds + LEAD_S) / 60.0:.1f} 分钟"
        )


@dataclass(frozen=True)
class Slot:
    """时间表里的一行:某一条提示的某一段,真正占了 `[t0, t1)`。

    存成 JSON 的样子和 2026-09-17 原型的 `timetable.json` 对得上(多一个 `kind`),
    所以那一批数据转过来就能用(`vision_side.scripts.import_runs`)。
    """

    rep: int
    kind: str
    key: str
    expect: str | None
    seg: str
    t0: float
    t1: float
    text: str
    label: str

    def to_json(self) -> dict:
        return {
            "rep": self.rep,
            "kind": self.kind,
            "key": self.key,
            "expect": self.expect,
            "seg": self.seg,
            "t0": round(self.t0, 3),
            "t1": round(self.t1, 3),
            "text": self.text,
            "label": self.label,
        }

    @staticmethod
    def from_json(raw: dict) -> Slot:
        return Slot(
            rep=int(raw["rep"]),
            kind=str(raw.get("kind", FIST_MOVE)),
            key=str(raw.get("key", "")),
            expect=raw.get("expect"),
            seg=str(raw["seg"]),
            t0=float(raw["t0"]),
            t1=float(raw["t1"]),
            text=str(raw.get("text", "")),
            label=str(raw.get("label", "")),
        )


@dataclass(frozen=True)
class RepWindow:
    """一条提示在时间轴上占的那一段,回放测试按它分派触发。

    `t0` 是准备段开头,`go` 是「做!」开头,`end` 是这一条自己结束(歇一下也完了),
    `t1` 是下一条准备段开头。**判这一次做成没做成看 `[go, end)`,准备段里冒出来的算
    误触发**,而 `end` 到 `t1` 之间是空档(显示卡了一下,或者这是最后一条),既不算
    做成也不算误触发(spec「回放测试」计数口径)。
    """

    index: int
    kind: str
    key: str
    expect: str | None
    label: str
    t0: float
    go: float
    end: float
    t1: float

    @property
    def prep(self) -> tuple[float, float]:
        """准备段那一截:`[t0, go)`。"""
        return self.t0, self.go

    @property
    def window(self) -> tuple[float, float]:
        """判这一次做成没做成的那一截:`[go, end)`。"""
        return self.go, self.end


class Timetable:
    """一次录制的提示时间表。一段一段记下来,录完存成 `timetable.json`。

    回放测试从它身上要两样东西:每一条提示的**窗口**(从准备段起到下一条准备段前),
    和窗口里的「做!」是从哪一刻开始的。
    """

    def __init__(self, slots: Sequence[Slot] = ()) -> None:
        self._slots: list[Slot] = list(slots)

    def __len__(self) -> int:
        return len(self._slots)

    def __iter__(self) -> Iterator[Slot]:
        return iter(self._slots)

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(self._slots)

    # ---- 录的时候 ------------------------------------------------------

    def open(self, rep: Rep, segment: Segment, t: float) -> None:
        """换到新的一段。上一段就在这一刻结束。"""
        self.close(t)
        self._slots.append(
            Slot(
                rep=rep.index,
                kind=rep.kind,
                key=rep.key,
                expect=rep.expect,
                seg=segment.name,
                t0=t,
                t1=t,
                text=segment.text,
                label=rep.label,
            )
        )

    def close(self, t: float) -> None:
        """把最后一段收在这一刻。录完调一次。"""
        if not self._slots:
            return
        last = self._slots[-1]
        if t > last.t1:
            self._slots[-1] = replace(last, t1=t)

    # ---- 回放的时候 ----------------------------------------------------

    def reps(self) -> list[RepWindow]:
        """每一条提示的窗口。**从准备段起**,到下一条的准备段前为止。

        最后一条到时间表末尾为止。没有准备段的(自由活动只有一段)就从那一段起。
        """
        by_rep: dict[int, list[Slot]] = {}
        for slot in self._slots:
            by_rep.setdefault(slot.rep, []).append(slot)
        order = sorted(by_rep, key=lambda i: min(s.t0 for s in by_rep[i]))
        windows = []
        for n, index in enumerate(order):
            mine = by_rep[index]
            first = min(mine, key=lambda s: s.t0)
            if n + 1 < len(order):
                t1 = min(s.t0 for s in by_rep[order[n + 1]])
            else:
                t1 = max(s.t1 for s in mine)
            go = next((s.t0 for s in mine if s.seg == GO), None)
            windows.append(
                RepWindow(
                    index=index,
                    kind=first.kind,
                    key=first.key,
                    expect=first.expect,
                    label=first.label,
                    t0=first.t0,
                    go=first.t0 if go is None else go,
                    end=max(s.t1 for s in mine),
                    t1=t1,
                )
            )
        return windows

    def at(self, t: float) -> Slot | None:
        """这一刻在哪一段上。不在任何一段上就是 `None`。"""
        for slot in self._slots:
            if slot.t0 <= t < slot.t1:
                return slot
        return None

    # ---- 存取 ----------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([s.to_json() for s in self._slots], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: Path) -> Timetable:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Timetable([Slot.from_json(r) for r in raw])


# ---- 屏幕上说什么 ------------------------------------------------------

#: 各个方向怎么说。平移那四个**箭头的起点和终点画在真实动作的位置**(见 `PromptView`),
#: 所以这里只说做什么,不说「往屏幕左边」——那是画出来的事。前后两个没有箭头,全靠这一句
#: (2026-09-24 录原型时用的就是这两句)。
MOVE_TEXT = {
    "left": "MOVE YOUR FIST TO YOUR LEFT",
    "right": "MOVE YOUR FIST TO YOUR RIGHT",
    "up": "MOVE YOUR FIST UP",
    "down": "MOVE YOUR FIST DOWN",
    "forward": "PUSH YOUR FIST TOWARD THE CAMERA ABOUT 20 CM, THEN HOLD",
    "back": "KNOCK DOWN TWICE WITH YOUR FIST, LIKE ON A TABLE",
}

#: 姿势保持那几个怎么做。和 `vision_side.overlay.POSE_HOWTO` 同一套说法。
POSE_TEXT = {
    "peace": "HOLD A V SIGN",
    "like": "HOLD A THUMBS UP",
    "ok": "HOLD AN OK SIGN",
    "rock": "HOLD A ROCK SIGN",
}

PREP_PREFIX = "GET READY - "
"""准备段那一行的开头。做什么和「做!」段说的是同一句,只多这个前缀。"""

GO_PREFIX = "GO - "
"""「做!」段那一行的开头。加粗框和高音另有(spec「录制工具」)。"""

REST_TEXT = "REST - LOWER YOUR HAND"
"""歇一下那一行。"""

STOP_PREP = "GET READY - LOWER YOUR HAND"
STOP_GO = "GO - OPEN YOUR PALM TO STOP"
"""叫停那一组:先放下手,再张开手掌。"""

FREE_TEXT = "MOVE AROUND NORMALLY - DO NOT MAKE ANY GESTURE"
"""自由活动那一组。什么都不该认出来。"""


def fist_move_rep(index: int, direction: str, label: str) -> Rep:
    what = MOVE_TEXT[direction]
    return Rep(
        index=index,
        kind=FIST_MOVE,
        key=direction,
        expect=direction,
        label=label,
        segments=(
            Segment(PREP, PREP_S, f"{PREP_PREFIX}MAKE A FIST, HOLD IT STILL"),
            Segment(GO, GO_S, f"{GO_PREFIX}{what}"),
            Segment(REST, REST_S, REST_TEXT),
        ),
    )


def pose_hold_rep(index: int, pose: str, label: str) -> Rep:
    what = POSE_TEXT.get(pose, f"HOLD {pose.upper()}")
    return Rep(
        index=index,
        kind=POSE_HOLD,
        key=pose,
        expect=pose,
        label=label,
        segments=(
            Segment(PREP, PREP_S, f"{PREP_PREFIX}{what}"),
            Segment(GO, GO_S, f"{GO_PREFIX}{what}"),
            Segment(REST, REST_S, REST_TEXT),
        ),
    )


def stop_rep(index: int, label: str) -> Rep:
    return Rep(
        index=index,
        kind=STOP_GESTURE,
        key="palm",
        expect="stop",
        label=label,
        segments=(
            Segment(PREP, PREP_S, STOP_PREP),
            Segment(GO, GO_S, STOP_GO),
            Segment(REST, REST_S, REST_TEXT),
        ),
    )


def free_rep(index: int, seconds: float, label: str) -> Rep:
    """自由活动:只有一段,什么也不要求做。误触发和误叫停全算在它头上。"""
    return Rep(
        index=index,
        kind=FREE,
        key="",
        expect=None,
        label=label,
        segments=(Segment(GO, seconds, FREE_TEXT),),
    )


def dirs_plan(rounds: int = 10) -> Plan:
    """四个方向各做 `rounds` 次。和 2026-09-17 原型的 `dirs` 同一个排法。"""
    order = ("up", "down", "left", "right")
    reps = []
    for r in range(rounds):
        for direction in order:
            reps.append(
                fist_move_rep(
                    len(reps),
                    direction,
                    f"FIST MOVE {direction.upper()} ({r + 1}/{rounds})",
                )
            )
    return Plan("dirs", tuple(reps))


def depth_plan(rounds: int = 10) -> Plan:
    """往前推、往下敲两下各做 `rounds` 次,交替着来。和 2026-09-24 定规则时那两批同一个排法。"""
    names = {"forward": "PUSH", "back": "KNOCK TWICE"}
    reps = []
    for r in range(rounds):
        for direction, name in names.items():
            reps.append(fist_move_rep(len(reps), direction, f"{name} ({r + 1}/{rounds})"))
    return Plan("depth", tuple(reps))


def poses_plan(rounds: int = 10, poses: Sequence[str] = ("peace", "like")) -> Plan:
    reps = []
    for r in range(rounds):
        for pose in poses:
            reps.append(
                pose_hold_rep(
                    len(reps), pose, f"HOLD {pose.upper()} ({r + 1}/{rounds})"
                )
            )
    return Plan("poses", tuple(reps))


def stop_plan(rounds: int = 10) -> Plan:
    """张开手掌叫停。整幅画面下做——执行中送的就是整幅画面(ADR-0008)。"""
    return Plan(
        "stop", tuple(stop_rep(n, f"STOP ({n + 1}/{rounds})") for n in range(rounds))
    )


def free_plan(rounds: int = 1, seconds: float = FREE_S) -> Plan:
    """自由活动 `rounds` 段,每段 `seconds` 秒。"""
    return Plan(
        "free",
        tuple(
            free_rep(n, seconds, f"FREE ACTIVITY ({n + 1}/{rounds})")
            for n in range(rounds)
        ),
    )


#: 命令行上 `--plan` 认得的几张计划。
#:
#: **每张的第一个参数都是「每种做几次」**:命令行照着 `--rounds` 原样传进来
#: (`vision_side/scripts/record.py`),哪一张的参数换了位置,`--rounds` 就会被当成
#: 别的东西——`poses_plan(poses, rounds)` 那会儿 `--plan poses --rounds 3` 直接崩,
#: `free_plan(seconds)` 那会儿 3 次被当成 3 秒,还不报错。
PLANS = {
    "dirs": dirs_plan,
    "depth": depth_plan,
    "poses": poses_plan,
    "stop": stop_plan,
    "free": free_plan,
}


@dataclass(frozen=True)
class PromptView:
    """这一刻屏幕上要显示的提示。画它是 `vision_side.overlay.Painter` 的事。

    `arrow` 是**画面坐标**里的一对点:起点和终点画在真实动作的位置(spec「录制工具」
    第 1 条)——握拳平移就是起点区中心和它朝那个方向移过 `move_hand_widths` 个手宽
    之后的位置,所以操作者照着箭头做,做出来的就正好够触发。姿势保持没有位移,
    `arrow` 是 `None`。
    """

    headline: str
    """大字:这一条要做什么(`FIST MOVE LEFT (3/10)`)。"""
    text: str
    """那一行说明。"""
    seg: str
    """现在在哪一段。`GO` 那一段有加粗框。"""
    left_s: float
    """这一段还剩多久。"""
    progress: float
    """整张计划走了几成,0 到 1。"""
    demo: np.ndarray | None = None
    """示范姿势图。准备段显示大的,「做!」段显示小的。没有图就只有字。"""
    arrow: tuple[tuple[float, float], tuple[float, float]] | None = None

    @property
    def big(self) -> bool:
        """准备段:示范图画大的(spec「录制工具」第 4 条)。"""
        return self.seg == PREP

    @property
    def going(self) -> bool:
        """「做!」段:加粗框。"""
        return self.seg == GO

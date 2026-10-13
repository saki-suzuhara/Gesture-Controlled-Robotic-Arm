"""回放测试算出来的那几个数,和「上一次比」这件事(spec「回放测试」)。

计数口径是票里写死的四条,每一条这里都有一条测试盯着:

1. 一条提示的窗口**从准备段算起**,判做成没做成只看「做!」那一截;
2. 两次误停间隔超过 1 秒才算两次;
3. 同一种触发 3 秒内只计一次;
4. 移动组只看窗口里的第一次触发,之后到下一条准备段之前的不计。

不读磁盘、不跑模型:时间表和认出来的东西都是这里编的。
"""

from __future__ import annotations

import pytest

from robot_side.config import FIST_MOVE, POSE_HOLD
from vision_side.prompts import (
    FREE,
    GO,
    PREP,
    REST,
    STOP_GESTURE,
    Slot,
    Timetable,
)
from vision_side.replay import (
    DIRECTION,
    HIT,
    MISS,
    STOP_HIT,
    STOP_MISS,
    WRONG,
    Fired,
    Report,
    score,
)


def rep_slots(
    index: int,
    kind: str,
    key: str,
    expect: str | None,
    t0: float,
    *,
    prep_s: float = 3.0,
    go_s: float = 3.0,
    rest_s: float = 2.0,
) -> list[Slot]:
    """一条提示的三段,从 `t0` 起。"""
    label = f"{kind} {key}"
    return [
        Slot(index, kind, key, expect, PREP, t0, t0 + prep_s, "", label),
        Slot(index, kind, key, expect, GO, t0 + prep_s, t0 + prep_s + go_s, "", label),
        Slot(
            index,
            kind,
            key,
            expect,
            REST,
            t0 + prep_s + go_s,
            t0 + prep_s + go_s + rest_s,
            "",
            label,
        ),
    ]


def dirs_table(directions: list[str], *, start: float = 10.0) -> Timetable:
    """一张握拳平移的时间表,每条 8 秒(3 + 3 + 2)。"""
    slots: list[Slot] = []
    for i, direction in enumerate(directions):
        slots += rep_slots(i, FIST_MOVE, direction, direction, start + i * 8.0)
    return Timetable(slots)


def free_table(seconds: float = 60.0, *, start: float = 10.0) -> Timetable:
    return Timetable([Slot(0, FREE, "", None, GO, start, start + seconds, "", "FREE")])


def stop_table(count: int, *, start: float = 10.0) -> Timetable:
    slots: list[Slot] = []
    for i in range(count):
        slots += rep_slots(i, STOP_GESTURE, "palm", "stop", start + i * 8.0)
    return Timetable(slots)


# ---- 窗口怎么划 --------------------------------------------------------


def test_窗口从准备段起到下一条准备段前():
    table = dirs_table(["up", "down"])
    windows = table.reps()
    assert [w.t0 for w in windows] == [10.0, 18.0]
    assert [w.go for w in windows] == [13.0, 21.0]
    assert [w.end for w in windows] == [18.0, 26.0]
    # 第一条的窗口一直延到第二条的准备段,中间不留缝
    assert windows[0].t1 == windows[1].t0


def test_准备段里的触发算误触发不算做成():
    table = dirs_table(["up"])
    # 11.0 落在准备段(10.0–13.0)里:操作者提前摆好手,冒出来一次
    report = score(table, [Fired(11.0, FIST_MOVE, "up")], [])
    assert report.false_triggers == 1
    assert report.hits == 0
    assert report.reps[0].verdict == MISS


def test_做那一段里方向对上就算认出来():
    table = dirs_table(["up"])
    report = score(table, [Fired(14.0, FIST_MOVE, "up")], [])
    assert report.hits == 1
    assert report.misses == 0
    assert report.false_triggers == 0
    assert report.reps[0].verdict == HIT


def test_方向认错单独算一档():
    table = dirs_table(["up"])
    report = score(table, [Fired(14.0, FIST_MOVE, "down")], [])
    assert report.direction_errors == 1
    assert report.hits == 0
    assert report.reps[0].verdict == DIRECTION


def test_认成另一类手势算认错不算方向判错():
    table = dirs_table(["up"])
    report = score(table, [Fired(14.0, POSE_HOLD, "peace")], [])
    assert report.wrong == 1
    assert report.direction_errors == 0
    assert report.reps[0].verdict == WRONG


def test_窗口里一次都没有算漏认():
    table = dirs_table(["up", "down"])
    report = score(table, [], [])
    assert report.misses == 2
    assert report.rate == pytest.approx(0.0)


# ---- 计数口径 ----------------------------------------------------------


def test_窗口里第一次之后的触发不计():
    """移动组只看第一次;之后到下一条准备段之前的相当于执行中(spec)。"""
    table = dirs_table(["up", "down"])
    report = score(
        table,
        [
            Fired(14.0, FIST_MOVE, "up"),  # 第一次,算数
            Fired(15.0, FIST_MOVE, "down"),  # 放手带出来的,不计
            Fired(17.5, FIST_MOVE, "left"),  # 还在这一条的窗口里,不计
        ],
        [],
    )
    assert report.hits == 1
    assert report.false_triggers == 0
    assert report.later == 2


def test_同一种触发三秒内只计一次():
    table = free_table()
    fired = [
        Fired(20.0, FIST_MOVE, "left"),
        Fired(21.5, FIST_MOVE, "left"),  # 1.5 秒,并进上一次
        Fired(22.9, FIST_MOVE, "left"),  # 离上一次 1.4 秒,还在并
        Fired(26.0, FIST_MOVE, "left"),  # 离上一次 3.1 秒,另算一次
    ]
    report = score(table, fired, [])
    assert report.false_triggers == 2
    assert report.merged == 2


def test_三秒内只并同一种不同种各算各的():
    table = free_table()
    report = score(
        table,
        [Fired(20.0, FIST_MOVE, "left"), Fired(20.5, FIST_MOVE, "right")],
        [],
    )
    assert report.false_triggers == 2
    assert report.merged == 0


def test_不要求认出什么的那几条触发全算误触发():
    """2026-09-17 原型的自然抬手组:`expect` 是空的,要的是「什么也别认出来」。

    这时认出来一次是误触发,一次都没有才是对的——不能当成漏认,也不能进识别率的分母。
    """
    table = Timetable(rep_slots(0, FIST_MOVE, "raise", None, 10.0))
    report = score(table, [Fired(14.0, FIST_MOVE, "up")], [])
    assert report.false_triggers == 1
    assert report.misses == 0
    assert report.direction_errors == 0
    assert report.gesture_reps == []

    quiet = score(table, [], [])
    assert quiet.false_triggers == 0
    assert quiet.misses == 0


def test_自由活动里的触发全算误触发():
    table = free_table()
    report = score(
        table,
        [Fired(20.0, FIST_MOVE, "left"), Fired(40.0, POSE_HOLD, "peace")],
        [],
    )
    assert report.false_triggers == 2
    assert report.hits == 0


# ---- 叫停 --------------------------------------------------------------


def test_叫停落在做那一段里算成功():
    table = stop_table(2)
    report = score(table, [], [14.0, 22.0])
    assert report.stop_hits == 2
    assert report.stop_rate == pytest.approx(1.0)
    assert report.false_stops == 0
    assert report.reps[0].verdict == STOP_HIT


def test_该叫停时没叫停算漏停():
    table = stop_table(2)
    report = score(table, [], [14.0])
    assert report.stop_hits == 1
    assert report.stop_misses == 1
    assert report.reps[1].verdict == STOP_MISS
    assert report.stop_rate == pytest.approx(0.5)


def test_不该叫停时叫停算误停():
    table = free_table()
    report = score(table, [], [20.0, 40.0])
    assert report.false_stops == 2


def test_两次误停间隔不到一秒只算一次():
    table = free_table()
    report = score(table, [], [20.0, 20.4, 20.9])
    assert report.false_stops == 1


def test_两次误停间隔超过一秒算两次():
    table = free_table()
    report = score(table, [], [20.0, 21.2])
    assert report.false_stops == 2


def test_叫停那一组的准备段里叫停算误停():
    """准备段说的是「放下手」,这时候叫停就是误停。"""
    table = stop_table(1)
    report = score(table, [], [11.0, 14.0])
    assert report.stop_hits == 1
    assert report.false_stops == 1


def test_比提示早一点抬手不该既算漏停又算误停():
    """并是为了不把同一下动作数成好几次**误停**,不该把该停的那一次也并掉。

    操作者早 0.1 秒抬手:13.9 秒一次、14.2 秒一次,「做!」从 14.0 秒起。先并再认领的
    话 14.2 被并进 13.9,窗口里一次都没有——同一个正确动作既算漏停、又算误停。
    """
    table = stop_table(1)
    report = score(table, [], [13.9, 14.2])
    assert report.stop_hits == 1, "窗口里那一次该认领得到"
    assert report.stop_misses == 0


def test_同一次叫停连着认到好几下只算一次误停():
    table = free_table()
    report = score(table, [], [20.0, 20.3, 20.6])
    assert report.false_stops == 1


def test_不在任何窗口里的触发单独记一笔():
    """开头走回操作位那几秒不在任何一条提示的窗口里,但那正是容易误触发的时候。

    不进逐项比(它不属于任何一条提示),但要数出来——不然这几件就悄悄没了。
    """
    table = dirs_table(["up"], start=10.0)
    report = score(table, [Fired(2.0, FIST_MOVE, "left")], [])
    assert report.orphans == 1
    assert report.false_triggers == 0, "不在窗口里的不算到哪一条头上"


# ---- 和上一次比 --------------------------------------------------------


def _report(**kwargs) -> Report:
    table = dirs_table(["up", "down", "left", "right"])
    return score(table, kwargs.pop("fired", []), kwargs.pop("stops", []))


def test_一模一样时没有一项变差():
    table = dirs_table(["up", "down"])
    fired = [Fired(14.0, FIST_MOVE, "up"), Fired(22.0, FIST_MOVE, "down")]
    now = score(table, fired, [])
    before = score(table, fired, [])
    assert now.compare(before) == []


def test_少认出一次就报出来():
    table = dirs_table(["up", "down"])
    before = score(
        table, [Fired(14.0, FIST_MOVE, "up"), Fired(22.0, FIST_MOVE, "down")], []
    )
    now = score(table, [Fired(14.0, FIST_MOVE, "up")], [])
    worse = now.compare(before)
    assert worse, "少认出一次,应当有一项变差"
    assert any(w.name == "认出" for w in worse)


def test_多一次误触发就报出来不设容差():
    table = free_table()
    before = score(table, [], [])
    now = score(table, [Fired(20.0, FIST_MOVE, "left")], [])
    worse = now.compare(before)
    assert [w.name for w in worse] == ["误触发"]
    assert worse[0].before == 0 and worse[0].now == 1


def test_变好不算变差但会记下来():
    table = free_table()
    before = score(table, [Fired(20.0, FIST_MOVE, "left")], [])
    now = score(table, [], [])
    assert now.compare(before) == []
    assert any(c.name == "误触发" for c in now.changes(before))


def test_漏停变成停住不算变差():
    """叫停那一组的好结局是 `STOP_HIT`,不是 `HIT`——只认 `HIT` 会把变好报成变差。"""
    table = stop_table(1)
    before = score(table, [], [])
    now = score(table, [], [14.0])
    changed = now.per_rep_changes(before)
    assert changed, "结局从漏停变成停住,该记一笔"
    assert not any(c.worse for c in changed), "这是变好,不该标成变差"
    assert now.compare(before) == []


def test_逐条比不只比总数():
    """总数一样、逐条不一样,也要报出来——不然换了个方向判错会被总数盖住。"""
    table = dirs_table(["up", "down"])
    before = score(
        table, [Fired(14.0, FIST_MOVE, "up"), Fired(22.0, FIST_MOVE, "down")], []
    )
    now = score(
        table, [Fired(14.0, FIST_MOVE, "down"), Fired(22.0, FIST_MOVE, "down")], []
    )
    worse = now.compare(before)
    assert any(w.name == "认出" for w in worse)
    assert any("#0" in w.name for w in now.per_rep_changes(before))

"""提示计划和提示时间表(第 05 票)。

纯数据,不开摄像头、不读磁盘。看的是**计划排得对不对、时间表存的是真占了哪段时间、
窗口怎么划**——窗口的划法是回放测试的地基,划错了后面所有的数都错。
"""

from __future__ import annotations

import json

import pytest

from robot_side.config import FIST_MOVE, POSE_HOLD
from vision_side import prompts
from vision_side.prompts import (
    FREE,
    GO,
    PREP,
    REST,
    STOP_GESTURE,
    Plan,
    PromptView,
    Timetable,
    dirs_plan,
    free_plan,
    poses_plan,
    stop_plan,
)


# ---- 计划 --------------------------------------------------------------


def test_四个方向各做十次一共四十条():
    plan = dirs_plan(10)
    assert len(plan.reps) == 40
    assert {r.key for r in plan.reps} == {"up", "down", "left", "right"}
    assert all(r.kind == FIST_MOVE for r in plan.reps)
    assert all(r.expect == r.key for r in plan.reps)


def test_前后那张计划推和敲交替各做十次():
    plan = prompts.depth_plan(10)
    assert len(plan.reps) == 20
    assert [r.key for r in plan.reps[:4]] == ["forward", "back", "forward", "back"]
    assert all(r.kind == FIST_MOVE and r.expect == r.key for r in plan.reps)
    # 前后画不出箭头,全靠那一句说明:说的是推、是敲,不是「往前」「往后」
    go = {r.key: r.segments[1].text for r in plan.reps[:2]}
    assert "PUSH" in go["forward"] and "KNOCK" in go["back"]
    assert prompts.PLANS["depth"] is prompts.depth_plan


def test_每条提示分三段准备段做和歇一下():
    rep = dirs_plan(1).reps[0]
    assert [s.name for s in rep.segments] == [PREP, GO, REST]
    assert rep.segments[0].seconds == prompts.PREP_S == 3.0


def test_条与条之间不重号():
    plan = dirs_plan(3)
    assert [r.index for r in plan.reps] == list(range(12))


def test_姿势保持那张计划用的是姿势不是方向():
    plan = poses_plan(2, ("peace", "like"))
    assert {r.key for r in plan.reps} == {"peace", "like"}
    assert all(r.kind == POSE_HOLD for r in plan.reps)


def test_叫停那张计划只有叫停一档():
    plan = stop_plan(10)
    assert len(plan.reps) == 10
    assert all(r.kind == STOP_GESTURE for r in plan.reps)


def test_自由活动什么也不要求做():
    plan = free_plan(seconds=120.0)
    rep = plan.reps[0]
    assert rep.kind == FREE
    assert rep.expect is None, "自由活动这一组不该认出任何东西"
    assert rep.seconds == 120.0


def test_四张计划的第一个参数都是每种做几次():
    """命令行把 `--rounds` 原样传给这四个(`vision_side/scripts/record.py`)。

    哪一张的参数换了位置,`--rounds` 就会被当成别的东西:`poses_plan(poses, rounds)`
    那会儿 `--plan poses --rounds 3` 直接崩,`free_plan(seconds)` 那会儿 3 次被当成
    3 秒,还不报错。
    """
    for name, build in prompts.PLANS.items():
        plan = build(3)
        assert plan.reps, f"{name} 排出来是空的"
        if name == "dirs":
            assert len(plan.reps) == 12, "四个方向各 3 次"
        elif name == "free":
            assert len(plan.reps) == 3
            assert plan.reps[0].seconds == prompts.FREE_S, "3 是次数,不是秒数"
        else:
            assert len(plan.reps) in (3, 6), f"{name} 该是 3 次(姿势两个就是 6 条)"


def test_屏幕提示一律英文():
    """`CONTEXT.md`「预览窗口」:文字一律英文,和正式程序的界面一致。"""
    texts = [
        s.text
        for plan in (dirs_plan(1), poses_plan(rounds=1), stop_plan(1), free_plan())
        for rep in plan.reps
        for s in rep.segments
    ] + [prompts.REST_TEXT, prompts.FREE_TEXT, prompts.STOP_GO]
    for text in texts:
        assert text.isascii(), f"这一行不是英文:{text}"
        assert text == text.upper(), f"这一行不是全大写:{text}"


def test_做那一段说的是怎么做不是往屏幕哪边动():
    """往哪边动由**画在真实动作位置**的箭头说(spec 第 1 条),字里不提屏幕方向。"""
    for rep in dirs_plan(1).reps:
        go = next(s for s in rep.segments if s.name == GO)
        assert "SCREEN" not in go.text
        assert "YOUR" in go.text or "UP" in go.text or "DOWN" in go.text


# ---- 时间表 ------------------------------------------------------------


def table_for(plan: Plan, *, start: float = 0.0, stretch: float = 1.0) -> Timetable:
    """照计划把时间表填出来,每一段按 `stretch` 倍的时长算。

    `stretch` 不是 1 就是「显示卡了一下,这一段实际比计划长」——时间表存的是真的
    占了多久,不是计划里写的。
    """
    table = Timetable()
    t = start
    for rep in plan.reps:
        for segment in rep.segments:
            table.open(rep, segment, t)
            t += segment.seconds * stretch
    table.close(t)
    return table


def test_时间表记的是真占了多久不是计划里写的():
    table = table_for(dirs_plan(1), stretch=2.0)
    prep = table.slots[0]
    assert prep.t1 - prep.t0 == pytest.approx(6.0), "卡了一倍,这一段就该记 6 秒"


def test_一条提示的窗口从准备段起():
    table = table_for(dirs_plan(1))
    window = table.reps()[0]
    assert window.t0 == 0.0
    assert window.go == prompts.PREP_S
    assert window.prep == (0.0, prompts.PREP_S)


def test_窗口一直延到下一条的准备段不留缝():
    windows = table_for(dirs_plan(2)).reps()
    for first, second in zip(windows, windows[1:]):
        assert first.t1 == second.t0, "两条之间留了缝,那一段的触发就没人认领了"


def test_最后一条的窗口到表尾为止():
    table = table_for(dirs_plan(1))
    window = table.reps()[0]
    assert window.t1 == window.end == dirs_plan(1).reps[0].seconds


def test_判做成没做成只看做那一截():
    table = table_for(dirs_plan(1))
    window = table.reps()[0]
    assert window.window == (prompts.PREP_S, prompts.PREP_S + prompts.GO_S + prompts.REST_S)


def test_自由活动只有一段窗口就是那一段():
    table = table_for(free_plan(seconds=40.0))
    window = table.reps()[0]
    assert window.t0 == window.go == 0.0, "没有准备段时,窗口从这一段本身算起"
    assert window.end == 40.0


def test_某一刻在哪一段上():
    table = table_for(dirs_plan(1))
    assert table.at(1.0).seg == PREP
    assert table.at(4.0).seg == GO
    assert table.at(7.0).seg == REST
    assert table.at(100.0) is None


def test_存下来读回去一模一样(tmp_path):
    table = table_for(dirs_plan(2))
    path = tmp_path / "timetable.json"
    table.save(path)
    again = Timetable.load(path)
    assert [s.to_json() for s in again] == [s.to_json() for s in table]
    assert [w.go for w in again.reps()] == [w.go for w in table.reps()]


def test_存出来的和原型那一批对得上(tmp_path):
    """原型的 `timetable.json` 少一个 `kind`,别的字段名要一样,那一批才转得过来。"""
    path = tmp_path / "t.json"
    table_for(dirs_plan(1)).save(path)
    row = json.loads(path.read_text(encoding="utf-8"))[0]
    assert set(row) >= {"rep", "key", "expect", "seg", "t0", "t1", "text", "label"}


# ---- 屏幕上那一份 ------------------------------------------------------


def test_准备段画大的做那一段有粗框():
    prep = PromptView("X", "Y", PREP, 1.0, 0.0)
    go = PromptView("X", "Y", GO, 1.0, 0.0)
    assert prep.big and not prep.going
    assert go.going and not go.big

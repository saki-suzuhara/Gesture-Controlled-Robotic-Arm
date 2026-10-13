"""`three2` 要不要加进停止标签:用已有录像离线算,不靠感觉(spec「停止手势与叫停」)。

    uv run -m vision_side.scripts.three2

事情的由来:2026-09-22 站着自由活动加叫停那一轮,第 1 次叫停时张开的手在整幅画面里
**被认成 `three2` 约 10 帧、晚了 0.45 秒,没漏停**。所以 `three2` 看起来像个能让叫停
更快的候选。但多认一个标签就多一条误叫停的路,加之前得先知道会多几次——
**算完再定加不加;在那之前维持 5 个标签**(票里写死的)。

算法就是把回放跑两遍:一遍用现在这 5 个标签,一遍再加上 `three2`,逐份比。比的是

- **多出几次误叫停**:不该停的时候停了几次(两次间隔超过 1 秒才算两次);
- **叫停早了多少**:该停的时候,加了以后是不是停得更早。

本机没有数据时什么也不跑、退出码 0。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from robot_side.config import STOP_GESTURE_LABELS, load_demo
from vision_side.recordings import RECORDINGS_DIR, Run, find_runs
from vision_side.replay import replay_log

CANDIDATE = "three2"
"""要算的那个标签。"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="算 three2 加进停止标签会多几次误叫停")
    parser.add_argument("--dir", type=Path, default=RECORDINGS_DIR)
    parser.add_argument("--run", default="", help="只算这一份")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    log = logging.getLogger("vision_side.replay")

    runs = [r for r in find_runs(args.dir) if not args.run or args.run in r.name]
    if not runs:
        print(f"{args.dir} 下没有能回放的录制,算不了。本机没有这批数据就跳过这一步。")
        return 0

    demo = load_demo()
    with_three2 = STOP_GESTURE_LABELS | {CANDIDATE}
    print(f"现在的停止标签:{'、'.join(sorted(STOP_GESTURE_LABELS))}")
    print(f"要算的:再加上 {CANDIDATE}\n")

    rows = []
    frames_seen = 0
    for run in runs:
        now = replay_log(run, demo, log=log)
        after = replay_log(run, demo, stop_labels=with_three2, log=log)
        rows.append((run, now, after))
        print(
            f"{run.name:42s} 误叫停 {now.false_stops} → {after.false_stops}"
            f"   叫停成功 {now.stop_hits} → {after.stop_hits}"
            f"   漏停 {now.stop_misses} → {after.stop_misses}"
            f"   停住得多快 {_ms(now.stop_latency)} → {_ms(after.stop_latency)}"
        )
        seen = _frames_with(run)
        frames_seen += seen
        if seen:
            print(f"{'':42s} 其中 {seen} 帧里有手被认成 {CANDIDATE}")
        _earlier(now, after)

    extra = sum(a.false_stops - n.false_stops for _r, n, a in rows)
    saved = sum(a.stop_hits - n.stop_hits for _r, n, a in rows)
    sooner = _sooner(rows)
    print("\n" + "-" * 70)
    print(
        f"合计:误叫停多 {extra} 次,叫停成功多 {saved} 次;"
        f"{frames_seen} 帧里有手被认成 {CANDIDATE}"
    )
    print(_verdict(extra, saved, sooner, frames_seen))
    return 0


def _ms(span: float | None) -> str:
    return "-" if span is None else f"{span:.2f}s"


def _frames_with(run: Run) -> int:
    """这一份里有多少帧被认成 `three2`。多不多得有个数,不然只看误叫停不知道为什么。"""
    return sum(
        any(h.label == CANDIDATE for h in record.hands) for record in run.frames()
    )


def _earlier(now, after) -> None:
    """哪一次停得更早了。逐条列出来——次数上看不出来的差别在这儿。"""
    for a, b in zip(now.stop_reps, after.stop_reps):
        if a.latency is None or b.latency is None or abs(a.latency - b.latency) < 1e-6:
            continue
        print(
            f"{'':42s} #{a.index} {_ms(a.latency)} → {_ms(b.latency)}"
            f"({a.latency - b.latency:+.2f}s)"
        )


def _sooner(rows) -> float:
    """加了以后一共早停了多少秒(正数 = 更早)。"""
    total = 0.0
    for _run, now, after in rows:
        for a, b in zip(now.stop_reps, after.stop_reps):
            if a.latency is not None and b.latency is not None:
                total += a.latency - b.latency
    return total


def _verdict(extra: int, saved: int, sooner: float, frames: int) -> str:
    """结论。**这只是这批数据上的结论**,换一批数据要重算。"""
    unchanged = extra == 0 and saved == 0 and abs(sooner) < 0.01
    if unchanged and frames == 0:
        return (
            f"这批数据上一帧都没被认成 {CANDIDATE},算不出差别。**维持 5 个标签。**"
        )
    if unchanged:
        return (
            f"这批数据上有 {frames} 帧被认成 {CANDIDATE},但加不加一模一样:"
            "误叫停没多,叫停也没早——那几帧连不成判定要的连续 3 帧,"
            "或者那几下本来就已经被别的标签停住了。"
            "\n**结论:维持 5 个标签,不加 three2。**"
            "\n再算一次的条件:站着录一批新的(整幅画面、有旁人入镜),"
            f"看 {CANDIDATE} 会不会连成 3 帧。"
        )
    if extra > 0 and saved <= 0 and sooner <= 0:
        return f"加了只多 {extra} 次误叫停,不多救回、也不更早:**不加**,维持 5 个标签。"
    if extra == 0 and saved > 0:
        return (
            f"加了不多误叫停,还多救回 {saved} 次漏停、早停 {sooner:.2f} 秒:"
            "**建议加**——漏一次叫停比多停一次危险(ADR-0012)。"
        )
    if extra == 0 and sooner > 0:
        return (
            f"加了一次误叫停都不多,也没救回漏停(本来就没漏),只是**早停 "
            f"{sooner:.2f} 秒**。\n"
            "也就是说:加它的好处是快一点,坏处这批数据上量不到。但这批数据里"
            f"只有 {frames} 帧被认成它,证据薄——**建议先维持 5 个标签**,"
            "等下一批数据(站着、整幅画面、有旁人入镜)再算一次;"
            "要是那时误叫停还是不多,再加。\n"
            "**加不加由用户定**,这里只给数。"
        )
    return (
        f"加了多 {extra} 次误叫停、多救回 {saved} 次、早停 {sooner:.2f} 秒,是个取舍。"
        "漏一次叫停比多停一次危险(ADR-0012),但多停会把演示打断;"
        "拿这个数去问用户,别自己定。"
    )


if __name__ == "__main__":
    sys.exit(main())

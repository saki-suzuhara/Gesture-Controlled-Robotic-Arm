"""回放测试:拿录下来的数据重跑判定,和上一次逐项比(spec「回放测试」)。

    uv run -m vision_side.scripts.replay                    # 全部录制,走逐帧日志(快)
    uv run -m vision_side.scripts.replay --run proto2-...   # 只跑一份
    uv run -m vision_side.scripts.replay --video            # 重跑识别(换模型、改预处理之后)
    uv run -m vision_side.scripts.replay --save             # 把这一次的结果存成新的基准

**只要有一项变差就报出来,不设容差**——同一段视频、同样的代码,结果应当逐条一致。
变差时退出码是 1,所以放进自动测试或者钩子里都行。

上一次的结果存在录制那个文件夹里(`replay-log.json` / `replay-video.json`),
**跟着数据走**:数据拷到别的机器上,比对的基准跟着一起过去。

本机没有这些数据时什么也不跑、退出码 0(spec「回放测试」最后一条)。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from robot_side.config import STOP_GESTURE_LABELS, load_demo
from vision_side.recordings import RECORDINGS_DIR, Run, find_runs
from vision_side.replay import Report, replay_log, replay_video, result_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回放测试:和上一次逐项比")
    parser.add_argument("--run", default="", help="只跑这一份(文件夹名,可以只写一截)")
    parser.add_argument("--dir", type=Path, default=RECORDINGS_DIR, help="录制放在哪儿")
    parser.add_argument(
        "--video",
        action="store_true",
        help="重跑识别(换模型 / 分辨率 / 预处理之后用);不给就读逐帧日志,快得多",
    )
    parser.add_argument(
        "--save", action="store_true", help="把这一次的结果存成新的基准"
    )
    parser.add_argument(
        "--stop-labels",
        default="",
        help="换一套停止标签算(逗号分隔)。算 three2 要不要加进去时用",
    )
    parser.add_argument("--quiet", action="store_true", help="只出汇总,不出逐条")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    log = logging.getLogger("vision_side.replay")

    runs = [r for r in find_runs(args.dir) if not args.run or args.run in r.name]
    if not runs:
        # 数据不在版本库里,本机没有是常事:说清楚就行,不算失败
        print(
            f"{args.dir} 下没有能回放的录制。"
            "\n先录一段(`uv run -m vision_side.scripts.record --plan dirs`),"
            "或者把原型那批转过来(`uv run -m vision_side.scripts.import_runs`)。"
        )
        return 0

    mode = "video" if args.video else "log"
    labels = (
        frozenset(s.strip() for s in args.stop_labels.split(",") if s.strip())
        if args.stop_labels
        else None
    )
    if labels is not None:
        print(f"停止标签换成:{'、'.join(sorted(labels))}(常规是 {'、'.join(sorted(STOP_GESTURE_LABELS))})")

    demo = load_demo()
    recognizer = None
    if args.video:
        from vision_side.gestures import GestureRecognizer

        recognizer = GestureRecognizer()

    worse_any = False
    for run in runs:
        if args.video and not run.has_video:
            print(f"{run.name}:没有录像,跳过")
            continue
        report = (
            replay_video(run, demo, recognizer, stop_labels=labels, log=log)
            if args.video
            else replay_log(run, demo, stop_labels=labels, log=log)
        )
        print()
        print(report.table() if not args.quiet else _short(report))
        # 换了停止标签算出来的数不作基准,**一份都不写**:没有基准时写一份看着方便,
        # 但写的要是「加了 three2 」那一版,往后每次比的就都是那一版,而且不声不响
        worse_any |= _compare(run, report, mode, baseline=labels is None, save=args.save)

    if labels is not None:
        print("\n(换了停止标签,这一次不作基准,也不和上一次比)")
        return 0
    if worse_any:
        print("\n有项变差。同一段数据、同样的代码,结果应当逐条一致——先查是什么改动带来的。")
        return 1
    print("\n没有一项变差。")
    return 0


def _short(report: Report) -> str:
    lines = [f"{report.run}({report.mode},{report.frames} 帧)"]
    lines.append("  ".join(f"{k} {v}" for k, v in report.totals().items()))
    if report.intervals:
        lines.append(report.interval_line())
    return "\n".join(lines)


def _compare(
    run: Run, report: Report, mode: str, *, baseline: bool, save: bool
) -> bool:
    """和上一次比。返回这一份有没有变差的项。

    `baseline=False` 时这一次的数**一份都不写**:换了停止标签算出来的不能当基准。
    """
    path = result_path(run, mode)
    if not path.exists():
        if not baseline:
            print("  (上一次的结果不在;这一次换了停止标签,不拿它当基准)")
            return False
        report.save(path)
        print(f"  (上一次的结果不在,这一次存成基准:{path.name})")
        return False

    before = Report.load(path)
    worse = report.compare(before)
    for change in report.changes(before):
        print(f"  {'✗' if change.worse else '·'} {change.line()}")
    for change in report.per_rep_changes(before):
        print(f"  ~ {change.name}:结局变了")
    if not worse and not report.changes(before):
        print("  和上一次逐项一致。")
    if save and baseline:
        report.save(path)
        print(f"  (存成新的基准:{path.name})")
    return bool(worse)


if __name__ == "__main__":
    sys.exit(main())

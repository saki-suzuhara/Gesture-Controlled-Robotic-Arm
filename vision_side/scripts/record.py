"""录制工具的入口:取景检查,或者按提示录一段(spec「录制工具」)。

    uv run -m vision_side.scripts.record --check            # 取景检查,不录
    uv run -m vision_side.scripts.record --plan dirs        # 四个方向各 10 次
    uv run -m vision_side.scripts.record --plan poses       # 比 V、竖拇指各 10 次
    uv run -m vision_side.scripts.record --plan stop        # 张开手掌叫停 10 次(整幅画面)
    uv run -m vision_side.scripts.record --plan free        # 自由活动 160 秒,什么都别做
    uv run -m vision_side.scripts.record --plan dirs --rounds 3 --no-video

先做一次**启动对准**(和正式程序同一步):站到操作位,把拳头举起来停住约 2 秒。
对上以后按提示一条一条做,录完**立刻在屏幕上出表**。

**第一次用先跑 `--check`**:只显示画面和识别结果,不录像,可以边调屏幕倾角边试做,
免得录完才发现取景不对。

录下来的放 `data/recordings/`,**只存本地,不提交、不分发**(里面有你本人)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_side.beeps import Beeper
from robot_side.config import ConfigError, load_demo, load_demo_config
from robot_side.runlog import close as close_log
from robot_side.runlog import start as start_log
from vision_side.frames import CameraFrames, FrameError, VideoFrames
from vision_side.gestures import GestureRecognizer, ModelsMissing
from vision_side.prompts import PLANS
from vision_side.record import RecordSession
from vision_side.recordings import RECORDINGS_DIR

EXIT_OK = 0
EXIT_VISION = 1
"""摄像头或识别不对(含摄像头占位图:拦住开始,不让白录一遍)。"""
EXIT_CONFIG = 2
EXIT_MODELS = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按提示录一段,或者只看取景")
    parser.add_argument(
        "--plan",
        choices=sorted(PLANS),
        default=None,
        help="录哪一张计划。不给就是取景检查",
    )
    parser.add_argument(
        "--check", action="store_true", help="取景检查:只显示,不录(等于不给 --plan)"
    )
    parser.add_argument("--rounds", type=int, default=None, help="每种做几次(默认 10)")
    parser.add_argument("--demo", default="", help="换一份演示配置(.toml 的路径)")
    parser.add_argument("--camera", type=int, default=None, help="摄像头编号")
    parser.add_argument("--video", type=Path, default=None, help="放录像代替摄像头(调画面用)")
    parser.add_argument("--out", type=Path, default=RECORDINGS_DIR, help="录到哪儿")
    parser.add_argument("--no-video", action="store_true", help="只录逐帧日志,不录像")
    parser.add_argument(
        "--seconds", type=float, default=None, help="自测用:跑这么多秒自动退出"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        demo = load_demo_config(Path(args.demo)) if args.demo else load_demo()
    except ConfigError as exc:
        print(f"配置有问题,什么也没做:\n  {exc}", file=sys.stderr)
        return EXIT_CONFIG

    plan = None
    if args.plan is not None and not args.check:
        build = PLANS[args.plan]
        plan = build(args.rounds) if args.rounds is not None else build()

    log, path = start_log("record" if plan is not None else "check")
    log.info("日志 %s", path)
    log.info("演示配置 %s", demo.path.name)
    if plan is not None:
        log.info("%s", plan.describe())
    else:
        log.info("取景检查:一帧也不会写下来")

    try:
        recognizer = GestureRecognizer()
    except ModelsMissing as exc:
        log.error("模型不在:%s", exc)
        print(f"模型不在,什么也没做:\n  {exc}", file=sys.stderr)
        close_log(log)
        return EXIT_MODELS

    if args.video is not None:
        source = VideoFrames(args.video)
    else:
        width, height = demo.vision.camera_px
        index = demo.vision.camera_index if args.camera is None else args.camera
        source = CameraFrames(index, width, height)

    beeps = Beeper()
    session = RecordSession(
        source,
        recognizer,
        demo,
        log,
        plan=plan,
        folder=args.out,
        video=not args.no_video,
        seconds=args.seconds,
        beeps=beeps,
    )
    try:
        return session.run_session()
    except FrameError as exc:
        # 摄像头打不开、迟迟没画面、**画面是一片均匀的**(隐私开关关着):
        # 说清是哪一步,一帧都没录下来
        log.error("启动检查 不通过(%s):%s", exc.reason, exc)
        print(f"{exc.reason}:{exc}\n什么也没录下来。", file=sys.stderr)
        return EXIT_VISION
    finally:
        beeps.close()
        close_log(log)


if __name__ == "__main__":
    sys.exit(main())

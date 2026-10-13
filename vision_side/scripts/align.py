"""启动对准跑一遍:开摄像头,站到操作位把拳头举起来停住,程序放下两个区域。

对上了就响一声、画出检测区域和起点区的边框,把对准结果写进日志,然后自己往下走
(正式程序里下一步是连机械臂,这个脚本里是显示几秒就退出)——**不用回电脑按键**。
一直对不准就停在这一步,屏幕上提示怎么站、怎么举拳;按 q / Esc 退出。

站得太近时屏幕上显示 TOO CLOSE,但**不拦着**对准:退不退一步由你定(ADR-0008)。

    uv run -m vision_side.scripts.align
    uv run -m vision_side.scripts.align --video data/recordings/xxx/video.mp4

这里是单线程的,和查看程序一样(`view.py`):真程序里识别和窗口分在不同线程
(spec「线程」)。所以别去拖窗口。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from robot_side import runlog
from robot_side.beeps import Beeper
from robot_side.config import load_demo
from vision_side.align import TOO_CLOSE_HINT, Aligner, Alignment
from vision_side.frames import (
    CameraFrames,
    FrameError,
    FrameSource,
    VideoFrames,
    first_frame,
)
from vision_side.gestures import GestureRecognizer, ModelsMissing
from vision_side.recognize import FrameResult, Region
from vision_side.scripts.view import (
    FONT,
    GREEN,
    GREY,
    RED,
    WHITE,
    YELLOW,
    FpsMeter,
    draw_boxes,
    next_frame,
    save_snap,
)

WINDOW = "ALIGN"

SHOW_S = 3.0
"""对上以后把两个边框显示多久再退出。正式程序里这里改成接着连机械臂。

不叫 hold:`VisionParams.hold_s` 是「拳头要停稳多久」,同一个词指两件事会看错。
"""

HINT_SCALE = 1.2
"""提示那一行的字号。站在 1–1.5 m 外要看得清(`CONTEXT.md`「预览窗口」)。"""


def draw_region(image, region: Region, color, thickness: int = 2) -> None:
    cv2.rectangle(
        image, (region.x, region.y), (region.right, region.bottom), color, thickness
    )


def centered(image, text: str, y: int, color, scale: float = HINT_SCALE) -> None:
    """把一行字画在画面正中那一列。对准时人站在远处,居中最好找。"""
    (width, _), _ = cv2.getTextSize(text, FONT, scale, 2)
    cv2.putText(
        image, text, ((image.shape[1] - width) // 2, y), FONT, scale, color, 2
    )


def draw(image, result: FrameResult, aligner: Aligner, *, fps: float) -> None:
    """把这一帧该显示的都画上去(原地改)。文字一律英文。"""
    done = aligner.done
    if done is not None:
        # 对上了:检测区域和送进去的那块是同一块,所以照常用绿的(灰色专留给「送整幅
        # 画面」那一档,`CONTEXT.md`「检测区域」);起点区另用一色,手该放哪儿一眼看得出
        draw_region(image, done.detection_region, GREEN)
        draw_region(image, done.start_zone, YELLOW, 3)
    elif aligner.send is not None:
        draw_region(image, aligner.send, GREEN)

    draw_boxes(image, result, with_id=False)

    corner = "FULL FRAME" if aligner.send is None else "REGION"
    cv2.putText(image, corner, (12, 26), FONT, 0.7, WHITE, 2)
    cv2.putText(image, f"{fps:5.1f} FPS", (12, 52), FONT, 0.7, WHITE, 2)

    height = image.shape[0]
    centered(image, aligner.hint, height - 90, GREEN if done else WHITE)
    if aligner.too_close:
        # 只提示,不拦着:对准照走
        centered(image, TOO_CLOSE_HINT, height - 140, RED)
    if done is not None:
        centered(
            image,
            f"HAND WIDTH {done.hand_width:.0f} PX",
            height - 40,
            GREEN,
            scale=0.8,
        )
    else:
        bar(image, aligner.progress)
        cv2.putText(
            image, "Q / ESC = QUIT", (12, height - 14), FONT, 0.6, GREY, 2
        )


def bar(image, progress: float) -> None:
    """停稳进度条。举着不动要举 2 秒,不画个东西的话不知道还要多久。"""
    height, width = image.shape[:2]
    left, right = width // 4, width - width // 4
    top = height - 70
    cv2.rectangle(image, (left, top), (right, top + 14), GREY, 1)
    filled = left + int((right - left) * progress)
    if filled > left:
        cv2.rectangle(image, (left, top), (filled, top + 14), YELLOW, -1)


def run(
    source: FrameSource,
    *,
    show_s: float,
    log,
    beeps: Beeper,
    snap: Path | None = None,
) -> int:
    """跑到对上、或者按 q / Esc、或者画面放完。对上了返回 0,没对上返回 1。"""
    demo = load_demo()
    recognizer = GestureRecognizer()
    frame = first_frame(source)
    aligner = Aligner(
        (frame.width, frame.height),
        demo.vision,
        # 换送法时清掉上游的跟踪记录:换了一块画面,同一只手的位置是跳变的(ADR-0008)
        on_switch=lambda _region: recognizer.clear_tracks(),
    )
    log.info(
        "对准 开始:画面 %dx%d,检测区域 %dx%d,停稳 %.1f 秒 + 量手宽 %.1f 秒",
        frame.width,
        frame.height,
        *demo.vision.detection_region_px,
        demo.vision.hold_s,
        demo.vision.measure_s,
    )

    headless = snap is not None
    if not headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, frame.width, frame.height)

    alignment: Alignment | None = None
    done_at: float | None = None
    warned_too_close = False
    meter = FpsMeter()
    last_image = None

    while frame is not None:
        result = recognizer.recognize(frame, aligner.send)
        alignment = aligner.update(result)

        if alignment is not None and done_at is None:
            done_at = time.monotonic()
            beeps.aligned()  # 站在远处也知道这一步过了
            log.info("%s", alignment.describe())
            log.info(
                "对准 整幅画面里量到的宽比检测区域里的大 %.1f%%(每次都不一样,不折算)",
                alignment.width_gap_percent,
            )
        if aligner.too_close and not warned_too_close:
            warned_too_close = True
            log.warning("对准 TOO CLOSE:整幅画面里的拳头手框偏宽,站太近了;只提示,不拦着")

        image = frame.image.copy()
        draw(image, result, aligner, fps=meter.tick())
        last_image = image

        if not headless:
            cv2.imshow(WINDOW, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                log.info("对准 按键退出")
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        # 对上了就自己往下走,不用回电脑按键
        if done_at is not None and time.monotonic() - done_at >= show_s:
            break

        frame = next_frame(source)

    cv2.destroyAllWindows()
    if snap is not None and last_image is not None:
        save_snap(snap, last_image)
    if alignment is None:
        log.warning("对准 没对上;正式程序会停在这一步,不往下连机械臂")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    demo = load_demo()
    parser = argparse.ArgumentParser(description="启动对准")
    parser.add_argument(
        "--camera", type=int, default=demo.vision.camera_index, help="摄像头编号"
    )
    parser.add_argument("--video", type=Path, default=None, help="放录像,不开摄像头")
    parser.add_argument(
        "--show", type=float, default=SHOW_S, help="对上以后把边框显示多久再退出"
    )
    parser.add_argument(
        "--snap", type=Path, default=None, help="自测用:不开窗口,跑完存一张画好的图"
    )
    args = parser.parse_args(argv)

    width, height = demo.vision.camera_px
    source: FrameSource = (
        VideoFrames(args.video)
        if args.video is not None
        else CameraFrames(args.camera, width, height)
    )
    log, _path = runlog.start("align")
    try:
        with source, Beeper() as beeps:
            return run(source, show_s=args.show, log=log, beeps=beeps, snap=args.snap)
    except ModelsMissing as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    except FrameError as exc:
        log.error("%s:%s", exc.reason, exc)
        return 1
    finally:
        runlog.close(log)


if __name__ == "__main__":
    raise SystemExit(main())

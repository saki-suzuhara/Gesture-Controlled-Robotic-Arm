"""最小的查看程序:开摄像头,窗口里画出画面、手框、姿势标签和帧率,按 q / Esc 退出。

用它确认三件事(第 01 票最后一条):1.2 m 处能稳定认出手;帧率在 30 fps 那一档,没有
掉到关不掉线程自旋的 9–15 fps;送检测区域和送整幅画面都走得通(`--region`)。

窗口上的字一律英文(`CONTEXT.md`「预览窗口」)。这里是单线程的——真程序里识别和窗口
分在不同线程(spec「线程」),因为拖着窗口标题栏不放会让同一线程里的识别停最长 6.8 秒
(2026-09-17 实测)。所以看帧率时别去拖窗口。

**颜色、字体、手框怎么画、帧率怎么数、怎么存图也放在这一份里**,启动对准那个脚本
(`align.py`)直接拿去用:两个小工具画的是同一批东西,分成两份迟早会各画各的。

    uv run -m vision_side.scripts.view
    uv run -m vision_side.scripts.view --region
    uv run -m vision_side.scripts.view --video data/recordings/xxx/video.mp4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from vision_side.frames import (
    CameraFrames,
    Frame,
    FrameError,
    FrameSource,
    VideoFrames,
    first_frame,
)
from robot_side.config import load_demo
from vision_side.gestures import GestureRecognizer, ModelsMissing

# 颜色和字体从正式程序的画面叠加那一份来,不另开一套:两个小工具和预览窗口画的是同一批
# 东西,分成几份迟早会各画各的。`align.py`、`triggers.py` 也是从这儿转手拿的。
from vision_side.overlay import FONT, GREEN, GREY, RED, WHITE, YELLOW
from vision_side.recognize import FrameResult, Region

WINDOW = "VISION CHECK"

REGION_SIZE = (800, 600)
"""`--region` 用的检测区域大小,放在画面正中(第 02 票才按拳头位置放)。"""

FPS_OVER = 30
"""帧率按最近这么多帧算。"""



class FpsMeter:
    """最近若干帧的帧率。两个脚本共用一份,免得同一个数各算各的。"""

    def __init__(self, over: int = FPS_OVER) -> None:
        self._stamps: deque[float] = deque(maxlen=over)

    def tick(self) -> float:
        """记下这一帧,返回眼下的帧率。头一帧还算不出来,给 0。"""
        self._stamps.append(time.perf_counter())
        return self.value

    @property
    def value(self) -> float:
        if len(self._stamps) > 1 and self._stamps[-1] > self._stamps[0]:
            return (len(self._stamps) - 1) / self.span
        return 0.0

    @property
    def frames(self) -> int:
        """眼下算在里面的帧数。"""
        return len(self._stamps)

    @property
    def span(self) -> float:
        """这几帧一共花了多久。"""
        return self._stamps[-1] - self._stamps[0] if len(self._stamps) > 1 else 0.0


def draw_boxes(image, result: FrameResult, *, with_id: bool = True) -> None:
    """把这一帧看见的手画上去(原地改):已确认的绿框,没确认的灰框,旁边写标签。"""
    for hand in result.hands:
        x1, y1, x2, y2 = hand.box
        color = GREEN if hand.confirmed else GREY
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{hand.label.upper()} #{hand.track_id}" if with_id else hand.label.upper()
        cv2.putText(image, text, (x1, max(18, y1 - 8)), FONT, 0.6, color, 2)


def save_snap(path: Path, image) -> None:
    """存一张画好的图。路径里有中文时 `cv2.imwrite` 写不出来,所以走 `tofile`。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", image)[1].tofile(str(path))
    print(f"存了一张 {path}")


def draw(
    image, result: FrameResult, *, fps: float, recognize_ms: float, region: Region | None
) -> None:
    """把这一帧的识别结果画到画面上(原地改)。"""
    if region is not None:
        cv2.rectangle(
            image, (region.x, region.y), (region.right, region.bottom), GREY, 2
        )
    draw_boxes(image, result)
    corner = "REGION" if region is not None else "FULL FRAME"
    cv2.putText(image, corner, (12, 26), FONT, 0.7, WHITE, 2)
    cv2.putText(
        image,
        f"{fps:5.1f} FPS   RECOGNIZE {recognize_ms:4.1f} MS   HANDS {len(result.hands)}",
        (12, 52),
        FONT,
        0.7,
        WHITE,
        2,
    )
    if result.flat:
        cv2.putText(image, "NO CAMERA IMAGE", (12, 84), FONT, 0.8, RED, 2)
    cv2.putText(
        image,
        "Q / ESC = QUIT",
        (12, image.shape[0] - 14),
        FONT,
        0.6,
        GREY,
        2,
    )


def next_frame(source: FrameSource) -> Frame | None:
    """取下一帧;这一次读空就再读,画面真的没有了才返回 `None`。"""
    while True:
        frame = source.next_frame()
        if frame is not None:
            return frame
        if not source.has_next():
            return None


def run(
    source: FrameSource, *, use_region: bool, seconds: float | None, snap: Path | None
) -> int:
    """跑到按 q / Esc、画面放完、或者 `--seconds` 到点。返回退出码。"""
    recognizer = GestureRecognizer()
    frame: Frame | None = first_frame(source)
    assert frame is not None
    if isinstance(source, CameraFrames):
        print(
            f"摄像头 {source.index} 号,后端 {source.backend},"
            f"首帧 {frame.width}x{frame.height}"
        )

    region = None
    if use_region:
        w, h = REGION_SIZE
        region = Region(
            (frame.width - w) // 2, (frame.height - h) // 2, w, h
        ).clamped_to(frame.width, frame.height)

    # 存图那一档不开窗口(开窗口会抢屏幕),跑完把最后一帧画好的图存下来。
    headless = snap is not None
    if not headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, frame.width, frame.height)

    meter = FpsMeter()
    costs: list[float] = []
    deadline = None if seconds is None else time.monotonic() + seconds
    frames = 0
    last_image = None

    while frame is not None:
        t0 = time.perf_counter()
        result = recognizer.recognize(frame, region)
        cost_ms = (time.perf_counter() - t0) * 1000.0
        costs.append(cost_ms)
        fps = meter.tick()
        frames += 1

        image = frame.image.copy()
        draw(image, result, fps=fps, recognize_ms=cost_ms, region=region)
        last_image = image

        if not headless:
            cv2.imshow(WINDOW, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        if deadline is not None and time.monotonic() > deadline:
            break

        frame = next_frame(source)

    cv2.destroyAllWindows()
    if snap is not None and last_image is not None:
        save_snap(snap, last_image)
    # 第一帧不算:那一帧带着模型预热,本机约 330 ms,混进来 p99 就只剩它了。
    rest = sorted(costs[1:])
    if rest:
        p50 = statistics.median(rest)
        p99 = rest[min(len(rest) - 1, int(len(rest) * 0.99))]
        print(
            f"{frames} 帧,识别每帧(不含首帧)p50 {p50:.1f} ms、p99 {p99:.1f} ms,"
            f"首帧 {costs[0]:.0f} ms,"
            f"最近 {meter.frames} 帧 {meter.value:.1f} fps"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    demo = load_demo()
    parser = argparse.ArgumentParser(description="视觉一侧的查看程序")
    parser.add_argument(
        "--camera", type=int, default=demo.vision.camera_index, help="摄像头编号"
    )
    parser.add_argument("--video", type=Path, default=None, help="放录像,不开摄像头")
    parser.add_argument(
        "--region", action="store_true", help="送画面正中的 800x600 检测区域,不送整幅画面"
    )
    parser.add_argument("--seconds", type=float, default=None, help="跑这么多秒自动退出")
    parser.add_argument(
        "--snap", type=Path, default=None, help="自测用:不开窗口,跑完存一张画好的图"
    )
    args = parser.parse_args(argv)
    if args.snap is not None and args.seconds is None and args.video is None:
        args.seconds = 3.0  # 存图配摄像头时不能没完没了地跑

    width, height = demo.vision.camera_px
    source: FrameSource = (
        VideoFrames(args.video)
        if args.video is not None
        else CameraFrames(args.camera, width, height)
    )
    try:
        with source:
            return run(
                source, use_region=args.region, seconds=args.seconds, snap=args.snap
            )
    except ModelsMissing as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    except FrameError as exc:
        print(f"{exc.reason}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

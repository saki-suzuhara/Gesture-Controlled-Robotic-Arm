"""触发判定走一遍:不接机械臂,屏幕上一步步带你做完六个触发手势加一次叫停。

先做启动对准(放下检测区域和起点区、量出手宽),然后**一次只提示一件事**:屏幕上
写着现在该做哪个手势,并在起点区上画出这一下该怎么走——箭头的起点和终点就画在真实
动作的位置。认出来了就打勾、自动进下一个;七件走完当场出一张表。

    uv run -m vision_side.scripts.triggers
    uv run -m vision_side.scripts.triggers --free    # 不提示,只显示认到了什么

按 n 跳过这一件,q / Esc 退出。**这一票不接机械臂**:认出来只是显示和写日志,真正
排进状态逻辑、真正叫停都是第 04 票的事。

送的是**检测区域**(和空闲时一样,ADR-0008),所以叫停那一步也请在大框里做;「旁人
站在检测区域以外叫停」要等第 04 票把整幅画面那一路接上。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from robot_side import runlog
from robot_side.beeps import Beeper
from robot_side.config import load_demo
from vision_side.align import Aligner, Alignment
from vision_side.frames import (
    CameraFrames,
    FrameError,
    FrameSource,
    VideoFrames,
    first_frame,
)
from vision_side.gestures import GestureRecognizer, ModelsMissing
from vision_side.overlay import FIST_HOWTO
from vision_side.recognize import FrameResult
from vision_side.scripts.align import YELLOW, centered, draw_region
from vision_side.scripts.align import draw as draw_align
from vision_side.scripts.view import (
    FONT,
    GREEN,
    GREY,
    RED,
    WHITE,
    FpsMeter,
    draw_boxes,
    next_frame,
    save_snap,
)
from vision_side.triggers import DIRECTIONS, FIST_MOVE, POSE_HOLD, Trigger, TriggerJudge

WINDOW = "TRIGGERS"

STOP_STEP = "stop"
"""走一遍里最后那一件:叫停。它不是触发手势,单独排一件。"""

NEXT_S = 1.2
"""认出来以后停多久再进下一件。够你看清打了勾,也够手放下来。"""

#: 姿势标签该怎么做,照 `CONTEXT.md` 的定义写成一句人能照做的话。**光写标签名不够**:
#: 2026-09-21 站着走一遍时,`HOLD PEACE` 那一件卡了 10.6 秒,因为屏幕上没说 peace
#: 是哪个手势。第 04 票的图例会配示范小图,在那之前先用文字。
POSE_SAYS = {
    "peace": "V SIGN (TWO FINGERS UP)",
    "like": "THUMBS UP",
    "ok": "OK SIGN (THUMB AND INDEX MAKE A RING)",
    "rock": "ROCK SIGN (INDEX AND PINKY UP)",
}

#: 握拳平移里前后那两个怎么做。画面里画不出前后的箭头,全靠这一句。
FIST_SAYS = {
    "forward": "PUSH TOWARD THE CAMERA, HOLD",
    "back": "KNOCK DOWN TWICE",
}


class Step:
    """走一遍里的一件:提示语、怎么画示意、认出来没有。"""

    def __init__(self, kind: str, key: str, says: str) -> None:
        self.kind = kind
        self.key = key
        self.says = says
        self.done_at: float | None = None
        self.started_at: float | None = None
        self.wrong: list[str] = []
        """中途认错成别的那几件。方向判错最要紧,这里记下来。"""

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.key}"

    @property
    def spent(self) -> float | None:
        if self.done_at is None or self.started_at is None:
            return None
        return self.done_at - self.started_at


def plan(demo) -> list[Step]:
    """按演示配置排出要走的七件:六个触发手势 + 一次叫停。"""
    steps = []
    for t in demo.triggers:
        if t.kind == FIST_MOVE:
            says = f"FIST MOVE {t.key.upper()}"
            if t.key in FIST_SAYS:
                # 前后两个光看名字猜不出怎么做,和姿势保持一样跟一句
                says += f" ({FIST_SAYS[t.key]})"
        else:
            # 标签名后面跟上这个姿势怎么做:照着标签名猜不出来
            says = f"HOLD {POSE_SAYS.get(t.key, t.key.upper())} FOR 1 S"
        steps.append(Step(t.kind, t.key, says))
    steps.append(Step(STOP_STEP, STOP_STEP, "OPEN PALM (STOP)"))
    return steps


class Walkthrough:
    """走一遍:七件按顺序来,一次只提示一件,认出来就打勾、隔一会儿自动进下一个。

    「走到第几件、这一件什么时候过的、下一件什么时候开始」是一起变的,所以收在这里,
    不摊成几个变量在主循环和画面之间来回传。
    """

    def __init__(self, steps: list[Step]) -> None:
        self.steps = steps
        self.current = 0
        self._next_at: float | None = None

    @property
    def step(self) -> Step | None:
        """现在该做哪一件。都走完了就是 `None`。"""
        return self.steps[self.current] if self.current < len(self.steps) else None

    @property
    def over(self) -> bool:
        return self.current >= len(self.steps)

    @property
    def all_passed(self) -> bool:
        return all(s.done_at is not None for s in self.steps)

    def start(self, t: float) -> None:
        """对准过了,从第一件开始计时。"""
        if self.steps:
            self.steps[0].started_at = t

    def score(self, got: Trigger | None, stopped: bool, t: float, log) -> None:
        """认出来的这一件算在谁头上。"""
        step = self.step
        if step is None:
            return
        if step.started_at is None:
            step.started_at = t
        hit = (got is not None and got.kind == step.kind and got.key == step.key) or (
            stopped and step.kind == STOP_STEP
        )
        if hit and step.done_at is None:
            step.done_at = t
            log.info("走一遍 %s 认出来了,用了 %.1f 秒", step.says, step.spent or 0.0)
            self._next_at = t + NEXT_S
            return
        if got is not None and not hit:
            # 方向判错最要紧:做左移却报了右移,这一条要看得见
            name = f"{got.kind}:{got.key}"
            if name not in step.wrong:
                step.wrong.append(name)
                log.warning("走一遍 做 %s 的时候认成了 %s", step.says, name)
        if self._next_at is not None and t >= self._next_at:
            self.skip(t)

    def skip(self, t: float, log=None) -> None:
        """这一件不等了,进下一件。"""
        if log is not None and self.step is not None:
            log.info("跳过 %s", self.step.says)
        self.current += 1
        self._next_at = None
        if self.step is not None:
            self.step.started_at = t

    def summary(self) -> str:
        """走完出的那张表。屏幕上和日志里是同一张。"""
        lines = ["", "走一遍的结果:"]
        # 列宽按最长那一件算,不写死:手势名从演示配置来,写死了就会参差不齐
        width = max(len(s.says) for s in self.steps)
        for step in self.steps:
            if step.done_at is None:
                mark = "没认出来"
            else:
                mark = f"认出来了,用了 {step.spent:.1f} 秒" if step.spent else "认出来了"
            wrong = f",中途认错成 {'、'.join(step.wrong)}" if step.wrong else ""
            lines.append(f"  {step.says:<{width}}  {mark}{wrong}")
        passed = sum(1 for s in self.steps if s.done_at is not None)
        lines.append(f"  合计 {passed}/{len(self.steps)} 件认出来了")
        return "\n".join(lines)


def draw_howto(image, step: Step, alignment: Alignment, params) -> None:
    """在起点区上画出这一下该怎么走。箭头的起点和终点画在真实动作的位置。"""
    zone = alignment.start_zone
    draw_region(image, zone, YELLOW, 2)
    cx, cy = zone.center
    if step.kind == FIST_MOVE and step.key in DIRECTIONS:
        ux, uy = DIRECTIONS[step.key]
        reach = params.move_hand_widths * alignment.hand_width
        # 起点画在区里靠反方向那一侧,终点是真的移过 1.5 个手宽以后的位置
        start = (int(cx - ux * reach / 2), int(cy - uy * reach / 2))
        end = (int(cx + ux * reach / 2), int(cy + uy * reach / 2))
        cv2.circle(image, start, 8, YELLOW, -1)
        cv2.arrowedLine(image, start, end, YELLOW, 4, tipLength=0.3)
    elif step.kind in (POSE_HOLD, FIST_MOVE):
        # 姿势保持、往前推、敲两下:都在起点区正中做,没有箭头可画
        cv2.circle(image, (int(cx), int(cy)), 10, YELLOW, 2)


CHECK_SCALE = 0.55

PANEL = (20, 20, 24)
PANEL_ALPHA = 0.55
"""字底下垫一层暗的。屏幕上的字要在 1–1.5 m 外看得清(`CONTEXT.md`「预览窗口」),
而画面背景是什么颜色不由我们定——浅色墙、窗帘上,灰字直接看不见。"""


def panel(image, x1: int, y1: int, x2: int, y2: int) -> None:
    """把这一块压暗一点,当字的衬底。"""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    block = image[y1:y2, x1:x2]
    cv2.addWeighted(block, 1.0 - PANEL_ALPHA, np.full_like(block, PANEL), PANEL_ALPHA, 0.0, block)


def draw_checklist(image, walk: Walkthrough) -> None:
    """右上角的清单:走到哪儿了、哪几件过了。

    靠最长那一行算左边界,不写死宽度——手势名是从演示配置里来的,写死了换个动作名
    就会被画面右边切掉。
    """
    steps = walk.steps
    widest = max(
        cv2.getTextSize(f"[x] {step.says}", FONT, CHECK_SCALE, 2)[0][0] for step in steps
    )
    x = max(12, image.shape[1] - widest - 16)
    panel(image, x - 10, 6, image.shape[1], 30 + len(steps) * 26)
    for i, step in enumerate(steps):
        if step.done_at is not None:
            mark, color = "[x]", GREEN
        elif i == walk.current:
            mark, color = "[>]", WHITE
        else:
            mark, color = "[ ]", GREY
        cv2.putText(
            image, f"{mark} {step.says}", (x, 30 + i * 26), FONT, CHECK_SCALE, color, 2
        )


def draw_live(image, judge: TriggerJudge, alignment: Alignment) -> None:
    """判定这一刻的样子:READY、方向箭头、保持进度。"""
    zone = alignment.start_zone
    if judge.ready:
        color = GREY if judge.ready_dim else GREEN
        cv2.putText(image, "READY", (zone.x, max(24, zone.y - 12)), FONT, 1.0, color, 3)
    if judge.direction is not None and judge.direction not in DIRECTIONS:
        cx, cy = zone.center
        # 往前推、敲两下画不出箭头,写上认成了哪一个(和预览窗口同一套字)
        cv2.putText(
            image, FIST_HOWTO[judge.direction], (int(cx), int(cy)), FONT, 1.0, GREEN, 3
        )
    elif judge.direction is not None:
        ux, uy = DIRECTIONS[judge.direction]
        cx, cy = zone.center
        reach = alignment.hand_width * 2
        cv2.arrowedLine(
            image,
            (int(cx), int(cy)),
            (int(cx + ux * reach), int(cy + uy * reach)),
            GREEN,
            5,
            tipLength=0.3,
        )
    if judge.hold_progress > 0.0:
        cx, cy = zone.center
        cv2.ellipse(
            image,
            (int(cx), int(cy)),
            (40, 40),
            -90.0,
            0.0,
            360.0 * judge.hold_progress,
            GREEN,
            5,
        )


def run(
    source: FrameSource,
    *,
    log,
    beeps: Beeper,
    walk: bool,
    snap: Path | None = None,
    seconds: float | None = None,
) -> int:
    """开摄像头 → 启动对准 → 走一遍。全走完返回 0,有没认出来的返回 1。"""
    demo = load_demo()
    params = demo.vision
    recognizer = GestureRecognizer()
    frame = first_frame(source)
    aligner = Aligner(
        (frame.width, frame.height),
        params,
        on_switch=lambda _region: recognizer.clear_tracks(),
    )
    steps = Walkthrough(plan(demo)) if walk else None
    poses = tuple(t.key for t in demo.triggers if t.kind == POSE_HOLD)

    headless = snap is not None
    if not headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, frame.width, frame.height)

    judge: TriggerJudge | None = None
    alignment: Alignment | None = None
    meter = FpsMeter()
    deadline = None if seconds is None else time.monotonic() + seconds
    last_image = None

    while frame is not None:
        result = recognizer.recognize(frame, aligner.send)
        image = frame.image.copy()

        if alignment is None:
            alignment = aligner.update(result)
            draw_align(image, result, aligner, fps=meter.tick())
            if alignment is not None:
                beeps.aligned()
                log.info("%s", alignment.describe())
                judge = TriggerJudge(
                    alignment,
                    params,
                    poses=poses,
                    stop_frames=demo.run.stop_gesture_frames,
                )
                if steps is not None:
                    steps.start(frame.t)
        else:
            assert judge is not None
            got = judge.update(result)
            if got is not None:
                log.info("%s", got.describe())
                beeps.trigger()
            if judge.stop:
                log.info("认到停止手势(这一票只报出来,不叫停)")
            if steps is not None:
                steps.score(got, judge.stop, frame.t, log)
            _paint(image, result, judge, alignment, steps, params, meter.tick())
            if steps is not None and steps.over:
                last_image = image
                break

        last_image = image
        if not headless:
            cv2.imshow(WINDOW, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                log.info("按键退出")
                break
            if key in (ord("n"), ord("N")) and steps is not None and alignment is not None:
                steps.skip(frame.t, log)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        if deadline is not None and time.monotonic() > deadline:
            break
        frame = next_frame(source)

    cv2.destroyAllWindows()
    if snap is not None and last_image is not None:
        save_snap(snap, last_image)
    if alignment is None:
        log.warning("没对上,走一遍没开始")
        return 1
    if steps is None:
        return 0
    # 日志本来就往终端抄一份,这里再 print 一遍会出两张表
    log.info("%s", steps.summary())
    return 0 if steps.all_passed else 1


def _paint(
    image,
    result: FrameResult,
    judge: TriggerJudge,
    alignment: Alignment,
    walk: Walkthrough | None,
    params,
    fps: float,
) -> None:
    """把这一帧该显示的都画上去。`walk` 是 `None` 就是自由模式,不提示也不计分。"""
    draw_region(image, alignment.detection_region, GREEN)
    draw_boxes(image, result, with_id=False)
    height = image.shape[0]
    if walk is not None and walk.step is not None:
        step = walk.step
        draw_howto(image, step, alignment, params)
        panel(image, 0, height - 130, image.shape[1], height - 78)
        centered(image, f"NOW: {step.says}", height - 100, WHITE)
        if step.done_at is not None:
            centered(image, "OK", height - 50, GREEN)
        draw_checklist(image, walk)
    else:
        draw_region(image, alignment.start_zone, YELLOW, 2)
    draw_live(image, judge, alignment)
    if judge.stop:
        centered(image, "STOP GESTURE", 60, RED)
    cv2.putText(image, f"{fps:5.1f} FPS", (12, 26), FONT, 0.7, WHITE, 2)
    keys = "N = SKIP   Q / ESC = QUIT" if walk is not None else "Q / ESC = QUIT"
    cv2.putText(image, keys, (12, height - 14), FONT, 0.6, GREY, 2)


def main(argv: list[str] | None = None) -> int:
    demo = load_demo()
    parser = argparse.ArgumentParser(description="触发判定走一遍")
    parser.add_argument(
        "--camera", type=int, default=demo.vision.camera_index, help="摄像头编号"
    )
    parser.add_argument("--video", type=Path, default=None, help="放录像,不开摄像头")
    parser.add_argument(
        "--free", action="store_true", help="不提示、不计分,只显示认到了什么"
    )
    parser.add_argument("--seconds", type=float, default=None, help="跑这么多秒自动退出")
    parser.add_argument(
        "--snap", type=Path, default=None, help="自测用:不开窗口,跑完存一张画好的图"
    )
    args = parser.parse_args(argv)
    if args.snap is not None and args.seconds is None and args.video is None:
        args.seconds = 5.0

    width, height = demo.vision.camera_px
    source: FrameSource = (
        VideoFrames(args.video)
        if args.video is not None
        else CameraFrames(args.camera, width, height)
    )
    log, _path = runlog.start("triggers")
    try:
        with source, Beeper() as beeps:
            return run(
                source,
                log=log,
                beeps=beeps,
                walk=not args.free,
                snap=args.snap,
                seconds=args.seconds,
            )
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

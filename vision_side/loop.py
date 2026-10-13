"""识别线程,以及它接到机器人一侧的那两条线(spec「线程」)。

一圈是:取帧 → 翻转(取帧那一层翻的)→ 识别 → 触发判定 → 结果进队列。认出触发手势
就往状态逻辑排一件事,认出停止手势**就地叫停**。机器人一侧的状态逻辑、叫停通道都
照用,走的是和调试按键、空格同一条路。

分成两半,为的是能不起线程就测完:

- `VisionLoop` 是一帧一步的那一圈(`step()`),测试里同步驱动,先后关系定死;
- `VisionThread` 只管把它放到自己的线程上跑,结果搁进一个位置让主线程来取。

几条定下来的规矩:

- **停止手势按「这一次」算,不按「这一帧」算。**判定那边只要手还举着就一帧一帧都说
  「认到了」,每帧都叫一次的话记录会越堆越多:执行中第一次叫停让状态逻辑进回待命,
  紧接着的几条就被当成「回待命途中又被叫停」,直接进已停止。所以只在**从没认到变成
  认到**的那一下叫一次(和空格只认按下去那一下是同一个道理)。
- **送哪块画面按状态定**(ADR-0008):空闲、冷却送检测区域,执行中、回待命、已停止送
  整幅画面。换的时候清跟踪记录和触发判定记录,**停止手势的连续帧计数不清**。
- **主线程交代的事不当场做。**主线程会来说「状态变了」「清一下判定记录」「按 R 重新
  对准」,这些都记下来,由识别线程在下一帧开头自己做——判定和识别器都是一个线程用的
  (`Recognizer`、`TriggerJudge` 上都写着),别的线程伸手进去就乱了。
- **识别中断判定在主线程**(`VisionLost`),因为它本来就在取队列里的结果。识别线程
  出错整条结束时,主线程也是从「收不到新结果」这一条发现的。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from robot_side.beeps import Beeper
from robot_side.config import STOP_GESTURE_LABELS, DemoConfig
from robot_side.state import Event, Machine, State
from robot_side.stop import StopSignal
from vision_side.align import Aligner, Alignment
from vision_side.frames import Frame, FrameError, FrameSource
from vision_side.recognize import FrameResult, Recognizer, Region
from vision_side.triggers import FIST_MOVE, POSE_HOLD, Trigger, TriggerJudge

VISION_LOST = "VISION LOST"
"""识别中断在预览窗口上那一行短句(`CONTEXT.md`「识别中断」)。"""

FULL_FRAME_STATES = (State.RUNNING, State.RETURNING, State.HALTED)
"""这几个状态送整幅画面,好让画面里任何人都能用停止手势叫停(ADR-0008)。

已停止时不认任何手势,送哪块都一样;跟着送整幅画面,一次运动就只切两次。
"""

READY_S = 1.0
"""启动检查第 2 步:要连续这么久都有识别结果(spec「启动顺序」)。"""

ARROW_S = 1.0
"""认出方向以后,箭头在屏幕上再留多久。

判定那边只在**认出来的那一帧**报方向(`TriggerJudge.direction` 每帧开头就清掉),
30 fps 下就是 33 ms——屏幕上根本看不见。按住一会儿,人才看得出程序认成了哪个方向
(spec 第 21 条 user story)。
"""

AFTER_TRIGGER_FULL_FRAME_S = 1.0
"""触发成立以后,先自作主张送多久整幅画面。

spec 说「触发成立以后从**下一帧**起送整幅画面」。光等主线程把新状态传下来是不够的:
主线程会卡住(拖预览窗口卡过 6.8 秒),那期间机械臂已经在动,而这边还只送检测区域,
站在区外的人做停止手势就看不见了(ADR-0008、ADR-0012)。所以认出触发的下一帧就切,
不等主线程。

**只有空闲时认出来的才切。**别的状态下状态逻辑本来就会忽略这一件,切过去纯属白切:
紧接着那一秒正好是人重做手势的时候,而整幅画面里拳头只有约 21 px、检测区域里约 34 px
(ADR-0008)。2026-09-22 站着验收时,17 次手势里有 5 次落在冷却上,每次都白切了一秒。

空闲时切了、状态却一直没变(演示配置里没有对应的预设动作之类),过了这么久就自己
退回检测区域——多送一秒整幅画面不会错过什么,但不能一直送下去。
"""

NOTICE_S = 1.6
"""认出来但没被受理的那一件,在屏幕上报多久。

比箭头留得久一点:箭头是「做对了」的回执,这一条是「白做了,再来一次」,人得先看懂
再重做。比冷却(2 秒)短一点,免得话还挂着、其实已经可以做了。
"""

IDLE_SLEEP_S = 0.002
"""识别线程这一次没读到画面时歇多久。摄像头本来就会把线程按在读帧上,这只挡住
「一路读空」时的空转。"""


SLOW_TICK_S = 0.2
"""主循环一轮超过这么久就记一条。

倒计时按秒显示、冷却 2 秒,卡 0.2 秒还看不出来;再久就该留痕迹了。
"""


# ---- 循环卡没卡 --------------------------------------------------------


class LoopHealth:
    """一条循环每轮花了多久,卡过几次。

    上一轮录制循环卡过 6 次 0.3–1.4 秒,不在读帧和识别,只剩显示和等按键
    (spec「录制工具」最后一条)。这一票把显示和读帧识别分到了不同线程,到底还卡不卡
    得有数才说得清——所以主循环和识别线程各挂一个,退出时把数写进日志。

    **自己不写日志**(除了卡过那几下):每轮都写反而会把循环拖慢,那正是要查的东西。
    """

    def __init__(
        self,
        what: str,
        log: logging.Logger | None = None,
        *,
        slow_s: float = SLOW_TICK_S,
    ) -> None:
        self.what = what
        self.slow_s = slow_s
        self._log = log if log is not None else logging.getLogger("vision_side.loop")
        self._spans: list[float] = []
        self.slow = 0

    def tick(self, spent_s: float) -> None:
        self._spans.append(spent_s)
        if spent_s >= self.slow_s:
            self.slow += 1
            self._log.warning("%s 这一轮卡了 %.2f 秒", self.what, spent_s)

    def describe(self) -> str:
        """退出时写进日志的那一行。"""
        if not self._spans:
            return f"{self.what} 一轮都没跑过"
        spans = sorted(self._spans)
        p99 = spans[min(len(spans) - 1, int(len(spans) * 0.99))]
        return (
            f"{self.what} {len(spans)} 轮:p50 {spans[len(spans) // 2] * 1000:.0f} ms、"
            f"p99 {p99 * 1000:.0f} ms、最长 {spans[-1] * 1000:.0f} ms;"
            f"卡过 {self.slow} 次(超过 {self.slow_s:.1f} 秒算)"
        )


# ---- 报上去的东西 ------------------------------------------------------


@dataclass(frozen=True)
class AlignView:
    """还在启动对准时,屏幕上要显示的东西。"""

    hint: str
    progress: float
    too_close: bool


@dataclass(frozen=True)
class JudgeView:
    """对上以后,触发判定这一刻的样子。屏幕上画 READY、箭头、进度圈用。

    判定本身只在识别线程里用,所以这里把要画的几样**抄一份**带出去,主线程不去读它。
    """

    ready: bool
    ready_dim: bool
    direction: str | None
    """刚认出来的那个方向。认出来之后**按住约 `ARROW_S`**,不然屏幕上只闪一帧。"""
    direction_track: int
    """做这一下的是哪只手,画箭头时按它找手框。`-1` 是不知道。

    不拿「画面里最大的手」顶替:触发那一下锁定已经解除了(ADR-0010),最大的手
    可能已经换成别人的了,箭头会画到旁人手上。
    """
    hold_progress: float
    holding: str | None


@dataclass(frozen=True)
class Ignored:
    """认出来了、但状态逻辑不会受理的一件触发手势。

    屏幕上要报一下,不然人做完什么也没发生,只能再做一次——2026-09-22 站着验收时
    17 次手势里有 5 次落在冷却上,每次都白做。`state` 是挡下它的那个状态。
    """

    kind: str
    key: str
    state: State


@dataclass(frozen=True)
class VisionUpdate:
    """识别线程报上来的一帧。主线程拿它画预览窗口、判识别中断。

    `image` 是**已经翻转过**的整幅画面,识别线程不再动它,所以主线程读它是安全的。
    `sent` 是这一帧送进检测器的那块(`None` 是整幅画面)。
    """

    t: float
    image: np.ndarray
    result: FrameResult
    sent: Region | None
    alignment: Alignment | None
    align: AlignView | None
    judge: JudgeView | None
    trigger: Trigger | None
    """这一帧认出来的触发手势。已经排出去了,这里只为画面和日志。"""
    stop: bool
    """这一帧认到停止手势没有。"""
    ignored: Ignored | None = None
    """认出来但不会被受理的那一件。认出来之后**按住约 `NOTICE_S`**,人才看得见。"""


# ---- 接到机器人一侧的两条线 --------------------------------------------


class RobotSide:
    """把识别线程认出来的东西送到机器人一侧**已有的**两条路上。

    - 触发手势 → 状态逻辑的事件队列(和调试按键同一条);
    - 停止手势 → 叫停通道(和空格同一条,ADR-0012:直达机械臂,不经过主线程)。

    **连机械臂之前也照样建起来**:启动对准在连机械臂之前做(spec「启动顺序」),
    那段时间认出来的东西没处排,就只记一笔。`attach()` 由主线程在连上以后调一次。
    """

    def __init__(
        self,
        demo: DemoConfig,
        *,
        machine: Machine | None = None,
        signal: StopSignal | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._demo = demo
        self._machine = machine
        self._signal = signal
        self._log = log if log is not None else logging.getLogger("vision_side.loop")

    def attach(self, machine: Machine | None, signal: StopSignal | None) -> None:
        """接上状态逻辑和叫停通道。主线程在连上机械臂以后调一次。

        识别线程只是读这两个名字(Python 里读写属性本身是原子的),接上之前读到的是
        `None`,那就只记一笔;不会读到半接不接的东西。
        """
        self._machine = machine
        self._signal = signal

    def to_event(self, trigger: Trigger) -> Event | None:
        """这件触发手势该排成哪一件事。演示配置里没有对应的预设动作就是 `None`。

        对应关系**查演示配置**,和调试按键一样(`robot_side.preview`),所以改了哪个
        手势对哪个动作,两边都不用改代码。
        """
        action = self._demo.action_for_trigger(trigger.kind, trigger.key)
        if action is None:
            return None
        return Event.trigger(action, f"gesture:{trigger.kind}:{trigger.key}")

    def trigger(self, trigger: Trigger) -> None:
        """认出一件触发手势。在识别线程上调。"""
        event = self.to_event(trigger)
        if event is None:
            self._log.warning(
                "触发 演示配置里没有 %s %s 对应的预设动作,忽略", trigger.kind, trigger.key
            )
            return
        machine = self._machine
        if machine is None:
            self._log.info("触发 %s(还没连机械臂,只记一笔)", trigger.describe())
            return
        machine.submit(event)

    def stop(self) -> None:
        """认到停止手势:就地叫停。在识别线程上调(ADR-0012)。

        机械臂那边报错时**不往外抛**:抛了会把识别线程整条带倒,之后就再也叫不了停。
        标记这时已经立住了,和按键线程的办法一样——记下来,线程接着活着。
        """
        signal = self._signal
        if signal is None:
            self._log.info("叫停 认到停止手势(还没连机械臂,只记一笔)")
            return
        try:
            signal.raise_stop("stop-gesture")
        except Exception as exc:
            self._log.warning("叫停 机械臂的「停止」报错:%s: %s", type(exc).__name__, exc)


# ---- 识别线程干的活 ----------------------------------------------------


class VisionLoop:
    """一帧一步的那一圈。**一个线程用**(识别线程)。

    `set_state` / `clear_records` / `realign` 可以从主线程调,它们只是留个话;
    真正做是在下一次 `step()` 的开头。
    """

    def __init__(
        self,
        source: FrameSource,
        recognizer: Recognizer,
        demo: DemoConfig,
        *,
        frame_size: tuple[int, int],
        robot: RobotSide,
        beeps: Beeper | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._source = source
        self._recognizer = recognizer
        self._demo = demo
        self._params = demo.vision
        self._robot = robot
        self._beeps = beeps
        self._log = log if log is not None else logging.getLogger("vision_side.loop")
        self._size = frame_size
        self._poses = tuple(t.key for t in demo.triggers if t.kind == POSE_HOLD)

        self._aligner = self._new_aligner()
        self._alignment: Alignment | None = None
        self._judge: TriggerJudge | None = None
        self._sending: Region | None = None
        self._stop_last_frame = False
        """上一帧算不算认到停止手势。只在从假变真的那一下叫停。"""
        self._arrow: tuple[str, int, float] | None = None
        """刚认出来的方向:方向、哪只手、哪一刻。按住 `ARROW_S` 再撤。"""
        self._notice: tuple[Ignored, float] | None = None
        """认出来但没被受理的那一件,和认出来的时刻。按住 `NOTICE_S` 再撤。"""
        self._state_now: State | None = None
        """这一帧按哪个状态判的(`_prepare` 记下,`_judge_frame` 用)。"""
        self._triggered_at: float | None = None
        """上一次认出触发手势是哪一刻。切整幅画面不等主线程,就看它。"""

        self._orders = threading.Lock()
        self._state: State | None = None
        self._pending_clear = False
        self._pending_realign = False

    # ---- 主线程会调的 --------------------------------------------------

    @property
    def alignment(self) -> Alignment | None:
        """启动对准的结果。还没对上就是 `None`。"""
        return self._alignment

    def set_state(self, state: State | None) -> None:
        """告诉它机械臂现在是什么状态。送哪块画面按这个定。

        `None` 是「还没连机械臂」(启动对准那一段),这时机械臂不会动,按空闲办。
        """
        with self._orders:
            self._state = state

    def clear_records(self) -> None:
        """清一次触发判定记录。状态逻辑进冷却前会调(它的 `on_clear_records`)。"""
        with self._orders:
            self._pending_clear = True

    def realign(self) -> None:
        """从头对一次(空闲时按 R)。对准期间判定不在,所以也就不会有触发手势。"""
        with self._orders:
            self._pending_realign = True

    # ---- 一帧一步 ------------------------------------------------------

    def step(self) -> VisionUpdate | None:
        """走一帧。这一次没读到画面就返回 `None`(不算错,摄像头偶尔读空)。"""
        frame = self._source.next_frame()
        if frame is None:
            return None
        region = self._prepare(frame.t)
        result = self._recognizer.recognize(frame, region)
        if self._alignment is None:
            return self._align(frame, result, region)
        return self._judge_frame(frame, result, region)

    # ---- 内部 ----------------------------------------------------------

    def _new_aligner(self) -> Aligner:
        # 换送法时清掉上游的跟踪记录:换了一块画面,同一只手的位置是跳变的(ADR-0008)。
        # 对准那一段的切换由 `Aligner` 自己管,所以挂在它的 `on_switch` 上。
        return Aligner(
            self._size,
            self._params,
            on_switch=lambda _region: self._recognizer.clear_tracks(),
        )

    def _prepare(self, t: float) -> Region | None:
        """这一帧开头:先办主线程交代的事,再定这一帧送哪块画面。"""
        with self._orders:
            state = self._state
            clear = self._pending_clear
            realign = self._pending_realign
            self._pending_clear = False
            self._pending_realign = False

        if realign:
            self._log.info("对准 收到重新对准,从头对一次")
            self._aligner = self._new_aligner()
            self._alignment = None
            self._judge = None
            self._sending = None
            # 停止手势的连续帧计数跟着判定一起没了。重新对准时机械臂停在待命位姿
            # 不动(`App._on_realign` 只在空闲时才让重新对准),漏不掉什么
            self._stop_last_frame = False
            self._arrow = None
            self._notice = None
            self._triggered_at = None

        # 这一帧按哪个状态判的,`_judge_frame` 认出触发时要回头看它
        self._state_now = state

        if self._alignment is None:
            # 还在对准:送哪块由对准自己说了算(先整幅画面找拳头,再切检测区域量手宽)
            return self._aligner.send

        # 触发成立以后**不等主线程**就切整幅画面(见 `AFTER_TRIGGER_FULL_FRAME_S`)
        just_triggered = (
            self._triggered_at is not None
            and (t - self._triggered_at) <= AFTER_TRIGGER_FULL_FRAME_S
        )
        full = state in FULL_FRAME_STATES or just_triggered
        want = None if full else self._alignment.detection_region
        switched = want != self._sending
        if switched:
            self._log.info(
                "送 %s → %s",
                "检测区域" if self._sending is not None else "整幅画面",
                "检测区域" if want is not None else "整幅画面",
            )
            self._sending = want
            self._recognizer.clear_tracks()
        if (switched or clear) and self._judge is not None:
            # **停止手势的连续帧计数不清**:它按画面计、不看跟踪编号,清了反而可能
            # 晚停约 0.1 秒(spec「送哪块画面」)。`TriggerJudge.clear` 保证这一点
            self._judge.clear()
        return want

    def _align(
        self, frame: Frame, result: FrameResult, region: Region | None
    ) -> VisionUpdate:
        """还在启动对准这一段。"""
        alignment = self._aligner.update(result)
        view = AlignView(
            hint=self._aligner.hint,
            progress=self._aligner.progress,
            too_close=self._aligner.too_close,
        )
        if alignment is not None:
            self._alignment = alignment
            self._sending = self._aligner.send
            self._judge = TriggerJudge(
                alignment,
                self._params,
                poses=self._poses,
                stop_frames=self._demo.run.stop_gesture_frames,
            )
            self._log.info("%s", alignment.describe())
            if alignment.too_close:
                self._log.warning("对准 TOO CLOSE:站太近了;只提示,不拦着")
            if self._beeps is not None:
                self._beeps.aligned()  # 站在远处也知道这一步过了
        return VisionUpdate(
            t=frame.t,
            image=frame.image,
            result=result,
            sent=region,
            alignment=alignment,
            align=view,
            judge=None,
            trigger=None,
            stop=False,
        )

    def _judge_frame(
        self, frame: Frame, result: FrameResult, region: Region | None
    ) -> VisionUpdate:
        """对上以后:三套规则走一遍,该叫停的叫停,该排的排。"""
        judge = self._judge
        assert judge is not None
        got = judge.update(result)

        # 叫停排在触发前面:同一帧两样都有时,先把机械臂停住
        stop = judge.stop
        if stop and not self._stop_last_frame:
            self._log.info("叫停 认到停止手势,就地叫停  %s", _stop_hands(result))
            self._robot.stop()
        self._stop_last_frame = stop

        if judge.direction is not None:
            # 哪只手做的,从这一件触发手势上拿。同一帧两套规则都成立时交出去的可能是
            # 姿势保持那一件(`TriggerJudge` 排了个小队),那就不知道是哪只手,画在
            # 起点区中间——认成了哪个方向照样看得见
            track = got.track_id if got is not None and got.kind == FIST_MOVE else -1
            self._arrow = (judge.direction, track, frame.t)
        if got is not None:
            self._log.info("%s", got.describe())
            if self._state_now is not None and self._state_now is not State.IDLE:
                # 认出来了,但状态逻辑不会受理。屏幕上报一下,不然人做完什么也没发生
                self._notice = (Ignored(got.kind, got.key, self._state_now), frame.t)
            if self._state_now is State.IDLE:
                # **只有空闲时认出来的才算数。**别的状态下状态逻辑本来就会忽略这一件
                # (冷却、执行中、已停止),这时切过去就是白切:紧接着的那一秒正好是
                # 人重做手势的时候,而整幅画面里拳头只有约 21 px、检测区域里约 34 px
                # (ADR-0008),白切反而让重做更难认。2026-09-22 站着验收时,17 次手势
                # 里有 5 次落在冷却上,每次都白切了一秒
                self._triggered_at = frame.t
            self._robot.trigger(got)

        direction, direction_track = self._arrow_now(frame.t)
        return VisionUpdate(
            t=frame.t,
            image=frame.image,
            result=result,
            sent=region,
            alignment=self._alignment,
            align=None,
            judge=JudgeView(
                ready=judge.ready,
                ready_dim=judge.ready_dim,
                direction=direction,
                direction_track=direction_track,
                hold_progress=judge.hold_progress,
                holding=judge.holding,
            ),
            trigger=got,
            stop=stop,
            ignored=self._notice_now(frame.t),
        )

    def _arrow_now(self, t: float) -> tuple[str | None, int]:
        """这一刻箭头该画哪个方向、画在哪只手旁边。过了 `ARROW_S` 就撤掉。"""
        if self._arrow is None:
            return None, -1
        direction, track_id, since = self._arrow
        if t - since > ARROW_S:
            self._arrow = None
            return None, -1
        return direction, track_id

    def _notice_now(self, t: float) -> Ignored | None:
        """这一刻要不要报「认出来了但没受理」。过了 `NOTICE_S` 就撤掉。"""
        if self._notice is None:
            return None
        ignored, since = self._notice
        if t - since > NOTICE_S:
            self._notice = None
            return None
        return ignored


def _stop_hands(result: FrameResult) -> str:
    """叫停的那一帧里是哪几只手张开着:姿势、跟踪编号、手框中心和宽(用户故事 37)。

    停止手势按整幅画面认、任何一只手都算,所以光说「认到了」查不出是谁:2026-09-23
    实跑五次叫停有四次紧跟在「拳头往上」后面 1–3 秒,日志里看不出是不是同一只手。
    """
    return "、".join(
        f"{h.label.upper()} 手 #{h.track_id} 在 ({h.center[0]:.0f}, {h.center[1]:.0f}) "
        f"宽 {h.width:.0f} px"
        for h in result.hands
        if h.label in STOP_GESTURE_LABELS
    )


class VisionThread:
    """把 `VisionLoop` 放到自己的线程上跑,最新的一帧搁在一个位置上等主线程来取。

    **只留最新的一帧,不排队。**主线程要画的就是「现在看见什么」;真排成队列的话,
    主线程一卡(拖窗口卡过 6.8 秒),攒下的几百帧会一帧一帧慢慢补画,越补越落后。
    识别中断也就分不清「没有新结果」和「队列还没排到」。

    线程出错就**结束**,不硬撑:主线程因此收不到新结果,按识别中断处理
    (spec「线程」)。错误留在 `error` 里,由主线程写进日志。
    """

    def __init__(self, loop: VisionLoop, *, log: logging.Logger | None = None) -> None:
        self._loop = loop
        self._log = log if log is not None else logging.getLogger("vision_side.loop")
        self._lock = threading.Lock()
        self._slot: VisionUpdate | None = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def error(self) -> BaseException | None:
        """识别线程出的错。没出错就是 `None`。"""
        return self._error

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()

    def take(self) -> VisionUpdate | None:
        """取一帧新的。上次取过之后没有新的就是 `None`(识别中断据此数时间)。"""
        with self._lock:
            update, self._slot = self._slot, None
            return update

    def close(self, timeout_s: float = 2.0) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                self._log.warning("识别线程 %.0f 秒还没结束;它是 daemon,不挡退出", timeout_s)
            self._thread = None

    def __enter__(self) -> VisionThread:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _run(self) -> None:
        while self._running.is_set():
            try:
                update = self._loop.step()
            except BaseException as exc:  # 出错就结束,主线程按识别中断处理
                self._error = exc
                self._log.exception("识别线程出错,这一条就到这儿了:%s", exc)
                return
            if update is None:
                time.sleep(IDLE_SLEEP_S)
                continue
            with self._lock:
                self._slot = update


# ---- 识别中断(主线程) ------------------------------------------------


class VisionLost:
    """识别中断判定(VISION LOST)。主线程每轮调一次 `update()`。

    两种算:队列里超过 `after_s` 没有新的识别结果;或者最近那一份说**画面均匀**
    ——「均匀」本身已经是「连着约 0.5 秒都均匀」了(`vision_side.recognize` 里数的),
    所以这里看一眼就够。**画面正常、只是没有手,不算。**
    """

    def __init__(
        self, after_s: float, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.after_s = after_s
        self._now = clock
        self._last_at = clock()
        self._flat = False
        self._lost = False
        self._reported = False
        """这一次中断已经排过一件停机没有。每轮都排的话已停止会被反复重报。"""

    @property
    def lost(self) -> bool:
        return self._lost

    @property
    def why(self) -> str:
        """写日志用的那半句。屏幕上一律是 `VISION_LOST`。"""
        if not self._lost:
            return ""
        return "画面是一片均匀的" if self._flat else f"超过 {self.after_s:.1f} 秒没有新结果"

    def update(self, update: VisionUpdate | None) -> bool:
        """喂这一轮取到的新结果(没有新的就给 `None`),返回这一刻算不算识别中断。"""
        now = self._now()
        if update is not None:
            self._last_at = now
            self._flat = update.result.flat
        self._lost = self._flat or (now - self._last_at) > self.after_s
        if not self._lost:
            self._reported = False  # 画面回来了,下次再断还要再报一次
        return self._lost

    def halt_event(self, state: State) -> Event | None:
        """这一轮要不要往状态逻辑排一件停机。

        执行中或回待命途中 → 进已停止;**空闲(含冷却)时只在预览窗口报错,不惊动
        机械臂**(它本来就停着)。已停止时也不必再报。同一次中断只排一件。
        """
        if not self._lost or self._reported:
            return None
        if state not in (State.RUNNING, State.RETURNING):
            return None
        self._reported = True
        return Event.halt(VISION_LOST, "vision")


# ---- 启动检查第 2 步 ---------------------------------------------------


def check_vision(
    source: FrameSource,
    recognizer: Recognizer,
    *,
    need_s: float = READY_S,
    timeout_s: float = 5.0,
    gap_s: float = 0.5,
    log: logging.Logger | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """要求连续约 `need_s` 都有识别结果,且画面不是均匀的(spec「启动顺序」第 2 步)。

    不满足就抛 `FrameError`,由入口报错退出——**这一步还没碰过机械臂**。
    「有识别结果」不是「看得见手」:画面里没有人也算数,只要识别跑得起来、画面有东西。
    """
    log = log if log is not None else logging.getLogger("vision_side.loop")
    deadline = clock() + timeout_s
    started: float | None = None
    last_t: float | None = None
    while True:
        frame = source.next_frame()
        if frame is None:
            if not source.has_next():
                raise FrameError("这一路画面已经没有了", reason="NO FRAMES")
            if clock() > deadline:
                raise FrameError(
                    f"等了 {timeout_s:.1f} 秒也没能连着取到画面", reason="NO CAMERA IMAGE"
                )
            time.sleep(IDLE_SLEEP_S)
            continue
        result = recognizer.recognize(frame, None)
        if result.flat:
            raise FrameError(
                "画面是一片均匀的(摄像头隐私开关关着?),看不见停止手势",
                reason="CAMERA BLANK",
            )
        if started is None or (last_t is not None and frame.t - last_t > gap_s):
            started = frame.t  # 中间断过就从头数:要的是「连续」这么久
        last_t = frame.t
        if frame.t - started >= need_s:
            log.info("启动检查 识别连着跑了 %.1f 秒,画面正常", frame.t - started)
            return
        if clock() > deadline:
            raise FrameError(
                f"等了 {timeout_s:.1f} 秒,识别还没连着跑满 {need_s:.1f} 秒",
                reason="VISION NOT READY",
            )

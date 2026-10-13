"""启动对准(ALIGN):放下检测区域和起点区,量出手宽。

站到操作位、把拳头举起来停住约 2 秒,程序就把区域放下——举在哪里都行,不要求胸前,
也不用回电脑按键(`CONTEXT.md`「启动对准」)。这一步在**连机械臂之前**做完:机械臂
连上以后第一件事就是回待命,那时停止手势必须已经能用(ADR-0008)。

分两段,中间换一次送进检测器的画面:

1. **整幅画面**里找拳头,停稳 `hold_s`——期间手框中心移动不超过半个手宽,至少八成帧
   是握拳类标签(没认出手的帧也算在内)。这一段量到的宽**只用来判断站没站太近**;
2. 以拳头位置为中心放下检测区域,**切过去以后再量手宽** `measure_s`,取这段时间拳头
   手框宽的中位数。两段合起来约 2 秒;这一段里认出的拳头太少就自动延长。

两处量到的宽差多少每次都不一样(原型实测 1–16%,2026-09-21 站着实测 11.6%),不能按
固定比例折算,所以**手宽一律是检测区域里量到的那一个**,整幅画面里的那个只留着判
TOO CLOSE 和记进日志。

几条定下来的规矩:

- **手宽量定以后不再变**:`Alignment` 是只读的,对上以后 `update()` 一直返回同一份。
  握拳平移的锁定怎么解除依赖手宽固定,改成逐帧量会让同一个动作时灵时不灵(spec)。
- **站太近只提示,不拦着**:整幅画面里的拳头手框宽超过 `too_close_px` 就把 `too_close`
  亮起来,对准照走;退不退一步由操作者自己定(第 7 条 user story)。
- **换送法时要清掉跟踪记录**:换了一块画面,同一只手在送进去的那块里的位置是跳变的
  (ADR-0008)。这里只管「什么时候换」,清由识别器做——把 `clear_tracks` 传进
  `on_switch` 即可。
- **这一层不碰摄像头、不碰模型**:吃的是识别结果那个缝出来的东西,所以喂编好的结果
  就能整条测完(`tests/test_align.py`)。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

from robot_side.config import FIST_LIKE_LABELS, VisionParams
from vision_side.recognize import FrameResult, HandResult, Region

MIN_WIDTHS = 3
"""一段里至少要量到几个拳头手框宽,中位数才算数。不够就延长(spec「启动对准」)。

这是「中位数要几个数才有意义」的下限,不是调参用的门槛,所以留在代码里不进演示配置
(和 `vision_side/recognize.py` 里那几个像素门槛同一档)。
"""

MEASURE_EXTEND = 2.0
"""量手宽那一段最多延长到 `measure_s` 的几倍。延到头还不够,这一次就不算。"""

SEEK_HINT = "STAND AT THE MARK, RAISE YOUR FIST"
"""还没找到拳头时屏幕上那一行。怎么站、怎么举都写在里面(一律英文,`CONTEXT.md`)。"""

HOLD_HINT = "HOLD YOUR FIST STILL"
"""找到了、正在停稳时那一行。"""

ALIGNED_HINT = "ALIGNED"
"""对上了。这一刻响一声、画出两个边框,自动继续。"""

TOO_CLOSE_HINT = "TOO CLOSE - STEP BACK"
"""站太近时**另加**的一行。只提示,不拦着对准。"""


@dataclass(frozen=True)
class Alignment:
    """一次启动对准的结果。后面所有判定的输入,量定以后不再变。

    `detection_region` 是检测区域,`start_zone` 是起点区,两个都是整幅画面的坐标;
    `hand_width` 是**检测区域里**量到的拳头手框宽中位数(单位像素,`CONTEXT.md`
    「手宽」);`full_frame_width` 是整幅画面里量到的那一个,只用来判 `too_close`;
    `t` 是对准成立的那一帧的时间戳。
    """

    detection_region: Region
    """检测区域(DETECTION REGION)。不简写成 detection——上游把每一帧检出来的手框
    也叫 detection,两个凑在一起会看错。"""
    start_zone: Region
    hand_width: float
    full_frame_width: float
    too_close: bool
    t: float

    @property
    def width_gap_percent(self) -> float:
        """整幅画面里量到的宽比检测区域里的大百分之几。

        原型在 1.2 m 处实测这个数是 1–16%,每次都不一样;2026-09-21 用户本人站在操作位
        上跑了一次,检测区域里 73.0 px、整幅画面里 81.5 px,差 11.6%,落在这个区间里。
        (同一天拿坐着近拍的那段录像跑,量到 35%——那是素材的事,不是站着的情形。)
        **不能拿它折算**,记下来只为事后对着看(spec「启动对准」)。
        """
        if self.hand_width <= 0:
            return 0.0
        return 100.0 * (self.full_frame_width - self.hand_width) / self.hand_width

    def describe(self) -> str:
        """写进日志的那一行(spec「启动对准」最后一条)。

        事后查误触发时,要能从日志里看出当时区域放在哪儿、手宽是多少
        (第 36 条 user story)。
        """
        line = (
            f"对准 DETECTION REGION {self.detection_region.x},{self.detection_region.y} "
            f"{self.detection_region.width}x{self.detection_region.height}  "
            f"START ZONE {self.start_zone.x},{self.start_zone.y} "
            f"{self.start_zone.width}x{self.start_zone.height}  "
            f"HAND WIDTH {self.hand_width:.1f} px  "
            f"FULL FRAME {self.full_frame_width:.1f} px"
        )
        if self.too_close:
            line += "  TOO CLOSE"
        return line


class _Span:
    """一段「停稳」:数帧、记下拳头的手框宽和中心,并盯着有没有动超过半个手宽。

    `frames` 数的是**这一段里的所有帧**,没认出手的也算——八成帧那一条就是按它算的
    (和姿势保持同一个口径,spec)。`widths` 只收握拳类的那些。
    """

    def __init__(
        self,
        t0: float,
        anchor: tuple[float, float],
        width: float,
        params: VisionParams,
    ) -> None:
        self.t0 = t0
        self.anchor = anchor
        self.frames = 0
        self.widths: list[float] = []
        self.centers: list[tuple[float, float]] = []
        self._width = width
        self._params = params

    @property
    def hand_width(self) -> float:
        """这一段量到的手宽:中位数。还没量到时用进来时那个估计值。"""
        return median(self.widths) if self.widths else self._width

    @property
    def share(self) -> float:
        """这一段里有几成帧是握拳类标签。"""
        return 100.0 * len(self.widths) / self.frames if self.frames else 0.0

    @property
    def enough(self) -> bool:
        """量到的拳头够不够算一个中位数。不够就延长。"""
        return len(self.widths) >= MIN_WIDTHS

    @property
    def center(self) -> tuple[float, float]:
        """这一段里拳头的位置:各取中位数,不受个别跳动的影响。"""
        if not self.centers:
            return self.anchor
        return (
            median(c[0] for c in self.centers),
            median(c[1] for c in self.centers),
        )

    def add(self, fist: HandResult | None) -> bool:
        """喂一帧。返回 `False` 表示动得太多,这一次不算。"""
        self.frames += 1
        if fist is None:
            return True
        moved = math.dist(fist.center, self.anchor)
        if moved > self._params.still_hand_widths * self.hand_width:
            return False
        self.widths.append(fist.width)
        self.centers.append(fist.center)
        return True


class Aligner:
    """启动对准的判定。吃逐帧识别结果,对上了就吐出一份 `Alignment`。

    用法(真程序和查看脚本都是这一圈):

        region = aligner.send                       # 这一帧该送哪块画面
        result = recognizer.recognize(frame, region)
        alignment = aligner.update(result)

    `on_switch` 在换送法时调一次,参数是新的那块(`None` 是整幅画面);把识别器的
    `clear_tracks` 传进来即可。一个线程用(识别线程)。
    """

    def __init__(
        self,
        frame_size: tuple[int, int],
        params: VisionParams,
        *,
        on_switch: Callable[[Region | None], None] | None = None,
    ) -> None:
        self._size = frame_size
        self._params = params
        self._on_switch = on_switch
        self._send: Region | None = None
        self.reset()

    # ---- 外面看得到的 --------------------------------------------------

    @property
    def send(self) -> Region | None:
        """这一帧该送哪块画面进检测器。`None` 是整幅画面。"""
        return self._send

    @property
    def done(self) -> Alignment | None:
        """对上了没有。对上了就是那份结果。"""
        return self._done

    @property
    def too_close(self) -> bool:
        """站得太不太近。**对准过程中就亮**,不然退一步来不及。

        要量到 `MIN_WIDTHS` 个手框宽才肯报:刚认出拳头的头一两帧手框常偏大
        (2026-09-21 在原型录像上实测,79 px 的手一开始报到过 107 px 以上),
        单帧就亮会白提示一次。
        """
        if self._done is not None:
            return self._done.too_close
        return (
            self._full_width is not None
            and self._full_width > self._params.too_close_px
        )

    @property
    def hint(self) -> str:
        """屏幕上那一行提示(英文)。太近那一行另看 `too_close`。"""
        if self._done is not None:
            return ALIGNED_HINT
        # 看 `_began` 不看 `_span`:切到检测区域以后等拳头重新被认出来的那几帧
        # `_span` 是空的,提示不该在这时跳回「举起拳头」
        return SEEK_HINT if self._began is None else HOLD_HINT

    @property
    def progress(self) -> float:
        """停稳进度,0 到 1。屏幕上画进度用,操作者知道还要举多久。"""
        if self._done is not None:
            return 1.0
        if self._began is None:
            return 0.0
        total = self._params.hold_s + self._params.measure_s
        return min(1.0, max(0.0, (self._last_t - self._began) / total))

    def reset(self) -> None:
        """从头对一次(空闲时按 R 走的也是这一条,由第 04 票接进状态逻辑)。"""
        self._done: Alignment | None = None
        self._span: _Span | None = None
        self._measuring = False
        self._anchor: tuple[float, float] | None = None
        self._detection_region: Region | None = None
        self._full_width: float | None = None
        self._began: float | None = None
        self._measure_from: float | None = None
        self._last_t = 0.0
        self._switch_to(None)

    def update(self, result: FrameResult) -> Alignment | None:
        """喂一帧识别结果。对上了返回那份结果,还没对上返回 `None`。

        对上以后再喂也一直返回同一份:手宽量定以后不再变。
        """
        if self._done is not None:
            return self._done
        self._last_t = result.t
        fist = _fist(result)

        if self._span is None:
            if fist is not None:
                self._seed(result.t, fist)
            elif self._measuring and _past(result.t, self._measure_from, self._params.measure_s):
                # 切过去以后一直没再认出拳头(手放下了、或者人不在区里):这一次不算
                self._restart()
            return None

        measuring = self._measuring
        if not self._span.add(fist):
            # 动得超过半个手宽:这一次不算,从头找
            self._restart()
            if not measuring and fist is not None:
                # 整幅画面那一段里挪了位置,这一帧仍是整幅画面的,就地当新的起点。
                # 检测区域那一段不这么办:那一帧的手框宽是区域里量的,混不得。
                self._seed(result.t, fist)
            return None

        elapsed = result.t - self._span.t0
        if measuring:
            return self._finish(result.t, elapsed)
        if self._span.enough:
            # 整幅画面里量到的宽,边停边更新:TOO CLOSE 要在对上之前就亮
            self._full_width = self._span.hand_width
        if elapsed >= self._params.hold_s:
            self._measure(result.t)
        return None

    # ---- 内部 ----------------------------------------------------------

    def _seed(self, t: float, fist: HandResult) -> None:
        """从这一帧起开始数。整幅画面那一段和检测区域那一段都从这里起算。

        **切到检测区域以后也要等拳头重新被认出来才起算**:上游的跟踪记录刚清过,同一只
        手要连上 3 帧才重新算「已确认」(`vision_side/gestures.py`),那几帧不该算进
        「八成帧是握拳类」里——30 fps 下勉强够,帧率一掉就会把好好举着的手判成不算。
        """
        if self._measuring:
            assert self._anchor is not None and self._full_width is not None
            self._span = _Span(t, self._anchor, self._full_width, self._params)
            if not self._span.add(fist):
                # 重新出现的手离停稳那一点太远(多半是别的手):这一帧不作数,
                # 等下一帧;一直等不到就由 `_measure_from` 那一条超时重来
                self._span = None
            return
        self._span = _Span(t, fist.center, fist.width, self._params)
        self._began = t
        self._span.add(fist)

    def _restart(self) -> None:
        """这一次不算,退回整幅画面重新找。

        **`_full_width` 不清**:站太近是「人站在哪儿」的事,不是「这一次对准」的事。
        清了的话,手一晃就要重新量满三帧,屏幕上的 TOO CLOSE 会一闪一闪;人退一步
        之前那一行本来就该一直亮着。下一次停稳量到新的宽就把它盖掉。
        """
        self._span = None
        self._measuring = False
        self._anchor = None
        self._detection_region = None
        self._began = None
        self._measure_from = None
        self._switch_to(None)

    def _measure(self, t: float) -> None:
        """停稳这一段到点了:放下检测区域,切过去开始量手宽。"""
        span = self._span
        assert span is not None
        if span.share < self._params.fist_percent:
            self._restart()  # 八成帧那一条没过,这一次不算
            return
        if not span.enough:
            return  # 量到的拳头太少,接着停(spec:不够就延长)

        self._anchor = span.center
        self._full_width = span.hand_width
        self._detection_region = self._around(
            self._anchor, self._params.detection_region_px
        )
        self._switch_to(self._detection_region)
        self._span = None  # 等拳头在检测区域里重新被认出来再起算(见 `_seed`)
        self._measure_from = t
        self._measuring = True

    def _finish(self, t: float, elapsed: float) -> Alignment | None:
        """量手宽这一段到点了:放下起点区,对准成立。"""
        span = self._span
        assert span is not None
        if elapsed < self._params.measure_s:
            return None
        if span.share < self._params.fist_percent or not span.enough:
            # **不够就延长**(spec「启动对准」):手好好举着、只是这一段漏检多,再给
            # 一倍时间接着量,不该把人赶回去从头停 2 秒。延到头还不够才算这一次不算。
            if elapsed >= self._params.measure_s * MEASURE_EXTEND:
                self._restart()
            return None

        assert self._anchor is not None and self._detection_region is not None
        assert self._full_width is not None
        hand_width = span.hand_width
        # 至少 1 px:手框偶尔会退化成 0 宽,边长算成 0 的话 `Region` 抛异常、
        # 把识别线程整条带倒。对准结果本身照样拿得到,离谱的手宽一看日志就知道。
        side = max(1, round(self._params.start_zone_hand_widths * hand_width))
        self._done = Alignment(
            detection_region=self._detection_region,
            start_zone=self._around(self._anchor, (side, side)),
            hand_width=hand_width,
            full_frame_width=self._full_width,
            too_close=self.too_close,
            t=t,
        )
        return self._done

    def _around(self, center: tuple[float, float], size: tuple[int, int]) -> Region:
        """以这个位置为中心放一块;超出画面时贴着画面边缘放,不缩小。"""
        cx, cy = center
        width, height = size
        return Region(
            round(cx - width / 2), round(cy - height / 2), width, height
        ).clamped_to(*self._size)

    def _switch_to(self, region: Region | None) -> None:
        """换送法。真换了才叫 `on_switch`——识别器每清一次跟踪记录,手就要重新确认。"""
        if region is self._send:
            return
        self._send = region
        if self._on_switch is not None:
            self._on_switch(region)


def _past(t: float, since: float | None, span_s: float) -> bool:
    """从 `since` 起算,过了 `span_s` 秒没有。`since` 是 `None` 时不算。"""
    return since is not None and (t - since) >= span_s


def _fist(result: FrameResult) -> HandResult | None:
    """这一帧里的拳头:画面里最大的那只已确认的手,而且是握拳类标签。

    对准这一步没有锁定(锁定是 READY 之后的事,ADR-0010),所以每帧都取最大的那只;
    「同一只手」由「移动不超过半个手宽」那一条管着。
    """
    hand = result.largest()
    if hand is None or hand.label not in FIST_LIKE_LABELS:
        return None
    return hand

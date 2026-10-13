"""触发判定:握拳平移、姿势保持、停止手势。

吃逐帧识别结果 + 启动对准的结果(起点区、手宽),吐出认出来的触发手势。**这一层只
报出来**:排进状态逻辑的事件队列、真正叫停机械臂,都是第 04 票的事。

三套规则互不相干,共用同一份逐帧结果:

- **握拳平移**(ADR-0009、ADR-0010):起点区里先停住 → READY 亮起并锁定这只手 →
  朝一个方向移过 1.5 个手宽。收手抑制挡住「做完一次往回收」那一下。前后两个方向
  (2026-09-23 用户定)也挂在锁定上:往前是握拳往镜头推(手框变大),往后是握拳像
  敲桌子往下点两下。
- **姿势保持**(ADR-0007):只看起点区里最大的那只手,同一个姿势保持约一秒;
  触发过要先松开才能再触发。**不用锁定**。
- **停止手势**(ADR-0012):按整幅画面计,任意一只手、任何人做都算,连续 3 帧。

几条定下来的规矩:

- **参数是六个原型扫参后的初值**,全部来自演示配置的 `[vision]` 一节,不在代码里另调
  ——改了就得重跑回放测试(第 05 票)。
- **没认出手的帧也算在帧占比里**(和启动对准、姿势保持同一个口径):锁定的手这一帧
  没被认出来,就当这一帧没有手。
- **清记录不清停止手势的连续帧计数**(spec「送哪块画面」):它按画面计、不看跟踪编号,
  清了反而可能晚停约 0.1 秒。
- **这一层不碰摄像头、不碰模型**,喂编好的逐帧结果就能整条测完(`tests/test_triggers.py`)。
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import NamedTuple

from robot_side.config import (
    FIST_LIKE_LABELS,
    FIST_MOVE,
    POSE_HOLD,
    STOP_GESTURE_LABELS,
    VisionParams,
)
from vision_side.align import Alignment
from vision_side.recognize import FrameResult, HandResult, Region

__all__ = ["BACK", "DIRECTIONS", "FIST_MOVE", "FORWARD", "POSE_HOLD", "Trigger", "TriggerJudge"]
"""`FIST_MOVE` / `POSE_HOLD` 原样从 `robot_side.config` 转出来:演示配置里
`[[triggers]]` 的 `kind` 用的就是它们,两边不能各拼各的。"""

#: 画面里平移的四个方向和它在画面上的单位向量。画面已经水平翻转过(取帧那一层翻的),
#: 所以屏幕上的左就是操作者的左;y 轴朝下,所以「上」是 -1。
DIRECTIONS: dict[str, tuple[float, float]] = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
}

FORWARD = "forward"
"""往前:握拳往镜头推出去停住。不在画面平面里,没有单位向量——看的是手框变大。"""
BACK = "back"
"""往后:握拳像敲桌子那样往下点两下(2026-09-23 用户定)。看的是手框中心上下走了两个来回。"""


@dataclass(frozen=True)
class Trigger:
    """认出来的一个触发手势。

    `kind` 是 `FIST_MOVE` 或 `POSE_HOLD`,`key` 是方向或姿势标签——两个合起来就能在
    演示配置里查到对应的预设动作(`DemoConfig.action_for_trigger`),所以这一层不用
    知道动作叫什么。`since` 是**从哪一帧开始算的**,`track_id` 是哪只手:两样都为了
    写日志(spec 第 37 条 user story)。
    """

    kind: str
    key: str
    t: float
    since: float
    track_id: int

    def describe(self) -> str:
        """写进日志的那一行。"""
        return (
            f"触发 {self.kind.upper()} {self.key.upper()}  手 #{self.track_id}  "
            f"从 {self.since:.3f} 起算,{self.t - self.since:.3f} 秒"
        )


class TriggerJudge:
    """三套规则合在一起。一个线程用(识别线程)。

    每帧 `update(result)`,认出触发手势就返回一件 `Trigger`,没有就是 `None`;
    停止手势另看 `stop`(它不是一件触发手势,走的是叫停那条路)。
    """

    def __init__(
        self,
        alignment: Alignment,
        params: VisionParams,
        *,
        poses: Sequence[str],
        stop_frames: int,
        stop_labels: frozenset[str] | None = None,
    ) -> None:
        self._fist = _FistMove(alignment, params)
        self._pose = _PoseHold(alignment.start_zone, params, poses)
        self._stop = _StopGesture(stop_frames, stop_labels)
        self._stopped = False
        self._pending: deque[Trigger] = deque()
        """认出来还没交出去的。两套规则同一帧都成立时,另一件在下一帧交出去——
        直接丢掉的话,姿势保持那边已经记下「触发过、要先松开」,这一次就白做了。"""

    # ---- 外面看得到的 --------------------------------------------------

    @property
    def stop(self) -> bool:
        """这一帧算不算认到了停止手势(连续帧数已经够)。"""
        return self._stopped

    @property
    def ready(self) -> bool:
        """READY 亮着没有——起点区里停稳了、锁定着那只拳头。"""
        return self._fist.ready

    @property
    def ready_dim(self) -> bool:
        """锁定的手这一帧没被认出来。READY 变暗,但锁定不解除。"""
        return self._fist.dim

    @property
    def direction(self) -> str | None:
        """刚认出来的那个方向。屏幕上在手框旁边画箭头用(往前、往后画不出箭头,写字),
        认出来的那一帧才有。"""
        return self._fist.direction

    @property
    def hold_progress(self) -> float:
        """姿势保持的进度,0 到 1。屏幕上画进度圈用。"""
        return self._pose.progress

    @property
    def holding(self) -> str | None:
        """正在保持的那个姿势。没有就是 `None`。"""
        return self._pose.holding

    def update(self, result: FrameResult) -> Trigger | None:
        """喂一帧,一次交出一件。三套规则都要走一遍——停止手势的计数一帧都不能漏。

        两套规则同一帧都成立时(比如一只手在做握拳平移、另一只手同时保持满一秒),
        先交握拳平移,剩下的下一帧交,**不丢**。
        """
        self._stopped = self._stop.update(result)
        for got in (self._fist.update(result), self._pose.update(result)):
            if got is not None:
                self._pending.append(got)
        return self._pending.popleft() if self._pending else None

    def clear(self) -> None:
        """清空判定记录(换送法时,ADR-0008)。

        **不清停止手势的连续帧计数**:它按画面计、不看跟踪编号,清了反而可能晚停约
        0.1 秒(spec「送哪块画面」)。姿势保持「触发过要先松开」那一条也留着——不然
        冷却一过,举着没放的手会立刻再触发一次。
        """
        self._fist.clear()
        self._pose.clear()


# ---- 停止手势 ----------------------------------------------------------


class _StopGesture:
    """连续几帧看见张开的手就算认到叫停。

    **不另设置信度阈值,也不要求「已确认」**(ADR-0012):漏一次叫停比多停一次危险,
    而新出现的手要连上 3 帧才算已确认,等它就白等约 0.1 秒。连续 3 帧本身就是防抖
    那一道(这一段约 63 ms,最多 89 ms)。
    """

    def __init__(self, frames: int, labels: frozenset[str] | None = None) -> None:
        self.frames = frames
        self.labels = STOP_GESTURE_LABELS if labels is None else labels
        """算叫停的那几个标签。**正式程序一律用 `STOP_GESTURE_LABELS`**,换成别的
        只有一处:回放测试离线算「加一个标签会多出几次误叫停」(spec「three2」)。
        加标签之前先拿已有录像算,不靠感觉——所以这里留一个口子,不改代码就能算。"""
        self._run = 0

    def update(self, result: FrameResult) -> bool:
        seen = any(h.label in self.labels for h in result.hands)
        self._run = self._run + 1 if seen else 0
        return self._run >= self.frames


# ---- 姿势保持 ----------------------------------------------------------


class _Watched:
    """盯着的一个姿势。触发过没有、多久没被认出来了、从哪一刻起算,都在这里。

    三样是一起变的,所以放一块儿,不摊成三张按姿势名索引的字典。
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.blocked = False
        """触发过了,要先松开才能再触发。"""
        self.missing_since: float | None = None
        self.since: float | None = None
        """从哪一刻起算。松开之后只算松开之后的帧。"""

    def saw(self, pose: str | None, t: float, release_s: float) -> None:
        """喂一帧看到的姿势;这个姿势够久没被认出来就算松开。"""
        if pose == self.name:
            self.missing_since = None
            return
        if self.missing_since is None:
            self.missing_since = t
            return
        if t - self.missing_since >= release_s and self.blocked:
            self.blocked = False
            self.since = t  # 松开以后只算松开之后的帧


class _PoseHold:
    """起点区里保持同一个姿势约一秒。区里有几只手取最大的,不用锁定。"""

    def __init__(self, zone: Region, params: VisionParams, poses: Sequence[str]) -> None:
        self._zone = zone
        self._params = params
        self._poses = tuple(_Watched(p) for p in poses)
        self._window: deque[tuple[float, str | None]] = deque()
        self._holding: str | None = None
        self._progress = 0.0
        self._restart = False

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def holding(self) -> str | None:
        return self._holding

    def clear(self) -> None:
        """清掉窗口和进度。**「触发过要先松开」那一条不清**(见 `TriggerJudge.clear`)。"""
        self._window.clear()
        self._holding = None
        self._progress = 0.0
        self._restart = True

    def update(self, result: FrameResult) -> Trigger | None:
        t = result.t
        if self._restart:
            self._restart = False
            for watched in self._poses:
                watched.since = t

        hand = _largest_in(result, self._zone)
        pose = hand.label if hand is not None else None
        self._window.append((t, pose))
        while self._window and self._window[0][0] < t - self._params.pose_hold_s:
            self._window.popleft()

        self._holding = pose if any(w.name == pose for w in self._poses) else None
        self._progress = 0.0

        got: Trigger | None = None
        for watched in self._poses:
            watched.saw(pose, t, self._params.release_s)
            if watched.blocked:
                continue
            if watched.since is None:
                watched.since = t
            held = t - watched.since
            if watched.name == self._holding:
                self._progress = min(1.0, held / self._params.pose_hold_s)
            if held < self._params.pose_hold_s:
                continue
            frames = [(ts, q) for ts, q in self._window if ts >= watched.since]
            if not frames:
                continue
            share = 100.0 * sum(1 for _, q in frames if q == watched.name) / len(frames)
            if share < self._params.pose_percent:
                continue
            watched.blocked = True  # 先松开才能再触发
            got = Trigger(
                kind=POSE_HOLD,
                key=watched.name,
                t=t,
                since=frames[0][0],
                track_id=hand.track_id if hand is not None else -1,
            )
        return got


# ---- 握拳平移 ----------------------------------------------------------


class _Sample(NamedTuple):
    """握拳平移记下的一帧。"""

    t: float
    center: tuple[float, float] | None
    """手框中心。这一帧没有手就是 None。"""
    fist: bool
    """是不是握拳类标签。"""
    size: tuple[float, float] | None
    """手框宽、高。这一帧没有手就是 None。"""


class _DepthAnchor(NamedTuple):
    """往前推从哪儿量起:连远近也停住的那一刻、那时的手框中心和宽高。"""

    t: float
    center: tuple[float, float]
    width: float
    height: float


class _FistMove:
    """起点区里停住 → READY 并锁定 → 朝一个方向移过 1.5 个手宽;或者往前推、往下敲两下。"""

    def __init__(self, alignment: Alignment, params: VisionParams) -> None:
        self._zone = alignment.start_zone
        self._hand_width = alignment.hand_width
        self._params = params
        self._keep_s = (
            max(params.reverse_guard_s, params.move_max_s, params.knock_window_s)
            + params.settle_s
        )
        self._path: deque[_Sample] = deque()
        """最近这一段的样子。"""
        self._track: int | None = None
        self._lock: int | None = None
        self._ready_at: float | None = None
        self._anchor: tuple[float, float] | None = None
        self._anchor_t: float | None = None
        self._depth: deque[_DepthAnchor] = deque()
        """最近 `move_max_s` 里「远近也停住」的那几刻。往前推从其中手框最小的那一刻量起。"""
        self._unit: float | None = None
        """最近一次停住时手框有多宽。敲两下按它量(见 `_knock_twice`)。"""
        self._knock = _Knock(params)
        self._since: float | None = None
        """这一段记录从哪一刻起。不满一个停住段就别急着判。"""
        self.dim = False
        self.direction: str | None = None

    @property
    def ready(self) -> bool:
        return self._lock is not None

    def clear(self) -> None:
        """清空判定记录,**含收手抑制的记录**。"""
        self._path.clear()
        self._track = None
        self._since = None
        self.dim = False
        self.direction = None
        self._unlock()

    def update(self, result: FrameResult) -> Trigger | None:
        t = result.t
        self.direction = None
        hand = self._pick(result)
        self.dim = self._lock is not None and hand is None

        if hand is not None and hand.track_id != self._track:
            # 交给判定的手换了跟踪编号:清掉旧记录,包括收手抑制的(补法 A)
            self._path.clear()
            self._since = None
            self._track = hand.track_id
        if self._since is None:
            self._since = t

        if hand is None:
            self._path.append(_Sample(t, None, False, None))
        else:
            x1, y1, x2, y2 = hand.box
            self._path.append(
                _Sample(t, hand.center, hand.label in FIST_LIKE_LABELS, (x2 - x1, y2 - y1))
            )
        while self._path and self._path[0].t < t - self._keep_s:
            self._path.popleft()

        if (
            self._lock is not None
            and self._ready_at is not None
            and t - self._ready_at > self._params.lock_timeout_s
        ):
            self._unlock()  # 锁定超时

        self._settle(t, hand)
        # 三样挂在同一个锁定上,哪样成立都会解除锁定,所以一帧最多出一件
        for judge in (self._move, self._push, self._knock_twice):
            got = judge(t, hand)
            if got is not None:
                return got
        return None

    # ---- 内部 ----------------------------------------------------------

    def _pick(self, result: FrameResult) -> HandResult | None:
        """这一帧把哪只手交给判定。

        锁定期间只认那个跟踪编号,**别的手一律不管**——另一只手抬到脸前时它一出现就
        比拳头大,一两帧的跳变就够触发一次(ADR-0010)。锁定之前取画面里最大的已确认手。
        """
        if self._lock is not None:
            return next((h for h in result.hands if h.track_id == self._lock), None)
        return result.largest()

    def _settle(self, t: float, hand: HandResult | None) -> None:
        """停住段:停稳了就把 READY 点亮、锁定这只手,并记下停住点。"""
        if hand is None or self._since is None:
            return
        if t - self._since < self._params.settle_s:
            return  # 记录还不满一个停住段
        window = [s for s in self._path if s.t >= t - self._params.settle_s]
        fists = [s.center for s in window if s.center is not None and s.fist]
        if not fists:
            return
        share = 100.0 * len(fists) / len(window)
        if share < self._params.settle_percent:
            return
        # 幅度按**所有看见手的帧**算,帧占比按握拳类算:spec 那两条是分开的
        # (「手框中心移动不超过半个手宽」「至少八成帧是握拳类标签」),
        # 只拿握拳帧量幅度的话,那两成非握拳的帧就白白免检了
        seen = [s for s in window if s.center is not None]
        xs = [s.center[0] for s in seen]
        ys = [s.center[1] for s in seen]
        spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if spread > self._params.settle_hand_widths * self._hand_width:
            return
        # 停住点取中位数,不取最新那一帧:手刚动起来的头两三帧还落在容差里,拿最新的
        # 会让停住点跟着往前挪,移动距离白白少算一截
        anchor = (median(c[0] for c in fists), median(c[1] for c in fists))
        if not self._zone.contains(anchor):
            return  # 停住点不在起点区里(ADR-0009)
        self._anchor = anchor
        self._anchor_t = t
        self._ready_at = t
        if self._lock is None:
            self._lock = hand.track_id
        # 往前推的起点另要「远近也停住」:推的时候中心几乎不动,只看中心的话推到一半
        # 还算停住,起点跟着往前挪,就量不出变大了多少(2026-09-24 原型里 20 次只认出 2 次)
        widths = [s.size[0] for s in seen if s.size is not None]
        width = median(widths)
        self._unit = width
        if max(widths) - min(widths) <= self._params.settle_size_percent / 100.0 * width:
            height = median(s.size[1] for s in seen if s.size is not None)
            self._depth.append(_DepthAnchor(t, anchor, width, height))

    def _move(self, t: float, hand: HandResult | None) -> Trigger | None:
        """移动段:够远、够快、够直,就是一次握拳平移。"""
        if hand is None or self._lock is None:
            return None
        if self._anchor is None or self._anchor_t is None:
            return None
        spent = t - self._anchor_t
        if spent <= 0 or spent > self._params.move_max_s:
            return None
        dx = hand.center[0] - self._anchor[0]
        dy = hand.center[1] - self._anchor[1]
        major, minor = max(abs(dx), abs(dy)), min(abs(dx), abs(dy))
        if major < self._params.move_hand_widths * self._hand_width:
            return None
        if major < self._params.move_axis_ratio * minor:
            return None  # 斜着移被忽略,这是有意的
        moving = [s for s in self._path if s.t >= self._anchor_t]
        if not moving:
            return None
        share = 100.0 * sum(1 for s in moving if s.fist) / len(moving)
        if share < self._params.move_percent:
            return None

        # 方向就照着 DIRECTIONS 挑投影最大的那个,不另写一串 if —— 一张表说了算
        direction = max(DIRECTIONS, key=lambda d: dx * DIRECTIONS[d][0] + dy * DIRECTIONS[d][1])
        self.direction = direction
        track_id = self._lock
        since = self._anchor_t
        # 触发过一次就解除锁定,**被收手抑制挡掉的那次也算**(ADR-0010)
        self._unlock()
        if self._reverse_guard(t, direction):
            return None
        return Trigger(kind=FIST_MOVE, key=direction, t=t, since=since, track_id=track_id)

    def _reverse_guard(self, t: float, direction: str) -> bool:
        """收手抑制(REVERSE GUARD):这之前一段时间里手朝反方向动过没有。**不看姿势**。

        看的是原始的手框中心,不要求那一段是一次合格的握拳平移——往回收的时候手常常
        已经松开了(第 16 条 user story)。

        **门槛是它自己的一套**(1 个手宽、轴比 2),不是上面那次移动的 1.5 个手宽和
        轴比 3:往回收的那一下通常比有意做的手势短,拿移动的门槛去量会漏掉
        (原型 `camera-prototype-4` 的 `zone_rule.py`,「另跑一个不看姿势的笔画检测器」)。
        """
        ux, uy = DIRECTIONS[direction]
        points = [
            s.center
            for s in self._path
            if s.center is not None and s.t >= t - self._params.reverse_guard_s
        ]
        back, sideways = _furthest_along(points, (-ux, -uy))
        if back < self._params.reverse_guard_hand_widths * self._hand_width:
            return False
        return back >= self._params.reverse_guard_axis_ratio * sideways

    def _push(self, t: float, hand: HandResult | None) -> Trigger | None:
        """往前推:连远近也停住以后,手框在移动段用时内变大到够大,而且还是拳头。

        宽、高都要变大:只看宽的话,食指往旁边一指手框也会变宽(9-19 自由活动里就有
        一次)。最近几帧都得是握拳类:拳头张开成手掌,手框也会一下子变大(同一份里
        叫停那一组有一次)。中心也不能走太远——那是平移,该由 `_move` 管。

        **从最近 `move_max_s` 里手框最小的那一刻量起**,不从最近那一刻:推得慢的时候,
        刚起步那一段每 0.3 秒只大一点,照样算「远近也停住」,只拿最近那一刻量的话起点
        跟着手往前挪,推完 1.6 倍也只量出 1.1 倍(2026-09-24 第三批 10 次漏了 4 次)。
        """
        while self._depth and t - self._depth[0].t > self._params.move_max_s:
            self._depth.popleft()
        if hand is None or self._lock is None or not self._depth:
            return None
        depth = min(self._depth, key=lambda d: d.width)
        if t <= depth.t:
            return None
        recent = [s for s in self._path if s.t > depth.t and s.size is not None]
        recent = recent[-self._params.push_frames:]
        if len(recent) < self._params.push_frames:
            return None
        for s in recent:
            assert s.size is not None
            width, height = s.size
            if not s.fist:
                return None
            if width < self._params.push_size_ratio * depth.width:
                return None
            if height < self._params.push_height_ratio * depth.height:
                return None
        if math.dist(hand.center, depth.center) > (
            self._params.move_hand_widths * self._hand_width
        ):
            return None
        return self._fire(FORWARD, t, depth.t)

    def _knock_twice(self, t: float, hand: HandResult | None) -> Trigger | None:
        """往下敲两下。只在锁定期间数,解除锁定就从头数(见 `_Knock`)。

        **一个手宽按最近停住时手框有多宽算**,不按启动对准量的那一个:往前推完拳头举在
        镜头前,手框有对准时的将近两倍宽,手随便晃一下就够 0.4 个「对准手宽」
        (2026-09-24 原型里推完就冒出过一次往后)。

        握拳帧占比另有一个门槛(`knock_percent`):说话时手指着旁边上下晃,幅度也够两下
        (9-19 自由活动里有一次,占 44%),但那时多半不是拳头;敲得快的时候模糊得认不出手,
        真敲两下最低也有 56%,所以比平移的 60% 低一档。不够就从头数。
        """
        if hand is None or self._lock is None:
            return None
        since = self._knock.update(t, hand.center[1], self._unit or self._hand_width)
        if since is None:
            return None
        knocking = [s for s in self._path if s.t >= since]
        share = 100.0 * sum(1 for s in knocking if s.fist) / len(knocking)
        if share < self._params.knock_percent:
            self._knock.clear()
            return None
        return self._fire(BACK, t, since)

    def _fire(self, direction: str, t: float, since: float) -> Trigger:
        """认出往前 / 往后:报出来,解除锁定。"""
        assert self._lock is not None
        self.direction = direction
        track_id = self._lock
        self._unlock()
        return Trigger(kind=FIST_MOVE, key=direction, t=t, since=since, track_id=track_id)

    def _unlock(self) -> None:
        self._lock = None
        self._ready_at = None
        self._anchor = None
        self._anchor_t = None
        self._depth.clear()
        self._knock.clear()


class _Knock:
    """敲两下:手框中心往下、再抬回来,算一下;两下挨得够近就是一次往后(BACK)。

    一下 = 从最近的高点往下过 `knock_hand_widths` 个手宽、再从最低点抬回来
    `knock_return_hand_widths` 个手宽。**第二下往下一过门槛就算**,不等它抬回来:
    2026-09-24 实测第二下常常敲得更重,往下能走 1.3 个手宽,等它抬回来的话,握拳平移
    的「往下」(1.5 个手宽)有可能抢在前面成立。实测一下约 1 秒,两下合起来要 2.3 秒,
    所以窗口是 3 秒。
    """

    def __init__(self, params: VisionParams) -> None:
        self._down = params.knock_hand_widths
        self._up = params.knock_return_hand_widths
        self._window_s = params.knock_window_s
        self.clear()

    def clear(self) -> None:
        self._going_down = False
        self._top: tuple[float, float] | None = None
        """这一下从哪儿起:最近的高点的 (时刻, y)。"""
        self._bottom = 0.0
        self._start = 0.0
        """正在往下的这一下是从哪一刻起的。"""
        self._first: float | None = None
        """敲完的第一下是从哪一刻起的。还没敲完一下就是 None。"""

    def update(self, t: float, y: float, hand_px: float) -> float | None:
        """喂这一帧锁定那只手的手框中心 y(没看见手的帧不喂),`hand_px` 是一个手宽几像素。

        第二下往下过门槛的那一帧,返回第一下是从哪一刻起的;别的时候返回 None。
        """
        if self._first is not None and t - self._first > self._window_s:
            self._first = None  # 第一下太久以前了,这一下重新当第一下数
        if self._going_down:
            self._bottom = max(self._bottom, y)
            if self._bottom - y >= self._up * hand_px:
                self._going_down = False  # 抬回来了,这一下敲完
                self._first = self._start
                self._top = (t, y)
            return None
        if self._top is None or y < self._top[1]:
            self._top = (t, y)
        if y - self._top[1] < self._down * hand_px:
            return None
        if self._first is not None:
            return self._first
        self._going_down = True
        self._bottom = y
        self._start = self._top[0]
        return None


# ---- 小工具 ------------------------------------------------------------


def _largest_in(result: FrameResult, zone: Region) -> HandResult | None:
    """起点区里最大的那只手。区外的一律不算(姿势保持用)。

    **不要求「已确认」**:spec 里「取画面里最大的**已确认**手」那一条只挂在握拳平移
    的锁定上,姿势保持写的是「只看起点区里的手,区里有几只取最大的」。反正要满一秒
    八成帧,新出现那几帧混不进什么来。
    """
    inside = [h for h in result.hands if zone.contains(h.center)]
    return max(inside, key=lambda h: h.area) if inside else None


def _furthest_along(
    points: list[tuple[float, float]], unit: tuple[float, float]
) -> tuple[float, float]:
    """这串点沿某个方向走出过多远,以及那一段往旁边偏了多少。

    先退回去再往前走的,不合起来算——就是「最大差值」那个老问题:一路记着到此为止
    最靠后的位置,拿当前位置去减。往旁边偏多少是拿那一段的两头算的,收手抑制要用它
    判这一下够不够直。
    """
    ux, uy = unit
    sx, sy = -uy, ux  # 垂直于 unit 的那个方向
    best = 0.0
    sideways = 0.0
    back: tuple[float, float] | None = None
    for x, y in points:
        along, across = x * ux + y * uy, x * sx + y * sy
        if back is None or along < back[0]:
            back = (along, across)
        if along - back[0] > best:
            best = along - back[0]
            sideways = abs(across - back[1])
    return best, sideways

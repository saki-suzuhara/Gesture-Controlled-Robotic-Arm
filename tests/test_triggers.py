"""触发判定:从识别结果那个缝进,不开摄像头、不跑模型。

三套规则共用同一份逐帧识别结果——握拳平移(起点区 + 锁定 + 收手抑制)、姿势保持
(起点区里最大的手 + 松开)、停止手势(整幅画面、连续 3 帧)。这里喂编好的结果,
看**排出了哪件触发**、**有没有认到叫停**,不断言内部状态。

参数用真的演示配置;手宽取 80 px,好让「半个手宽 40 px」「1.5 个手宽 120 px」这类
数一眼算得过来。**参数是六个原型扫参后的初值,测试不另调**(票 03)。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from robot_side.config import load_demo
from vision_side.align import Alignment
from vision_side.recognize import FrameResult, HandResult, Region
from vision_side.triggers import TriggerJudge

FPS = 30.0
HAND = 80.0
"""这批测试里的手宽。起点区因此是 240 见方,移动要过 120 px,停住只能晃 40 px。"""

MIDDLE = (640.0, 360.0)

DEMO = load_demo()
PARAMS = DEMO.vision
POSES = tuple(t.key for t in DEMO.triggers if t.kind == "pose_hold")

ZONE = Region(
    round(MIDDLE[0] - 1.5 * HAND), round(MIDDLE[1] - 1.5 * HAND), round(3 * HAND), round(3 * HAND)
)

ALIGNMENT = Alignment(
    detection_region=Region(240, 60, 800, 600),
    start_zone=ZONE,
    hand_width=HAND,
    full_frame_width=88.0,
    too_close=False,
    t=0.0,
)

SETTLE = HAND * PARAMS.settle_hand_widths  # 停住段能晃多少:40 px
MOVE = HAND * PARAMS.move_hand_widths  # 移动段要过多少:120 px
GUARD = HAND * PARAMS.reverse_guard_hand_widths  # 反方向动过多少才算:80 px

JITTER = 30.0
"""「在原地晃着等」用的幅度。来回晃 ±30 拉开 60 px:比 SETTLE 大,停不稳;比 GUARD
小,又算不成一次「朝反方向动过」。两边都卡着,所以单拎出来写明白。"""

assert 2 * JITTER > SETTLE and 2 * JITTER < GUARD


STOP_FRAMES = 3
"""停止手势那几条用例按「连续 3 帧」写(帧数、`range(3)`、「两帧还不算」),所以钉在 3,
不读 `config/demo.toml`:现场把 `stop_gesture_frames` 调大调小,判定的逻辑没变,用例不该跟着红。"""


def make(*, params=PARAMS, poses=POSES, stop_frames=STOP_FRAMES) -> TriggerJudge:
    return TriggerJudge(ALIGNMENT, params, poses=poses, stop_frames=stop_frames)


def hand(
    center, *, label="fist", track_id=1, confirmed=True, width=HAND, height=None
) -> HandResult:
    """一只手。`height` 不给就是方框(和宽一样)。"""
    cx, cy = center
    half_w = width / 2
    half_h = (width if height is None else height) / 2
    return HandResult(
        box=(round(cx - half_w), round(cy - half_h), round(cx + half_w), round(cy + half_h)),
        label=label,
        track_id=track_id,
        confirmed=confirmed,
    )


def still(seconds: float, center=MIDDLE, **kwargs) -> list[tuple[HandResult, ...]]:
    """举着不动的一段。"""
    return [(hand(center, **kwargs),)] * round(seconds * FPS)


def sweep(start, end, seconds: float, **kwargs) -> list[tuple[HandResult, ...]]:
    """从一处匀速移到另一处。"""
    n = max(1, round(seconds * FPS))
    out = []
    for i in range(1, n + 1):
        out.append(
            (
                hand(
                    (
                        start[0] + (end[0] - start[0]) * i / n,
                        start[1] + (end[1] - start[1]) * i / n,
                    ),
                    **kwargs,
                ),
            )
        )
    return out


def nothing(seconds: float) -> list[tuple[HandResult, ...]]:
    """一只手都没有的一段。"""
    return [()] * round(seconds * FPS)


def drive(judge: TriggerJudge, script, *, start_t: float = 0.0):
    """逐帧喂进去,返回认出来的第一件触发手势。"""
    for i, hands in enumerate(script):
        got = judge.update(FrameResult(t=start_t + i / FPS, hands=hands))
        if got is not None:
            return got
    return None


def drive_all(judge: TriggerJudge, script, *, start_t: float = 0.0) -> list:
    """逐帧喂进去,把认出来的每一件都收下来。"""
    out = []
    for i, hands in enumerate(script):
        got = judge.update(FrameResult(t=start_t + i / FPS, hands=hands))
        if got is not None:
            out.append(got)
    return out


def fist_move(direction: str, *, settle_s: float | None = None, move_s: float = 0.4, **kwargs):
    """一次完整的握拳平移:在起点区正中停住,再朝一个方向移过 1.5 个手宽多一点。"""
    step = MOVE + 20
    ends = {
        "left": (MIDDLE[0] - step, MIDDLE[1]),
        "right": (MIDDLE[0] + step, MIDDLE[1]),
        "up": (MIDDLE[0], MIDDLE[1] - step),
        "down": (MIDDLE[0], MIDDLE[1] + step),
    }
    hold = PARAMS.settle_s + 0.1 if settle_s is None else settle_s
    return still(hold, MIDDLE, **kwargs) + sweep(MIDDLE, ends[direction], move_s, **kwargs)


# ---- 握拳平移:四个方向 ------------------------------------------------


@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
def test_四个方向各认得出(direction):
    got = drive(make(), fist_move(direction))
    assert got is not None
    assert (got.kind, got.key) == ("fist_move", direction)
    assert got.track_id == 1


def test_认出来的那件记着是从哪一帧开始算的():
    got = drive(make(), fist_move("left"))
    # 日志里要写「从哪一帧开始算的」(spec 第 37 条 user story)
    assert got.since < got.t
    assert got.t - got.since <= PARAMS.move_max_s


# ---- 握拳平移:起点区 --------------------------------------------------


def test_停在区外再移动不触发():
    outside = (ZONE.x - 60.0, MIDDLE[1])
    script = still(PARAMS.settle_s + 0.1, outside) + sweep(
        outside, (outside[0] - MOVE - 20, outside[1]), 0.4
    )
    assert drive(make(), script) is None


#: 起点区只有 3 个手宽(240 px),移 1.5 个手宽(120 px)不一定出得了区。从区里偏左
#: 一点出发,移到区里偏右一点,整段都在区里——收手那几条测试也用这一对。
LEFT_IN = (MIDDLE[0] - 65.0, MIDDLE[1])
RIGHT_IN = (MIDDLE[0] + 65.0, MIDDLE[1])


def test_停住点在区里就算不要求移动的终点出区():
    assert ZONE.contains(LEFT_IN) and ZONE.contains(RIGHT_IN)
    assert RIGHT_IN[0] - LEFT_IN[0] > MOVE  # 确实移过了 1.5 个手宽
    script = still(PARAMS.settle_s + 0.1, LEFT_IN) + sweep(LEFT_IN, RIGHT_IN, 0.4)
    got = drive(make(), script)
    assert got is not None and got.key == "right"


# ---- 握拳平移:停住段 --------------------------------------------------


def test_没停够就移动不触发():
    script = still(PARAMS.settle_s - 0.15) + sweep(
        MIDDLE, (MIDDLE[0] - MOVE - 20, MIDDLE[1]), 0.4
    )
    assert drive(make(), script) is None


def test_停住段晃太多不算停住():
    script = []
    for i in range(round((PARAMS.settle_s + 0.2) * FPS)):
        dx = SETTLE + 10 if i % 2 else -(SETTLE + 10)
        script.append((hand((MIDDLE[0] + dx, MIDDLE[1])),))
    script += sweep(MIDDLE, (MIDDLE[0] - MOVE - 20, MIDDLE[1]), 0.4)
    assert drive(make(), script) is None


def test_停住段里张着手不算():
    assert drive(make(), fist_move("left", label="palm")) is None


# ---- 握拳平移:移动段 --------------------------------------------------


def test_移动不够远不触发():
    short = (MIDDLE[0] - MOVE + 30, MIDDLE[1])
    script = still(PARAMS.settle_s + 0.1) + sweep(MIDDLE, short, 0.4)
    assert drive(make(), script) is None


def test_慢慢挪位置不触发():
    # 匀速慢慢挪:停住点会跟着一起挪,永远凑不满一次移动
    script = still(PARAMS.settle_s + 0.1) + sweep(
        MIDDLE, (MIDDLE[0] - MOVE - 20, MIDDLE[1]), PARAMS.move_max_s + 0.5
    )
    assert drive(make(), script) is None


def dawdle(hover_s: float):
    """停稳 → 先挪开一点(停住点就此定住)→ 磨蹭一阵 → 再补完剩下的路。

    专门用来单独验「用时不超过 1.2 秒」那一条:停住点定在 LEFT_IN 不动,磨蹭多久
    就算多久。磨蹭时手一直小幅晃(晃幅 90 px:停不稳,也够不上一次移动)。
    """
    near = (LEFT_IN[0] + 50.0, LEFT_IN[1])
    far = (LEFT_IN[0] + MOVE + 10, LEFT_IN[1])
    script = still(PARAMS.settle_s + 0.1, LEFT_IN) + [(hand(near),)]
    for i in range(round(hover_s * FPS)):
        dx = JITTER if i % 2 else -JITTER
        script.append((hand((near[0] + dx, near[1])),))
    return script + sweep(near, far, 0.2)


def patient() -> TriggerJudge:
    """把锁定超时放大的一个判定,专门用来单独看「移动用时」那一条。

    出厂参数里 `move_max_s` 和 `lock_timeout_s` 都是 1.2 秒,而两个都从停稳那一刻起算
    —— 锁定超时总是先到,用时那一条平时根本看不出来。放大锁定超时才验得了它。
    """
    return make(params=replace(PARAMS, lock_timeout_s=5.0))


def test_移动太慢不触发():
    # 停住点定住以后磨蹭超过 move_max_s,最后补完的那一下不算
    assert drive(patient(), dawdle(PARAMS.move_max_s + 0.3)) is None


def test_同样的路走得快就触发():
    # 同一条路、同样的距离,只是没磨蹭 —— 证明上一条卡的是用时,不是别的
    got = drive(patient(), dawdle(0.3))
    assert got is not None and got.key == "right"


def test_出厂参数下锁定超时先到():
    # 记一笔:按出厂的 1.2 / 1.2,磨蹭太久是被锁定超时挡下的,不是用时那一条
    judge = make()
    drive(judge, dawdle(PARAMS.move_max_s + 0.3))
    assert not judge.ready


def test_斜着移被忽略():
    # 主方向的位移不到另一方向的 3 倍
    diagonal = (MIDDLE[0] - MOVE - 20, MIDDLE[1] - MOVE - 20)
    script = still(PARAMS.settle_s + 0.1) + sweep(MIDDLE, diagonal, 0.4)
    assert drive(make(), script) is None


def test_稍微偏一点还算得上():
    # 主方向是另一方向的 5 倍,过得了 3 倍那一条
    mostly_left = (MIDDLE[0] - MOVE - 20, MIDDLE[1] - (MOVE + 20) / 5)
    script = still(PARAMS.settle_s + 0.1) + sweep(MIDDLE, mostly_left, 0.4)
    got = drive(make(), script)
    assert got is not None and got.key == "left"


def test_移动段里松开手太多帧不触发():
    script = still(PARAMS.settle_s + 0.1)
    moving = sweep(MIDDLE, (MIDDLE[0] - MOVE - 20, MIDDLE[1]), 0.5)
    for i in range(len(moving)):
        if i % 2:  # 一半的帧是张开的手,过不了六成
            moving[i] = (hand(moving[i][0].center, label="palm"),)
    assert drive(make(), script + moving) is None


# ---- 握拳平移:锁定(ADR-0010)-----------------------------------------


def test_READY之后另一只手成为最大的也不触发():
    script = still(PARAMS.settle_s + 0.1)
    # 另一只手抬到脸前:更大、也被认成握拳类,而且一出现就在别处
    other = (MIDDLE[0] - MOVE - 20, MIDDLE[1])
    for i in range(round(0.4 * FPS)):
        script.append(
            (
                hand(MIDDLE, track_id=1),
                hand(other, track_id=2, width=HAND * 2),
            )
        )
    assert drive(make(), script) is None


def test_锁定的手几帧不见锁定不解除只是变暗():
    judge = make()
    drive(judge, still(PARAMS.settle_s + 0.1))
    assert judge.ready and not judge.ready_dim
    drive(judge, nothing(0.1), start_t=1.0)
    assert judge.ready  # 锁定还在
    assert judge.ready_dim  # 只是变暗
    # 手回来以后接着移动,照样触发
    got = drive(
        judge, sweep(MIDDLE, (MIDDLE[0] - MOVE - 20, MIDDLE[1]), 0.3), start_t=1.1
    )
    assert got is not None and got.key == "left"


def test_锁定超时以后再移动不触发():
    judge = make()
    drive(judge, still(PARAMS.settle_s + 0.1))
    assert judge.ready
    # 举着不动干等,超过锁定超时;中途不再停稳(手一直在小幅晃,停不住)
    late = PARAMS.lock_timeout_s + 0.3
    script = []
    for i in range(round(late * FPS)):
        dx = SETTLE + 10 if i % 2 else -(SETTLE + 10)
        script.append((hand((MIDDLE[0] + dx, MIDDLE[1])),))
    drive(judge, script, start_t=1.0)
    assert not judge.ready  # 锁定已经超时解除


def test_拳头松开不解除锁定():
    judge = make()
    drive(judge, still(PARAMS.settle_s + 0.1))
    drive(judge, still(0.1, label="palm"), start_t=1.0)
    assert judge.ready  # 松开不解除


def test_换了手要自己重新停稳():
    judge = make()
    # 1 号手停稳、READY 亮起
    drive(judge, still(PARAMS.settle_s + 0.1))
    assert judge.ready
    # 1 号手不见了、换成 2 号:2 号没停稳过,直接移动不算
    gone = nothing(PARAMS.lock_timeout_s + 0.2)
    drive(judge, gone, start_t=1.0)
    assert not judge.ready
    start_t = 1.0 + len(gone) / FPS
    got = drive(
        judge,
        sweep(MIDDLE, (MIDDLE[0] - MOVE - 20, MIDDLE[1]), 0.3, track_id=2),
        start_t=start_t,
    )
    assert got is None


def test_换手连收手抑制的记录一起清():
    # 补法 A:1 号手刚朝右移过,2 号手接着朝左移 —— 那条「刚朝右动过」是别人的记录
    judge = make()
    out, _ = there_and_back(track_id=1)
    assert [t.key for t in drive_all(judge, out)] == ["right"]
    _, back = there_and_back(track_id=2)
    got = drive(judge, back, start_t=len(out) / FPS)
    assert got is not None and got.key == "left"
    assert got.track_id == 2


# ---- 握拳平移:收手抑制 ------------------------------------------------


def there_and_back(track_id: int = 1):
    """先在区里向右移一次(会触发),再把手收回去——整段都在起点区里。"""
    out = still(PARAMS.settle_s + 0.1, LEFT_IN, track_id=track_id) + sweep(
        LEFT_IN, RIGHT_IN, 0.3, track_id=track_id
    )
    back = still(PARAMS.settle_s + 0.1, RIGHT_IN, track_id=track_id) + sweep(
        RIGHT_IN, LEFT_IN, 0.3, track_id=track_id
    )
    return out, back


def test_收手不会被判成反方向的一次触发():
    judge = make()
    out, back = there_and_back()
    assert [t.key for t in drive_all(judge, out)] == ["right"]
    # 收回去这一下同样是「移过 1.5 个手宽」,但 1.5 秒内刚朝右动过
    assert drive(judge, back, start_t=len(out) / FPS) is None


def test_过了收手抑制那段时间就不挡了():
    judge = make()
    out, back = there_and_back()
    drive_all(judge, out)
    t1 = len(out) / FPS
    # 在原地等够收手抑制的时间。手一直小幅晃着:不会又停稳成新的 READY,
    # 晃的幅度也够不上「朝反方向动过」
    wait = []
    for i in range(round((PARAMS.reverse_guard_s + 0.3) * FPS)):
        dx = JITTER if i % 2 else -JITTER
        wait.append((hand((RIGHT_IN[0] + dx, RIGHT_IN[1])),))
    drive(judge, wait, start_t=t1)
    got = drive(judge, back, start_t=t1 + len(wait) / FPS)
    assert got is not None and got.key == "left"


def test_被收手抑制挡掉的那次也解除锁定():
    judge = make()
    out, back = there_and_back()
    drive_all(judge, out)
    drive(judge, back, start_t=len(out) / FPS)
    assert not judge.ready  # 挡掉了,但锁定照样解除


# ---- 握拳平移:往前推 --------------------------------------------------
#
# 往镜头推:手框中心几乎不动,手框变大。2026-09-24 录的两批里推 20 cm 手框宽变 1.5–2.3 倍。

BIG = HAND * (PARAMS.push_size_ratio + 0.2)
"""推到头时的手框宽:比门槛大一点。"""


def grow(start: float, end: float, seconds: float, *, center=MIDDLE, height_ratio=1.0, **kwargs):
    """手框在原地从 `start` 宽匀速变到 `end` 宽;高按 `height_ratio` 跟着变。"""
    n = max(1, round(seconds * FPS))
    out = []
    for i in range(1, n + 1):
        width = start + (end - start) * i / n
        height = start + (start * height_ratio - start + end - start) * i / n
        if height_ratio == 1.0:
            height = width
        out.append((hand(center, width=width, height=height, **kwargs),))
    return out


def push(**kwargs):
    """一次往前推:在起点区正中停住,再在 0.3 秒内把手框推大到门槛以上、停住。"""
    return (
        still(PARAMS.settle_s + 0.1, **kwargs)
        + grow(HAND, BIG, 0.3, **kwargs)
        + still(0.3, width=BIG, **kwargs)
    )


def test_往前推认得出():
    got = drive(make(), push())
    assert got is not None
    assert (got.kind, got.key) == ("fist_move", "forward")
    assert got.track_id == 1
    assert got.since < got.t  # 从停住那一刻起算


def test_起步慢的往前推也认得出():
    # 2026-09-24 第三批:刚起步那一段每 0.3 秒只大一点,照样算「远近也停住」;只从最近那
    # 一刻量的话起点跟着手往前挪,推完 1.6 倍只量得出 1.3 倍。从最近 1.2 秒里最小的量起
    slow_until = HAND * 1.2
    creeping = grow(HAND, slow_until, 0.6)
    assert (slow_until - HAND) / HAND / 0.6 * PARAMS.settle_s < PARAMS.settle_size_percent / 100
    script = (
        still(PARAMS.settle_s + 0.1)
        + creeping
        + grow(slow_until, BIG, 0.3)
        + still(0.3, width=BIG)
    )
    got = drive(make(), script)
    assert got is not None and got.key == "forward"


def test_认出往前推那一下屏幕上有得画():
    judge = make()
    script = push()
    for i, hands in enumerate(script):
        if judge.update(FrameResult(t=i / FPS, hands=hands)) is not None:
            break
    assert judge.direction == "forward"


def test_推完解除锁定():
    judge = make()
    drive(judge, push())
    assert not judge.ready


def test_没停住就推不算():
    # 手一出现就在变大:没有 READY,也没有量起的那一刻
    script = grow(HAND, BIG, 0.3)
    assert drive(make(), script) is None


def test_慢慢挪近不算往前推():
    # 慢慢挪近:每 0.3 秒只大一点,一直算「远近也停住」,起点跟着挪,凑不满 1.4 倍
    script = still(PARAMS.settle_s + 0.1) + grow(HAND, BIG, 4.0)
    assert drive(make(), script) is None


def test_手框只变宽不变高不算往前推():
    # 食指往旁边一指:手框变宽,高不变(9-19 自由活动里就有一次)
    script = still(PARAMS.settle_s + 0.1) + [
        (hand(MIDDLE, width=w, height=HAND),) for w in (HAND * 1.2, HAND * 1.5, BIG, BIG, BIG)
    ]
    assert drive(make(), script) is None


def test_手框变大但高不够也不算():
    tall_enough = HAND * PARAMS.push_height_ratio
    short = still(PARAMS.settle_s + 0.1) + [
        (hand(MIDDLE, width=BIG, height=tall_enough - 4),) for _ in range(5)
    ]
    assert drive(make(), short) is None
    tall = still(PARAMS.settle_s + 0.1) + [
        (hand(MIDDLE, width=BIG, height=tall_enough + 4),) for _ in range(5)
    ]
    got = drive(make(), tall)
    assert got is not None and got.key == "forward"


def test_拳头张开手框变大不算往前推():
    # 拳头张开成手掌手框也会变大(9-19 叫停那一组里有一次):那是叫停,不是往前推
    script = (
        still(PARAMS.settle_s + 0.1)
        + grow(HAND, BIG, 0.2)[:-3]
        + [(hand(MIDDLE, width=BIG, label="three2"),)] * 5
    )
    assert drive(make(), script) is None


def test_一两帧手框跳大不算往前推():
    # 连着 push_frames 次才算:只跳两帧就缩回去
    script = (
        still(PARAMS.settle_s + 0.1)
        + [(hand(MIDDLE, width=BIG),)] * (PARAMS.push_frames - 1)
        + still(0.3)
    )
    assert drive(make(), script) is None


def test_手挪远了再变大不算往前推():
    # 先斜着挪出 1.5 个手宽(平移不认斜的),再在那儿变大:离停住点太远,不算往前推
    far = (MIDDLE[0] - MOVE - 20, MIDDLE[1] - MOVE - 20)
    script = (
        still(PARAMS.settle_s + 0.1)
        + sweep(MIDDLE, far, 0.2)
        + grow(HAND, BIG, 0.2, center=far)
        + still(0.2, far, width=BIG)
    )
    assert drive(make(), script) is None


# ---- 握拳平移:敲两下(往后) --------------------------------------------
#
# 像敲桌子那样往下点两下。2026-09-24 实测每一下往下 0.5–1.3 个手宽、约 1 秒。

KNOCK = HAND * (PARAMS.knock_hand_widths + 0.2)
"""这批测试里一下往下多少:比门槛大一点。"""


def stroke(depth: float = KNOCK, seconds: float = 0.3, *, width=HAND, **kwargs):
    """往下一下、再抬回来。"""
    low = (MIDDLE[0], MIDDLE[1] + depth)
    return sweep(MIDDLE, low, seconds / 2, width=width, **kwargs) + sweep(
        low, MIDDLE, seconds / 2, width=width, **kwargs
    )


def knock(depth: float = KNOCK, **kwargs):
    """一次敲两下:停住,再往下点两下。"""
    return still(PARAMS.settle_s + 0.1, **kwargs) + stroke(depth, **kwargs) + stroke(depth, **kwargs)


def test_敲两下认得出():
    got = drive(make(), knock())
    assert got is not None
    assert (got.kind, got.key) == ("fist_move", "back")
    assert got.since < got.t  # 从第一下起算


def test_第二下往下一过门槛就算不等抬回来():
    judge = make()
    script = still(PARAMS.settle_s + 0.1) + stroke()
    low = (MIDDLE[0], MIDDLE[1] + KNOCK)
    going_down = sweep(MIDDLE, low, 0.15)
    got = drive(judge, script + going_down)
    assert got is not None and got.key == "back"


def test_只敲一下不算():
    script = still(PARAMS.settle_s + 0.1) + stroke() + still(1.0)
    assert drive(make(), script) is None


def test_敲得太轻不算():
    light = HAND * (PARAMS.knock_hand_widths - 0.15)
    assert drive(make(), knock(light)) is None


def test_两下隔太久不算():
    # 中间停住干等(一直 READY 着),第一下过了窗口就不作数
    wait = PARAMS.knock_window_s + 0.5
    script = still(PARAMS.settle_s + 0.1) + stroke() + still(wait) + stroke()
    assert drive(make(), script) is None


def test_第二下敲得重也先认成敲两下不认成往下():
    # 实测第二下常常更重,能往下走 1.3 个手宽;第二下一过门槛就算,赶在「往下」前面
    heavy = HAND * 1.6
    script = still(PARAMS.settle_s + 0.1) + stroke() + stroke(heavy, 0.3)
    got = drive_all(make(), script)
    assert [(t.kind, t.key) for t in got][:1] == [("fist_move", "back")]
    assert ("fist_move", "down") not in [(t.kind, t.key) for t in got]


def test_敲两下按停住时的手框量不按对准时的手宽():
    # 推完拳头举在镜头前,手框有对准时的两倍宽:晃一下的幅度按对准手宽够了,按那时的
    # 手框就不够(2026-09-24 原型里推完就冒出过一次往后)
    near = HAND * 2
    wobble = HAND * (PARAMS.knock_hand_widths + 0.2)  # 按对准手宽过门槛,按 2 倍手框不过
    assert wobble < near * PARAMS.knock_hand_widths
    script = still(PARAMS.settle_s + 0.1, width=near) + stroke(wobble, width=near) + stroke(
        wobble, width=near
    )
    assert drive(make(), script) is None
    # 同样的比例放在 2 倍手框上就认得出
    deep = near * (PARAMS.knock_hand_widths + 0.2)
    script = still(PARAMS.settle_s + 0.1, width=near) + stroke(deep, width=near) + stroke(
        deep, width=near
    )
    got = drive(make(), script)
    assert got is not None and got.key == "back"


def test_上下晃的多半不是拳头不算敲两下():
    # 说话时手指着旁边上下晃:幅度够,但那时多半不是拳头(9-19 自由活动里有一次,44%)
    script = still(PARAMS.settle_s + 0.1)
    for s in (stroke(), stroke()):
        script += [(hand(h[0].center, label="two_right"),) for h in s]
    assert drive(make(), script) is None


def test_敲的时候模糊得认不出手也还认得出():
    # 敲得快会模糊:每三帧丢一帧,握拳帧还在五成以上
    script = still(PARAMS.settle_s + 0.1)
    for i, hands in enumerate(stroke() + stroke()):
        script.append(() if i % 3 == 0 else hands)
    got = drive(make(), script)
    assert got is not None and got.key == "back"


def test_没有READY不数敲两下():
    # 手一出现就在上下点:没停住过,不锁定,不数
    assert drive(make(), stroke() + stroke()) is None


def test_清记录也清掉敲了几下():
    judge = make()
    first = still(PARAMS.settle_s + 0.1) + stroke()
    drive(judge, first)
    judge.clear()
    # 清完得重新停住;接着再敲一下,和清之前那一下凑不成两下
    got = drive(judge, still(PARAMS.settle_s + 0.1) + stroke() + still(0.5), start_t=len(first) / FPS)
    assert got is None


def test_往下平移照样认得出():
    # 一次干脆的往下:只有一下,不会被当成敲两下,也不被敲两下挡住
    got = drive(make(), fist_move("down"))
    assert got is not None and got.key == "down"


# ---- 姿势保持(ADR-0007)------------------------------------------------


@pytest.mark.parametrize("pose", ["peace", "like"])
def test_保持一秒就触发(pose):
    got = drive(make(), still(PARAMS.pose_hold_s + 0.1, label=pose))
    assert got is not None
    assert (got.kind, got.key) == ("pose_hold", pose)


def test_没保持够不触发():
    assert drive(make(), still(PARAMS.pose_hold_s - 0.2, label="peace")) is None


def test_不松开不会再触发():
    judge = make()
    script = still(PARAMS.pose_hold_s * 3, label="peace")
    out = drive_all(judge, script)
    assert [t.key for t in out] == ["peace"]  # 一直举着也只触发一次


def test_松开以后可以再触发():
    judge = make()
    hold = still(PARAMS.pose_hold_s + 0.1, label="peace")
    out = drive_all(judge, hold)
    assert len(out) == 1
    t1 = len(hold) / FPS
    letting_go = nothing(PARAMS.release_s + 0.1)
    drive(judge, letting_go, start_t=t1)
    t2 = t1 + len(letting_go) / FPS
    again = drive(judge, still(PARAMS.pose_hold_s + 0.1, label="peace"), start_t=t2)
    assert again is not None and again.key == "peace"


def test_刚松开又举回来不会立刻再触发():
    judge = make()
    hold = still(PARAMS.pose_hold_s + 0.1, label="peace")
    drive_all(judge, hold)
    t1 = len(hold) / FPS
    letting_go = nothing(PARAMS.release_s + 0.05)
    drive(judge, letting_go, start_t=t1)
    t2 = t1 + len(letting_go) / FPS
    again = drive(judge, still(PARAMS.pose_hold_s - 0.4, label="peace"), start_t=t2)
    assert again is None


def test_松开以后只算松开之后的帧():
    """松开那一段把窗口稀释掉之外,**旧帧本身也不该算**。

    出厂参数下(保持 1 秒、八成帧、松开 0.3 秒)这一条看不出来:松开那 0.3 秒本身就
    占掉窗口的三成,怎么算都到不了八成。把松开门槛调小才验得了它——spec 说的正是
    「保持时长长或帧占比门槛低时,窗口里剩下的旧帧就够触发」。
    """
    judge = make(params=replace(PARAMS, release_s=0.1))
    hold = still(PARAMS.pose_hold_s + 0.1, label="peace")
    assert len(drive_all(judge, hold)) == 1
    t1 = len(hold) / FPS
    # 换成别的姿势 0.15 秒:够算一次松开,但窗口里绝大多数还是刚才那些 peace 帧
    letting_go = still(0.15, label="ok")
    # 松开成立的那一帧最容易出事:窗口里旧帧还占着八成,一松开就又触发一次
    assert drive(judge, letting_go, start_t=t1) is None
    t2 = t1 + len(letting_go) / FPS
    again = drive(judge, still(0.1, label="peace"), start_t=t2)
    assert again is None  # 只举了 0.1 秒,不够一次保持


def test_区外的手不算保持():
    outside = (ZONE.x - 60.0, MIDDLE[1])
    assert drive(make(), still(PARAMS.pose_hold_s + 0.3, outside, label="peace")) is None


def test_区里有几只手取最大的():
    judge = make()
    script = []
    small_at = (ZONE.x + 20.0, MIDDLE[1])
    for _ in range(round((PARAMS.pose_hold_s + 0.1) * FPS)):
        script.append(
            (
                hand(small_at, label="peace", track_id=1, width=HAND / 2),
                hand(MIDDLE, label="like", track_id=2, width=HAND),
            )
        )
    got = drive(judge, script)
    assert got is not None and got.key == "like"


def test_没有手的帧也算在保持里():
    # 一半的帧没有手,到不了八成
    script = []
    for i in range(round((PARAMS.pose_hold_s + 0.3) * FPS)):
        script.append(() if i % 2 else (hand(MIDDLE, label="peace"),))
    assert drive(make(), script) is None


def test_偶尔漏一帧还算保持():
    script = []
    for i in range(round((PARAMS.pose_hold_s + 0.2) * FPS)):
        script.append(() if i % 10 == 0 else (hand(MIDDLE, label="peace"),))
    assert drive(make(), script) is not None


def test_保持的时候屏幕上有进度():
    judge = make()
    drive(judge, still(PARAMS.pose_hold_s * 0.5, label="peace"))
    assert 0.0 < judge.hold_progress < 1.0


def test_握拳类不会被当成保持的姿势():
    # 配置校验已经挡住了这种配法,判定这边也不该把拳头数成一次保持
    out = drive_all(make(), still(PARAMS.pose_hold_s + 0.5, label="fist"))
    assert [t.kind for t in out] == []


# ---- 停止手势 ----------------------------------------------------------


@pytest.mark.parametrize("label", ["palm", "stop", "stop_inverted", "three", "four"])
def test_五个标签连续三帧都算叫停(label):
    judge = make()
    drive(judge, still(3 / FPS, label=label))
    assert judge.stop


def test_不连续就不叫停():
    judge = make()
    script = []
    for i in range(30):
        script.append((hand(MIDDLE, label="palm"),) if i % 2 else (hand(MIDDLE, label="fist"),))
    drive(judge, script)
    assert not judge.stop


def test_两帧还不算():
    judge = make()
    drive(judge, still(2 / FPS, label="palm"))
    assert not judge.stop


def test_区外的手做停止手势也算():
    judge = make()
    far = (50.0, 50.0)
    drive(judge, still(3 / FPS, far, label="palm"))
    assert judge.stop


def test_锁定期间别的手做停止手势也算():
    judge = make()
    drive(judge, still(PARAMS.settle_s + 0.1))
    assert judge.ready
    script = []
    for _ in range(3):
        script.append((hand(MIDDLE, track_id=1), hand((100.0, 100.0), label="palm", track_id=9)))
    drive(judge, script, start_t=1.0)
    assert judge.stop


def test_没确认的手做停止手势也算():
    # 漏一次叫停比多停一次危险;连续 3 帧本身就是防抖那一道
    judge = make()
    drive(judge, still(3 / FPS, label="palm", confirmed=False))
    assert judge.stop


def test_这一票只报出来不动机械臂():
    judge = make()
    drive(judge, still(3 / FPS, label="palm"))
    assert judge.stop
    # 停止手势不走触发那条路:它不是一件 Trigger
    assert drive(make(), still(0.5, label="palm")) is None


# ---- 清记录 ------------------------------------------------------------


def test_清记录清掉握拳平移和保持但不清停止手势的计数():
    judge = make()
    drive(judge, still(PARAMS.settle_s + 0.1))
    assert judge.ready
    judge.clear()
    assert not judge.ready
    # 清完接着移动:没停稳过,不触发
    assert drive(judge, sweep(MIDDLE, (MIDDLE[0] - MOVE - 20, MIDDLE[1]), 0.3), start_t=1.0) is None


def test_清记录不清停止手势的连续帧计数():
    judge = make()
    # 先攒两帧停止手势
    drive(judge, still(2 / FPS, label="palm"))
    assert not judge.stop
    judge.clear()  # 换送法
    # 再来一帧就够 3 帧:清了的话会晚约 0.1 秒才停
    drive(judge, still(1 / FPS, label="palm"), start_t=1.0)
    assert judge.stop


def test_清记录也清掉保持的进度():
    judge = make()
    drive(judge, still(PARAMS.pose_hold_s * 0.6, label="peace"))
    assert judge.hold_progress > 0.0
    judge.clear()
    assert judge.hold_progress == 0.0


# ---- 走一遍那个脚本画得出来 --------------------------------------------
#
# 画错了(坐标算成小数、清单被画面边切掉)要在这里就撞出来,不能等站到摄像头前面。


def walkthrough(current: int = 0):
    """走到第几件的一个走一遍。"""
    from vision_side.scripts import triggers as script

    walk = script.Walkthrough(script.plan(DEMO))
    walk.current = current
    return walk


def screen(judge, walk, *, result=None):
    """按判定现在的样子画一帧,返回画好的图。`walk` 给 None 就是自由模式。"""
    import numpy as np

    from vision_side.scripts import triggers as script

    image = np.zeros((720, 1280, 3), np.uint8)
    if result is None:
        result = FrameResult(t=0.0, hands=(hand(MIDDLE),))
    script._paint(image, result, judge, ALIGNMENT, walk, PARAMS, 29.5)
    return image


def test_走一遍的每一件都画得出来():
    walk = walkthrough()
    assert len(walk.steps) == len(DEMO.triggers) + 1  # 每个触发手势 + 一次叫停
    for i in range(len(walk.steps) + 1):  # 多一轮:都走完了也要画得出来
        assert screen(make(), walkthrough(i)).any()


def test_自由模式也画得出来():
    assert screen(make(), None).any()


def test_清单不会被画面右边切掉():
    # 手势名是从演示配置里来的,写死清单宽度的话换个动作名就会被切掉
    image = screen(make(), walkthrough())
    # 最右边那几列只该有衬底那层暗色(约 11),不该有字(亮的)
    assert image[:, -3:].max() < 60


def test_清单再长也切不到():
    import numpy as np

    from vision_side.scripts import triggers as script

    long = script.Walkthrough([script.Step("fist_move", "left", "FIST MOVE LEFT " * 3)])
    image = np.zeros((720, 1280, 3), np.uint8)
    script.draw_checklist(image, long)
    # 挤不下的时候靠左贴边,宁可左边溢出,也不让右边把字切了
    assert image[:, -3:].max() < 60


def test_READY和方向箭头和进度圈都画得出来():
    judge = make()
    drive(judge, still(PARAMS.settle_s + 0.1))
    assert judge.ready
    assert screen(judge, walkthrough(0)).any()
    # 往前推、敲两下没有箭头,写字
    for script in (push(), knock()):
        judge3 = make()
        for i, hands in enumerate(script):
            if judge3.update(FrameResult(t=i / FPS, hands=hands)) is not None:
                break
        assert judge3.direction in ("forward", "back")
        assert screen(judge3, walkthrough(0)).any()
    judge2 = make()
    drive(judge2, still(PARAMS.pose_hold_s * 0.5, label="peace"))
    assert judge2.hold_progress > 0.0
    assert screen(judge2, walkthrough(4)).any()


def test_走完以后出得了那张表():
    walk = walkthrough()
    walk.steps[0].started_at, walk.steps[0].done_at = 0.0, 1.5
    walk.steps[1].wrong.append("fist_move:left")
    text = walk.summary()
    assert f"1/{len(DEMO.triggers) + 1}" in text
    assert "FIST MOVE LEFT" in text
    assert "fist_move:left" in text  # 方向判错要看得见
    assert not walk.all_passed


def test_认出来就打勾并且隔一会儿进下一件():
    import logging

    from vision_side.triggers import Trigger

    walk = walkthrough()
    log = logging.getLogger("test.walk")
    first = walk.step
    walk.score(Trigger(kind=first.kind, key=first.key, t=1.0, since=0.5, track_id=1), False, 1.0, log)
    assert first.done_at == 1.0
    assert walk.step is first  # 还停一会儿,让人看清打了勾
    walk.score(None, False, 1.0 + 99.0, log)
    assert walk.step is not first  # 到点了自动进下一件


def test_做错了会记下来认成了什么():
    import logging

    from vision_side.triggers import Trigger

    walk = walkthrough()
    log = logging.getLogger("test.walk")
    walk.score(Trigger(kind="fist_move", key="up", t=1.0, since=0.5, track_id=1), False, 1.0, log)
    assert walk.step.done_at is None
    assert walk.step.wrong == ["fist_move:up"]


# ---- 收手抑制自己的门槛 ------------------------------------------------
#
# 它另跑一个不看姿势的笔画检测器:距离 1 个手宽、轴比 2,和上面那次移动的 1.5 个手宽、
# 轴比 3 不是一套(原型 camera-prototype-4 的 zone_rule.py)。往回收那一下通常更短,
# 拿移动的门槛去量会漏掉。

FROM = (700.0, 360.0)


def retract_then_go(back: tuple[float, float]):
    """先从 FROM 退到 back,在那儿停稳,再朝右移过 1.5 个手宽。"""
    assert ZONE.contains(FROM) and ZONE.contains(back)
    ahead = (back[0] + MOVE + 10, back[1])
    return (
        still(PARAMS.settle_s + 0.1, FROM)
        + sweep(FROM, back, 0.3)
        + still(PARAMS.settle_s + 0.1, back)
        + sweep(back, ahead, 0.3)
    )


def test_退了一个手宽就算朝反方向动过():
    # 退 88 px:够不上一次移动(120 px),但过得了收手抑制的 80 px
    back = (FROM[0] - GUARD - 8, FROM[1])
    assert MOVE > FROM[0] - back[0] > GUARD
    assert drive(make(), retract_then_go(back)) is None


def test_只退了一点不算朝反方向动过():
    # 退 60 px:够不上 80 px,这次右移照样算数
    back = (FROM[0] - 60.0, FROM[1])
    assert FROM[0] - back[0] < GUARD
    got = drive(make(), retract_then_go(back))
    assert got is not None and got.key == "right"


def test_斜着退的不算朝反方向动过():
    # 左 88、上 88:轴比 1,过不了收手抑制自己的轴比 2
    back = (FROM[0] - GUARD - 8, FROM[1] - GUARD - 8)
    got = drive(make(), retract_then_go(back))
    assert got is not None and got.key == "right"


# ---- 两件同一帧成立 ----------------------------------------------------


def test_两件同一帧成立也不会丢():
    """握拳平移和姿势保持同一帧都成立时,两件都要报出来。

    两只手各做各的:1 号手先单独停稳、锁定,随后 2 号手(更大)进区保持比 V;
    1 号手发难的那一帧换着来,总有一个位置正好和「保持满一秒」撞在一起,所以扫一遍,
    每个位置都要两件齐全。
    """
    for jump in range(28, 49):
        judge = make()
        script = []
        for i in range(60):
            hands = [hand(RIGHT_IN if i >= jump else LEFT_IN, track_id=1)]
            if i >= 12:
                hands.append(hand(MIDDLE, label="peace", track_id=2, width=HAND * 2))
            script.append(tuple(hands))
        kinds = [t.kind for t in drive_all(judge, script)]
        assert "fist_move" in kinds, f"jump={jump} 没认出握拳平移"
        assert "pose_hold" in kinds, f"jump={jump} 丢了姿势保持"


def test_停住段里那两成非握拳的帧也要老实待着():
    # 「移动不超过半个手宽」和「八成帧是握拳类」是分开的两条:非握拳的那几帧不免检。
    # 这一帧手在别处、只是没被认成握拳类,整段就不算停住
    script = still(PARAMS.settle_s + 0.1)
    # 放在第 9 帧:再早一点,它会在移动把幅度撑开之前就滑出停住段的窗口,
    # 中间留出一个空档又停稳了,那就不是在测这一条了
    odd = round(PARAMS.settle_s * FPS) - 1
    script[odd] = (hand((MIDDLE[0] + SETTLE + 20, MIDDLE[1]), label="palm"),)
    script += sweep(MIDDLE, (MIDDLE[0] - MOVE - 20, MIDDLE[1]), 0.4)
    assert drive(make(), script) is None


def test_保持那几件要说清楚手势怎么做():
    # 2026-09-21 站着走一遍时,HOLD PEACE 那一件卡了 10.6 秒 —— 屏幕上只写了标签名,
    # 没说 peace 是哪个手势。光有标签名不算提示
    from vision_side.scripts import triggers as script

    for step in script.plan(DEMO):
        if step.kind != "pose_hold":
            continue
        assert step.says != f"HOLD {step.key.upper()} FOR 1 S"
        assert script.POSE_SAYS[step.key] in step.says


def test_前后两件要说清楚怎么做():
    # FIST MOVE FORWARD / BACK 光看名字猜不出是推、是敲
    from vision_side.scripts import triggers as script

    says = {step.key: step.says for step in script.plan(DEMO) if step.kind == "fist_move"}
    assert "PUSH" in says["forward"]
    assert "KNOCK" in says["back"]


def test_没配说法的姿势也不会画不出来():
    from vision_side.scripts import triggers as script

    step = script.Step("pose_hold", "ok", f"HOLD {script.POSE_SAYS.get('ok', 'OK')} FOR 1 S")
    assert "OK" in step.says

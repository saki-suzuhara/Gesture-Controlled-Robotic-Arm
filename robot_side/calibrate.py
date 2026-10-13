"""标定工作台坐标系:算角度、改配置、排验证移动。纯计算,不碰 RoboDK。

**要标的只有绕 Z 的那个角度。** 工作台坐标系相对机械臂基坐标系就是"绕 Z 转一个角度
加一个可选的原点平移"(`robot_side.motion.worktable_pose`);Z 轴两边是同一个,
原点平移由人量着填,所以现场唯一要量的就是角度。换机械臂时只重新标它,同一套路点
照用(ADR-0003、用户故事 33)。

**怎么标:**把机械臂手动挪到「操作者正前方那条线」上——也就是从底座转轴看过去,
法兰和操作位在同一个方位上,离转轴远一点。这时法兰在工作台坐标系里应当落在 **−Y**
上(待命点是 (0, −400, 550),`config/arms/*.toml`),于是

    工作台坐标系的角度 = 法兰相对底座的方位角 − (−90°)

`TOWARD_OPERATOR_BEARING_DEG` 就是那个 −90°。**算的时候要用基坐标系里的法兰位姿**
(`RoboDKArm.flange_pose_in_base`),不能用换算到工作台坐标系的那一个——工作台坐标系
正是这会儿还没定的东西。

改配置只改 `[worktable] angle_deg` **那一行的那一个数**,注释和别的字段一行不动:
这两份 TOML 是人手写、带满注释的,整份重写会把它们冲掉。
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from robodk.robomath import Mat, transl

from robot_side.guard import (
    GuardRefused,
    MoveLimits,
    PowerState,
    check_joint_travel,
    check_linear_travel,
    check_pose_readback,
    check_powered,
)

TOWARD_OPERATOR_BEARING_DEG = -90.0
"""标定时法兰**应当**落在工作台坐标系里的哪个方位(度)。

−90° 就是 −Y:待命点是 (0, −400, 550),六个预设动作的尺寸都相对它量
(`CONTEXT.md`「待命点」)。两台机械臂共用同一个待命点,所以这个数和机械臂无关。
"""

MIN_RADIUS_MM = 200.0
"""法兰离底座转轴(Z 轴)在水平面上至少要有这么远,算出来的方位角才有意义。

离得太近时,水平面上一点点位置误差就能把方位角甩出几十度。够不到这个距离就拒绝标,
让人把机械臂往外伸一点再来。
"""

VERIFY_MOVE_MM = 50.0
"""验证移动走多远:沿工作台坐标系 −Y(朝操作者)走这么多毫米。

选朝操作者走,是因为它**直接验证标出来的角度对不对**:角度标对了,机械臂就是朝人
走过来的那条线;标错 90°,一眼就看得出来走岔了。距离取 50 mm,和回待命的抬起同一
个量级——看得见,又短到出岔子也来得及按急停。
"""

VERIFY_MAX_AXIS_DEG = 15.0
"""验证移动时每一轴最多准转多少度(`robot_side.guard.MoveLimits`)。

50 mm 的直线在臂展一米多的机械臂上换成关节转角远小于这个数;设成 15° 是留给姿态
插值和不同构型的余地,同时把「目标算错了、要大转一圈」那一类挡在外面。
"""

VERIFY_MAX_TURN_DEG = 1.0
"""验证移动只挪位置、不转姿态,所以转角上限给一个很小的数。"""

VERIFY_COUNTDOWN_S = 3
"""验证移动前倒计时几秒(第 05 票第 8 条:按键确认 → 倒计时 3 秒 → 慢速移动)。

**写死在这里,不读演示配置。** 演示配置里那个倒计时管的是演示时的回待命,是另一件
事;把这一下挂在它身上,等于让一次配置改动能悄悄把真机动起来之前的那几秒去掉。
"""


VERIFY_LIMITS = MoveLimits(axes=(0, 1, 2, 3, 4, 5), max_delta_deg=VERIFY_MAX_AXIS_DEG)
"""验证移动准动哪几轴。

六轴都准动:50 mm 的直线本来就是六轴一起走一点点。挡住乱来的不是「只准动某一轴」,
而是那个很小的每轴上限(`VERIFY_MAX_AXIS_DEG`)加上直线段的距离上限——目标算错了、
坐标系写错了,先撞上的就是这两条。
"""

TRAVEL_SLACK_MM = 5.0
"""直线距离上限比 `VERIFY_MOVE_MM` 多留这么点,给往返一趟的舍入。"""


class CalibrationRefused(Exception):
    """这一次标定不能往下走。调用方停下,不猜、不将就。"""


def check_verify_move(
    *,
    power: PowerState,
    here_joints: Sequence[float],
    here_pose: Mat,
    wanted: Mat,
    staged: Mat,
    solved: Sequence[float] | None,
) -> None:
    """验证移动下发之前的四道护栏。过不了就抛,**调用方一律不发**。

    每一道卡的都是**读回来的那个值**,不是「我算出来的那个值」(spec 第 51 条;
    上一轮出事就出在只核对了后者,见 `robot_side.guard`):

    1. `power`:临发之前**再查一次**上电使能——倒计时这几秒里人可能按了急停;
    2. `staged`:目标放进 RoboDK 之后从**那个目标身上**读回来的位姿,和 `wanted`
       逐项核对;
    3. `staged` → `solved`:拿**读回来的**目标求出来的逆解,逐轴核对行程;
    4. `here_pose` → `staged`:直线段另外卡走过的距离和转过的角度。

    第 3 和第 4 卡的是同一段运动的两头(关节和法兰),算错了哪一头都拦得下。
    """
    check_powered(power)
    check_pose_readback(wanted, staged)
    check_linear_travel(
        here_pose,
        staged,
        max_mm=VERIFY_MOVE_MM + TRAVEL_SLACK_MM,
        max_deg=VERIFY_MAX_TURN_DEG,
    )
    if solved is None:
        raise GuardRefused("从读回来的目标求不出逆解(够不着或在奇异点上),不下发")
    check_joint_travel(here_joints, solved, VERIFY_LIMITS)


def normalize_deg(angle: float) -> float:
    """把角度折到 (−180, 180]。"""
    folded = math.fmod(angle, 360.0)
    if folded <= -180.0:
        folded += 360.0
    elif folded > 180.0:
        folded -= 360.0
    return folded


def worktable_angle_deg(
    flange_in_base: Mat | Sequence[float],
    origin_mm: Sequence[float] = (0.0, 0.0, 0.0),
) -> float:
    """由**基坐标系里**的法兰位姿,算出工作台坐标系绕 Z 转过的角度。

    前提是人已经把法兰挪到了操作者正前方那条线上(见模块说明)。`origin_mm` 是配置里
    的原点平移:方位角从那个原点量起,所以原点不在底座上时也算得对。
    """
    position = flange_in_base.Pos() if isinstance(flange_in_base, Mat) else flange_in_base
    x = float(position[0]) - float(origin_mm[0])
    y = float(position[1]) - float(origin_mm[1])
    radius = math.hypot(x, y)
    if radius < MIN_RADIUS_MM:
        raise CalibrationRefused(
            f"法兰离底座转轴只有 {radius:.0f} mm(水平面上),不到 {MIN_RADIUS_MM:.0f} mm,"
            "这么近算出来的方位角靠不住;把机械臂往操作者那边多伸一点再标"
        )
    bearing = math.degrees(math.atan2(y, x))
    return normalize_deg(bearing - TOWARD_OPERATOR_BEARING_DEG)


def verify_target(from_pose: Mat, mm: float = VERIFY_MOVE_MM) -> Mat:
    """验证移动的目标:从现在这个法兰位姿,沿**工作台坐标系** −Y 走一小段。

    `from_pose` 和返回值都在工作台坐标系里。姿态不动——左乘一个纯平移,转的部分原样
    留着(和 `robot_side.motion.plan_returning` 里抬起那一段同一个写法)。
    """
    return transl(0.0, -abs(mm), 0.0) * from_pose


# ---- 改配置:只动 [worktable] angle_deg 那一行 ---------------------------

STAMP_MARK = "标定于"
"""角度那一行后面记一笔是什么时候标的。重标时整句换掉,不往上叠。"""

_SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_ANGLE = re.compile(r"^(\s*angle_deg\s*=\s*)([^#\n]*?)(\s*(?:#.*)?)$")


def rewrite_angle(text: str, angle_deg: float, *, stamp: str) -> str:
    """把 `[worktable]` 里的 `angle_deg` 改成新值,**别的一行都不动**。

    找不到那一节、那一节里没有这个字段、或者写了不止一个,一律拒绝改:这两份配置是
    人手写的,拿不准改哪一行时,不动比猜着改好。

    `stamp` 是标定的日期,记进那一行的行尾注释里,好让人事后知道这个数是哪天量的。
    """
    lines = text.splitlines(keepends=True)
    section = ""
    hits = []
    for index, line in enumerate(lines):
        found = _SECTION.match(line)
        if found:
            section = found.group(1).strip()
            continue
        if section == "worktable" and _ANGLE.match(line):
            hits.append(index)
    if not hits:
        raise CalibrationRefused(
            "配置里找不到 [worktable] 一节下的 angle_deg,没有改;"
            "照 config/arms/jaka.toml 的样子补上这一节再来"
        )
    if len(hits) > 1:
        raise CalibrationRefused(
            f"[worktable] 里写了 {len(hits)} 个 angle_deg,不知道该改哪一个,没有改"
        )
    index = hits[0]
    head, _, _ = _ANGLE.match(lines[index]).groups()
    ending = "\n" if lines[index].endswith("\n") else ""
    lines[index] = f"{head}{angle_deg:.4f}  # {STAMP_MARK} {stamp}{ending}"
    return "".join(lines)

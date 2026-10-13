"""两份 TOML:机械臂配置(一台一份)和演示配置(各台共用)。

字段和单位照 `docs/grill-decisions.md` 第二节,术语照 `CONTEXT.md`:

- 位置 mm、角度 °、直线速度 mm/s、停顿秒;关节段写占最大关节速度的百分比。
- 路点是**法兰**在工作台坐标系里的位姿(ADR-0013),按相对待命点的偏移写。
- 和机械臂有关的运行参数(回待命的关节速度等)放机械臂配置,和机械臂无关的
  (倒计时、冷却、识别中断、停止手势帧数、抬起高度)放演示配置。
- 视觉一侧的判定参数(启动对准的那几个,在 `[vision]` 一节)也和机械臂无关,
  同样放演示配置;用它们的代码在 `vision_side/`。

**速度**:仿真里照写的走。真机上看机械臂配置的 `program_sets_speed`——`false` 时程序设的
速度到不了控制柜,快慢全看示教器(ADR-0011 的 2026-09-20 更新);`true` 时经包过的驱动
发出去,实际速度 = 这里写的 × 示教器上的速度倍率(2026-09-23:原驱动每行指令都把速度重置
成 100,那就是 9-20「四条通道全部无效」的原因,`robot_side/drivers/jaka_keep_speed.py`)。

配置里写错的东西一律在**启动时**报成 `ConfigError`,报里说清是哪个文件、哪个字段、
为什么不对:字段缺失或多写、单位不对(数值落在这个单位说不通的范围里)、触发手势
用了会被认成停止手势的姿势。
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

DEFAULT_ORIENTATION_DEG = (180.0, 0.0, 0.0)
"""默认姿态:法兰 Z 轴竖直朝下。演示配置里不写 `default_orientation_deg` 时用它。"""

STOP_GESTURE_LABELS = frozenset({"palm", "stop", "stop_inverted", "three", "four"})
"""会被认成停止手势的姿势标签(`CONTEXT.md`「停止手势」)。触发手势不能用这些。"""

FIST_LIKE_LABELS = frozenset({"fist", "grip", "point", "little_finger", "fist_inverted"})
"""握拳类标签。姿势保持不能用这些:拳头要先停住,停久一点就会先触发姿势保持。"""

HOME_ACTION = "home"
"""触发手势可以对应的一个**保留名**:不是预设动作,是回待命——和笔记本上按 H 同一件事
(只在空闲、冷却时有效,已经在待命位姿上就不动)。演示配置里不用、也不能为它写 [[actions]]。
2026-09-23 用户定 rock 手势对应它。"""

HOME_DISPLAY = "HOME"

FIST_MOVE = "fist_move"
POSE_HOLD = "pose_hold"
TRIGGER_KINDS = (FIST_MOVE, POSE_HOLD)
FIST_MOVE_DIRECTIONS = ("left", "right", "up", "down", "forward", "back")
"""握拳平移认得的方向。前四个是在画面里平移;`forward` 是握拳往镜头推出去停住,`back`
是握拳像敲桌子那样往下点两下(2026-09-23 用户定,2026-09-24 录了原型定的判定规则,
见 `vision_side.triggers`)。"""

POWER_CHECK_NONE = "none"
POWER_CHECK_JAKA = "jaka_jkrc"
POWER_CHECKS = (POWER_CHECK_NONE, POWER_CHECK_JAKA)
"""怎么查这台机械臂上电使能了没有。

`none` 是**查不了**——查不了就不许连真机(第 05 票第 3 条:没有上电使能不下发任何
运动,那就必须先查得出来)。Nova 5 怎么接还没定,它就是这一种。

`jaka_jkrc` 是经 RoboDK 自带的嵌入式 Python 跑一个只读小脚本,用 JAKA 官方的 `jkrc`
读控制柜状态位。**RoboDK 的 JAKA 驱动本身查不了**:驱动里连 `power_on` / `enable_robot`
都没有(`notes/jaka-first-connection-results.md`),状态位也不经它往外报。
"""


class ConfigError(Exception):
    """配置里写错的东西。启动时抛出,程序不动机械臂。"""


# ---- 读出来的东西 ------------------------------------------------------


@dataclass(frozen=True)
class Worktable:
    """工作台坐标系:相对机械臂基坐标系绕 Z 轴转过的角度,加上可选的原点平移。"""

    angle_deg: float = 0.0
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ArmConfig:
    """一台机械臂的配置。换机械臂就换这一份。"""

    path: Path
    name: str
    robot_name: str
    """工作站里这台机械臂的名字,程序按它找对应的那台。"""
    driver: str
    ip: str | None
    worktable: Worktable
    standby_joints_deg: tuple[float, ...]
    pointer_offset_mm: tuple[float, float, float]
    """指针偏移:只作参考,不参与运动计算(ADR-0013)。"""
    max_joint_degs: float
    max_linear_mms: float
    return_joint_percent: float
    """回待命的关节段:占最大关节速度的百分比。算不出尖端走多远时(空实现)才用它。"""
    program_sets_speed: bool = False
    """连着真机时,程序设的速度要不要真的发给控制柜。

    `true`:经包过的驱动下发(`robot_side/drivers/jaka_keep_speed.py`),每段按演示配置里的
    速度走,再乘控制柜上的速度倍率;`max_joint_degs` / `max_linear_mms` 是发出去的上限。
    `false`:用 RoboDK 原来的驱动,程序设的速度到不了控制柜(那个驱动每行指令都把速度
    重置成 100),快慢全看示教器(ADR-0011 的 2026-09-20 更新)。写在 `[speed]` 里,默认 false。
    只跑仿真时不看它:仿真里速度本来就生效。
    """
    power_check: str = POWER_CHECK_NONE
    """怎么查这台机械臂上电使能了没有(`POWER_CHECKS`)。

    写在 `[connection]` 一节里,不写就是 `none`——查不了。**查不了就不许连真机**
    (第 05 票第 3 条),但仿真照跑:仿真里没有上电使能这回事。
    """
    noarm_segment_s: float = 1.0
    """空实现(NO ARM)每段运动固定等多久。只有空实现看这个数,别的实现不理它。

    写在 `[noarm]` 一节里,默认 1 秒;不写这一节就是默认值(`CONTEXT.md`「空实现」)。
    """

    @property
    def return_joint_degs(self) -> float:
        """回待命的关节速度(°/s,只用于仿真)。"""
        return self.max_joint_degs * self.return_joint_percent / 100.0


@dataclass(frozen=True)
class Waypoint:
    """一个路点:法兰相对**出发点**的偏移,加上可选的姿态、点头角和到达后的停顿。

    出发点是这个预设动作开始时机械臂停着的那个停靠点;从待命位姿出发时就是待命点
    (2026-09-23 起动作做完不回待命,从哪儿停着就从哪儿算)。

    `orientation_deg` 是工作台坐标系下的**绝对**姿态(`CONTEXT.md`「路点」);不写就用
    演示配置里的默认姿态。点头角是在姿态之上再绕平行工作台 X 轴、过法兰的轴转过的角度;
    法兰位置不变,指针尖端会扫出弧。
    """

    dx_mm: float = 0.0
    dy_mm: float = 0.0
    dz_mm: float = 0.0
    orientation_deg: tuple[float, float, float] | None = None
    nod_deg: float = 0.0
    pause_s: float = 0.0


END_STAY = "stay"
"""做完停在最后一个路点上(左右上下这种:一次挪一步,可以叠加)。"""
END_BACK = "back"
"""做完直线回到出发点(画方块、点头这种:原地做一遍,做完还在原处)。"""
ACTION_ENDS = (END_STAY, END_BACK)


@dataclass(frozen=True)
class Action:
    """一个预设动作:从现在停着的地方出发,逐段直线走过各路点,做完按 `end` 停下。

    **做完不回待命位姿**(2026-09-23 用户定):`stay` 的停在最后一个路点上,`back` 的
    直线回到出发点。回待命只在叫停之后、已停止按回车之后、以及在笔记本上按 H 的时候。
    """

    name: str
    display: str
    speed_mms: float
    waypoints: tuple[Waypoint, ...]
    end: str = END_BACK

    @property
    def shift_mm(self) -> tuple[float, float, float]:
        """做完以后停靠点挪了多少:`stay` 是最后一个路点的偏移,`back` 是零。"""
        if self.end != END_STAY:
            return (0.0, 0.0, 0.0)
        last = self.waypoints[-1]
        return (last.dx_mm, last.dy_mm, last.dz_mm)


@dataclass(frozen=True)
class TriggerRule:
    """一个触发手势和它对应的预设动作。`key` 是握拳平移的方向或姿势保持的姿势标签。"""

    kind: str
    key: str
    action: str


@dataclass(frozen=True)
class RunParams:
    """和机械臂无关的运行参数。"""

    countdown_s: float
    cooldown_s: float
    vision_lost_after_s: float
    """多久拿不到新的识别结果算识别中断(`CONTEXT.md`「识别中断」,VISION LOST)。"""
    stop_gesture_frames: int
    lift_mm: float
    return_speed_mms: float
    """回待命两段(先抬起、再回去)指针尖端走多快。和预设动作一个量纲,好让它们一样快。

    这两段都是关节运动:关节速度由执行器按「指针尖端要走多远」现算
    (`robot_side.executor.settle`),所以写的是 mm/s,不是 °/s。
    """
    range_mm: tuple[float, float, float]
    """活动范围:停靠点离待命点最多多远,工作台坐标系里 ±X、±Y、±Z 各一个数。

    动作做完要停到这个范围外面的,那一次就不动(OUT OF RANGE)。只管停靠点,不管
    途中:画方块在边上做,方块的一角可以伸出去一点,够不够得着由可达性检查说了算。
    """


@dataclass(frozen=True)
class VisionParams:
    """视觉一侧的判定参数,演示配置里的 `[vision]` 一节。

    和哪台机械臂都无关,所以和运行参数一样放在演示配置里,调参不用改代码。
    现在只有启动对准那几个(`CONTEXT.md`「启动对准」);握拳平移、姿势保持、摄像头
    编号和分辨率由第 03 票加进来,还是这一张表、这一个类。
    """

    detection_region_px: tuple[int, int]
    """检测区域多大,宽高像素。"""
    hold_s: float
    """整幅画面里拳头要停稳多久,之后才切到检测区域。"""
    measure_s: float
    """切到检测区域以后量手宽量多久。和 `hold_s` 合起来约 2 秒。"""
    still_hand_widths: float
    """停稳期间手框中心最多移动几个手宽。"""
    fist_percent: float
    """停稳期间至少几成帧是握拳类标签。"""
    start_zone_hand_widths: float
    """起点区边长是几个手宽。"""
    too_close_px: float
    """整幅画面里的拳头手框宽超过多少就提示 TOO CLOSE。只提示,不拦着对准。"""

    camera_index: int
    """摄像头编号。笔记本自带的 RGB 是 0 号。"""
    camera_px: tuple[int, int]
    """摄像头分辨率,宽高像素。"""

    settle_s: float
    """握拳平移的停住段:拳头要在起点区里停住多久。"""
    settle_hand_widths: float
    """停住段手框中心最多晃几个手宽。对准那一段另有 `still_hand_widths`。"""
    settle_percent: float
    """停住段至少几成帧是握拳类标签。"""
    move_hand_widths: float
    """移动段:要移过几个手宽才算。"""
    move_max_s: float
    """移动段最多用多久。超了不算——慢慢挪位置不该触发。"""
    move_percent: float
    """移动段至少几成帧是握拳类标签。比停住段松:快速移动会漏检。"""
    move_axis_ratio: float
    """主方向的位移至少是另一方向的几倍。斜着移被忽略,这是有意的。"""
    reverse_guard_s: float
    """收手抑制(REVERSE GUARD):触发前这么久内朝反方向动过(不看姿势)就不触发。"""
    reverse_guard_hand_widths: float
    """反方向「动过」要动过几个手宽。比一次移动的门槛小——往回收那一下通常更短。"""
    reverse_guard_axis_ratio: float
    """反方向那一下也要够直。和移动段的轴比各是各的。"""
    lock_timeout_s: float
    """锁定超时:最后一次 READY 之后过了这么久就解除锁定。"""

    settle_size_percent: float
    """往前推(FIST MOVE FORWARD)从哪儿量起:停住段里手框宽最多变几成,才算连远近也停住了。

    平移只看手框中心;往镜头推的时候中心几乎不动、手框在变大,不另看大小的话,推的
    过程中一直算「停住」,起点跟着手往前挪,永远量不出变大了多少。"""
    push_size_ratio: float
    """往前推:手框宽至少变成停住时的几倍。"""
    push_height_ratio: float
    """往前推:手框高至少变成停住时的几倍。只看宽的话,食指往旁边一指手框也会变宽。"""
    push_frames: int
    """往前推:要连着几次看见手框够大(中间没看见手的帧不算)。一两帧的手框跳变不算数。"""
    knock_hand_widths: float
    """敲两下(FIST MOVE BACK):每一下往下至少几个手宽。"""
    knock_return_hand_widths: float
    """敲两下:每一下之后再抬回来至少几个手宽,才算敲完这一下。"""
    knock_window_s: float
    """敲两下:从第一下起、到第二下往下过门槛,最多多久。"""
    knock_percent: float
    """敲两下那一段里至少几成帧是握拳类标签(没有手的帧也算在内)。"""

    pose_hold_s: float
    """姿势保持:一个姿势要保持多久。"""
    pose_percent: float
    """保持期间至少几成帧是同一个姿势。没有手的帧也算在内。"""
    release_s: float
    """松开:这个姿势连续这么久没被认出,才算松开。松开之前同一个姿势不会再触发。"""


@dataclass(frozen=True)
class DemoConfig:
    """演示配置:各台机械臂共用。"""

    path: Path
    default_orientation_deg: tuple[float, float, float]
    run: RunParams
    vision: VisionParams
    actions: tuple[Action, ...]
    triggers: tuple[TriggerRule, ...]

    def action(self, name: str) -> Action:
        for a in self.actions:
            if a.name == name:
                return a
        raise KeyError(f"演示配置里没有叫 {name!r} 的预设动作")

    def display_for(self, name: str) -> str:
        """触发手势对应的那件事在屏幕上叫什么:预设动作用它的英文名,回待命是 HOME。"""
        return HOME_DISPLAY if name == HOME_ACTION else self.action(name).display

    def action_for_trigger(self, kind: str, key: str) -> str | None:
        for t in self.triggers:
            if t.kind == kind and t.key == key:
                return t.action
        return None


# ---- 读 TOML:一层薄壳,把「哪个文件、哪个字段」一路带着 ----------------


class _Table:
    """一张 TOML 表。读过的键会被划掉,`done()` 时还有剩的就是多写的字段。"""

    def __init__(self, path: Path, data: dict[str, Any], prefix: str = "") -> None:
        self.path = path
        self.data = dict(data)
        self.prefix = prefix

    def _where(self, key: str) -> str:
        return f"{self.path.name} 的 {self.prefix}{key}"

    def fail(self, key: str, why: str) -> ConfigError:
        return ConfigError(f"{self._where(key)}:{why}")

    def table(self, key: str, *, required: bool = True) -> _Table:
        value = self.data.pop(key, None)
        if value is None:
            if required:
                raise self.fail(key, "缺这一节")
            value = {}
        if not isinstance(value, dict):
            raise self.fail(key, f"应当是一节 [{self.prefix}{key}],读到的是 {type(value).__name__}")
        return _Table(self.path, value, f"{self.prefix}{key}.")

    def string(self, key: str, *, default: str | None = None) -> str:
        value = self.data.pop(key, default)
        if value is None:
            raise self.fail(key, "缺这个字段")
        if not isinstance(value, str):
            raise self.fail(key, f"应当是一段文字,读到的是 {value!r}")
        return value

    def number(
        self,
        key: str,
        *,
        unit: str,
        lo: float,
        hi: float,
        default: float | None = None,
    ) -> float:
        """读一个数。不在 [lo, hi] 里时,按「单位多半写错了」报出来。"""
        value = self.data.pop(key, default)
        if value is None:
            raise self.fail(key, f"缺这个字段(单位 {unit})")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self.fail(key, f"应当是一个数(单位 {unit}),读到的是 {value!r}")
        value = float(value)
        if not lo <= value <= hi:
            raise self.fail(
                key,
                f"{value:g} 不在 {lo:g}–{hi:g} 之间;这个字段的单位是 {unit},是不是单位写错了",
            )
        return value

    def boolean(self, key: str, *, default: bool | None = None) -> bool:
        value = self.data.pop(key, default)
        if value is None:
            raise self.fail(key, "缺这个字段")
        if not isinstance(value, bool):
            raise self.fail(key, f"应当是 true 或 false,读到的是 {value!r}")
        return value

    def integer(self, key: str, *, lo: int, hi: int, default: int | None = None) -> int:
        value = self.data.pop(key, default)
        if value is None:
            raise self.fail(key, "缺这个字段")
        if isinstance(value, bool) or not isinstance(value, int):
            raise self.fail(key, f"应当是一个整数,读到的是 {value!r}")
        if not lo <= value <= hi:
            raise self.fail(key, f"{value} 不在 {lo}–{hi} 之间")
        return value

    def numbers(
        self,
        key: str,
        *,
        count: int,
        unit: str,
        lo: float,
        hi: float,
        default: tuple[float, ...] | None = None,
    ) -> tuple[float, ...]:
        value = self.data.pop(key, default)
        if value is None:
            raise self.fail(key, f"缺这个字段(单位 {unit})")
        if not isinstance(value, (list, tuple)):
            raise self.fail(key, f"应当是 {_cn_count(count)}个数,读到的是 {value!r}")
        if len(value) != count:
            raise self.fail(
                key, f"应当是{_cn_count(count)}个数(单位 {unit}),读到 {len(value)} 个"
            )
        out = []
        for i, v in enumerate(value):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise self.fail(key, f"第 {i + 1} 个应当是一个数,读到的是 {v!r}")
            v = float(v)
            if not lo <= v <= hi:
                raise self.fail(
                    key,
                    f"第 {i + 1} 个是 {v:g},不在 {lo:g}–{hi:g} 之间;"
                    f"这个字段的单位是 {unit},是不是单位写错了",
                )
            out.append(v)
        return tuple(out)

    def vector3(
        self, key: str, *, unit: str, lo: float, hi: float, default=None
    ) -> tuple[float, float, float]:
        """读三个数(位置或姿态),读出来就是个三元组。"""
        x, y, z = self.numbers(key, count=3, unit=unit, lo=lo, hi=hi, default=default)
        return (x, y, z)

    def tables(self, key: str) -> list[_Table]:
        """读一串 [[key]]。"""
        value = self.data.pop(key, None)
        if value is None:
            raise self.fail(key, "一条都没有")
        if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
            raise self.fail(key, f"应当是一串 [[{self.prefix}{key}]]")
        return [
            _Table(self.path, v, f"{self.prefix}{key}[{i}].") for i, v in enumerate(value)
        ]

    def done(self) -> None:
        if self.data:
            extra = "、".join(sorted(self.data))
            raise ConfigError(
                f"{self.path.name} 的 {self.prefix or '最外层'}里有认不出的字段:{extra}"
                "(字段名写错了?)"
            )


def _cn_count(n: int) -> str:
    return {2: "两", 3: "三", 6: "六"}.get(n, str(n))


def _read(path: Path) -> _Table:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"没有配置文件 {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name} 不是合法的 TOML:{exc}") from exc
    return _Table(path, data)


# ---- 机械臂配置 --------------------------------------------------------


def load_arm_config(path: Path | str) -> ArmConfig:
    """读一份机械臂配置。写错的东西在这里就报出来。"""
    root = _read(Path(path))
    name = root.string("name")
    robot_name = root.string("robot_name")

    conn = root.table("connection", required=False)
    driver = conn.string("driver", default="robodk")
    ip = conn.string("ip", default="") or None
    power_check = conn.string("power_check", default=POWER_CHECK_NONE)
    if power_check not in POWER_CHECKS:
        raise conn.fail(
            "power_check",
            f"只能是 {' 或 '.join(POWER_CHECKS)},读到的是 {power_check!r}",
        )
    conn.done()

    wt = root.table("worktable", required=False)
    worktable = Worktable(
        angle_deg=wt.number("angle_deg", unit="°", lo=-360.0, hi=360.0, default=0.0),
        origin_mm=wt.vector3(
            "origin_mm", unit="mm", lo=-2000.0, hi=2000.0, default=(0.0, 0.0, 0.0)
        ),
    )
    wt.done()

    standby = root.table("standby")
    joints = standby.numbers("joints_deg", count=6, unit="°", lo=-360.0, hi=360.0)
    standby.done()

    pointer = root.table("pointer", required=False)
    offset = pointer.vector3(
        "offset_mm", unit="mm", lo=-1000.0, hi=1000.0, default=(0.0, 0.0, 0.0)
    )
    pointer.done()

    speed = root.table("speed")
    max_joint_degs = speed.number("max_joint_degs", unit="°/s", lo=1.0, hi=720.0)
    max_linear_mms = speed.number("max_linear_mms", unit="mm/s", lo=1.0, hi=2000.0)
    return_joint_percent = speed.number("return_joint_percent", unit="%", lo=1.0, hi=100.0)
    program_sets_speed = speed.boolean("program_sets_speed", default=False)
    speed.done()

    # 空实现每段等多久。真机那两份配置里不写这一节,用默认的 1 秒。
    noarm = root.table("noarm", required=False)
    noarm_segment_s = noarm.number("segment_s", unit="秒", lo=0.0, hi=60.0, default=1.0)
    noarm.done()

    root.done()
    return ArmConfig(
        path=Path(path),
        name=name,
        robot_name=robot_name,
        driver=driver,
        ip=ip,
        power_check=power_check,
        worktable=worktable,
        standby_joints_deg=joints,
        pointer_offset_mm=offset,
        max_joint_degs=max_joint_degs,
        max_linear_mms=max_linear_mms,
        return_joint_percent=return_joint_percent,
        program_sets_speed=program_sets_speed,
        noarm_segment_s=noarm_segment_s,
    )


# ---- 演示配置 ----------------------------------------------------------


def load_demo_config(path: Path | str) -> DemoConfig:
    """读演示配置。触发手势用了会被认成停止手势的姿势,也在这里报出来。"""
    root = _read(Path(path))
    orientation = root.vector3(
        "default_orientation_deg", unit="°", lo=-360.0, hi=360.0, default=DEFAULT_ORIENTATION_DEG
    )

    run_table = root.table("run")
    run = RunParams(
        countdown_s=run_table.number("countdown_s", unit="秒", lo=0.0, hi=60.0),
        cooldown_s=run_table.number("cooldown_s", unit="秒", lo=0.0, hi=60.0),
        vision_lost_after_s=run_table.number(
            "vision_lost_after_s", unit="秒", lo=0.05, hi=10.0
        ),
        stop_gesture_frames=run_table.integer("stop_gesture_frames", lo=1, hi=60),
        lift_mm=run_table.number("lift_mm", unit="mm", lo=1.0, hi=500.0),
        return_speed_mms=run_table.number("return_speed_mms", unit="mm/s", lo=1.0, hi=2000.0),
        range_mm=run_table.vector3("range_mm", unit="mm", lo=0.0, hi=2000.0),
    )
    run_table.done()

    vision_table = root.table("vision")
    vision_w, vision_h = vision_table.numbers(
        "detection_region_px", count=2, unit="像素", lo=80.0, hi=4000.0
    )
    camera_w, camera_h = vision_table.numbers(
        "camera_px", count=2, unit="像素", lo=80.0, hi=8000.0
    )
    vision = VisionParams(
        detection_region_px=(int(vision_w), int(vision_h)),
        hold_s=vision_table.number("hold_s", unit="秒", lo=0.1, hi=10.0),
        measure_s=vision_table.number("measure_s", unit="秒", lo=0.1, hi=10.0),
        still_hand_widths=vision_table.number(
            "still_hand_widths", unit="个手宽", lo=0.05, hi=5.0
        ),
        fist_percent=vision_table.number("fist_percent", unit="%", lo=1.0, hi=100.0),
        start_zone_hand_widths=vision_table.number(
            "start_zone_hand_widths", unit="个手宽", lo=0.5, hi=20.0
        ),
        too_close_px=vision_table.number("too_close_px", unit="像素", lo=10.0, hi=4000.0),
        camera_index=vision_table.integer("camera_index", lo=0, hi=16),
        camera_px=(int(camera_w), int(camera_h)),
        settle_s=vision_table.number("settle_s", unit="秒", lo=0.05, hi=10.0),
        settle_hand_widths=vision_table.number(
            "settle_hand_widths", unit="个手宽", lo=0.05, hi=5.0
        ),
        settle_percent=vision_table.number("settle_percent", unit="%", lo=1.0, hi=100.0),
        move_hand_widths=vision_table.number(
            "move_hand_widths", unit="个手宽", lo=0.1, hi=20.0
        ),
        move_max_s=vision_table.number("move_max_s", unit="秒", lo=0.05, hi=10.0),
        move_percent=vision_table.number("move_percent", unit="%", lo=1.0, hi=100.0),
        move_axis_ratio=vision_table.number("move_axis_ratio", unit="倍", lo=1.0, hi=20.0),
        reverse_guard_s=vision_table.number("reverse_guard_s", unit="秒", lo=0.0, hi=10.0),
        reverse_guard_hand_widths=vision_table.number(
            "reverse_guard_hand_widths", unit="个手宽", lo=0.1, hi=20.0
        ),
        reverse_guard_axis_ratio=vision_table.number(
            "reverse_guard_axis_ratio", unit="倍", lo=1.0, hi=20.0
        ),
        lock_timeout_s=vision_table.number("lock_timeout_s", unit="秒", lo=0.05, hi=10.0),
        settle_size_percent=vision_table.number(
            "settle_size_percent", unit="%", lo=1.0, hi=100.0
        ),
        push_size_ratio=vision_table.number("push_size_ratio", unit="倍", lo=1.05, hi=5.0),
        push_height_ratio=vision_table.number(
            "push_height_ratio", unit="倍", lo=1.0, hi=5.0
        ),
        push_frames=vision_table.integer("push_frames", lo=1, hi=30),
        knock_hand_widths=vision_table.number(
            "knock_hand_widths", unit="个手宽", lo=0.05, hi=5.0
        ),
        knock_return_hand_widths=vision_table.number(
            "knock_return_hand_widths", unit="个手宽", lo=0.05, hi=5.0
        ),
        knock_window_s=vision_table.number("knock_window_s", unit="秒", lo=0.1, hi=10.0),
        knock_percent=vision_table.number("knock_percent", unit="%", lo=1.0, hi=100.0),
        pose_hold_s=vision_table.number("pose_hold_s", unit="秒", lo=0.05, hi=10.0),
        pose_percent=vision_table.number("pose_percent", unit="%", lo=1.0, hi=100.0),
        release_s=vision_table.number("release_s", unit="秒", lo=0.05, hi=10.0),
    )
    vision_table.done()

    actions = tuple(_read_action(t) for t in root.tables("actions"))
    _check_unique(
        [a.name for a in actions], lambda n: f"{root.path.name} 里有两个叫 {n!r} 的预设动作"
    )

    triggers = tuple(_read_trigger(t, {a.name for a in actions}) for t in root.tables("triggers"))
    _check_unique(
        [f"{t.kind}:{t.key}" for t in triggers],
        lambda n: f"{root.path.name} 里有两个 {n.split(':', 1)[1]!r} 触发手势",
    )

    root.done()
    return DemoConfig(
        path=Path(path),
        default_orientation_deg=orientation,
        run=run,
        vision=vision,
        actions=actions,
        triggers=triggers,
    )


def _read_action(table: _Table) -> Action:
    name = table.string("name")
    display = table.string("display")
    speed_mms = table.number("speed_mms", unit="mm/s", lo=1.0, hi=2000.0)
    raw = table.data.pop("waypoints", None)
    if raw is None:
        raise table.fail("waypoints", "缺这个字段")
    if not isinstance(raw, list) or not all(isinstance(w, dict) for w in raw):
        raise table.fail("waypoints", "应当是一串路点,比如 [{ dx_mm = -150.0 }]")
    if not raw:
        raise ConfigError(f"{table.path.name} 里的预设动作 {name!r} 一个路点都没有")
    waypoints = []
    for i, w in enumerate(raw):
        wt = _Table(table.path, w, f"{table.prefix}waypoints[{i}].")
        waypoints.append(
            Waypoint(
                dx_mm=wt.number("dx_mm", unit="mm", lo=-2000.0, hi=2000.0, default=0.0),
                dy_mm=wt.number("dy_mm", unit="mm", lo=-2000.0, hi=2000.0, default=0.0),
                dz_mm=wt.number("dz_mm", unit="mm", lo=-2000.0, hi=2000.0, default=0.0),
                # 路点可以自己写姿态(工作台坐标系下的绝对姿态);不写就用默认姿态
                orientation_deg=(
                    wt.vector3("orientation_deg", unit="°", lo=-360.0, hi=360.0)
                    if "orientation_deg" in wt.data
                    else None
                ),
                nod_deg=wt.number("nod_deg", unit="°", lo=-180.0, hi=180.0, default=0.0),
                pause_s=wt.number("pause_s", unit="秒", lo=0.0, hi=60.0, default=0.0),
            )
        )
        wt.done()
    end = table.string("end")
    if end not in ACTION_ENDS:
        raise table.fail("end", f"只能是 {' 或 '.join(ACTION_ENDS)},读到的是 {end!r}")
    last = waypoints[-1]
    if end == END_STAY and (last.nod_deg != 0.0 or last.orientation_deg is not None):
        # 停靠点一律是默认姿态:下一个动作的路点都按默认姿态摆,停在歪着的姿态上,
        # 下一段就会一边走一边转回来
        raise table.fail("end", "停在终点的动作,最后一个路点不能改姿态(点头角、姿态都不写)")
    table.done()
    return Action(
        name=name,
        display=display,
        speed_mms=speed_mms,
        waypoints=tuple(waypoints),
        end=end,
    )


def _read_trigger(table: _Table, action_names: set[str]) -> TriggerRule:
    kind = table.string("kind")
    if kind not in TRIGGER_KINDS:
        raise table.fail("kind", f"只能是 {' 或 '.join(TRIGGER_KINDS)},读到的是 {kind!r}")
    if kind == "fist_move":
        key = table.string("direction")
        if key not in FIST_MOVE_DIRECTIONS:
            raise table.fail(
                "direction", f"只能是 {' / '.join(FIST_MOVE_DIRECTIONS)},读到的是 {key!r}"
            )
    else:
        key = table.string("pose")
        if key in STOP_GESTURE_LABELS:
            raise table.fail(
                "pose",
                f"{key!r} 会被认成停止手势(停止手势的标签:"
                f"{'、'.join(sorted(STOP_GESTURE_LABELS))}),不能用作触发手势",
            )
        if key in FIST_LIKE_LABELS:
            raise table.fail(
                "pose",
                f"{key!r} 是握拳类标签,不能用作姿势保持的姿势:"
                "握拳平移要先停住,拳头停久一点就会先触发姿势保持",
            )
    action = table.string("action")
    if action not in action_names and action != HOME_ACTION:
        raise table.fail(
            "action",
            f"演示配置里没有叫 {action!r} 的预设动作(回待命写 {HOME_ACTION!r})",
        )
    table.done()
    return TriggerRule(kind=kind, key=key, action=action)


def _check_unique(names: list[str], complain: Callable[[str], str]) -> None:
    """名字有重的就报错;`complain` 把重了的那个名字说成一句话。"""
    seen = set()
    for n in names:
        if n in seen:
            raise ConfigError(complain(n))
        seen.add(n)


# ---- 项目里那三份配置 --------------------------------------------------

DEFAULT_ARM = "jaka"
"""不指定机械臂配置时用 JAKA 那份,而且只在仿真里动(连真机必须显式指定,第 05 票)。"""


def arm_config_path(arm: str = DEFAULT_ARM) -> Path:
    return CONFIG_DIR / "arms" / f"{arm}.toml"


def resolve_arm_config(arm: str = DEFAULT_ARM) -> Path:
    """命令行上的 `--arm` 指向哪一份:`.toml` 结尾就当路径,否则当 `config/arms/` 里的名字。

    三支脚本都要这么认一次,所以收在这里一处(`action_demo`、`state_demo`、
    `calibrate_worktable`)。
    """
    return Path(arm) if arm.endswith(".toml") else arm_config_path(arm)


def load_arm(arm: str = DEFAULT_ARM) -> ArmConfig:
    return load_arm_config(arm_config_path(arm))


def over_linear_cap(demo: DemoConfig, arm: ArmConfig) -> list[tuple[str, float]]:
    """演示配置里哪些预设动作的速度超过了这台机械臂的直线速度上限 `max_linear_mms`。

    超过的发出去会被截在上限上(`RoboDKArm.set_speed`),包过的 JAKA 驱动也不许比原来的
    100 mm/s 快。启动时报一声:2026-09-24 用户把速度改成 200,看上去和 100 一样快。
    """
    return [(a.name, a.speed_mms) for a in demo.actions if a.speed_mms > arm.max_linear_mms]


def load_demo() -> DemoConfig:
    return load_demo_config(CONFIG_DIR / "demo.toml")

"""连真机这一侧:启动开关、驱动连接、上电使能检查、掉线判断。

**默认不连真机。**要连必须两样都有:启动开关(`--real`),以及明确指定是哪一台的
机械臂配置(`--arm`,不能靠默认值)——`choose_arm` 就干这一件事。加了开关还要这份
配置本身允许连(`check_real_allowed`):驱动得是 RoboDK 的、控制柜 IP 得填了、
上电使能得查得出来。

已定的启动顺序(`docs/grill-decisions.md` 第二节,第四十、四十一轮改过)是:

    打开工作站 → 按配置写入 → 可达性检查(这时还没连真机)
      → 要连真机才经驱动连接 → 检查上电使能 → 回待命 → 空闲

可达性检查在**连真机之前**,不通过就不碰真机;上电使能**只检查、不执行**
(用户故事 27),没有就提示人去示教器上做,并停在这一步,不下发任何运动。
程序不管倍率,也不改控制柜的负载、碰撞等级和 0 号坐标系、工具 0(用户故事 28 / 29)。
速度只在机械臂配置写了 `program_sets_speed = true` 时才发(经包过的驱动,实际再乘示教器
倍率);没写就不发,快慢全看示教器(ADR-0011 的 2026-09-20 更新)。

**掉线判断不只看一个返回值**(见 `link_verdict`)。

这一整个模块都没在真机上跑过:上一次上 JAKA 只做完第 1 段(连通),第 2–6 段中止
(`notes/jaka-first-connection-results.md`)。所以每一处拿不准的地方都往「停下问人」
那边倒,不往「大概行吧」那边倒。
"""

from __future__ import annotations

import enum
import json
import logging
import socket
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from robodk import robolink

from robot_side.arm import ArmError
from robot_side.config import POWER_CHECK_JAKA, POWER_CHECK_NONE, ArmConfig
from robot_side.executor import SegmentTimeout
from robot_side.guard import (
    MoveLimits,
    PowerState,
    check_joint_readback,
    check_joint_travel,
    check_powered,
)

log = logging.getLogger(__name__)

JAKA_COMMAND_PORT = 10000
"""JAKA 控制柜的命令端口。RoboDK 的驱动连的也是这一个。"""

CONNECT_TIMEOUT_S = 30.0
"""等驱动连上控制柜最多等多久。实测 0.2–0.6 秒就到 `ROBOTCOM_READY`。"""

CONNECT_POLL_S = 0.2
"""等驱动连上控制柜时,多久问一次它的连接状态。"""

PROBE_TIMEOUT_S = 0.3
"""探控制柜命令端口通不通,等多久算不通。"""

POWER_READ_TIMEOUT_S = 20.0
"""读一次上电使能最多等多久(要起一个 Python 进程、登录控制柜、再登出)。

控制柜真在那儿时,登录 0.2–0.6 秒就完事。**IP 写错时会一直卡到这个时限**:
`jkrc.login()` 对着一个不存在的地址要卡 50 秒才返回(2026-09-22 实测),
所以这里不等它,到点就掐掉,按「读不出来」处理。
"""

EMBEDDED_PYTHON = Path("C:/RoboDK/Python-Embedded/python.exe")
"""RoboDK 自带的嵌入式 Python 3.10。

**不能用项目自己的 3.13**:`jkrc.pyd`(2023-08 编译)在 3.12 起 `import` 就抛
`SystemError`(`notes/jaka-first-connection-results.md`)。编译好的驱动只覆盖
v37–v312,所以查上电使能只能借 RoboDK 这一份 Python 跑。
"""

POWER_HELPER = Path(__file__).resolve().parent / "scripts" / "read_jaka_status.py"
"""那个只读小脚本。它跑在 3.10 上,所以**一行 `robot_side` 都不许 import**。"""


class RealArmRefused(Exception):
    """不许连真机,或者连不上。**调用方一律停下**,不降级、不重试。"""


# ---- 开关:默认不连真机 --------------------------------------------------


def choose_arm(arm_arg: str, *, real: bool, default: str) -> str:
    """定这一次用哪一份机械臂配置。

    不加开关时,`--arm` 空着就用默认那份(只在仿真里动)。**加了开关就必须明说**
    是哪一台:连错台比不连更糟,而默认值恰恰是最容易连错的那一种(用户故事 2)。
    """
    if arm_arg:
        return arm_arg
    if real:
        raise RealArmRefused(
            "加了连真机的开关,就必须用 --arm 明确指定是哪一台的机械臂配置;"
            "不能靠默认值——连错台比不连更糟(用户故事 2)"
        )
    return default


def check_real_allowed(config: ArmConfig) -> None:
    """这份机械臂配置允不允许连真机。不允许就说清缺哪一样。"""
    if config.driver != "robodk":
        raise RealArmRefused(
            f"{config.path.name} 的 connection.driver 是 {config.driver!r},"
            "不是 'robodk';只有经 RoboDK 驱动的那一种能连真机(ADR-0011)"
        )
    if not config.ip:
        raise RealArmRefused(
            f"{config.path.name} 里没填 connection.ip(控制柜 IP),连不了真机;"
            "填上再来,或者去掉开关只跑仿真"
        )
    if config.power_check == POWER_CHECK_NONE:
        raise RealArmRefused(
            f"{config.path.name} 的 connection.power_check 是 'none':"
            "查不出这台机械臂上电使能了没有。查不了就不连真机——"
            "没有上电使能不许下发任何运动(第 05 票第 3 条)"
        )


# ---- 掉线判断:不只看一个返回值 ------------------------------------------

class Link(enum.Enum):
    """真机这条连接看上去怎么样。三个取值,不是三个裸字符串——仓里已有 `State` 这个先例。"""

    CONNECTED = "connected"
    LOST = "lost"
    UNSURE = "unsure"


CONNECTED = Link.CONNECTED
LOST = Link.LOST
UNSURE = Link.UNSURE
"""三个取值的简写,判定逻辑和调用方都用它,省得每处都写 `Link.`。"""

LOST_CODES = frozenset(
    {
        robolink.ROBOTCOM_DISCONNECTED,
        robolink.ROBOTCOM_NOT_CONNECTED,
        robolink.ROBOTCOM_PROBLEMS,
    }
)
"""驱动明说「没连着」的那几个编码。

`ROBOTCOM_UNKNOWN` **不在里面**:`Disconnect()` 之后报的就是它(带一句
`'Sending command...'`),而它也出现在一切说不准的场合
(`notes/jaka-first-connection-results.md`)。`WORKING` / `WAITING` 同理,连接
途中就是这两个。
"""

STUBBORN_READS = 3
"""只有一路说坏时,连着几次都不见好才判掉线。"""


@dataclass(frozen=True)
class LinkRead:
    """看一眼真机这条连接。

    - `code` / `message`:RoboDK 驱动报的(`ConnectedState()`)。
    - `controller_reachable`:另一路独立信号说这台控制柜在不在。`None` 是**没探**,
      也是**默认值**——原因见 `network_probe_is_honest`。

    `message` 只进日志,**不参与判定**:实测里它恰恰是最不能信的那一样
    (`Disconnect()` 之后报 `'Sending command...'`,听着像还在忙)。
    """

    code: int
    message: str = ""
    controller_reachable: bool | None = None

    @property
    def driver_good(self) -> bool:
        return self.code == robolink.ROBOTCOM_READY

    @property
    def driver_bad(self) -> bool:
        return self.code in LOST_CODES

    @property
    def controller_bad(self) -> bool:
        return self.controller_reachable is False

    def describe(self) -> str:
        where = {
            True: "控制柜端口通",
            False: "控制柜端口不通",
            None: "控制柜端口没探",
        }[self.controller_reachable]
        return f"驱动 {self.code}({self.message or '无'}),{where}"


def link_verdict(reads: Sequence[LinkRead]) -> Link:
    """看最近这几次读,判这条连接是连着、掉了,还是说不准。

    **不凭一个返回值下「掉线」这个结论**(`notes/jaka-first-connection-results.md`:
    `Disconnect()` 之后 `ConnectedState()` 报的是 `ROBOTCOM_UNKNOWN 'Sending
    command...'`,不是 `ROBOTCOM_DISCONNECTED`)。要判掉线,得满足下面两条之一:

    - **两路独立的信号一起说坏**:驱动报的编码在 `LOST_CODES` 里,而且另一路探针也
      说这台控制柜不在。两路来源不同,一起说坏就认。
    - **同一路连着 `STUBBORN_READS` 次都说坏**。一次不算数,反复就算;中间只要有一次
      不说坏,就不算——`[说不准, 说不准, 说坏]` 判的仍然是最后那一个返回值,不认。

    **这台笔记本上只有后面那一条管用**:网络探针在这儿说假话,所以默认不探
    (`network_probe_is_honest`),每次读都只有驱动那一路。即便如此,判掉线要的仍然
    是连着几次的读数,不是某一次的返回值。

    驱动一报 `ROBOTCOM_READY` 就是连着,而且把前面的坏记录清零——中间好过一次,
    说明那几次是抖动,不是掉线。

    只有「说不准」的信号(`UNKNOWN`、`WORKING`)时,读多少次都还是说不准:它们出现在
    连接途中,也出现在断开之后,分不出来。真掉了的话,下一条指令发出去就会报错,
    那一条由 `robot_side.robodk_arm` 翻成「断开」(用户故事 31)。
    """
    if not reads:
        return UNSURE
    last = reads[-1]
    if last.driver_good:
        return CONNECTED
    if last.driver_bad and last.controller_bad:
        return LOST
    if len(reads) >= STUBBORN_READS:
        recent = reads[-STUBBORN_READS:]
        # 同一路连着说坏才算。混着「说不准」的那种不算——那等于还是凭最后一个返回值。
        if all(r.driver_bad for r in recent) or all(r.controller_bad for r in recent):
            return LOST
    return UNSURE


def tcp_probe(ip: str, port: int = JAKA_COMMAND_PORT) -> bool:
    """控制柜的命令端口通不通。绕开 RoboDK,只拿 socket 探一下,不登录、不发指令。

    **这台笔记本上它没用**,见 `network_probe_is_honest`。留着是因为换一台机器、
    或者那个中间层撤掉之后,它就是最省事的第二路信号;要用就显式传进来。
    """
    try:
        with socket.create_connection((ip, port), PROBE_TIMEOUT_S):
            return True
    except OSError:
        return False


def no_probe(ip: str) -> None:
    """不探。默认就是它——探不出真话的话,宁可说「没探」。"""
    return None


UNREACHABLE_SENTINEL_IP = "192.0.2.1"
"""RFC 5737 留给文档用的地址,网络上不应该有东西在这儿应答。拿它试探针诚不诚实。"""


def network_probe_is_honest(probe: Callable[[str], bool | None] = tcp_probe) -> bool:
    """这条网络探针说的是不是真话:对着一个不该有人应答的地址,它该说「不通」。

    **2026-09-22 在这台笔记本上实测:说了假话。** TCP 连 `192.0.2.1:10000` 30 ms
    就「连上」了,`ping 192.0.2.1` 也「0 ms 收到回复」——机器上有个中间层替所有地址
    应答。所以网络探针在这台机器上**认不出掉线**,它永远说「通」。

    一路永远说「好」的信号,比没有这一路更糟:它会让 `link_verdict` 看上去有两路
    独立信号,实际上只有一路。所以默认不探(`no_probe`),判定退回到"连着几次都
    不见好"那一条——那仍然不是只看一个返回值。

    换一台网络正常的机器时,拿这个函数试一下,真话就把 `tcp_probe` 传进去。
    """
    try:
        return probe(UNREACHABLE_SENTINEL_IP) is False
    except Exception:
        return False


class DriverPort(Protocol):
    """RoboDK 那一侧提供的三件事。策略不在这里,在上面那几个函数里。

    由 `robot_side.robodk_arm.RoboDKArm` 实现;测试里由一个假的替上。
    """

    def connect_driver(self, ip: str, port: int) -> None:
        """让 RoboDK 的驱动去连控制柜。**发出即返回**,连没连上要另外问。"""

    def driver_state(self) -> tuple[int, str]:
        """驱动现在报的连接状态:`ConnectedState()` 的编码和原文。"""

    def disconnect_driver(self) -> None:
        """断开驱动这一侧。断之前先停住当前运动;不回待命、不下使能。"""

    def raise_fault(self, error: ArmError) -> None:
        """从外面挂一个报错上去,`Arm.error()` 随后就读得到。

        盯梢判定掉线之后要用的就是它:挂上「断开」,状态逻辑下一轮就进已停止
        (用户故事 31)。列在这里是因为 `LinkMonitor` 的调用方一定要有这一条——
        机械臂接口(`robot_side.arm.Arm`)上没有它,空实现也没有。
        """


Reachable = Callable[[str], bool | None]


def connect_real(
    port: DriverPort,
    ip: str,
    controller_port: int = JAKA_COMMAND_PORT,
    *,
    timeout_s: float = CONNECT_TIMEOUT_S,
    poll_s: float = CONNECT_POLL_S,
    reachable: Reachable = no_probe,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """经驱动连上控制柜,连上才返回;连不上就把驱动这一侧收干净再抛。

    判「连上了」只认驱动报 `ROBOTCOM_READY`;判「掉了」走 `link_verdict` 那两路。
    判掉线就立刻停下,不等满超时——等下去也只是等一条已经没了的连接。

    **这里一步都不动机械臂**:只连接、只读状态。
    """
    port.connect_driver(ip, controller_port)
    reads: list[LinkRead] = []
    deadline = now() + timeout_s
    while True:
        reads.append(_read_link(port, ip, reachable))
        verdict = link_verdict(reads)
        if verdict == CONNECTED:
            log.info("驱动已连上控制柜,用了 %d 次轮询", len(reads))
            return
        if verdict == LOST:
            _give_up(port, f"驱动报掉线:{reads[-1].describe()}")
        if now() >= deadline:
            _give_up(
                port,
                f"{timeout_s:g} 秒内连不上控制柜,最后一次看到的是 "
                f"{reads[-1].describe()}",
            )
        sleep(poll_s)


def _give_up(port: DriverPort, why: str) -> None:
    """连不上就收干净再抛:驱动那一侧不能留着一个连了一半的连接。"""
    try:
        port.disconnect_driver()
    except Exception as exc:  # 收摊时再报错也没什么可做的,记一条就是
        log.warning("连不上之后断开驱动又报错了:%s", exc)
    raise RealArmRefused(why)


def _read_link(port: DriverPort, ip: str, reachable: Reachable) -> LinkRead:
    """读一次两路信号。读状态本身抛错,也算驱动这一路说坏。"""
    try:
        code, message = port.driver_state()
    except Exception as exc:
        code, message = robolink.ROBOTCOM_PROBLEMS, f"{type(exc).__name__}: {exc}"
    try:
        alive: bool | None = reachable(ip)
    except Exception as exc:
        log.warning("探控制柜端口时出错:%s", exc)
        alive = None
    return LinkRead(code=code, message=message, controller_reachable=alive)


class LinkMonitor:
    """盯着真机这条连接。判定掉线就报一次,不重复报。

    盯梢自己**什么也不做**——只把「掉了」这件事交给 `on_lost`。谁先发现要停,谁就
    直接叫停(ADR-0012),所以调用方一般把 `on_lost` 接成「挂上断开这个报错 + 叫停」。
    """

    def __init__(
        self,
        port: DriverPort,
        ip: str,
        *,
        on_lost: Callable[[str], None],
        reachable: Reachable = no_probe,
        keep: int = STUBBORN_READS,
    ) -> None:
        self._port = port
        self._ip = ip
        self._on_lost = on_lost
        self._reachable = reachable
        self._keep = max(keep, STUBBORN_READS)
        self._reads: list[LinkRead] = []
        self._reported = False

    def poll(self) -> Link:
        """看一眼。返回这一次的判定。"""
        self._reads.append(_read_link(self._port, self._ip, self._reachable))
        del self._reads[: -self._keep]
        verdict = link_verdict(self._reads)
        if verdict == LOST and not self._reported:
            self._reported = True
            why = f"真机这条连接掉了:{self._reads[-1].describe()}"
            log.error("%s", why)
            self._on_lost(why)
        return verdict


# ---- 上电使能:只检查、不执行 --------------------------------------------

Runner = Callable[[Sequence[str]], tuple[int, str, str]]
"""跑一次那个只读小脚本,返回(退出码, stdout, stderr)。测试里替成一个假的。"""


def _run_helper(command: Sequence[str]) -> tuple[int, str, str]:
    done = subprocess.run(  # noqa: S603 — 跑的是本仓库里那一份小脚本,不拼字符串
        list(command),
        capture_output=True,
        text=True,
        timeout=POWER_READ_TIMEOUT_S,
    )
    return done.returncode, done.stdout, done.stderr


def read_power_state(ip: str, *, runner: Runner = _run_helper) -> PowerState:
    """读控制柜上电、使能了没有。**只读,不上电、不使能**(用户故事 27)。

    经 RoboDK 自带的嵌入式 Python 3.10 跑 `scripts/read_jaka_status.py`:`jkrc.pyd`
    在项目用的 3.13 里 import 就报错,而 RoboDK 的 JAKA 驱动本身又没有这两个状态位
    (`notes/jaka-first-connection-results.md`),所以只能这么读。

    **读不出来一律返回「读不出来」(`None`),绝不猜成「有」。** 判不出上电使能就不许
    下发运动(`robot_side.guard.check_powered`),宁可停下问人。
    """
    command = [str(EMBEDDED_PYTHON), str(POWER_HELPER), ip]
    try:
        code, out, err = runner(command)
    except subprocess.TimeoutExpired:
        return PowerState(
            None,
            None,
            detail=(
                f"{POWER_READ_TIMEOUT_S:g} 秒内没读到控制柜状态;"
                "多半是 connection.ip 写错了或者网线不通"
            ),
        )
    except Exception as exc:
        return PowerState(None, None, detail=f"{type(exc).__name__}: {exc}")
    if code != 0:
        return PowerState(None, None, detail=(err or out or f"退出码 {code}").strip())
    try:
        data = json.loads(out.strip().splitlines()[-1] if out.strip() else "")
    except (ValueError, IndexError):
        return PowerState(
            None, None, detail=f"小脚本吐的不是 JSON:{out.strip()[:200]!r}"
        )
    if not data.get("ok"):
        return PowerState(None, None, detail=str(data.get("error", "小脚本说没读成")))
    powered = _bit(data.get("powered"))
    enabled = _bit(data.get("enabled"))
    if powered is None or enabled is None:
        return PowerState(
            powered, enabled, detail=f"状态位读成了 {data.get('status_head')!r}"
        )
    return PowerState(powered, enabled)


def _bit(value: object) -> bool | None:
    """状态位只认 0 和 1。别的一律当读不出来,不当成「大概是真的」。"""
    if value in (0, False):
        return False
    if value in (1, True):
        return True
    return None


WARMUP_TIMEOUT_S = 90.0
"""热身那一条指令最多等多久。实测约 21–25 秒,给三倍余量。"""

REAL_SEGMENT_TIMEOUT_S = 90.0
"""连着真机时,一段运动等多久算不对劲。

**不按仿真速度估。**`robot_side.executor.segment_timeout_s` 那条「估算用时 × 2 + 5 秒」
是拿演示配置里的速度算的,而那些速度只对仿真成立;真机走多快由示教器上设的值决定,
程序既不设也读不到(ADR-0011)。2026-09-22 实测:示教器 10% 时关节约 12–14°/s,配置里
写的是 10°/s,本来差不多——但**连上之后第一条指令要等约 21 秒才执行**(见 `warm_up`),
那条估算式子当场就崩(第一次上真机就是这么超时的)。

时限本来也不是安全机制,它只是为了「别永远卡住」(用户故事 19);真正兜底的是空格、
急停和那 2.0 m。所以真机上给一个宽的固定值,比拿一个假的估算装准要诚实。

**开了 `program_sets_speed` 时不够用**(`real_segment_timeout`):程序按演示配置里的速度
发,从远处回待命那一段光程序速度就要一分多钟,再乘示教器倍率就是好几分钟。
"""

MIN_OVERRIDE = 0.10
"""开了 `program_sets_speed` 时,按示教器速度倍率**至少** 10% 估每段要走多久。

倍率程序读不到。调得比这还低,长的那几段(从远处回待命)会按超时报出来、进已停止。
"""


def real_segment_timeout(config: ArmConfig) -> float | SegmentTimeout:
    """连着真机时每段等多久。

    没开 `program_sets_speed`:固定 `REAL_SEGMENT_TIMEOUT_S`——程序不设速度,估不出来。
    开了:按这一段的估算用时 ÷ `MIN_OVERRIDE` 再 × 2 + 5 秒,不少于 `REAL_SEGMENT_TIMEOUT_S`
    (那 21 秒的首条延迟由热身付掉了,留着这个下限是再保一手)。
    """
    if config.program_sets_speed:
        return SegmentTimeout(slowdown=1.0 / MIN_OVERRIDE, floor_s=REAL_SEGMENT_TIMEOUT_S)
    return REAL_SEGMENT_TIMEOUT_S

WARMUP_LIMITS = MoveLimits(axes=(), max_delta_deg=0.0)
"""热身那一条指令**一轴都不准动**。

发的是「回到你现在所在的关节角」,所以本来就是零行程;护栏写成「没有任何一轴准动」,
万一读回来的目标不是当前位置,它当场拒发,而不是让机械臂真动起来。
"""


def executed_rpy(rpy: Sequence[float]) -> tuple[float, float, float]:
    """下发这个姿态,驱动**实际会执行成**什么。`(RX, RY, RZ)` → `(RX, −RY, −RZ)`。

    2026-09-22 真机实测(`notes/jaka-second-connection-results.md`):经 RoboDK 的
    JAKA 驱动下发直线运动时,位置是对的,姿态却被按 `Rx(180)` 共轭了一次——等价于
    RY 和 RZ 各取一次反。两次独立实验残差 ≤0.7°(就是模型和真机那 1.9 mm 的 DH 差)。

    角是 **RoboDK `pose_2_xyzrpw` 的 ZYX 顺序**,也就是控制柜自己报的那一套
    (实测两边对得上)。**不是**项目 `motion.orientation_pose` 的 XYZ 顺序。

    拿它干什么:真机脚本下发之前,把**下发的**和**驱动会翻出来的**两种都过一遍姿态
    护栏(`guard.check_orientation_change`)。这样即使镜像还在,也不会因此转出格。
    `RY = 0、RZ = 0` 是它的不动点,Home 就定在那儿(`config/arms/jaka.toml`)。

    **这条规律只在 RX≈180(指针朝下)的两个位姿上验过**,换位形没验。所以它只用来
    **多卡一道**,不用来「预测然后补偿」——补偿一个没吃透的 bug 比绕开它危险。
    """
    rx, ry, rz = (float(v) for v in rpy)
    return (rx, -ry, -rz)


class Warmable(Protocol):
    """热身要用到的四件事。由 `robot_side.robodk_arm.RoboDKArm` 实现。"""

    def joints(self) -> list[float]: ...
    def stage_joint_target(self, joints: Sequence[float]) -> list[float]: ...
    def move_joints_staged(self) -> None: ...
    def is_moving(self) -> bool: ...


def warm_up(
    arm: Warmable,
    *,
    log: logging.Logger = log,
    timeout_s: float = WARMUP_TIMEOUT_S,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """发一条**零行程**的关节运动,把 jkrc 登录留下的那笔延迟先付掉。返回用了多久。

    **2026-09-22 真机实测:我们自己那次 `jkrc.login()`(查上电使能的小脚本)会把
    RoboDK 驱动的会话搅了,之后第一条运动指令要等 20–24 秒机械臂才真的动。**
    A/B/C 对照实验,同一条连接、同样的 J6 ±5°,唯一差别是 B 段前面插了一次查使能:

        A  前面没有 jkrc 登录   →  0.63 s
        B  前面刚查过使能       → 24.20 s
        C  前面没有 jkrc 登录   →  0.04 s
        D  前面没有 jkrc 登录   →  0.09 s

    所以这笔开销**不是**「连上之后第一条」的固定成本(我一开始是这么以为的,错了),
    而是**每一次 jkrc 登录都会重新欠上一次**。规矩因此是:
    **查上电使能和热身必须成对**(见 `check_power_then_warm`)。

    不先付掉的话有两个后果,第二个是危险的那个:

    1. 下一个动作会莫名其妙迟到二十秒——人以为没反应,走开了它才动;
    2. **`Busy()` 在第一条指令上会在机械臂还没动之前就落下**(实测 21.00 s 落下、
       21.10 s 才开始动)。执行器「发一段 → 等 `Busy()` 不动 → 发下一段」的循环因此
       会在上一段还没执行时就把下一段发出去——实测里正是这样撞上的。

    热身发的是「回到你现在所在的关节角」,零行程;护栏(`WARMUP_LIMITS`)写成**一轴都
    不准动**,所以万一读回来的目标不对,它当场拒发,不会让机械臂动起来。

    **2026-09-22 在真机上验过了**:日志里「热身用了 20.1 秒」,紧接着的那一段只用了
    0.63 秒。
    """
    here = arm.joints()
    readback = arm.stage_joint_target(here)
    check_joint_readback(here, readback)
    check_joint_travel(here, readback, WARMUP_LIMITS)  # 一轴都不准动
    log.info("热身:发一条零行程的关节运动,把首条指令的延迟先付掉(实测约 21 秒)……")
    started = now()
    arm.move_joints_staged()
    deadline = started + timeout_s
    while arm.is_moving():
        if now() > deadline:
            raise RealArmRefused(
                f"热身那一条指令 {timeout_s:g} 秒还没结束;驱动或控制柜不对劲,不往下走"
            )
        sleep(0.02)
    spent = now() - started
    log.info("热身用了 %.1f 秒。后面每一段就不用再等这一笔了。", spent)
    return spent


def check_power_then_warm(
    arm: Warmable,
    config: ArmConfig,
    *,
    log: logging.Logger = log,
    read_power: Callable[[], PowerState] | None = None,
) -> PowerState:
    """查一次上电使能,**然后立刻热身**。两件事必须成对,别拆开。

    为什么成对:查上电使能要用 jkrc 登录控制柜(`read_power_state`),而那一下会把
    RoboDK 驱动的会话搅了——**登录之后的第一条运动指令要等 20–24 秒才执行**
    (2026-09-22 A/B/C 对照实测,见 `warm_up`)。热身发的是一条零行程的关节运动,
    专门用来把这笔延迟吃掉;吃掉之后,真正要发的那一段就只要零点几秒。

    **所以:凡是查了上电使能,后面就不能直接发真运动。** 想「临发之前再查一次」的话,
    要么用这一个(查完顺带热身),要么干脆别在那个位置查。第 05 票里两处原来就是直接
    查完就发,都改成用这一个了。

    没上电使能时抛 `robot_side.guard.GuardRefused`,**热身那一条也不会发出去**。
    """
    state = (read_power or power_reader(config))()
    log.info("上电使能:%s", state.describe())
    check_powered(state)
    warm_up(arm, log=log)
    return state


LINK_POLL_S = 1.0
"""主循环多久看一眼真机这条连接。

**只在机械臂停着的时候看**(空闲、冷却、已停止):那正是「下一条指令要发出去了」的
时刻,也是掉线最该被发现的时刻。运动中不看——那一下要占着叫停也要用的那条连接的
锁(非阻塞 `MoveL` 自己就占 63–146 ms),没必要为一件运动中反正会被下一条指令撞出来
的事去抢锁。
"""


def bring_up_real(
    port: DriverPort,
    config: ArmConfig,
    *,
    log: logging.Logger = log,
    read_power: Callable[[], PowerState] | None = None,
    warm: bool = True,
) -> None:
    """按已定顺序把真机接上:**经驱动连接 → 检查上电使能 → 热身**。都过了才返回。

    调用方负责在这之前已经做完「打开工作站 → 按配置写入 → 可达性检查」
    (`robot_side.startup.prepare`),而且检查**通过了**——不通过就不该走到这里
    (第 05 票第 2 条)。

    上电使能**只检查、不执行**:没有就抛 `robot_side.guard.GuardRefused`,由调用方
    把提示打出来并停在这一步,**不下发任何运动**(用户故事 27、第 05 票第 3 条)。

    最后那一步**发一条零行程的关节运动**(`warm_up`),把「连上之后第一条指令」那约
    21 秒的延迟先付掉——不付的话第一个动作会迟到二十秒,而且 `Busy()` 会在机械臂还没
    动之前就落下。它一轴都不准动,护栏保证。只有这一步会下发指令,别的都不动机械臂;
    这里不设速度、不改控制柜的任何现场设置(用户故事 26、28、29)。
    """
    check_real_allowed(config)
    ip = config.ip or ""
    log.warning("连真机:%s,控制柜 %s。人和笔记本站开,急停拿在手里。", config.name, ip)
    connect_real(port, ip)
    log.info("驱动已连上控制柜,还没发过任何运动。")

    if warm:
        check_power_then_warm(port, config, log=log, read_power=read_power)
    else:
        state = (read_power or power_reader(config))()
        log.info("上电使能:%s", state.describe())
        check_powered(state)
    if config.program_sets_speed:
        log.info(
            "上电使能都有,可以往下走。速度按演示配置里的发(经包过的驱动),"
            "实际再乘示教器上的速度倍率。"
        )
    else:
        log.info("上电使能都有,可以往下走。程序不设速度——慢不慢看示教器上调的值。")


def power_reader(config: ArmConfig) -> Callable[[], PowerState]:
    """按机械臂配置挑一条查上电使能的路。查不了的配置早就被挡在外面了。"""
    if config.power_check == POWER_CHECK_JAKA:
        ip = config.ip or ""
        return lambda: read_power_state(ip)
    raise RealArmRefused(
        f"{config.path.name} 的 connection.power_check 是 {config.power_check!r},"
        "查不了上电使能"
    )

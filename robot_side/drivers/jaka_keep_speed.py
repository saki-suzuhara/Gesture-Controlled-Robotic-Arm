"""RoboDK 的 JAKA 驱动,外面包一层:让程序设的速度真的传到控制柜。

**由 RoboDK 的嵌入式 Python 3.10 跑**(`C:/RoboDK/Python-Embedded/python.exe`),不 import
项目里的任何东西。RoboDK 按机械臂的驱动路径(`PathDriver`)启动它,程序在连真机之前
把路径指到这个文件(`robot_side.robodk_arm.RoboDKArm.connect_driver`)。

**为什么要包**——反编译 `C:/RoboDK/api/Robot/v310/JakaDriver.pyc`
(RoboDK Driver for JAKAZu v1.1.2)看到的:驱动每收到一行指令,`RunCommand` 开头先把
速度重置成写死的值:

    350  ROBOT_JOINT_SPEED = math.radians(100)
    351  ROBOT_SPEED = 100
    352  ROBOT_JOINT_ACCEL = math.radians(30)
    353  ROBOT_ACCEL = 30

`SPEED` 那一行确实把程序设的速度存进这四个全局变量,可下一行 `MOVJ` / `MOVL` 一进来
又被重置回去。所以 9-20 试过的「经 RoboDK 设速度」全部无效,实际速度一直是
**100 × 控制柜的速度倍率**:倍率 13% 时关节约 13 °/s、直线约 13 mm/s,和 9-22 实测的
12–14 °/s、约 12 mm/s 对得上。关节 13 °/s 落到伸出去一米的法兰上是每秒一两百毫米,
这就是回待命、点头看上去比左右上下快得多的原因。

**包法**:只把那四行(14 条指令)换成空操作,别的一个字节不动;再给这四个全局变量
一个初值,就是原来写死的那几个数——程序不设速度时,和原驱动一模一样。另外两条护栏:

- **只准比原来慢**:程序发来的速度超过原来写死的值,就按原来的值。包过的驱动不可能
  比原驱动快。
- **不是正数的不收**(RoboDK 用 −1 表示「不改」),留着上一次的。

要换掉的那 14 条指令先逐条核对,对不上(驱动换了版本)就**不启动**:RoboDK 那边会
连不上。宁可连不上,也不拿没核对过的字节码跑真机。

每收到一行指令在 `data/logs/jaka-driver-<日期>.log` 里记一笔,连同那一刻生效的速度——
程序设的速度到底有没有传到,事后看这份就知道。
"""

from __future__ import annotations

import dis
import importlib.machinery
import importlib.util
import math
import sys
import types
from datetime import datetime
from pathlib import Path

STOCK = {
    "ROBOT_SPEED": 100.0,
    "ROBOT_JOINT_SPEED": math.radians(100.0),
    "ROBOT_ACCEL": 30.0,
    "ROBOT_JOINT_ACCEL": math.radians(30.0),
}
"""原驱动每行指令都重置成的值。也是初值,也是上限。关节的两个是弧度(驱动里就这么存)。"""

RESET = (
    ("LOAD_GLOBAL", "math"),
    ("LOAD_METHOD", "radians"),
    ("LOAD_CONST", 100),
    ("CALL_METHOD", 1),
    ("STORE_GLOBAL", "ROBOT_JOINT_SPEED"),
    ("LOAD_CONST", 100),
    ("STORE_GLOBAL", "ROBOT_SPEED"),
    ("LOAD_GLOBAL", "math"),
    ("LOAD_METHOD", "radians"),
    ("LOAD_CONST", 30),
    ("CALL_METHOD", 1),
    ("STORE_GLOBAL", "ROBOT_JOINT_ACCEL"),
    ("LOAD_CONST", 30),
    ("STORE_GLOBAL", "ROBOT_ACCEL"),
)
"""`RunCommand` 开头那四行重置速度的指令,逐条(操作名,参数)。换掉之前先核对。"""

LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"
"""和程序的运行日志放一起(`robot_side.runlog`)。`data/` 整个被 git 忽略。"""


class DriverMismatch(Exception):
    """原驱动的字节码和核对过的那一版对不上。不启动。"""


def load_driver() -> types.ModuleType:
    """按文件路径载入编译好的原驱动(`api/Robot/v310/JakaDriver.pyc`)。

    **不能照 RoboDK 自己那个加载器写 `from v310.JakaDriver import *`**:RoboDK 启动驱动时
    把 `C:/RoboDK/Posts` 放进了 PYTHONPATH,那里也有一个 `v310` 包(后处理器的),排在
    前面就先被找到,于是「没有 v310.JakaDriver」(2026-09-23 实测)。它自己的加载器没事,
    是因为它就放在 `api/Robot` 里,脚本所在目录排第一。
    """
    api = Path(sys.executable).resolve().parent.parent / "api" / "Robot"
    if not api.is_dir():
        api = Path("C:/RoboDK/api/Robot")
    version = f"v{sys.version_info[0]}{sys.version_info[1]}"
    path = api / version / "JakaDriver.pyc"
    loader = importlib.machinery.SourcelessFileLoader("JakaDriver", str(path))
    spec = importlib.util.spec_from_loader("JakaDriver", loader, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["JakaDriver"] = module
    loader.exec_module(module)
    return module


def without_reset(code: types.CodeType) -> types.CodeType:
    """把开头那 14 条重置速度的指令换成空操作,别的一个字节不动。对不上就抛。"""
    head = list(dis.get_instructions(code))[: len(RESET)]
    got = tuple((i.opname, i.argval) for i in head)
    if got != RESET or any(i.offset != 2 * n for n, i in enumerate(head)):
        raise DriverMismatch(
            "原驱动 RunCommand 开头不是核对过的那四行重置速度的指令,不启动。"
            f"读到的是 {got}"
        )
    nop = bytes((dis.opmap["NOP"], 0))
    size = 2 * len(RESET)
    return code.replace(co_code=nop * len(RESET) + code.co_code[size:])


def install(base: types.ModuleType, *, log_dir: Path | None = LOG_DIR) -> types.FunctionType:
    """换上不重置速度的 `RunCommand`,返回原来那个(测试拿它对照)。"""
    original = base.RunCommand
    patched = types.FunctionType(
        without_reset(original.__code__),
        base.__dict__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    for name, value in STOCK.items():
        setattr(base, name, value)
    log = _Log(log_dir)
    log.write(f"启动:{base.DRIVER_VERSION},速度初值同原驱动 {_speeds(base)}")

    def run_command(linecmd: str) -> None:
        before = {name: getattr(base, name) for name in STOCK}
        command = linecmd.strip()
        verb = command.split(" ", 1)[0]
        # 执行之前就记:CONNECT、运动这些一执行就可能卡住很久,甚至被 RoboDK 直接收掉
        if verb in ("MOVJ", "MOVL", "MOVC"):
            log.write(f"{verb} 按 {_speeds(base)}")
        elif verb not in ("SPEED", "CJNT"):  # CJNT 是读关节角,来得太勤,不记
            log.write(f"指令 {command[:60]!r}")
        patched(linecmd)
        for name, stock in STOCK.items():
            value = getattr(base, name, None)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                setattr(base, name, before[name])  # −1 是「不改」:留着上一次的
            elif value > stock:
                setattr(base, name, stock)  # 只准比原驱动慢
        if verb == "SPEED":
            log.write(f"SPEED 收到 {command[5:].strip()!r},现在 {_speeds(base)}")

    base.RunCommand = run_command  # `RunDriver` 按模块里的这个名字找
    return original


def _speeds(base: types.ModuleType) -> str:
    return (
        f"直线 {base.ROBOT_SPEED:g} mm/s、关节 {math.degrees(base.ROBOT_JOINT_SPEED):g} °/s"
        "(再乘控制柜倍率)"
    )


class _Log:
    """往文件里记一笔。写不了就算了:记日志不能把驱动带倒。"""

    def __init__(self, log_dir: Path | None) -> None:
        self._path = None
        if log_dir is None:
            return
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._path = log_dir / f"jaka-driver-{datetime.now():%Y%m%d}.log"
        except OSError:
            self._path = None

    def write(self, text: str) -> None:
        if self._path is None:
            return
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%H:%M:%S.%f}"[:-3] + f" {text}\n")
        except OSError:
            pass


def main() -> None:
    base = load_driver()
    install(base)
    base.RunMain()


if __name__ == "__main__":
    main()

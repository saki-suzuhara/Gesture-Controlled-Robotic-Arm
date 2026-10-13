"""机械臂接口的 RoboDK 实现(SIM)。

仿真和真机共用这一份(ADR-0011);第 01 票只做仿真,连真机的开关是第 05 票。

连接时做四件事:RoboDK 没开就自己启动它;打开工作站文件(已经开着就用开着的那个);
仿真速度设成 1 倍——RoboDK 默认是 5 倍,不设的话仿真里的耗时和配置里写的速度对不上;
关掉碰撞检查——可达性检查本来就不查会不会碰到东西(`CONTEXT.md`「可达性检查」)。

**不动用户原有的东西**:原来开着的工作站不关、不改;原来的仿真速度、原来的碰撞检查
开关和原来激活的工作站在断开时恢复;只有自己打开的那个工作站才由自己关掉。

碰撞检查的开关是**整个 RoboDK 一个**,不是每个工作站一个(2026-09-20 实测:新建工作站
里读到的是同一个值),所以更要在退出时恢复。

**RoboDK 中途被关掉**(用户故事 31)走的是下面的 `_fail`:API 抛什么这里都接住,翻成
机械臂一侧的报错挂起来,上层照常进已停止。翻成「断开」的那一种之后**不再下发任何东西**
——`robolink` 会自己去重连,连上的可能是一个新开的、什么都没载入的 RoboDK,往那里发指令
不如不发。要恢复就重启程序。

只有别人来调时才会发现断开:程序空闲、一条指令都不发的时候,这里不主动去 ping
(那是每轮一次 API 来回)。机械臂那会儿本来就停着,下一次要动它时立刻就会发现。
"""

from __future__ import annotations

import logging
import math
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from robodk import robolink
from robodk.robomath import Mat, eye, invH, pose_angle_between

from robot_side.arm import (
    Arm,
    ArmBusyError,
    ArmError,
    MotionError,
    NotConnectedError,
    joint_drift,
)
from robot_side.config import ArmConfig
from robot_side.motion import Segment, joint_path_mm, worktable_pose
from robot_side.reach import ReachReport, check_reach

T = TypeVar("T")

log = logging.getLogger(__name__)

ROBODK_API_PORT = 20500

ROBODK_EXE = Path("C:/RoboDK/bin/RoboDK.exe")
"""要启动哪一个 RoboDK。明写出来,不靠注册表去找。

`robolink` 默认是读注册表 `HKLM\\SOFTWARE\\RoboDK\\INSTDIR`;本机没有这个键,所以它会打印
「RoboDK was not installed properly」再退回这个路径。明写以后那句吓人的话也就不出现了,
而且万一以后装了别的版本,也不会被它抢走。
"""

API_SETTLE_S = 3.0
"""自己启动 RoboDK 之后,等它的 API 服务稳下来,再换一条连接干活。

2026-09-20 实测:由 API 启动的那一下,最早那条连接会被掐掉——`Version()` 还答得上,
紧接着的 `AddFile` 就报 `ConnectionAbortedError`(WinError 10053),工作站也就打不开。
RoboDK 本来就开着时不用等。
"""

ROBODK_VERSION = "5.9"
"""只认这一版(2026-09-20 用户定:要 5.9,不要 6.x)。

所有路点、待命位姿和实测数都是在 5.9.0.25042 上得出的;6.x 还要另一份许可证。
连上的不是这一版就停下,不往下走——宁可不动,也不拿没验过的版本跑机械臂。
"""

SIMULATION_SPEED = 1.0
"""仿真速度设成 1 倍:RoboDK 默认是 5 倍,不设的话仿真里的耗时和配置里的速度对不上。"""

TOOL_TOL_MM = 0.01
"""核对工具 0 时容许的偏差。"""

IK_POS_TOL_MM = 0.5
IK_ANG_TOL_DEG = 0.1
IK_ITERS = 10
"""逆解用正解核对时的容差和迭代次数。

JAKA 的逆解是数值迭代的,一次调用可能差几 mm 还照样返回,所以不能把「返回不为空」
当作够得着:每次都用正解核对,不过就把结果当初值再求一次。
"""

STILL_TOL_DEG = 1e-3
"""可达性检查前后核对「机械臂没动过」时容许的偏差。"""

WORKTABLE_FRAME_PREFIX = "Worktable "
"""工作台坐标系在工作站里的名字前缀,后面接机械臂的名字。"""

STAGED_TARGET_NAME = "Guarded target"
STAGED_JOINT_TARGET_NAME = "Guarded joint target"
"""护栏用的两个目标在工作站里叫什么。放进去、读回来、再发的就是它们。

分两个:一个存位姿(直线运动用),一个存关节角(关节运动用)。合成一个的话,上一次
存进去的那一种会被下一次盖掉,而「读回来核对」核对的就正好是这个目标。"""

#: 工作站文件里两台机械臂的名字(由建站脚本从模型文件载入后得到)
JAKA = "JAKA Zu20"
NOVA5 = "Dobot Nova 5"

KEEP_SPEED_DRIVER = Path(__file__).resolve().parent / "drivers" / "jaka_keep_speed.py"
"""包过的 JAKA 驱动:程序设的速度真的传到控制柜(那个文件开头的说明)。

机械臂配置里 `program_sets_speed = true` 时,连真机之前把 RoboDK 里这台机械臂的驱动路径
(`PathDriver`)指到它;断开时改回原来的。只改 RoboDK 内存里的工作站,不存盘。
"""

MIN_PATH_MM = 1.0
"""关节运动时指针尖端走不到这么远,就不按尖端速度换算:几乎原地转,换算出来的数没意义。"""


def _read_collision_state(rdk: robolink.Robolink) -> int | None:
    """读整个 RoboDK 的碰撞检查开关(1 开、0 关);读不出来返回 None。

    读不出来就**不去动这个开关**:恢复不了原设置的话,宁可不关——关掉再瞎猜着恢复,
    等于替用户改了设置。不关的后果只是可能查出碰撞,而那一条本来就有兜底:记日志、
    提示,不当作够不着(第 02 票第 9 条)。
    """
    try:
        return int(str(rdk.Command("CollisionActive")).strip())
    except (ValueError, TypeError):
        return None


def _open_api(*, started_robodk: bool) -> robolink.Robolink:
    """连上 RoboDK 的 API;是这一下把它启起来的,就等它稳下来再干活。

    明写 RoboDK 的路径,不靠注册表找(见 `ROBODK_EXE`)。

    `close_std_out=True` **不是为了少打几行字**:不设它的话,`robolink` 会开一个**非守护**
    线程去读 RoboDK 的 stdout,那个线程要等 RoboDK 退出才结束,于是活儿干完了我们的进程
    也退不出去(2026-09-20 实测:一直挂到被杀,关掉 API 连接也没用)。设了它就直接把管子
    关掉、不开那个线程;RoboDK 留在后台,下一次跑省掉十几秒的启动。
    """
    path = str(ROBODK_EXE) if ROBODK_EXE.is_file() else None
    rdk = robolink.Robolink(robodk_path=path, close_std_out=True)
    if started_robodk:
        time.sleep(API_SETTLE_S)
    return rdk


LOST_MARKS = (
    "Unable to connect",
    "Communication problems",
    "Problems running function",
)
"""`robolink` 报成一般 `Exception` 的那几句,其实是「这条连接没了」。

`robolink` 在每次调用前会自己试着重连,连不上才抛这几句里的一句;所以见到它们就是
RoboDK 那边已经没了,而不是这一条指令有问题。
"""


def robodk_is_running(timeout_s: float = 0.3) -> bool:
    """RoboDK 现在开着吗(API 端口通不通)。只为把「是不是自己启动的」记准。"""
    try:
        with socket.create_connection(("localhost", ROBODK_API_PORT), timeout_s):
            return True
    except OSError:
        return False


DRIVER_MARKERS = ("JakaDriver", "jaka_keep_speed")
"""机械臂驱动进程的命令行里认得出的字样:RoboDK 原来的 JAKA 驱动,和包过的那一份。"""


def arm_driver_processes() -> list[int] | None:
    """现在在跑的机械臂驱动进程号。**查不了返回 None**(调用方当「可能有」处理)。

    驱动在跑,就是有人经 RoboDK 连着真机(或者上一次的没退干净)。这时别的程序再连
    同一个 RoboDK、动工作站里那台机械臂,指令会顺着驱动发到控制柜:2026-09-24 用户
    连着真机做手势时,自动测试里的 `apply_config` 经驱动发了两次 `SETTOOL`,控制柜的
    当前工具被切到了 10 号。只看进程,不连 RoboDK——查这一下本身不能再碰它。
    """
    command = (
        "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
        "ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }"
    )
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    pids = []
    for line in done.stdout.splitlines():
        pid, _, cmdline = line.strip().partition(" ")
        if pid.isdigit() and any(m in cmdline for m in DRIVER_MARKERS):
            pids.append(int(pid))
    return pids


class RoboDKArm(Arm):
    """经 RoboDK 的 Python API 驱动一台机械臂。

    运动一律非阻塞发出:阻塞的 `MoveJ` / `MoveL` 在整段运动期间一直占着这条连接的
    锁,同一条连接上的 `Stop()` 就要等运动走完才发得出去。非阻塞加 `Busy()` 轮询时,
    `Stop()` 本身 0.4–4.4 ms 就返回(2026-09-19 实测),所以叫停和运动走同一条连接,
    不另开一条。

    非阻塞的 `MoveL` 本身也要 63–146 ms 才返回,这期间占着锁;正好在这一刻叫停,
    最坏多等约 0.15 秒,仍在 0.5 秒以内。
    """

    def __init__(self, station_file: Path, robot_name: str) -> None:
        self.station_file = Path(station_file)
        self.robot_name = robot_name
        self._rdk: robolink.Robolink | None = None
        self._robot: robolink.Item | None = None
        self._station: robolink.Item | None = None
        self._opened_station = False
        self._started_robodk = False
        self._speed_before: float | None = None
        self._station_before: robolink.Item | None = None
        self._collision_before: int | None = None
        self._worktable: Mat = eye(4)
        """工作台坐标系相对底座的位姿。写入配置之前是单位阵(即和底座重合)。"""
        self._fault: ArmError | None = None
        """机械臂一侧现在挂着的报错;`error()` 读的就是它。"""
        self._real_ip: str | None = None
        """连着真机时是控制柜的 IP;只连仿真时一直是 `None`(第 05 票)。"""
        self._target: robolink.Item | None = None
        """护栏用的位姿目标(`stage_target`)。只有真机脚本用得上。"""
        self._joint_target: robolink.Item | None = None
        """护栏用的关节目标(`stage_joint_target`)。同上。"""
        self._pointer: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._max_joint_degs: float | None = None
        self._max_linear_mms: float | None = None
        self._program_sets_speed = False
        """这四个从机械臂配置来(`apply_config`):尖端速度换算、速度上限、真机上发不发速度。"""
        self._driver_before: str | None = None
        """换成包过的驱动之前,这台机械臂原来的驱动路径;断开时改回去。没换过就是 None。"""
        # stop() 可能和 connect() / disconnect() 撞上,拿这个护住那几个字段
        self._lock = threading.Lock()

    # ---- 机械臂接口 ----------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._robot is not None

    def connect(self) -> None:
        if self.connected:
            return
        if not self.station_file.is_file():
            raise FileNotFoundError(
                f"没有工作站文件 {self.station_file};先跑建站脚本 "
                "(uv run python -m robot_side.scripts.build_station)"
            )
        with self._lock:
            self._fault = None  # 重新连上就是新的一轮
        self._started_robodk = not robodk_is_running()
        rdk = _open_api(started_robodk=self._started_robodk)
        with self._lock:
            self._rdk = rdk
        try:
            version = str(rdk.Version())
            if not version.startswith(ROBODK_VERSION + "."):
                raise NotConnectedError(
                    f"连上的 RoboDK 是 {version},这个项目只认 {ROBODK_VERSION}.x"
                    f"(路点和实测数都是在它上面得出的,6.x 还要另一份许可证)。"
                    f"先关掉现在开着的 RoboDK,再用 {ROBODK_EXE} 打开"
                )
            self._station = self._open_station(rdk)
            robot = rdk.Item(self.robot_name, robolink.ITEM_TYPE_ROBOT)
            if not robot.Valid():
                names = [
                    item.Name() for item in rdk.ItemList(robolink.ITEM_TYPE_ROBOT)
                ]
                raise NotConnectedError(
                    f"工作站 {self.station_file.name} 里没有叫 {self.robot_name!r} 的"
                    f"机械臂;有的是 {names}"
                )
            # ADR-0013:路点按法兰量,工具保持 0。这里只核对、不写——写入配置是
            # 第 02 票,而且用户故事 29 要求程序不去动工具 0。
            tool = robot.PoseTool().Pos()
            if max(abs(v) for v in tool) > TOOL_TOL_MM:
                raise NotConnectedError(
                    f"{self.robot_name} 在工作站里启用了工具偏移 "
                    f"{[round(v, 2) for v in tool]};ADR-0013 要求保持工具 0,"
                    "重新跑建站脚本"
                )
            self._speed_before = rdk.SimulationSpeed()
            rdk.setSimulationSpeed(SIMULATION_SPEED)
            self._collision_before = _read_collision_state(rdk)
            if self._collision_before is not None:
                rdk.setCollisionActive(robolink.COLLISION_OFF)
        except Exception:
            # 连一半失败也要清场,不然用户的仿真速度和工作站就留在改过的状态
            self.disconnect()
            raise
        with self._lock:
            self._robot = robot

    def disconnect(self) -> None:
        with self._lock:
            rdk, station, robot = self._rdk, self._station, self._robot
            self._rdk = self._station = self._robot = None
        if rdk is None:
            return
        # 清场的每一步都可能扑空:RoboDK 已经被关掉时,这些调用一个都做不成
        # (用户故事 31)。做不成就算了,程序本来就要退出或者重启。
        try:
            # 退出时先停住当前运动再断开;不自动回待命、不下使能(用户故事 32)
            if robot is not None:
                robot.Stop()
                if self._real_ip is not None:
                    # 连着真机就先把驱动这一侧断掉,再去收拾工作站:关工作站会把
                    # 这台机械臂一起带走,那时就没人断得了真机的连接了
                    robot.Disconnect()
                self._restore_driver(robot)
            if self._speed_before is not None:
                rdk.setSimulationSpeed(self._speed_before)
            if self._collision_before is not None:
                # 碰撞检查的开关是整个 RoboDK 一个,一定要恢复成用户原来的设置
                rdk.setCollisionActive(self._collision_before)
            if self._opened_station and station is not None and station.Valid():
                rdk.setActiveStation(station)
                rdk.CloseStation()
            if self._station_before is not None:
                # 只有它还开着才切回去:`Valid()` 只看指针,认不出「这个工作站已经没了」。
                # RoboDK 载入文件时会把没动过的空工作站关掉,原来激活的那个常常就是它。
                if self._station_before in rdk.getOpenStations():
                    rdk.setActiveStation(self._station_before)
        except Exception as exc:
            # 正常退出、RoboDK 已经被关掉,都会走到这里。记一条,但**不挂报错**:
            # 这一下是收摊,不是「机械臂出了事要人来看」;挂上反而让下一次
            # `error()` 说瞎话。
            log.warning("断开时清场没做完(RoboDK 可能已经关了):%s", exc)
        finally:
            self._speed_before = None
            self._collision_before = None
            self._opened_station = False
            self._station_before = None
            self._real_ip = None
            self._target = None
            self._joint_target = None
            self._driver_before = None
        # 最后把这条 API 连接关掉。自己启动的那个 RoboDK 要等它断开才退出,而 `robolink`
        # 为启动它的那条连接开的 stdout 读取线程**不是守护线程**,要等 RoboDK 退出才结束
        # ——不关的话,活儿干完了进程还退不出去(2026-09-20 实测:要等到被杀)。
        # RoboDK 本来就开着时,关这条连接不会把它关掉。
        rdk.Disconnect()

    def apply_config(self, config: ArmConfig) -> tuple[float, float, float]:
        """**只在还没连真机的时候写。**连着真机时拒绝,理由见下。

        已定的启动顺序把「按配置写入」放在「经驱动连接」**之前**,不是随手排的:这里
        会调 `setPoseFrame` 和 `setPoseTool`,而 RoboDK 的 JAKA 驱动里**有**
        `set_tool_data` / `set_tool_id` / `set_user_frame_id` 这几个调用
        (`notes/jaka-first-connection-results.md` 从驱动里提取的字符串)。连着真机时
        写,就可能把控制柜的工具 0 和 0 号坐标系改掉——用户故事 29 明说不许动它们,
        因为别人还要用这台机械臂。

        驱动到底会不会在这一下push 过去,**没有实测过**。没验过的事就按最坏的算:
        连着就不写。真要改配置,先 `disconnect_driver()`,写完再连回来。
        """
        if self._real_ip is not None:
            raise ArmError(
                "连着真机的时候不写配置:`setPoseFrame` / `setPoseTool` 可能被驱动"
                "push 到控制柜,改掉那里的工具 0 和 0 号坐标系(用户故事 29)。"
                "先 disconnect_driver(),写完再连回来"
            )
        robot = self._require_robot()
        self._worktable = worktable_pose(config.worktable)
        self._pointer = tuple(config.pointer_offset_mm)
        self._max_joint_degs = config.max_joint_degs
        self._max_linear_mms = config.max_linear_mms
        self._program_sets_speed = config.program_sets_speed
        frame = self._worktable_frame(robot)
        frame.setPose(self._worktable)  # 界面上改过的数,这里按配置盖掉
        robot.setPoseFrame(frame)
        robot.setPoseTool(eye(4))  # ADR-0013:控制器保持工具 0
        robot.setJointsHome(list(config.standby_joints_deg))
        standby = self.flange_pose(config.standby_joints_deg).Pos()
        return (float(standby[0]), float(standby[1]), float(standby[2]))

    def set_speed(
        self, *, linear_mms: float | None = None, joints_degs: float | None = None
    ) -> None:
        """设这一段的速度,不超过机械臂配置里的上限。

        **连着真机时,只有换上了包过的驱动才发**(`program_sets_speed`,`connect_driver`)。
        没换的话一次都不调,直接返回:原驱动每行指令都把速度重置成 100,程序设的到不了
        控制柜;可「到不了」只是那一版驱动的毛病,不是一条保证——哪天它修好了,照旧调下去
        就等于程序在人没同意的情况下开始替人设真机的速度(第 05 票第 4 条要挡的)。

        发出去的数先按上限截住(`max_linear_mms` / `max_joint_degs`);包过的驱动那边还有
        一道:不许比原驱动的 100 快。真机上实际再乘示教器上的速度倍率。
        """
        if self._real_ip is not None and self._driver_before is None:
            return
        if linear_mms is not None and self._max_linear_mms is not None:
            linear_mms = min(linear_mms, self._max_linear_mms)
        if joints_degs is not None and self._max_joint_degs is not None:
            joints_degs = min(joints_degs, self._max_joint_degs)
        self._require_robot().setSpeed(
            speed_linear=-1 if linear_mms is None else linear_mms,
            speed_joints=-1 if joints_degs is None else joints_degs,
        )

    def joint_speed_for(
        self, from_joints: Sequence[float], to_joints: Sequence[float], tip_mms: float
    ) -> float | None:
        """按指针尖端走多远换算关节速度(`Arm.joint_speed_for`),不超过 `max_joint_degs`。"""
        path = joint_path_mm(self.flange_pose, from_joints, to_joints, self._pointer)
        if path < MIN_PATH_MM or tip_mms <= 0:
            return None
        span = max(abs(float(a) - float(b)) for a, b in zip(from_joints, to_joints, strict=True))
        degs = span * tip_mms / path
        if self._max_joint_degs is not None:
            degs = min(degs, self._max_joint_degs)
        return degs

    def move_joints(self, joints: Sequence[float]) -> None:
        self._guard("发关节运动", lambda robot: robot.MoveJ(list(joints), blocking=False))

    def move_linear(self, pose: Mat) -> None:
        self._guard("发直线运动", lambda robot: robot.MoveL(pose, blocking=False))

    def is_moving(self) -> bool:
        return bool(self._guard("问还在不在动", lambda robot: robot.Busy()))

    def stop(self) -> None:
        # 任意线程可调、重复调没有副作用;没连接时什么也不做
        with self._lock:
            fault, robot = self._fault, self._robot
        if robot is None or (fault is not None and not fault.clearable):
            # 连接已经没了:停不了,也没什么可停的
            return
        try:
            robot.Stop()
        except Exception as exc:
            raise self._fail("叫停", exc) from exc

    def joints(self) -> list[float]:
        if self.is_moving():
            # 运动中读位置会让 RoboDK 崩进程(2026-09-20 实测)
            raise ArmBusyError("还在动,不能读关节角")
        return self._guard(
            "读关节角", lambda robot: [float(v) for v in robot.Joints().list()[:6]]
        )

    def pose(self) -> Mat:
        """读法兰当前的位姿(工作台坐标系里)。和 `joints()` 一样,**只在不运动时调**。

        **连着真机时,先读一次关节角把模型同步过来。** `robot.Pose()` 算的是
        **工作站模型**里那组关节角的正解,而模型只有在读关节角时才从驱动同步——
        不先读,拿到的就是上一段的旧位姿。

        2026-09-22 真机实测:连上之后直接读位姿得到 `(4.45, −944.82, 895.13)`,
        先读一次关节角再读得到 `(46.49, −945.63, 895.08)`,**差 42 mm**。执行器的
        「到位核对」(`executor.arrived`)就是栽在这上面:机械臂明明到了(差 2.2 mm),
        它拿着旧位姿比,永远判不到位,最后按超时报了出来。

        只在连着真机时多读这一下:只跑仿真时模型就是唯一的真相,不用同步,省一次
        API 来回。
        """
        if self.is_moving():
            raise ArmBusyError("还在动,不能读位姿")

        def read(robot: robolink.Item) -> Mat:
            if self._real_ip is not None:
                robot.Joints()  # 把模型从驱动同步过来,下面那一下才算的是真机的位姿
            return robot.Pose()

        return self._guard("读位姿", read)

    def error(self) -> ArmError | None:
        """机械臂一侧现在挂着什么报错。只读挂着的那个,不去问 RoboDK(`Arm.error`)。"""
        with self._lock:
            return self._fault

    def clear_error(self) -> None:
        """把挂着的报错清掉。

        仿真这边没有要清的东西:报错本来就是这一层翻出来的,清的也就是这一层挂着的
        那个。**断开的那一种清不掉**,原样抛出去——`robolink` 就算自己重连上了,连上的
        也可能是另一个 RoboDK,要恢复就重启程序。

        **连着真机时,这里清的仍然只是我们自己挂着的那个。** RoboDK 的 JAKA 驱动里
        **没有** `clear_error`、也没有 `collision_recover`
        (`notes/jaka-first-connection-results.md` 从驱动里提取的字符串),所以控制柜
        那边要是真报了错(碰撞保护、急停),只能由人在示教器上清、必要时重新上电使能
        ——和上电使能一样,程序只提示、不替人做(用户故事 27)。清完要是控制柜还挂着
        错,下一段运动发出去就会再失败一次,照旧进已停止。

        真机上碰撞保护和急停到底报不报得出来、报出来长什么样,**还没实测过**
        (第 05 票明说这一项另开 session 和用户一起做),所以这里不写任何「替人恢复」
        的逻辑,只把话说清楚。
        """
        with self._lock:
            fault = self._fault
            if fault is not None and not fault.clearable:
                raise fault
            self._fault = None
            real = self._real_ip is not None
        if real:
            log.warning(
                "清的只是程序这一侧挂着的报错。控制柜那边要是也报了错(碰撞、急停),"
                "驱动里没有清错误这一条,请到示教器上清掉、必要时重新上电使能。"
            )

    def reach_check(
        self, plans: Mapping[str, Sequence[Segment]], standby_joints: Sequence[float]
    ) -> ReachReport:
        """查完把仿真里的模型放回原处,并核对它确实没动。

        RoboDK 的 `MoveL_Test` 查完会把模型挪到终点(或碰撞点),所以这里查之前记下
        关节角、查完放回去。放回去用的 `set_joints` 只动仿真里的模型:按已定的启动顺序,
        可达性检查在连真机**之前**做,不通过就不碰真机。
        """
        before = self.joints()
        try:
            report = check_reach(self, plans, standby_joints)
        finally:
            self.set_joints(before)
        drift = joint_drift(before, self.joints())
        if drift > STILL_TOL_DEG:
            raise ArmError(
                f"可达性检查之后机械臂没回到原处(最大差 {drift:.4f}°),停下不动"
            )
        return report

    # ---- 可达性检查要用的运动学(reach.Kinematics)-----------------------

    def flange_pose(self, joints: Sequence[float]) -> Mat:
        """一组关节角对应的法兰位姿,在**工作台坐标系**里。不动机械臂。"""
        return invH(self._worktable) * self._require_robot().SolveFK(list(joints))

    def solve_ik(self, pose: Mat, seed: Sequence[float]) -> list[float] | None:
        """求逆解,每次都用正解核对、不过就把结果当初值再求;始终不收敛就算够不着。"""
        robot = self._require_robot()
        seed = list(seed)
        for _ in range(IK_ITERS):
            solved = robot.SolveIK(pose, seed, eye(4), self._worktable)
            if solved.size(0) * solved.size(1) == 0:
                return None
            seed = [float(v) for v in solved.list()[:6]]
            got = self.flange_pose(seed)
            dp = math.dist(got.Pos(), pose.Pos())
            da = math.degrees(pose_angle_between(got, pose))
            if dp < IK_POS_TOL_MM and da < IK_ANG_TOL_DEG:
                return seed
        return None

    def linear_test(self, from_joints: Sequence[float], pose: Mat) -> int:
        """一段直线走不走得通:0 走得通,−1 走不通,−2 够不着,大于 0 是碰撞编号。"""
        return int(self._require_robot().MoveL_Test(list(from_joints), pose))

    # ---- 机械臂接口里没有的:真机这一侧和可达性检查要用的运动学 ----------
    #
    # 真机那几条(第 05 票)只搭线,策略在 `robot_side.real` 里。

    def connect_driver(self, ip: str, port: int) -> None:
        """让 RoboDK 的驱动去连控制柜。**发出即返回**,连没连上要另外问。

        按已定的启动顺序,这一步在**按配置写入和可达性检查之后**才做:检查不通过
        就根本不碰真机。这里一步都不动机械臂——只设连接参数、只发连接。
        """
        robot = self._require_robot()
        if self._program_sets_speed:
            self._use_keep_speed_driver(robot)
        robot.setConnectionParams(ip, port, "/", "", "")
        robot.Connect(ip, blocking=False)
        self._real_ip = ip

    def _use_keep_speed_driver(self, robot: robolink.Item) -> None:
        """把这台机械臂的驱动换成包过的那个,**从 RoboDK 里读回来核对**,对不上就不连。

        读回来用的是 `setParam("PathDriver", "")`:空值就是「报现在的」(2026-09-23 实测)。
        """
        before = str(robot.setParam("PathDriver", ""))
        wanted = KEEP_SPEED_DRIVER.as_posix()
        if not KEEP_SPEED_DRIVER.is_file():
            raise NotConnectedError(f"没有包过的驱动 {wanted},不连真机")
        robot.setParam("PathDriver", wanted)
        got = str(robot.setParam("PathDriver", ""))
        if got != wanted:
            robot.setParam("PathDriver", before)
            raise NotConnectedError(
                f"驱动路径没换成:要的是 {wanted},RoboDK 里读回来是 {got!r};不连真机"
            )
        if self._driver_before is None:
            self._driver_before = before
        log.info("驱动换成包过的那个(程序设的速度会发给控制柜):%s", wanted)

    def _restore_driver(self, robot: robolink.Item) -> None:
        """驱动路径改回换之前的。没换过就什么也不做。"""
        if self._driver_before is None:
            return
        try:
            robot.setParam("PathDriver", self._driver_before)
        finally:
            self._driver_before = None

    def driver_state(self) -> tuple[int, str]:
        """驱动现在报的连接状态。

        **一个返回值说明不了问题**:`Disconnect()` 之后这里报的是
        `ROBOTCOM_UNKNOWN 'Sending command...'`。怎么判掉线见 `real.link_verdict`。
        """
        code, message = self._require_robot().ConnectedState()
        return int(code), str(message)

    def disconnect_driver(self) -> None:
        """断开驱动这一侧。**先停住当前运动再断**;不回待命、不下使能(用户故事 32)。

        **断成了才放下「连着真机」那个标记。** 顺序反过来会开一条绕过护栏的路:
        `apply_config` 和 `set_speed` 都是看这个标记决定拒不拒绝的,先放下再去断、
        断又失败,就成了「程序以为断了、其实还连着」,那两下会照发不误
        (第 05 票第 4 条要挡的正是这个)。所以断不成就原样抛出去,标记留着。
        """
        with self._lock:
            robot = self._robot
        if robot is None:
            self._real_ip = None
            return
        robot.Stop()
        robot.Disconnect()
        self._real_ip = None
        self._restore_driver(robot)

    @property
    def busy_is_reliable(self) -> bool:
        """只跑仿真时 `Busy()` 是准的,不用再核对到位;连着真机时不信它。

        仿真里核对到位反而会误报:RoboDK 给 JAKA 模型走直线用的是数值迭代的逆解,终点
        会差一点(2026-09-23 实测 0.65 mm、1.71 mm,在活动范围左上角画方块时姿态差到
        2.14°,超过到位容差 2°,整段按超时进了已停止)。那几次 `Busy()` 落下时和 1 秒后
        读到的位置一模一样——没有「早落」,只是仿真走不准。真机的直线由控制柜自己走。
        """
        return self._real_ip is None

    @property
    def real_ip(self) -> str | None:
        """连着真机的话,连的是哪一台;只连仿真时是 `None`。"""
        return self._real_ip

    def raise_fault(self, error: ArmError) -> None:
        """从外面挂一个报错上去(比如盯梢发现掉线)。`error()` 随后就读得到。

        只挂,不停:要停的人自己去调 `stop()`(ADR-0012:谁先发现要停谁就直接调)。
        """
        with self._lock:
            self._fault = error

    def stage_target(self, pose: Mat) -> Mat:
        """把一个目标位姿放进工作站,再从**那个目标身上读回来**,返回读到的位姿。

        护栏要的「从指令里把目标读回来」就是这一下(`robot_side.guard`)。上一轮出事
        正是因为只核对了自己算出来的值,没有回头看指令里真正指向的点
        (`notes/jaka-first-connection-results.md`「一个操作上的失误」)。

        目标挂在工作台坐标系下,所以读回来的也是工作台坐标系里的位姿——写错坐标系、
        写错单位,都会在这一步显出来。发的时候用 `move_linear_staged()`,动的就是
        这个被读回来核对过的目标本身。
        """
        def stage(robot: robolink.Item) -> Mat:
            rdk = self._rdk
            if rdk is None:
                raise NotConnectedError("还没连接;先调 connect()")
            if self._target is None:
                # 挂在工作台坐标系下——和 `apply_config` 设给机械臂的是同一个坐标系,
                # 所以读回来的位姿就是工作台坐标系里的位姿
                frame = self._worktable_frame(robot)
                self._target = rdk.AddTarget(STAGED_TARGET_NAME, frame, robot)
            # 明写成笛卡尔目标,不靠默认值 —— 关节目标那一边就是栽在「靠默认」上的
            self._target.setAsCartesianTarget()
            self._target.setPose(pose)
            return self._target.Pose()

        return self._guard("放目标进工作站", stage)

    def stage_joint_target(self, joints: Sequence[float]) -> list[float]:
        """把一组关节角放进工作站的目标里,再从**那个目标身上读回来**,返回读到的关节角。

        `stage_target` 的关节版。关节运动不经位姿换算,所以它要核对的是关节角本身
        (`robot_side.guard.check_joint_readback`),不是位姿。
        """

        def stage(robot: robolink.Item) -> list[float]:
            rdk = self._rdk
            if rdk is None:
                raise NotConnectedError("还没连接;先调 connect()")
            if self._joint_target is None:
                self._joint_target = rdk.AddTarget(
                    STAGED_JOINT_TARGET_NAME, self._worktable_frame(robot), robot
                )
            # **必须标成关节目标。** `AddTarget` 建出来的默认是笛卡尔目标,只 `setJoints`
            # 的话 `MoveJ` 用的仍是那个目标身上从没设过的位姿 —— 2026-09-22 在真机上
            # 踩到:下发的关节角一点没执行,机械臂原地不动,而 `Busy()` 还真了 20 秒。
            self._joint_target.setAsJointTarget()
            self._joint_target.setJoints(list(joints))
            return [float(v) for v in self._joint_target.Joints().list()[:6]]

        return self._guard("放关节目标进工作站", stage)

    def move_joints_staged(self) -> None:
        """关节运动到 `stage_joint_target()` 放进去的那个目标。发出即返回,不等它走完。"""
        target = self._joint_target
        if target is None:
            raise MotionError("还没放关节目标进去;先调 stage_joint_target()")
        self._guard(
            "发关节运动(按放好的目标)",
            lambda robot: robot.MoveJ(target, blocking=False),
        )

    def move_linear_staged(self) -> None:
        """直线走到 `stage_target()` 放进去的那个目标。发出即返回,不等它走完。"""
        target = self._target
        if target is None:
            raise MotionError("还没放目标进去;先调 stage_target()")
        self._guard("发直线运动(按放好的目标)", lambda robot: robot.MoveL(target, blocking=False))

    def joint_limits(self) -> tuple[list[float], list[float]]:
        """工作站模型里每一轴的下限和上限(度)。只读,不动机械臂。"""
        low, high, _ = self._guard("读关节限位", lambda robot: robot.JointLimits())
        return [float(v) for v in low.list()[:6]], [float(v) for v in high.list()[:6]]

    def flange_pose_in_base(self, joints: Sequence[float]) -> Mat:
        """一组关节角对应的法兰位姿,在**机械臂基坐标系**里。不动机械臂。

        和 `flange_pose()` 的差别只有一个:那一个换算到工作台坐标系,这一个不换。
        标定工作台坐标系要用这一个——工作台坐标系正是这时候还没定的那个东西,
        拿它换算出来的数没有意义。
        """
        return self._require_robot().SolveFK(list(joints))

    def set_joints(self, joints: Sequence[float]) -> None:
        """把仿真里的模型直接摆到一组关节角,不走运动。

        只用来摆仿真里那一段运动的出发点(实测脚本每次叫停前都回到同一个位姿)。

        **连着真机时会怎样,没有实测过**;第 05 票接真机之前不要用它。
        """
        self._require_robot().setJoints(list(joints))

    # ---- 内部 ----------------------------------------------------------

    def _open_station(self, rdk: robolink.Robolink) -> robolink.Item:
        """打开工作站文件;已经开着就用开着的那个,不重复打开。"""
        active = rdk.ActiveStation()
        self._station_before = active if active.Valid() else None
        wanted = self.station_file.stem
        for station in rdk.getOpenStations():
            if station.Name() == wanted:
                rdk.setActiveStation(station)
                self._opened_station = False
                return station
        station = rdk.AddFile(str(self.station_file.resolve()))
        if not station.Valid():
            raise NotConnectedError(f"打不开工作站文件 {self.station_file}")
        rdk.setActiveStation(station)
        self._opened_station = True
        return station

    def _worktable_frame(self, robot: robolink.Item) -> robolink.Item:
        """工作台坐标系那个坐标系:挂在这台机械臂的底座下,没有就建一个。"""
        name = WORKTABLE_FRAME_PREFIX + self.robot_name
        base = robot.Parent()
        for child in base.Childs():
            if child.Type() == robolink.ITEM_TYPE_FRAME and child.Name() == name:
                return child
        rdk = self._rdk
        if rdk is None:
            raise NotConnectedError("还没连接;先调 connect()")
        return rdk.AddFrame(name, base)

    def _require_robot(self) -> robolink.Item:
        with self._lock:
            fault, robot = self._fault, self._robot
        if fault is not None and not fault.clearable:
            # 连接已经没了:再发就可能发到 `robolink` 自己重连上的另一个 RoboDK 里
            raise fault
        if robot is None:
            raise NotConnectedError("还没连接;先调 connect()")
        return robot

    def _guard(self, what: str, call: Callable[[robolink.Item], T]) -> T:
        """调一次 RoboDK 的 API;抛什么都翻成机械臂一侧的报错,并挂起来。"""
        robot = self._require_robot()
        try:
            return call(robot)
        except Exception as exc:
            raise self._fail(what, exc) from exc

    def _fail(self, what: str, exc: Exception) -> ArmError:
        """把 RoboDK 抛的那个翻成机械臂一侧的报错,挂起来,再交给调用方抛。

        分三类:

        - **断开**:socket 那一层的错,加上 `robolink` 重连不上时报的那几句
          (`LOST_MARKS`)。RoboDK 被关掉、拔网线都落在这里。
        - **运动错误**:`TargetReachError`(够不着)、`StoppedError`(这一段被停掉)
          和 RoboDK 报上来的别的错误字符串。
        - 已经是机械臂一侧的报错了(比如可达性检查自己抛的),原样挂起来。

        碰撞保护单独没有一类:RoboDK 仿真里碰撞是查出来的,不是报上来的
        (`CONTEXT.md`「可达性检查」);真机上是什么样,等第 05 票实测。
        """
        if isinstance(exc, ArmError):
            error: ArmError = exc
        elif isinstance(exc, OSError) or any(
            mark in str(exc) for mark in LOST_MARKS
        ):
            error = NotConnectedError(f"{what}时和 RoboDK 断了:{exc}")
        else:
            error = MotionError(f"{what}时 RoboDK 报错:{exc}")
        with self._lock:
            self._fault = error
        return error

    def __enter__(self) -> RoboDKArm:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.disconnect()

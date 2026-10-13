"""机械臂接口:状态逻辑和机械臂之间唯一的缝(spec「模块划分」)。

RoboDK 实现、空实现(NO ARM)和测试里的假机械臂(FAKE ARM)都实现这同一组方法,
上层因此不知道背后是仿真还是真机。

有的是:连接 / 断开、写入配置、关节运动、直线运动、是否还在动、停止、读关节角、
读法兰位姿、设仿真速度、可达性检查、查错误状态、清除错误。

两条从实测来的规矩,写在方法的说明里,也由实现强制:

- **运动用非阻塞方式发出**:`move_joints` / `move_linear` 发出即返回,执行线程
  只轮询 `is_moving()`,等这一段结束才发下一段。上一段还在走时发下一段,会占着
  连接的锁一直等,同一条连接上的叫停也就卡住了。
- **运动中不读关节角**:运动中调 `joints()` 会让 RoboDK 崩进程(2026-09-20 实测),
  而且运动中本来也读不到位置曲线。所以 `joints()` 在还在动时直接报错。
"""

from __future__ import annotations

import abc
import time
from collections.abc import Mapping, Sequence

from robodk.robomath import Mat

from robot_side.config import ArmConfig
from robot_side.motion import Segment
from robot_side.reach import ReachReport

MOVE_POLL_S = 0.02
"""等一段运动结束时,多久查一次「是否还在动」。

真机上 `Busy()` 的分辨率只有 50–61 ms(2026-09-20 实测),查得再密也没有更准。
"""

# ---------------------------------------------------------------------------
# **真机上「是否还在动」这个判据本身不成立 —— 还没解决。**
#
# 2026-09-22 第二次上 JAKA 实测(`notes/jaka-second-connection-results.md`),用
# RoboDK 自带的 3.10 跑 jkrc 记录器、直读控制柜命令端口取到运动曲线,发现经 RoboDK 的
# JAKA 驱动读回来的 `Busy()` 两头都不准:
#
# - **连上之后第一条指令**:`Busy()` 在下发后 21.00 秒落下,而机械臂 21.10 秒才开始动
#   —— 它在机械臂 ** 还没动之前 ** 就说「不动了」。执行器「发一段 → 等不动 → 发下一段」
#   的循环因此会在上一段还没执行时就把下一段发出去,实测里正是这样撞上的。
# - **后面几条**:又拖 1–2.6 秒才落下(机械臂早停了)。
#
# 第一条那笔延迟现在由 `robot_side.real.warm_up` 先付掉,所以「早落」这一种在主流程上
# 被绕开了;但 `Busy()` 本身仍然不是一个可信的「这一段走完了没有」。要真解决,得换一条
# 通道(候选:经 jkrc 直读控制柜,拿关节角或 `inpos` 位判到位),而那是一次不小的改动
# —— 涉及一个常驻的 3.10 子进程,也动到 ADR-0011 的前提。 ** 留给用户拍板,没有自作主张。 **
#
# 仿真里 `Busy()` 是准的,所以这条只影响真机。
# ---------------------------------------------------------------------------


def joint_drift(before: Sequence[float], after: Sequence[float]) -> float:
    """两组关节角里差得最多的那一轴差了多少度。用来核对「机械臂没动过」。"""
    return max(abs(a - b) for a, b in zip(before, after, strict=True))


class ArmError(RuntimeError):
    """机械臂一侧的错误。状态逻辑见到它就进已停止。

    每一种报错自己说清四件事,状态逻辑不用挨个 `isinstance`:

    - `reason`:预览窗口上那一行短句,**英文**(`CONTEXT.md`「预览窗口」)。
    - `hint`:接下来要人做什么,也是英文短句;没有就空着。
    - `needs_power_on`:恢复时要不要提示人去示教器上重新上电使能。程序自己
      不上电、不使能(用户故事 27),只提示。
    - `clearable`:清除报错之后还能不能接着用这台机械臂。断开是唯一不能的:连接
      已经没了,清哪个报错都接不回来,要恢复就重启程序(用户故事 31)。
    """

    reason = "ARM ERROR"
    hint = ""
    needs_power_on = False
    clearable = True


class CollisionError(ArmError):
    """碰撞保护被触发。"""

    reason = "COLLISION"


class MotionError(ArmError):
    """运动错误:走不通、够不着、控制器把这一段判失败。"""

    reason = "MOTION ERROR"


class EmergencyStopError(ArmError):
    """急停被按下。

    松开急停之后控制柜一般是下了使能的,所以恢复时要提示人去示教器上重新上电使能
    ——程序只提示,不替人做(用户故事 27)。
    """

    reason = "E-STOP"
    hint = "RELEASE E-STOP, THEN RE-ENABLE ON THE PENDANT"
    needs_power_on = True


class NotConnectedError(ArmError):
    """还没连接,或已经断开(RoboDK 被关掉也算)。"""

    reason = "ARM DISCONNECTED"
    hint = "RESTART THE PROGRAM"
    clearable = False


class ArmBusyError(ArmError):
    """机械臂还在动的时候做了只能在静止时做的事(比如读关节角)。"""


class ArmTimeoutError(ArmError):
    """一段运动迟迟不结束。

    不是机械臂报上来的,是程序自己等不下去了(`executor.segment_timeout_s`:
    估算用时 × 2 + 5 秒),**当作机械臂一侧报错处理**(用户故事 19)。
    """

    reason = "TIMEOUT"


class Arm(abc.ABC):
    """机械臂接口(ARM)。

    `stop()` 可以从任意线程调、重复调没有副作用;其余方法由一个线程(执行线程)调。
    """

    @abc.abstractmethod
    def connect(self) -> None:
        """连上机械臂。已经连着时再调没有副作用。"""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开。断开前先停住当前运动;没连过时再调没有副作用。"""

    @property
    @abc.abstractmethod
    def connected(self) -> bool:
        """现在是不是连着。"""

    @property
    def busy_is_reliable(self) -> bool:
        """「是否还在动」说不动了,是不是就真的走完了。

        **默认不信**:执行器还要读一次位置、核对到没到目标(`executor.wait_until_done`)。
        经 RoboDK 的 JAKA 驱动,真机的 `Busy()` 会在机械臂还没走完、甚至还没开始走时就
        落下(2026-09-22 实测)。只跑 RoboDK 仿真时它是准的,`RoboDKArm` 那时说 True。
        """
        return False

    def joint_speed_for(
        self, from_joints: Sequence[float], to_joints: Sequence[float], tip_mms: float
    ) -> float | None:
        """这一段关节运动要让指针尖端走 `tip_mms` 那么快,关节速度该给多少(°/s)。不动机械臂。

        按「尖端走多远 ÷ 尖端速度 = 要走多久」,再拿转得最多的那一轴除以这个时间
        (`robot_side.executor.settle`、`motion.joint_path_mm`)。**没有运动学的实现算不出,
        返回 None**,执行器就照旧用那一段自己写的 °/s。
        """
        return None

    def solve_ik(self, pose: Mat, seed: Sequence[float]) -> list[float] | None:
        """工作台坐标系里这个法兰位姿对应的关节角,从 `seed` 那一组接着求。不动机械臂。

        回待命「抬起」那一段要用:先求出抬起以后的关节角,再用关节运动过去
        (`robot_side.startup.returning`)。**没有运动学的实现求不出,返回 None**,
        抬起那一段就排不出来,状态逻辑停下等人。
        """
        return None

    @abc.abstractmethod
    def apply_config(self, config: ArmConfig) -> tuple[float, float, float]:
        """把机械臂配置写进机械臂一侧,返回**待命点**(待命位姿下法兰在工作台坐标系里的位置)。

        写的是工作台坐标系和待命位姿;工具保持 0(ADR-0013)。每次启动都照配置写一遍,
        所以在 RoboDK 界面上改过的数下次启动会被覆盖。
        """

    @abc.abstractmethod
    def reach_check(
        self, plans: Mapping[str, Sequence[Segment]], standby_joints: Sequence[float]
    ) -> ReachReport:
        """逐路点求逆解、每段直线单独检查,**全程不移动机械臂**。

        判定本身在 `robot_side.reach` 里;实现这一层只负责把自己的运动学接上去,并保证
        查完机械臂还在原处(RoboDK 的直线检查会把仿真里的模型挪到终点)。
        """

    @abc.abstractmethod
    def set_speed(
        self, *, linear_mms: float | None = None, joints_degs: float | None = None
    ) -> None:
        """设这一段的速度:直线 mm/s、关节 °/s。

        真机上发不发得出去,看机械臂配置的 `program_sets_speed`:不发的话快慢全看示教器;
        发的话实际速度还要乘示教器上的速度倍率。

        不模拟速度的实现(空实现、测试里的假机械臂)收下就算,不用做什么。
        """

    @abc.abstractmethod
    def move_joints(self, joints: Sequence[float]) -> None:
        """发一段关节运动到给定的关节角(度)。**发出即返回,不等它走完。**"""

    @abc.abstractmethod
    def move_linear(self, pose: Mat) -> None:
        """发一段直线运动到给定的法兰位姿。**发出即返回,不等它走完。**

        位姿在机械臂当前的坐标系里、按当前的工具算;第 01 票不写入配置,
        所以用的是载入模型时的基坐标系和工具 0(ADR-0013:控制器保持工具 0)。
        """

    @abc.abstractmethod
    def is_moving(self) -> bool:
        """还在动吗。执行线程等一段运动结束,只许查这一个。"""

    @abc.abstractmethod
    def stop(self) -> None:
        """立刻停在半路。

        **可以从任意线程调,重复调没有副作用**(ADR-0012:谁先发现要停,谁就直接
        调它,不经过主线程)。没连接时什么也不做。
        """

    @abc.abstractmethod
    def joints(self) -> list[float]:
        """读当前关节角(度)。**只在不运动时调**,还在动时抛 `ArmBusyError`。"""

    @abc.abstractmethod
    def pose(self) -> Mat:
        """读法兰当前的位姿(工作台坐标系里)。和 `joints()` 一样,**只在不运动时调**。

        回待命要从「现在停在哪」算起,所以这一个和 `joints()` 一样重要。
        """

    @abc.abstractmethod
    def error(self) -> ArmError | None:
        """机械臂一侧现在挂着什么报错;没有就是 `None`。

        **不许自己去问机械臂**:报错是各实现在自己的调用里接住、挂起来的,这里只是
        把挂着的那个读出来。状态逻辑每轮都问一次,问一次要是一次 API 来回,
        一秒就是几十次(`robot_side.state.Machine.tick`)。

        挂起来而不只是抛出来,是因为有些报错发生时我们这边一条指令都没发
        (急停被人按下、控制柜自己报碰撞),没有哪一次调用能把它抛出来。
        """

    @abc.abstractmethod
    def clear_error(self) -> None:
        """清除机械臂一侧挂着的报错。没有挂着的就什么也不做。

        已停止后按回车走这一步(用户故事 17)。清不掉就**原样抛出**挂着的那个错
        (比如连接已经断了,清了也接不回来);要不要提示人重新上电使能,由调用方
        看 `ArmError.needs_power_on`——程序自己不上电、不使能(用户故事 27)。
        """

    def wait_until_not_moving(
        self, timeout_s: float, poll_s: float = MOVE_POLL_S
    ) -> None:
        """轮询 `is_moving()`,等当前这一段走完。

        中途被叫停也算走完——叫停之后机械臂就不动了。超过 `timeout_s` 抛
        `ArmTimeoutError`;什么时候算超时由调用方定
        (`robot_side.executor.segment_timeout_s`:估算用时 × 2 + 5 秒)。

        (名字不叫 idle:IDLE 在 `CONTEXT.md` 里专指空闲那个状态,不是「这一段走完了」。)
        """
        deadline = time.monotonic() + timeout_s
        while self.is_moving():
            if time.monotonic() > deadline:
                raise ArmTimeoutError(f"一段运动超过 {timeout_s:.1f} 秒还没结束")
            time.sleep(poll_s)


def open_arm(config: ArmConfig) -> Arm:
    """按机械臂配置挑一个实现。`driver = "none"` 就是空实现,不连任何东西。

    两个实现都**用得上才 import**:没开 RoboDK 的时候(`--arm noarm`)不该因为
    import 就把 RoboDK 那一套拖进来。
    """
    if config.driver == "none":
        from robot_side.noarm import NoArm

        return NoArm(config.noarm_segment_s)
    from robot_side.robodk_arm import RoboDKArm
    from robot_side.station import STATION_FILE

    return RoboDKArm(STATION_FILE, config.robot_name)

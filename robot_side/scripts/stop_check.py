"""叫停实测:在仿真里连做 10 次叫停,看每次是不是 0.5 秒内停住、停在半路。

    uv run python -m robot_side.scripts.stop_check [--robot "JAKA Zu20"] [--trials 10]

四组:
1. **走到一半叫停** —— 运动发出 1 秒后叫停,连做 10 次;
2. **正在下发运动指令的那一刻叫停** —— 非阻塞的 MoveL 本身要 63–146 ms 才返回,
   这期间占着连接的锁,叫停最坏多等约 0.15 秒;这一组专门打这个窗口;
3. **叫停正好落在两段之间** —— 这时机械臂本来就不动,拦住下一段的是「已叫停」标记;
4. **重复叫停、从别的线程叫停** —— 停止是幂等的,不该有副作用。

叫停走的是正式的那条路:`StopSignal` 先立「已叫停」标记,再调机械臂接口的「停止」,
而且是从另一个线程调的(ADR-0012)。这里不按空格——空格那条路要人来按,见
`stop_demo`。

事先定好、不按结果调的数:直线 150 mm、50 mm/s、发出 1 秒后叫停、10 次、过关线
0.5 秒;执行线程每 10 ms 查一次「是否还在动」。完整结果另存 data/stop-check/。
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from robodk.robomath import Mat, transl

from robot_side.arm import ArmTimeoutError
from robot_side.executor import run_segments
from robot_side.robodk_arm import JAKA, RoboDKArm
from robot_side.station import ROOT, STATION_FILE
from robot_side.stop import StopSignal

LINE_MM = 150.0
SPEED_MMS = 50.0
STOP_AFTER_S = 1.0
TRIALS = 10
PASS_S = 0.5
POLL_S = 0.01
MOVE_TIMEOUT_S = 20.0
SETTLE_S = 0.3
DRIFT_CHECK_S = 0.5
DRIFT_TOL_MM = 0.5
AT_END_MM = LINE_MM - 1.0

OUT_DIR = ROOT / "data" / "stop-check"

# 找一个能走 150 mm 直线的起点:先试模型自带的位姿,不行再试几种弯肘的
JOINT_OFFSETS = (
    (0, 0, 0, 0, 0, 0),
    (0, 20, -40, 20, 0, 0),
    (0, -20, 40, -20, 0, 0),
    (0, 30, -60, 30, 0, 0),
    (0, -30, 60, -30, 0, 0),
)
DIRECTIONS = (
    ("+X", (1, 0, 0)), ("-X", (-1, 0, 0)),
    ("+Y", (0, 1, 0)), ("-Y", (0, -1, 0)),
    ("-Z", (0, 0, -1)), ("+Z", (0, 0, 1)),
)


@dataclass
class TestMove:
    """实测用的那一段运动:从哪组关节角出发,往哪个方向,150 mm 外的目标位姿。"""

    from_joints: list[float]
    direction_name: str
    direction: tuple[float, float, float]
    target: Mat


@dataclass
class Trial:
    """一次实测的结果。"""

    n: int
    group: str
    stop_s: float | None = None
    """叫停(立标记)到机械臂不动了,秒。过关线 0.5。"""
    stop_call_s: float | None = None
    """机械臂接口的「停止」本身花了多久。"""
    move_s: float | None = None
    dispatch_s: float | None = None
    """非阻塞下发本身花了多久(已知 63–146 ms)。"""
    along_mm: float = 0.0
    perp_mm: float = 0.0
    drift_mm: float = 0.0
    passed: bool = False
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def judge_ran_to_end(self) -> None:
        """对照组:没叫停,就该一直走到终点。"""
        self.passed = self.along_mm >= AT_END_MM

    def judge_stopped_in_time(self) -> None:
        """叫停组:0.5 秒内停住,停在半路,而且停了就不再动。"""
        self.passed = (
            self.stop_s is not None
            and self.stop_s <= PASS_S
            and self.along_mm < AT_END_MM
            and self.drift_mm < DRIFT_TOL_MM
        )


def along_perp(
    point: list[float], origin: list[float], direction: tuple[float, float, float]
) -> tuple[float, float]:
    """沿直线走了多远、偏离直线多远(mm)。"""
    v = [point[i] - origin[i] for i in range(3)]
    along = sum(v[i] * direction[i] for i in range(3))
    perp = [v[i] - along * direction[i] for i in range(3)]
    return along, math.dist(perp, [0, 0, 0])


def plan_test_move(arm: RoboDKArm, line_mm: float = LINE_MM) -> TestMove:
    """挑一个出发位姿和方向,从那里走的这段直线走得通。

    对着走不通的目标硬发是自找麻烦,所以下发之前先查一遍。
    """
    model_joints = arm.joints()
    for offset in JOINT_OFFSETS:
        joints = [model_joints[i] + offset[i] for i in range(6)]
        arm.set_joints(joints)
        pose = arm.pose()
        for name, d in DIRECTIONS:
            target = transl(*(v * line_mm for v in d)) * pose
            if arm.linear_test(joints, target) == 0:
                arm.set_joints(joints)
                return TestMove(joints, name, d, target)
    raise SystemExit(
        f"{arm.robot_name}:找不到能走 {line_mm:.0f} mm 直线的出发位姿,"
        "先停下,不要硬跑"
    )


def run_trial(
    arm: RoboDKArm,
    signal: StopSignal,
    move: TestMove,
    n: int,
    group: str,
    stop_delay_s: float | None,
    stop_from_dispatch_start: bool = False,
) -> Trial:
    """走一次 150 mm 直线;`stop_delay_s` 不是 None 时,另一个线程到点就叫停。

    `stop_from_dispatch_start` 为真时,延时从「开始下发」算起,想让叫停落在非阻塞
    MoveL 还没返回、占着连接锁的那一段里;否则从「下发完」算起。打没打中那个窗口
    是量出来的,不是假定的——见结果里的注记。
    """
    trial = Trial(n=n, group=group)
    signal.clear()
    records_before = len(signal.records)
    arm.set_joints(move.from_joints)
    time.sleep(SETTLE_S)
    if arm.is_moving():
        trial.error = "起点还在动"
        return trial
    origin = arm.pose().Pos()

    dispatching = threading.Event()
    dispatched = threading.Event()
    times: dict[str, float] = {}
    errors: list[str] = []

    def executor() -> None:
        try:
            dispatching.set()
            times["dispatch_start"] = time.monotonic()
            arm.move_linear(move.target)
            times["dispatch_end"] = time.monotonic()
            dispatched.set()
            arm.wait_until_not_moving(MOVE_TIMEOUT_S, POLL_S)
            times["not_moving"] = time.monotonic()
        except ArmTimeoutError as exc:
            errors.append(f"执行线程:{exc}")
        except Exception as exc:  # 记下来,不吞掉
            errors.append(f"执行线程:{type(exc).__name__}: {exc}")
        finally:
            dispatching.set()
            dispatched.set()

    def stopper() -> None:
        gate = dispatching if stop_from_dispatch_start else dispatched
        gate.wait(MOVE_TIMEOUT_S)
        time.sleep(stop_delay_s or 0.0)
        try:
            signal.raise_stop(f"stop-check/{group}")
        except Exception as exc:  # 记下来,不吞掉
            errors.append(f"叫停:{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=executor, name="executor", daemon=True)]
    if stop_delay_s is not None:
        threads.append(threading.Thread(target=stopper, name="stopper", daemon=True))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(MOVE_TIMEOUT_S + 5)
        if thread.is_alive():
            errors.append(f"{thread.name} 线程没有结束")

    if "dispatch_end" in times:
        trial.dispatch_s = times["dispatch_end"] - times["dispatch_start"]
    if "not_moving" in times:
        trial.move_s = times["not_moving"] - times["dispatch_end"]
    records = signal.records[records_before:]
    if stop_delay_s is not None and not records:
        errors.append("叫停线程没有留下记录")
    if records:
        record = records[0]
        trial.stop_call_s = record.stopped_at - record.flag_at
        if "not_moving" in times:
            trial.stop_s = times["not_moving"] - record.flag_at
        if record.error:
            errors.append(f"叫停:{record.error}")
        if stop_from_dispatch_start and "dispatch_end" in times:
            trial.notes.append(_where_stop_landed(record.flag_at, times))

    if errors:
        trial.error = ";".join(errors)
        return trial
    if arm.is_moving():
        trial.error = "这一段结束后机械臂还在动"
        return trial

    stopped_at = arm.pose().Pos()
    time.sleep(DRIFT_CHECK_S)
    later = arm.pose().Pos()
    trial.along_mm, trial.perp_mm = along_perp(stopped_at, origin, move.direction)
    trial.drift_mm = math.dist(stopped_at, later)
    if arm.is_moving():
        trial.error = "叫停 0.5 秒后还在动"
        return trial

    if stop_delay_s is None:
        trial.judge_ran_to_end()
    else:
        trial.judge_stopped_in_time()
    return trial


def _where_stop_landed(flag_at: float, times: dict[str, float]) -> str:
    """这一次叫停到底落在下发之前、下发当中,还是下发之后。"""
    if flag_at < times["dispatch_start"]:
        return "叫停落在下发之前(没打中窗口)"
    if flag_at <= times["dispatch_end"]:
        return "叫停落在下发当中"
    return "叫停落在下发之后(没打中窗口)"


def check_stop_is_idempotent(arm: RoboDKArm, signal: StopSignal) -> Trial:
    """重复叫停、从两个线程同时叫停:不该报错,机械臂也该照样停着。"""
    trial = Trial(n=1, group="重复叫停")
    errors: list[str] = []

    def stop_twice() -> None:
        try:
            signal.raise_stop("stop-check/重复")
            arm.stop()
        except Exception as exc:  # 记下来,不吞掉
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=stop_twice, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    arm.stop()  # 主线程再来一次

    if errors:
        trial.error = ";".join(errors)
        return trial
    if arm.is_moving():
        trial.error = "重复叫停之后机械臂还在动"
        return trial
    if not signal.raised:
        trial.error = "叫停之后「已叫停」标记没立住"
        return trial
    trial.notes.append(f"一共叫停 {len(signal.records)} 次,都没报错")
    trial.passed = True
    return trial


def check_stop_between_segments(
    arm: RoboDKArm, signal: StopSignal, move: TestMove
) -> Trial:
    """叫停正好落在两段之间:第一段走完就叫停,第二段不该再发出去。

    这一段机械臂本来就不动,「停止」没什么可停的——拦住下一段的是「已叫停」标记。
    """
    trial = Trial(n=1, group="两段之间")
    signal.clear()
    arm.set_joints(move.from_joints)
    time.sleep(SETTLE_S)
    origin = arm.pose().Pos()

    # 第一段:走完整的 150 mm,不叫停。第二段:关节运动回到出发位姿——它一定走得通,
    # 所以真要是发出去了,机械臂会往回走,量出来就能看见
    first = [lambda: arm.move_linear(move.target)]
    second = [lambda: arm.move_joints(move.from_joints)]

    try:
        sent_first = run_segments(arm, signal, first, MOVE_TIMEOUT_S, POLL_S)
        at_end, _ = along_perp(arm.pose().Pos(), origin, move.direction)
        # 第一段刚走完、第二段还没发出去的那一刻叫停
        signal.raise_stop("stop-check/两段之间")
        sent_second = run_segments(arm, signal, second, MOVE_TIMEOUT_S, POLL_S)
    except Exception as exc:  # 记下来,不吞掉
        trial.error = f"{type(exc).__name__}: {exc}"
        return trial

    trial.along_mm, trial.perp_mm = along_perp(arm.pose().Pos(), origin, move.direction)
    if sent_first != 1:
        trial.error = f"第一段没发出去(发出 {sent_first} 段)"
        return trial
    if sent_second != 0:
        trial.error = "已叫停之后还是发出了第二段"
        return trial
    if abs(trial.along_mm - at_end) > DRIFT_TOL_MM:
        trial.error = (
            f"第二段没发出,机械臂却动了:{at_end:.1f} → {trial.along_mm:.1f} mm"
        )
        return trial
    trial.notes.append(f"第二段没有发出;停在第一段的终点 {trial.along_mm:.1f} mm")
    trial.passed = True
    return trial


def print_row(trial: Trial) -> None:
    def num(value: float | None, width: int, digits: int) -> str:
        return " " * (width - 1) + "-" if value is None else f"{value:>{width}.{digits}f}"

    print(
        f"{trial.n:>2} {trial.group:<12} {num(trial.stop_s, 9, 3)} "
        f"{num(trial.stop_call_s, 10, 4)} {num(trial.dispatch_s, 9, 3)} "
        f"{trial.along_mm:>8.1f} {trial.drift_mm:>9.2f}  "
        f"{'过' if trial.passed else '不过'}  "
        f"{trial.error or ';'.join(trial.notes)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="仿真里的叫停实测")
    parser.add_argument("--robot", default=JAKA, help="工作站里机械臂的名字")
    parser.add_argument("--trials", type=int, default=TRIALS, help="走到一半叫停做几次")
    args = parser.parse_args(argv)

    trials: list[Trial] = []
    started = datetime.now()
    with RoboDKArm(STATION_FILE, robot_name=args.robot) as arm:
        signal = StopSignal(arm)
        arm.set_speed(linear_mms=SPEED_MMS)
        move = plan_test_move(arm)
        print(
            f"机械臂 {arm.robot_name};出发关节角 "
            f"{[round(v, 1) for v in move.from_joints]},方向 {move.direction_name},"
            f"走 {LINE_MM:.0f} mm、{SPEED_MMS:.0f} mm/s"
        )

        print("\n对照:不叫停,走完全程")
        baseline = run_trial(arm, signal, move, 0, "不叫停", None)
        trials.append(baseline)
        print(
            f"  全程 {baseline.move_s or float('nan'):.3f} s,终点 "
            f"{baseline.along_mm:.1f} mm,偏离 {baseline.perp_mm:.2f} mm,"
            f"{'过' if baseline.passed else '不过'} {baseline.error or ''}"
        )
        if not baseline.passed:
            print("对照就没走完,后面的数不可信,停下")
            return 1

        print(
            f"\n{'#':>2} {'组':<12} {'叫停用时s':>9} {'停止耗时s':>10} "
            f"{'下发耗时s':>9} {'停在mm':>8} {'0.5s后动mm':>9}  结果  错误"
        )
        for i in range(args.trials):
            trial = run_trial(arm, signal, move, i + 1, "走到一半", STOP_AFTER_S)
            trials.append(trial)
            print_row(trial)
        for i in range(3):
            trial = run_trial(
                arm, signal, move, i + 1, "下发中", 0.0, stop_from_dispatch_start=True
            )
            trials.append(trial)
            print_row(trial)

        between = check_stop_between_segments(arm, signal, move)
        trials.append(between)
        print_row(between)

        idempotent = check_stop_is_idempotent(arm, signal)
        trials.append(idempotent)
        print_row(idempotent)

    print("\n汇总")
    failed = 0
    for group in ("走到一半", "下发中", "两段之间", "重复叫停"):
        rows = [t for t in trials if t.group == group]
        if not rows:
            continue
        times = [t.stop_s for t in rows if t.stop_s is not None]
        passed = sum(t.passed for t in rows)
        failed += len(rows) - passed
        span = (
            f";叫停用时 最短 {min(times):.3f} / 最长 {max(times):.3f} s"
            if times
            else ""
        )
        stops = [t.along_mm for t in rows]
        print(
            f"  {group}:过关 {passed}/{len(rows)}{span}"
            f";停在 {min(stops):.1f}–{max(stops):.1f} mm(全程 {LINE_MM:.0f} mm)"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"run-{started:%Y%m%d-%H%M%S}.json"
    out.write_text(
        json.dumps(
            {
                "started": started.isoformat(timespec="seconds"),
                "robot": args.robot,
                "params": {
                    "line_mm": LINE_MM, "speed_mms": SPEED_MMS,
                    "stop_after_s": STOP_AFTER_S, "pass_s": PASS_S, "poll_s": POLL_S,
                },
                "test_move": {
                    "from_joints": move.from_joints,
                    "direction": move.direction_name,
                },
                "trials": [asdict(t) for t in trials],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"完整结果:{out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""活动范围全查一遍的结果存下来,配置不变就不再查(2026-09-23 用户要的)。

连真机之前把活动范围里每个停靠点出发的每个动作都查一遍(`startup.check_range`),
JAKA 上要 80 秒左右,每次 `--real` 启动都等这一下。可查出来的东西只取决于:

- 机械臂配置里和运动学有关的那几样:哪台机械臂、待命位姿、工作台坐标系;
- 演示配置里的默认姿态、活动范围、各个预设动作的路点和做完怎么停;
- RoboDK 工作站文件(机械臂模型就在里面);
- 这一套检查怎么查(`CHECK_VERSION`,改了检查的算法就加一)。

这几样都没变,结果就不会变,所以按它们算一个指纹,结果存在 `data/cache/` 下
(`data/` 整个不进 git)。**控制柜 IP 不算在里面**:仿真里跑一次攒下的,真机上照用。
**速度、停顿、显示的名字也不算**:够不够得着和走多快无关。原来整个预设动作都算进去,
2026-09-24 用户在真机上调速度,每改一次就重查一遍 75 个停靠点(JAKA 约 5 分钟)。

读不出来(没有、坏了、指纹对不上)就重新查、重新存——宁可多等一次,不拿对不上的
结果去放行运动。
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path

from robot_side.arm import Arm
from robot_side.config import ROOT, Action, ArmConfig, DemoConfig
from robot_side.reach import ReachFailure
from robot_side.startup import RangeFailure, RangeReport, StartupPlans, check_range

CACHE_DIR = ROOT / "data" / "cache"
"""存在哪儿。`data/` 整个被 git 忽略。"""

CHECK_VERSION = 1
"""这一套检查的版本。改了 `check_range` / `check_reach` 怎么查,就把它加一,旧结果作废。"""


def fingerprint(arm_config: ArmConfig, demo: DemoConfig, station_file: Path) -> str:
    """决定查出来什么结果的那几样东西的指纹(模块说明里列的那些)。"""
    parts = {
        "version": CHECK_VERSION,
        "robot": arm_config.robot_name,
        "standby": [round(v, 6) for v in arm_config.standby_joints_deg],
        "worktable": [arm_config.worktable.angle_deg, *arm_config.worktable.origin_mm],
        "orientation": list(demo.default_orientation_deg),
        "range": list(demo.run.range_mm),
        "actions": [_geometry(action) for action in demo.actions],
    }
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8"))
    digest.update(Path(station_file).read_bytes())
    return digest.hexdigest()[:16]


def _geometry(action: Action) -> list:
    """一个预设动作里决定够不够得着的那几样:名字、做完怎么停、每个路点到哪儿、什么姿态。

    `speed_mms`、`pause_s`、`display` 不在里面——改了它们,查出来的结果一样。
    """
    return [
        action.name,
        action.end,
        [
            [w.dx_mm, w.dy_mm, w.dz_mm, list(w.orientation_deg or ()), w.nod_deg]
            for w in action.waypoints
        ],
    ]


def cache_file(arm_config: ArmConfig, key: str, cache_dir: Path = CACHE_DIR) -> Path:
    name = arm_config.robot_name.replace(" ", "-")
    return Path(cache_dir) / f"range-{name}-{key}.json"


def save(report: RangeReport, path: Path, key: str) -> None:
    """存下来。先写临时文件再换名,写到一半断电也不会留下一份半截的。"""
    data = {
        "fingerprint": key,
        "rest_points": report.rest_points,
        "checked_segments": report.checked_segments,
        "reachable": [[*rest, name, ok] for (rest, name), ok in report.reachable.items()],
        "failures": [
            [*f.rest, f.failure.action, f.failure.waypoint, f.failure.label, f.failure.why]
            for f in report.failures
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load(path: Path, key: str) -> RangeReport | None:
    """读回来。没有、读不懂、指纹对不上,一律 None(调用方重新查)。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["fingerprint"] != key:
            return None
        reachable = {
            ((float(x), float(y), float(z)), str(name)): bool(ok)
            for x, y, z, name, ok in data["reachable"]
        }
        failures = tuple(
            RangeFailure(
                (float(x), float(y), float(z)),
                ReachFailure(str(action), int(waypoint), str(label), str(why)),
            )
            for x, y, z, action, waypoint, label, why in data["failures"]
        )
        return RangeReport(
            reachable=reachable,
            rest_points=int(data["rest_points"]),
            checked_segments=int(data["checked_segments"]),
            failures=failures,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def check_range_cached(
    arm: Arm,
    arm_config: ArmConfig,
    demo: DemoConfig,
    prepared: StartupPlans,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    log: logging.Logger | None = None,
    cache_dir: Path = CACHE_DIR,
    station_file: Path | None = None,
) -> tuple[RangeReport, bool]:
    """配置和模型都没变就用存下来的结果,不然查一遍再存。返回(结果, 是不是存下来的)。"""
    log = log if log is not None else logging.getLogger("robot_side.range_cache")
    if station_file is None:
        from robot_side.station import STATION_FILE  # 用得上才 import(它要 robolink)

        station_file = STATION_FILE
    key = fingerprint(arm_config, demo, station_file)
    path = cache_file(arm_config, key, cache_dir)
    stored = load(path, key)
    if stored is not None:
        log.info("活动范围:配置和模型都没变,用存下来的结果(%s),不再查", path.name)
        return stored, True
    log.info("活动范围:没存过或配置变了,查一遍(查完存到 %s)", path.name)
    report = check_range(arm, arm_config, demo, prepared, on_progress=on_progress)
    try:
        save(report, path, key)
    except OSError as exc:  # 存不下来不耽误这一次,下次再查就是了
        log.warning("活动范围的结果没存下来:%s", exc)
    return report, False

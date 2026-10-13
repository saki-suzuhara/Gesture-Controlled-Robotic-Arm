"""建站脚本:从 model/ 里的两个模型建出工作站文件,可以重复跑。

    uv run python -m robot_side.scripts.build_station [--keep]

每跑一次都从零新建,存成 model/demo-station.rdk。同名工作站已经开着时先关掉它
(它本来就是这个脚本生成的)。**不动用户原有的工作站**:别的工作站不关、不改,
跑完把原来激活的那个恢复回去。

--keep 留着新建的工作站不关,方便在 RoboDK 界面里看一眼。
"""

from __future__ import annotations

import sys

from robodk import robolink

from robot_side.robodk_arm import robodk_is_running
from robot_side.station import (
    PLACEMENTS,
    SPACING_MM,
    STATION_FILE,
    TABLE_TOP_MM,
    build_station,
    clearance_mm,
)


def close_stale_station(rdk: robolink.Robolink, name: str) -> None:
    """同名的工作站已经开着就关掉——重跑一次就是要整个重建。"""
    for station in rdk.getOpenStations():
        if station.Name() == name:
            print(f"工作站 {name} 已经开着,先关掉再重建")
            rdk.setActiveStation(station)
            rdk.CloseStation()


def main(argv: list[str] | None = None) -> int:
    keep = "--keep" in (argv if argv is not None else sys.argv[1:])

    if not robodk_is_running():
        print("RoboDK 没开,正在启动 ...")
    rdk = robolink.Robolink()
    print(f"RoboDK 版本:{rdk.Version()}")

    active_before = rdk.ActiveStation()
    stations_before = [s.Name() for s in rdk.getOpenStations()]
    print(f"原有工作站:{stations_before or '无'}")

    close_stale_station(rdk, STATION_FILE.stem)
    station, robots = build_station(rdk)

    print(f"\n工作站 {station.Name()}:")
    for placement, robot in zip(PLACEMENTS, robots, strict=True):
        base = robot.Parent().Pose().Pos()
        print(
            f"  {robot.Name():<14} 底座 ({base[0]:.0f}, {base[1]:.0f}, {base[2]:.0f}) mm"
            f",臂展 {placement.reach_mm:.0f} mm,桌子 {placement.table_name}"
        )
    print(f"  桌面离地 {TABLE_TOP_MM:.0f} mm;两台相隔 {SPACING_MM:.0f} mm")
    print(f"  互相够不着:两只手最近还差 {clearance_mm():.0f} mm")
    print(f"\n已存成 {STATION_FILE}")

    if keep:
        print("--keep:新建的工作站留着不关")
    else:
        rdk.setActiveStation(station)
        rdk.CloseStation()
        if active_before.Valid():
            rdk.setActiveStation(active_before)
        print("已关掉新建的工作站,原来激活的那个已恢复")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""按布置建出工作站:两台机械臂各站在一张离地 0.5 m 的桌子上,相隔约 3 m。

工作站文件只放模型和桌子这类摆设;工作台坐标系、待命位姿这些数以机械臂配置为准,
程序每次启动写进 RoboDK(第 02 票)。

相隔 3 m 是为了两台互相够不着:JAKA Zu 20 臂展 1780 mm、Dobot Nova 5 臂展 850 mm,
加起来 2630 mm,离 3000 mm 还差 370 mm。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robodk import robolink
from robodk.robomath import transl

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
STATION_FILE = MODEL_DIR / "demo-station.rdk"

TABLE_TOP_MM = 500.0
"""桌面离地高度:机械臂底座就装在这个高度上。"""

TABLE_SIDE_MM = 800.0
"""桌子的边长(正方形),桌子中心在机械臂底座正下方。"""

SPACING_MM = 3000.0
"""两台机械臂底座之间的距离。"""

TABLE_COLOR = [0.55, 0.55, 0.58, 1.0]


@dataclass(frozen=True)
class ArmPlacement:
    """一台机械臂在工作站里的摆放。"""

    model_file: str
    x_mm: float
    reach_mm: float
    table_name: str


PLACEMENTS = (
    ArmPlacement("JAKA-Zu20.robot", 0.0, 1780.0, "Table JAKA"),
    ArmPlacement("Dobot-Nova-5.robot", SPACING_MM, 850.0, "Table Nova 5"),
)


def box_vertices(
    x0: float, x1: float, y0: float, y1: float, z0: float, z1: float
) -> list[list[float]]:
    """一个长方体的 12 个三角形,共 36 个顶点(RoboDK 的 AddShape 要三个一组)。"""
    corner = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    quads = [
        (0, 3, 2, 1),  # 底
        (4, 5, 6, 7),  # 顶
        (0, 1, 5, 4),  # -Y
        (2, 3, 7, 6),  # +Y
        (1, 2, 6, 5),  # +X
        (3, 0, 4, 7),  # -X
    ]
    vertices: list[list[float]] = []
    for a, b, c, d in quads:
        vertices += [corner[a], corner[b], corner[c]]
        vertices += [corner[a], corner[c], corner[d]]
    return vertices


def add_table(rdk: robolink.Robolink, name: str, x_mm: float) -> robolink.Item:
    """在 x_mm 处放一张桌子:从地面到桌面的一个方块。"""
    half = TABLE_SIDE_MM / 2
    table = rdk.AddShape(
        box_vertices(x_mm - half, x_mm + half, -half, half, 0.0, TABLE_TOP_MM)
    )
    table.setName(name)
    table.setColor(TABLE_COLOR)
    return table


def add_arm(
    rdk: robolink.Robolink, station: robolink.Item, placement: ArmPlacement
) -> robolink.Item:
    """载入一台机械臂的模型,把它的底座放到桌面上。"""
    model = MODEL_DIR / placement.model_file
    if not model.is_file():
        raise FileNotFoundError(f"没有模型文件 {model}")
    robot = rdk.AddFile(str(model), station)
    if not robot.Valid() or robot.Type() != robolink.ITEM_TYPE_ROBOT:
        raise RuntimeError(f"{model.name} 载入的不是机械臂:{robot}")
    # 载入模型时 RoboDK 会给它建一个底座坐标系,机械臂挂在下面;移这个坐标系
    robot.Parent().setPose(transl(placement.x_mm, 0.0, TABLE_TOP_MM))
    return robot


def clearance_mm() -> float:
    """两台机械臂的手最近能差多远:底座间距减去两个臂展。大于 0 就是互相够不着。

    按实际摆位算,不按 `SPACING_MM` —— 改了摆位就该跟着变。
    """
    return min(
        abs(a.x_mm - b.x_mm) - a.reach_mm - b.reach_mm
        for i, a in enumerate(PLACEMENTS)
        for b in PLACEMENTS[i + 1 :]
    )


def build_station(
    rdk: robolink.Robolink, station_file: Path = STATION_FILE
) -> tuple[robolink.Item, list[robolink.Item]]:
    """新建工作站、摆好两台机械臂和两张桌子,存成工作站文件。

    工作站的名字和文件名一致,`RoboDKArm` 按这个名字认出它已经开着。
    调用方负责清场(关掉这个工作站、恢复原来激活的那个)。
    """
    if clearance_mm() <= 0:
        raise ValueError(
            f"两台按 {SPACING_MM:.0f} mm 摆会互相够得着(差 {clearance_mm():.0f} mm)"
        )
    station = rdk.AddStation(station_file.stem)
    if rdk.ActiveStation() != station:
        # 桌子是加到当前工作站里的,加错地方就是在动别人的工作站
        raise RuntimeError("新建的工作站没有成为当前工作站,停下")
    robots = []
    for placement in PLACEMENTS:
        add_table(rdk, placement.table_name, placement.x_mm)
        robots.append(add_arm(rdk, station, placement))
    station_file.parent.mkdir(parents=True, exist_ok=True)
    rdk.Save(str(station_file.resolve()))
    return station, robots

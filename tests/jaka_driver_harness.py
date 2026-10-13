"""给 `tests/test_jaka_keep_speed.py` 用:在 RoboDK 的嵌入式 Python 3.10 里载入**真的**原驱动,
把控制柜换成一个只记账的假的,喂几行指令,看 `joint_move` / `linear_move` 收到什么速度。

不连任何东西:原驱动只在收到 CONNECT 那一行时才去连控制柜,这里一行都不喂。
最后一行打印一份 JSON,测试读它。
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_side" / "drivers"))

import jaka_keep_speed as wrapper  # noqa: E402


class FakeJaka:
    """只记账:每次运动收到的速度。"""

    def __init__(self) -> None:
        self.moves: list[tuple[str, float]] = []

    def joint_move(self, joints, mode, block, speed):
        self.moves.append(("MOVJ", math.degrees(speed)))
        return (0,)

    def linear_move(self, pose, mode, block, speed):
        self.moves.append(("MOVL", speed))
        return (0,)

    def get_joint_position(self):
        return (0, [0.0] * 6)


MOVJ = "MOVJ 0 0 0 0 0 0"
MOVL = "MOVL 0 0 0 0 0 0 100 0 500 180 0 0"


def feed(base, fake, lines):
    fake.moves.clear()
    for line in lines:
        base.RunCommand(line + "\n")
    return list(fake.moves)


def main(log_dir: str) -> None:
    base = wrapper.load_driver()
    fake = FakeJaka()
    base.ROBOT = types.SimpleNamespace(robot=fake)
    for name, value in wrapper.STOCK.items():  # 原驱动模块里本来没有这几个,第一行指令才建
        setattr(base, name, value)

    out = {}
    # 原驱动:SPEED 收下了,下一行 MOVJ 又被重置回 100
    out["stock"] = feed(base, fake, ["SPEED 20 5 100 30", MOVJ, MOVL])

    wrapper.install(base, log_dir=Path(log_dir))
    out["kept"] = feed(base, fake, ["SPEED 20 5 100 30", MOVJ, MOVL, MOVJ])
    out["capped"] = feed(base, fake, ["SPEED 500 400 1000 1000", MOVJ, MOVL])
    out["unchanged"] = feed(base, fake, ["SPEED 20 5 100 30", "SPEED -1 -1 -1 -1", MOVJ, MOVL])
    out["partial"] = feed(base, fake, ["SPEED 30 -1 -1 -1", MOVJ, MOVL])

    other = types.SimpleNamespace(RunCommand=types.FunctionType(
        (lambda line: None).__code__, {}, "RunCommand"
    ))
    try:
        wrapper.install(other, log_dir=None)
        out["mismatch"] = "installed"
    except wrapper.DriverMismatch as exc:
        out["mismatch"] = str(exc)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])

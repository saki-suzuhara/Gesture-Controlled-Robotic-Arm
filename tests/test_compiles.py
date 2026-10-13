"""所有源文件都编得过。

几支脚本(`robot_side/scripts/`、`vision_side/scripts/`)没有测试 import 它们,写坏了要等有人
跑那一支才撞出来:2026-09-23 `action_demo.py` 里一个 `\\n` 被写成了真换行,一直到 10-12
打便携版、预编译字节码时才发现。
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "main.py"] + sorted(
    path
    for folder in ("robot_side", "vision_side", "packaging")
    for path in (ROOT / folder).rglob("*.py")
)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_编得过(path):
    compile(path.read_bytes(), str(path), "exec")

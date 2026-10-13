"""从 HaGRID 的官方示例图里裁出手势图例要用的几张示范小图。

图例上那几张小图取自 HaGRID 仓库的 `images/gestures.png`(HaGRIDv2 论文里的同一张)。
**只存本地,不提交、不分发**(spec「分支与提交」):裁出来的放 `data/legend/`,那一层
被 git 忽略。屏幕上标出处的那一行在 `vision_side/overlay.py` 的 `LEGEND_SOURCE`。

源图是 CC BY-SA 4.0 的,所以要标出处;**这个脚本不下载**,源图由人自己存到本机再传给
它——要不要从网上取、取哪一份,是人的决定,不是程序的。

    # 先把 https://github.com/hukenovs/hagrid/raw/master/images/gestures.png 存到本机
    uv run -m vision_side.scripts.legend_images --from ~/Downloads/gestures.png
    uv run -m vision_side.scripts.legend_images --from ~/Downloads/gestures.png --sheet

没有这几张图程序照样跑,图例只显示文字(`vision_side.overlay.load_legend_images`)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from robot_side.config import load_demo
from vision_side.overlay import LEGEND_DIR, legend_poses

#: 主网格第一格的左上角、每格多大、格与格的间距,都按**源图宽高的比例**写。
#: 数是 2026-09-22 照着 `images/gestures.png`(2910x1588)量的;源图换了版式就得重量,
#: 所以 `--sheet` 存一张拼图出来,一眼看得出裁歪没有。
GRID_X0, GRID_Y0 = 0.065, 0.064
CELL_W, CELL_H = 0.0988, 0.1835
COL_STEP, ROW_STEP = 0.110, 0.2477

#: 每个姿势在主网格里的位置(列、行,都从 0 起)。
#: 第 1 行:call dislike fist four like mute
#: 第 2 行:ok one palm peace peace-inv rock
CELLS: dict[str, tuple[int, int]] = {
    "fist": (2, 0),
    "like": (4, 0),
    "palm": (2, 1),
    "peace": (3, 1),
    "ok": (0, 1),
    "rock": (5, 1),
}

OUT_PX = 220
"""裁出来存多大。图例上按字高再缩一次,这里存得比要用的大一点就够。"""


def crop(image: np.ndarray, col: int, row: int) -> np.ndarray:
    """按比例把某一格裁出来。"""
    height, width = image.shape[:2]
    x = round((GRID_X0 + col * COL_STEP) * width)
    y = round((GRID_Y0 + row * ROW_STEP) * height)
    w = round(CELL_W * width)
    h = round(CELL_H * height)
    x, y = max(0, x), max(0, y)
    return image[y : min(height, y + h), x : min(width, x + w)]


def fit(image: np.ndarray, size: int = OUT_PX) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    return cv2.resize(
        image,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def save(path: Path, image: np.ndarray) -> None:
    """存一张。路径里有中文时 `cv2.imwrite` 写不出来,所以走 `tofile`。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", image)[1].tofile(str(path))


def sheet(pieces: dict[str, np.ndarray]) -> np.ndarray:
    """把裁出来的几张拼成一张,底下写上名字。裁歪没有,看这一张就知道。"""
    names = list(pieces)
    cell = OUT_PX + 40
    canvas = np.full((cell, cell * len(names), 3), 30, np.uint8)
    for i, name in enumerate(names):
        piece = fit(pieces[name])
        h, w = piece.shape[:2]
        x = i * cell + (cell - w) // 2
        canvas[8 : 8 + h, x : x + w] = piece
        cv2.putText(
            canvas, name.upper(), (i * cell + 12, cell - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2,
        )
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="裁出手势图例要用的示范小图")
    parser.add_argument(
        "--from", dest="source", type=Path, required=True, help="存到本机的 gestures.png"
    )
    parser.add_argument(
        "--out", type=Path, default=LEGEND_DIR, help=f"裁出来放哪儿(默认 {LEGEND_DIR})"
    )
    parser.add_argument(
        "--sheet", action="store_true", help="另存一张拼图,用来看裁歪没有"
    )
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"找不到源图 {args.source}", file=sys.stderr)
        return 1
    image = cv2.imdecode(np.fromfile(str(args.source), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"读不了源图 {args.source}", file=sys.stderr)
        return 1

    want = legend_poses(load_demo())
    missing = [p for p in want if p not in CELLS]
    if missing:
        # 演示配置换了姿势、这里还没量出它在网格里的位置:说清楚,别悄悄少几张
        print(f"这几个姿势在这张图里的位置还没量过:{'、'.join(missing)}", file=sys.stderr)

    pieces = {p: crop(image, *CELLS[p]) for p in want if p in CELLS}
    for name, piece in pieces.items():
        path = args.out / f"{name}.png"
        save(path, fit(piece))
        print(f"存了 {path}({piece.shape[1]}x{piece.shape[0]} → {OUT_PX} 以内)")
    if args.sheet:
        path = args.out / "_sheet.png"
        save(path, sheet(pieces))
        print(f"存了拼图 {path},看一眼裁歪没有")
    print("这几张只存本地,不提交、不分发;出处已经写在预览窗口一角。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

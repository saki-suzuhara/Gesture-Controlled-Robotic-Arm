"""把已有的原型录像转成录制工具的格式(spec「回放测试」第 5 条)。

    uv run -m vision_side.scripts.import_runs             # 两批都转
    uv run -m vision_side.scripts.import_runs --only fist # 只转握拳平移那批
    uv run -m vision_side.scripts.import_runs --no-video

`.scratch/` 下的原文件**一律只读,不动**;转出来的放 `data/recordings/`(整个 `data/`
被 git 忽略;录像里有用户本人,只存本地、不提交、不分发)。

两批数据各有各的用处:

- **握拳平移原型**(2026-09-17,`camera-prototype-2/runs/`,5 份):票里指名的第一批。
  **坐着**做,握拳平移从肩膀出发,录的时候还没有启动对准、也没有起点区——所以这边的
  检测区域和起点区是**从数据本身倒推的**(`alignment_source = "derived"`)。拿它当基准
  是为了「改了判定规则,数会不会变差」,不是为了看绝对值好不好看。
- **站着自由活动加叫停原型**(2026-09-19,`camera-prototype-7/runs/*-free`):
  **站在操作位**上录的,有真的启动对准结果(`recorded`)、10 次叫停提示、160 秒自由活动。
  叫停成功率和误叫停的基准在这一批上才有意义;`three2` 那件事也是在这一批的整幅画面里
  发生的(spec「停止手势与叫停」),所以整幅画面和检测区域各转一份。

要转的有三处:

- **姿势标签**:原型存的是上游 `targets` 里的下标,这边存名字;
- **提示时间表**:原型没有 `kind` 那一列,按这一批是什么补上;
- **时间轴**:两批都是**对准做完才开始录像**,所以录像的 0 秒对应日志里的十几秒;
  差的那一截记成 `video_t0`,回放视频时加回去(`vision_side.recordings.Meta`)。

视频原样复制:原型存的也是**翻转之前**的画面,和这边的格式一致
(`vision_side.recordings.RunWriter._write_video`)。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

from robot_side.config import FIST_LIKE_LABELS, FIST_MOVE, DemoConfig, load_demo
from vision_side.align import Alignment
from vision_side.dynamic_gestures.utils.enums import targets
from vision_side.prompts import (
    FREE,
    FREE_TEXT,
    GO,
    PREP,
    STOP_GESTURE,
    STOP_GO,
    STOP_PREP,
    Slot,
    Timetable,
)
from vision_side.recognize import UNKNOWN_LABEL, HandResult, Region
from vision_side.recordings import (
    DETECTION_REGION,
    FRAMES_NAME,
    FULL_FRAME,
    RECORDINGS_DIR,
    VIDEO_NAME,
    FrameRecord,
    Meta,
    alignment_to_json,
)

FIST_SOURCE = Path(".scratch") / "camera-prototype-2" / "runs"
"""握拳平移原型那一批在哪儿。**只读,不动它**(票里写明)。"""

FREE_SOURCE = Path(".scratch") / "camera-prototype-7" / "runs"
"""站着自由活动加叫停原型那一份在哪儿。"""

FIST_PREFIX = "proto2"
FREE_PREFIX = "proto7"
"""转出来的文件夹前缀,一眼看得出是哪一批。"""


def label_of(index: object) -> str:
    """上游的下标翻成名字。和 `vision_side.gestures.label_name` 同一套。"""
    if index is None:
        return UNKNOWN_LABEL
    i = int(index)
    return targets[i] if 0 <= i < len(targets) else UNKNOWN_LABEL


def convert_frames(lines: list[str], *, region: Region | None = None) -> list[FrameRecord]:
    """原型的逐帧日志转成这边的。

    取的是 `trk`(跟踪结果),不是 `raw`(每帧的检出):正式程序用的就是跟踪那一路
    ——它带跟踪编号和「已确认」,握拳平移的锁定靠它(ADR-0010)。

    `region` 是这一批录的时候送的那块画面,原样记进每一帧:回放时按它重跑,数据和
    正式程序的画面范围才一致(spec「录制工具」第 3 条)。
    """
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        hands = tuple(
            HandResult(
                box=(int(h[2]), int(h[3]), int(h[4]), int(h[5])),
                label=label_of(h[1]),
                track_id=int(h[0]),
                confirmed=bool(h[6]),
            )
            for h in raw.get("trk") or ()
        )
        out.append(
            FrameRecord(
                index=int(raw["i"]),
                t=float(raw["t"]),
                hands=hands,
                # 原型的 `ph` / `blank` 就是「这一帧是摄像头占位图」,和 flat 同一个意思
                flat=bool(raw.get("ph", raw.get("blank", 0))),
                region=region,
                interval_ms=None if raw.get("iv") is None else float(raw["iv"]),
                video_index=None if raw.get("vi") is None else int(raw["vi"]),
            )
        )
    return out


def video_t0_of(records: list[FrameRecord]) -> float:
    """录像第 0 帧对应日志里的哪一刻。两批都是对准做完才开始录像。"""
    for record in records:
        if record.video_index == 0:
            return record.t
    return 0.0


def intervals_from(path: Path) -> dict[int, float]:
    """从**录的时候那份**逐帧日志里取每帧的间隔,按帧号索引。

    事后用录像重跑出来的那两份(`replay-*.jsonl`)没有这一列——它们是离线跑的,
    快慢跟当时的录制循环没关系。而「录制循环卡没卡」正是要查的事
    (spec「录制工具」最后一条),所以得从现场那份把这一列接过来。
    """
    if not path.exists():
        return {}
    out: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if raw.get("iv") is not None:
            out[int(raw["i"])] = float(raw["iv"])
    return out


def convert_timetable(raw: list[dict]) -> Timetable:
    """原型的 `timetable.json` 补上 `kind` 那一列。那一批全是握拳平移。"""
    return Timetable(
        [
            Slot(
                rep=int(r["rep"]),
                kind=FIST_MOVE,
                key=str(r.get("key", "")),
                expect=r.get("expect"),
                seg=str(r["seg"]),
                t0=float(r["t0"]),
                t1=float(r["t1"]),
                text=str(r.get("text", "")),
                label=str(r.get("label", "")),
            )
            for r in raw
        ]
    )


def derive_alignment(
    records: list[FrameRecord], timetable: Timetable, demo: DemoConfig, size: tuple[int, int]
) -> Alignment | None:
    """从数据本身倒推一份启动对准的结果。那一轮还没有这一步。

    手宽取**整段里握拳类手框宽的中位数**(对准量的就是这个);区域中心取**准备段里**
    拳头的中位位置——准备段说的是「握拳,放在某处,停住」,所以那时手在哪儿,就相当于
    对准时拳头停在哪儿。

    一个拳头都没有就返回 `None`:这一份转不了,说清楚,别悄悄编一个出来。
    """
    preps = [(s.t0, s.t1) for s in timetable if s.seg == "prep"]
    widths: list[float] = []
    centers: list[tuple[float, float]] = []
    for record in records:
        for hand in record.hands:
            if hand.label not in FIST_LIKE_LABELS:
                continue
            widths.append(hand.width)
            if any(t0 <= record.t < t1 for t0, t1 in preps):
                centers.append(hand.center)
    if not widths or not centers:
        return None

    hand_width = float(median(widths))
    cx = float(median(c[0] for c in centers))
    cy = float(median(c[1] for c in centers))
    region_w, region_h = demo.vision.detection_region_px
    detection = Region(
        round(cx - region_w / 2), round(cy - region_h / 2), region_w, region_h
    ).clamped_to(*size)
    zone = round(hand_width * demo.vision.start_zone_hand_widths)
    start_zone = Region(
        round(cx - zone / 2), round(cy - zone / 2), max(1, zone), max(1, zone)
    ).clamped_to(*size)
    return Alignment(
        detection_region=detection,
        start_zone=start_zone,
        hand_width=hand_width,
        # 那一轮只送整幅画面,所以两处量到的是同一个数
        full_frame_width=hand_width,
        too_close=hand_width > demo.vision.too_close_px,
        t=0.0,
    )


def _write(
    out: Path,
    records: list[FrameRecord],
    timetable: Timetable,
    meta: Meta,
    *,
    video_src: Path | None = None,
) -> str:
    """把转好的东西写下去。返回「带不带录像」那半句。"""
    out.mkdir(parents=True, exist_ok=True)
    with (out / FRAMES_NAME).open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(
                json.dumps(record.to_json(), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    timetable.save(out / "timetable.json")
    (out / "meta.json").write_text(
        json.dumps(meta.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if video_src is not None and video_src.exists():
        # 原样复制:原型存的也是翻转之前的画面,格式一致
        shutil.copy2(video_src, out / VIDEO_NAME)
        return ",带录像"
    return ""


def import_fist(
    src: Path, out: Path, demo: DemoConfig, *, video: bool = True
) -> tuple[Path | None, str]:
    """转一份握拳平移原型的录制。返回转到哪儿和一行说明。"""
    for name in (FRAMES_NAME, "timetable.json", "meta.json"):
        if not (src / name).exists():
            return None, f"{src.name}:少了 {name},跳过"

    old = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    size = _frame_size(old)
    records = convert_frames((src / FRAMES_NAME).read_text(encoding="utf-8").splitlines())
    timetable = convert_timetable(
        json.loads((src / "timetable.json").read_text(encoding="utf-8"))
    )
    alignment = derive_alignment(records, timetable, demo, size)
    if alignment is None:
        return None, f"{src.name}:准备段里一个拳头都没有,倒推不出检测区域,跳过"

    meta = Meta(
        plan=str(old.get("plan", src.name.split("-")[-1])),
        started=str(old.get("started", datetime.now().isoformat(timespec="seconds"))),
        frame_size=size,
        send=FULL_FRAME,  # 那一轮只送整幅画面,没有检测区域
        video_t0=video_t0_of(records),
        alignment=alignment_to_json(alignment),
        alignment_source="derived",
        vision=_vision_json(demo),
        versions=old.get("versions") or {},
        note=(
            f"2026-09-17 握拳平移原型 {src.name} 转过来的。"
            "**坐着**做,握拳平移从肩膀出发,录的时候没有起点区、也没有启动对准;"
            "这里的检测区域和起点区是从准备段里拳头的位置倒推的,只送整幅画面。"
        ),
    )
    got_video = _write(
        out, records, timetable, meta,
        video_src=(src / VIDEO_NAME) if video else None,
    )
    return out, (
        f"{src.name} → {out.name}:{len(records)} 帧、{len(timetable)} 段提示,"
        f"手宽 {alignment.hand_width:.1f} px{got_video}"
    )


def free_timetable(stops: list[dict], segs: list[dict]) -> Timetable:
    """站着那一份的提示时间表:先 10 次叫停,再 4 段自由活动。

    叫停那 10 次在 `stops.json` 里(每次带准备段和「做!」段的时刻),自由活动 4 段
    在 `timetable.json` 里。两样拼成一张表,窗口的划法和自己录的一样。
    """
    slots: list[Slot] = []
    for item in stops:
        index = int(item["rep"])
        label = f"STOP ({index + 1}/{len(stops)})"
        slots.append(
            Slot(index, STOP_GESTURE, "palm", "stop", PREP,
                 float(item["prep"]), float(item["go"]), STOP_PREP, label)
        )
        slots.append(
            Slot(index, STOP_GESTURE, "palm", "stop", GO,
                 float(item["go"]), float(item["end"]), STOP_GO, label)
        )
    base = len(stops)
    for n, seg in enumerate(segs):
        slots.append(
            Slot(base + n, FREE, str(seg.get("key", "")), None, GO,
                 float(seg["t0"]), float(seg["t1"]), FREE_TEXT,
                 f"FREE {str(seg.get('key', '')).upper()}")
        )
    return Timetable(slots)


def import_free(
    src: Path, out_dir: Path, demo: DemoConfig, *, video: bool = True
) -> list[tuple[Path | None, str]]:
    """转站着自由活动加叫停那一份。**整幅画面和检测区域各转一份。**

    录的时候叫停段两路都跑过,事后又用录像把整条重跑了两遍
    (`replay-full.jsonl` / `replay-region.jsonl`),所以两路各有一份完整的逐帧日志。
    叫停要看整幅画面(执行中送的就是它,ADR-0008),误触发要看检测区域(空闲送的是
    它)——分成两份,各比各的,不互相盖。
    """
    need = ("stops.json", "timetable.json", "meta.json")
    for name in need:
        if not (src / name).exists():
            return [(None, f"{src.name}:少了 {name},跳过")]

    old = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    size = _frame_size(old)
    alignment = recorded_alignment(old)
    if alignment is None:
        return [(None, f"{src.name}:meta 里没有对准结果,跳过")]
    timetable = free_timetable(
        json.loads((src / "stops.json").read_text(encoding="utf-8")),
        json.loads((src / "timetable.json").read_text(encoding="utf-8")),
    )

    out = []
    for suffix, source_name, send in (
        ("full", "replay-full.jsonl", FULL_FRAME),
        ("region", "replay-region.jsonl", DETECTION_REGION),
    ):
        path = src / source_name
        if not path.exists():
            out.append((None, f"{src.name}:没有 {source_name},跳过这一路"))
            continue
        region = alignment.detection_region if send == DETECTION_REGION else None
        records = convert_frames(
            path.read_text(encoding="utf-8").splitlines(), region=region
        )
        # 每帧的间隔从现场那份接过来(离线重跑的快慢说明不了录制循环卡没卡)
        live = intervals_from(src / FRAMES_NAME)
        if live:
            records = [
                dataclasses.replace(r, interval_ms=live.get(r.index)) for r in records
            ]
        where = out_dir / f"{FREE_PREFIX}-{src.name}-{suffix}"
        meta = Meta(
            plan=f"free+stop({suffix})",
            started=str(old.get("started", "")),
            frame_size=size,
            send=send,
            video_t0=video_t0_of(records),
            alignment=alignment_to_json(alignment),
            alignment_source="recorded",
            vision=_vision_json(demo),
            versions=old.get("versions") or {},
            note=(
                f"2026-09-19 站着自由活动加叫停原型 {src.name} 的 {source_name} 转过来的。"
                "**站在操作位**上录的,对准结果是当时真对出来的;"
                f"这一份送的是{'整幅画面' if send == FULL_FRAME else '检测区域'}。"
                "10 次叫停提示 + 160 秒自由活动。"
            ),
        )
        got_video = _write(
            where, records, timetable, meta,
            # 两份指的是同一段录像,只复制到整幅画面那一份下,免得占两份磁盘
            video_src=(src / VIDEO_NAME) if (video and suffix == "full") else None,
        )
        out.append(
            (
                where,
                f"{src.name}/{source_name} → {where.name}:{len(records)} 帧、"
                f"{len(timetable)} 段提示,手宽 {alignment.hand_width:.1f} px{got_video}",
            )
        )
    return out


def recorded_alignment(old: dict) -> Alignment | None:
    """站着那一份 meta 里**真对出来的**检测区域、起点区和手宽。

    原型存的是 `[x1, y1, x2, y2]`,这边的 `Region` 是左上角加宽高,换一下。
    """
    region, zone, calib = old.get("region"), old.get("zone"), old.get("calib") or {}
    if not region or not zone or not calib.get("hw"):
        return None
    hand_width = float(calib["hw"])
    return Alignment(
        detection_region=_corners(region),
        start_zone=_corners(zone),
        hand_width=hand_width,
        # 那一轮的手宽是在整幅画面里量的(对准时还没切到检测区域)
        full_frame_width=hand_width,
        too_close=bool(old.get("too_close")),
        t=0.0,
    )


def _corners(box: list) -> Region:
    x1, y1, x2, y2 = (round(float(v)) for v in box)
    return Region(x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _frame_size(old: dict) -> tuple[int, int]:
    camera = old.get("camera") or {}
    actual = camera.get("actual") or camera.get("requested") or [1280, 720]
    return int(actual[0]), int(actual[1])


def _vision_json(demo: DemoConfig) -> dict:
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in vars(demo.vision).items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把原型那几批录制转成录制工具的格式")
    parser.add_argument("--fist-from", type=Path, default=FIST_SOURCE)
    parser.add_argument("--free-from", type=Path, default=FREE_SOURCE)
    parser.add_argument("--out", type=Path, default=RECORDINGS_DIR)
    parser.add_argument(
        "--only",
        choices=("fist", "free", "all"),
        default="all",
        help="只转哪一批。fist = 握拳平移(坐着),free = 自由活动加叫停(站着)",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="不复制录像(只改判定规则时用不上)"
    )
    parser.add_argument("--force", action="store_true", help="已经转过的也重转一遍")
    args = parser.parse_args(argv)

    demo = load_demo()
    video = not args.no_video
    done = missing = 0

    if args.only in ("fist", "all"):
        if not args.fist_from.exists():
            print(f"找不到 {args.fist_from},跳过握拳平移那批", file=sys.stderr)
            missing += 1
        else:
            for src in sorted(p for p in args.fist_from.iterdir() if p.is_dir()):
                out = args.out / f"{FIST_PREFIX}-{src.name}"
                if out.exists() and not args.force:
                    print(f"{out.name} 已经在了,跳过(要重转加 --force)")
                    continue
                path, line = import_fist(src, out, demo, video=video)
                print(line)
                done += path is not None

    if args.only in ("free", "all"):
        if not args.free_from.exists():
            print(f"找不到 {args.free_from},跳过自由活动加叫停那份", file=sys.stderr)
            missing += 1
        else:
            for src in sorted(
                p for p in args.free_from.iterdir() if p.is_dir() and "free" in p.name
            ):
                if not args.force and any(
                    (args.out / f"{FREE_PREFIX}-{src.name}-{s}").exists()
                    for s in ("full", "region")
                ):
                    print(f"{FREE_PREFIX}-{src.name}-* 已经在了,跳过(要重转加 --force)")
                    continue
                for path, line in import_free(src, args.out, demo, video=video):
                    print(line)
                    done += path is not None

    print(f"转了 {done} 份 → {args.out}。**只存本地,不提交、不分发。**")
    # 两批都找不到才算不成:本机没有这些数据是常事,相关测试会自己跳过
    return 1 if done == 0 and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

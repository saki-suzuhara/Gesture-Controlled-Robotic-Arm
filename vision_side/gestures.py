"""真识别器:上游 `dynamic_gestures` 的检测 + 跟踪 + 分类,两个 ONNX 模型(ADR-0002)。

上游代码在 `vision_side/dynamic_gestures/`(Apache-2.0,声明随代码一起),移植时带上了
三处修补,都在那份代码里直接改好,不用运行时补丁:

1. `ocsort/kalmanfilter.py` 的 `w / float(h)` 改成 `w / h`——numpy ≥ 2.4 起原写法在手
   跟丢几帧再匹配上时会崩;
2. 关掉 onnxruntime 的线程自旋(`session.intra_op.allow_spinning=0`)——不关的话读帧
   线程被饿住,画面里有手时只有 9.4–14.5 fps,关掉后稳定在 30 fps;
3. `provider_options` 为空时不传——本机只有 CPU,传空列表会刷一段 EP Error。

(都是 2026-09-17 原型在本机实测的,见 `notes/camera-prototype-results.md`。)

**上游自带的动态手势事件不用**(左滑、右滑、点击那些,ADR-0007):这里只取每一帧的
手框、姿势标签、跟踪编号和确认与否,握拳平移和姿势保持由我们自己判(第 03 票)。
"""

from __future__ import annotations

from pathlib import Path

from vision_side.dynamic_gestures.main_controller import MainController
from vision_side.dynamic_gestures.utils.enums import targets
from vision_side.frames import Frame
from vision_side.recognize import (
    UNKNOWN_LABEL,
    FrameResult,
    HandResult,
    Recognizer,
    Region,
    UniformWatch,
)

MODELS_DIR = Path(__file__).resolve().parent / "dynamic_gestures" / "models"
DETECTOR_MODEL = MODELS_DIR / "hand_detector.onnx"
CLASSIFIER_MODEL = MODELS_DIR / "crops_classifier.onnx"

DETECTOR_INPUT = (320, 240)
"""检测器要的输入大小。送进去的那块(整幅画面或检测区域)都缩到这么大。

1.2 m 处的拳头在整幅画面里只有约 21 px,缩到 320×240 之后更小;检测区域里约 34 px
—— 这就是检测区域存在的全部理由(ADR-0008)。
"""


class ModelsMissing(FileNotFoundError):
    """两个 ONNX 模型不在。模型跟代码一起提交,所以这只会发生在文件被删掉时。

    测试见到它就跳过,不算失败(spec「Testing Decisions」)。
    """


class GestureRecognizer(Recognizer):
    """真识别器。一个线程用(识别线程)。

    建起来要载两个 ONNX 模型,约几百毫秒,所以整个程序只建一次,别每帧新建。
    """

    def __init__(
        self,
        detector: Path = DETECTOR_MODEL,
        classifier: Path = CLASSIFIER_MODEL,
    ) -> None:
        missing = [str(p) for p in (detector, classifier) if not Path(p).exists()]
        if missing:
            raise ModelsMissing("找不到模型:" + "、".join(missing))
        self._ctrl = MainController(str(detector), str(classifier))
        self._uniform = UniformWatch()

    def recognize(self, frame: Frame, region: Region | None = None) -> FrameResult:
        """吃一帧,出这一帧的结果。

        `region` 是这一帧送进检测器的那块(坐标先贴到画面里),`None` 是整幅画面;
        手框一律换算回整幅画面的坐标。「画面均匀」算的是**整幅画面**,不是送进去的那块
        —— 检测区域里恰好一片均匀不等于摄像头没画面。
        """
        if region is not None:
            region = region.clamped_to(frame.width, frame.height)
            sub = region.crop(frame.image)
        else:
            sub = frame.image
        flat = self._uniform.update(frame.image, frame.t)
        self._ctrl(sub)
        return FrameResult(
            t=frame.t, hands=self._hands(region), flat=flat, region=region
        )

    def clear_tracks(self) -> None:
        """清掉上游的跟踪记录(换送法时,ADR-0008)。

        只丢轨迹,不把 `frame_count` 归零:归零会让接下来几帧的新轨迹白捡一个「已确认」
        (上游 `frame_count <= min_hits` 那一条),手还没跟稳就被当成确认的手。
        跟踪编号是类上的计数器,清完再出现的手自然拿到新的编号。

        **「画面均匀」数的那半秒不清**:它看的是整幅画面亮不亮,和送哪块、跟踪到哪只手
        没关系;清了反而会让换送法那一刻的识别中断晚半秒才报出来。
        """
        self._ctrl.tracks.clear()

    # ---- 内部 ----------------------------------------------------------

    def _hands(self, region: Region | None) -> tuple[HandResult, ...]:
        """把上游这一帧的跟踪结果翻成 `HandResult`。

        只取**这一帧真的看见了**的手:上游给跟丢的轨迹补一个空手框(`bbox=None`)接着
        往下算,那些不报上来。不能只看 `time_since_update`——整帧一个检测都没有时上游
        直接给每条轨迹补空手框就返回,既不推进 `time_since_update` 也不推进
        `frame_count`,拿它判断会把几秒前的手当成这一帧的。
        """
        ox, oy = region.origin if region is not None else (0, 0)
        ctrl = self._ctrl
        hands = []
        for track in ctrl.tracks:
            deque, tracker = track["hands"], track["tracker"]
            if not len(deque):
                continue
            hand = deque[-1]
            if hand.bbox is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in hand.bbox[:4])
            # 和上游 `MainController.update` 报出轨迹时用的是同一个条件
            confirmed = tracker.time_since_update < 1 and (
                tracker.hit_streak >= ctrl.min_hits or ctrl.frame_count <= ctrl.min_hits
            )
            hands.append(
                HandResult(
                    box=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                    label=label_name(hand.gesture),
                    # +1:上游报出去的编号也加了 1(MOT 那边要求正数)
                    track_id=int(tracker.id) + 1,
                    confirmed=bool(confirmed),
                )
            )
        return tuple(hands)


def label_name(gesture: object) -> str:
    """把上游的姿势标签(`targets` 里的下标)翻成名字;没有就是 `UNKNOWN_LABEL`。"""
    if gesture is None:
        return UNKNOWN_LABEL
    index = int(gesture)
    if 0 <= index < len(targets):
        return targets[index]
    return UNKNOWN_LABEL

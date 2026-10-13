"""识别结果:视觉一侧的第二个缝(spec「两个缝的形状」)。

吃一帧,出这一帧的结果——每只手的手框(**整幅画面的坐标**)、姿势标签、跟踪编号、
已确认与否;整帧带时间戳和一个「画面均匀」的判断。真识别器(`vision_side.gestures`)、
逐帧日志回放和测试里编好的结果都实现这同一组方法。

几条定下来的规矩:

- **送哪块画面是 `recognize()` 的入参,不是两个识别器**(ADR-0008):空闲送检测区域,
  执行中送整幅画面。识别器内部把送进去的那块缩到检测器要的 320×240,手框换算回整幅
  画面的坐标再报上来,上层因此不用管送的是哪块。
- **切换送法时由识别器清掉上游跟踪器的状态**(`clear_tracks()`):换了一块画面,同一只
  手的位置在送进去的那块里是跳变的,旧轨迹接着算会乱。触发判定那边清自己的记录。
- **「画面均匀」在这里算**,因为只有这一层有像素;算的是**整幅画面**,不是送进去的那块。
  上层拿编好的结果就能测,而这条规则本身可以从取帧那个缝喂编好的帧来测
  (`tests/test_flat.py`)。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import cv2
import numpy as np

from vision_side.frames import Frame

UNKNOWN_LABEL = "unknown"
"""上游这一帧没给出姿势标签时用的标签。"""

FLAT_SMALL = (160, 90)
"""算「画面均匀」之前先缩到这么大。缩完再算,快且不受单个坏像素影响。"""

FLAT_STD = 6.0
"""整幅画面的灰度标准差低于这个数就算均匀(全黑、纯色背景都在这一档)。"""

PLACEHOLDER_STD = 25.0
PLACEHOLDER_MEAN = 140.0
PLACEHOLDER_MEAN_TOL = 30.0
"""摄像头占位图(隐私开关关着时那张灰底加相机图标的图)的门槛:灰度均值在 140±30、
标准差低于 25。2026-09-17 握拳平移原型现场遇到过一次,整段录制作废,所以要认出来。"""

UNIFORM_FOR_S = 0.5
"""画面均匀持续多久才算「看不见」(spec「识别中断」)。和识别中断的 0.5 秒同一档。"""


@dataclass(frozen=True)
class Region:
    """画面里的一块矩形,整幅画面的坐标,左上角加宽高。

    检测区域和起点区都是这个东西(第 02 票放下它们);这一票只用它当 `recognize()`
    的入参——「这一帧送哪块画面」。
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"矩形的宽高要大于 0:{self.width}x{self.height}")

    @property
    def origin(self) -> tuple[int, int]:
        """左上角,也就是把手框换算回整幅画面时要加的偏移。"""
        return self.x, self.y

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def contains(self, point: tuple[float, float]) -> bool:
        """这个点在不在这一块里。边上算在里面。

        起点区拿它判「手框中心落在区里没有」(第 03 票);检测区域拿它判送进去的那块
        看不看得见某只手。
        """
        x, y = point
        return self.x <= x <= self.right and self.y <= y <= self.bottom

    def clamped_to(self, width: int, height: int) -> Region:
        """挪到画面里。超出画面时**贴着画面边缘放**,不缩小(spec「启动对准」)。

        画面比这块还小时才裁,那只会出现在换了分辨率的录像上。
        """
        w = min(self.width, width)
        h = min(self.height, height)
        x = max(0, min(self.x, width - w))
        y = max(0, min(self.y, height - h))
        return Region(x, y, w, h)

    def crop(self, image: np.ndarray) -> np.ndarray:
        """从整幅画面里切出这一块。切出来的是视图,不复制。"""
        return image[self.y : self.bottom, self.x : self.right]


@dataclass(frozen=True)
class HandResult:
    """这一帧看见的一只手。

    `box` 是手框 `(x1, y1, x2, y2)`,**整幅画面的坐标**(送检测区域时已经加过偏移)。
    `label` 是姿势标签的名字(`vision_side.dynamic_gestures.utils.enums.targets` 里那些,
    比 V 是 peace、竖拇指是 like),没有就是 `UNKNOWN_LABEL`。`track_id` 是跟踪编号,
    同一只手连续跟上时不变。`confirmed` 是上游认不认这条轨迹(连上几帧才算)。
    """

    box: tuple[int, int, int, int]
    label: str
    track_id: int
    confirmed: bool

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def width(self) -> float:
        """手框宽。手宽(HAND WIDTH)量的就是它。"""
        x1, _, x2, _ = self.box
        return float(x2 - x1)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return float(x2 - x1) * float(y2 - y1)


@dataclass(frozen=True)
class FrameResult:
    """一帧的识别结果。

    `hands` 只包含**这一帧真的看见了的手**:上游会给跟丢的轨迹补一个空手框接着往下算,
    那些不报上来。`flat` 是「画面已经均匀了约 0.5 秒」,识别中断判定看它。`region` 是
    这一帧送进检测器的那块,`None` 表示送的是整幅画面。
    """

    t: float
    hands: tuple[HandResult, ...] = ()
    flat: bool = False
    region: Region | None = field(default=None)

    def largest(self) -> HandResult | None:
        """画面里最大的那只**已确认**的手。没有就是 `None`。

        锁定之前交给判定的就是这一只(ADR-0010);两只手离镜头差不多远时会挑错,
        2026-09-17 原型实测 17 次里误拒 3 次,这是已知的、用锁定兜底的取舍。
        """
        confirmed = [h for h in self.hands if h.confirmed]
        if not confirmed:
            return None
        return max(confirmed, key=lambda h: h.area)


def is_flat(image: np.ndarray) -> bool:
    """单帧:这幅画面是不是一片均匀(含摄像头占位图)。

    门槛是 2026-09-17 原型在本机实测定下的:缩到 160×90 转灰度,标准差低于 6 算均匀
    (全黑、纯色),或者均值 140±30、标准差低于 25 —— 那是隐私开关关着时的占位图。
    """
    small = cv2.resize(image, FLAT_SMALL)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sd = float(small.std())
    mean = float(small.mean())
    placeholder = sd < PLACEHOLDER_STD and abs(mean - PLACEHOLDER_MEAN) < PLACEHOLDER_MEAN_TOL
    return bool(sd < FLAT_STD or placeholder)


class UniformWatch:
    """数「画面均匀」持续了多久。单帧均匀不算,要连着约 0.5 秒才算。

    一帧亮回来就从头数:占位图和全黑会一直均匀,而正常画面里偶尔一帧过曝不该算。
    """

    def __init__(self, for_s: float = UNIFORM_FOR_S) -> None:
        self.for_s = for_s
        self._since: float | None = None

    def reset(self) -> None:
        self._since = None

    def update(self, image: np.ndarray, t: float) -> bool:
        """喂一帧(**整幅画面**),返回这一刻算不算「画面均匀」。"""
        if not is_flat(image):
            self._since = None
            return False
        if self._since is None:
            self._since = t
        return (t - self._since) >= self.for_s


class Recognizer(abc.ABC):
    """识别结果接口(RECOGNIZER)。

    一个线程用(识别线程,spec「线程」)。
    """

    @abc.abstractmethod
    def recognize(self, frame: Frame, region: Region | None = None) -> FrameResult:
        """吃一帧,出这一帧的结果。

        `region` 是这一帧送进检测器的那块,`None` 是整幅画面。手框一律按**整幅画面的
        坐标**报上来。
        """

    @abc.abstractmethod
    def clear_tracks(self) -> None:
        """清掉跟踪记录。之后出现的手会拿到新的跟踪编号。

        换送法(整幅画面 ↔ 检测区域)时调一次(ADR-0008)。
        """

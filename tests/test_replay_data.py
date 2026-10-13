"""拿本机真有的那几份录制跑回放,和上一次逐项比(第 05 票)。

**本机没有这些数据时全部跳过,不算失败**(spec「回放测试」最后一条):录像里有用户
本人,`data/` 整个被 git 忽略,别人拉下代码是没有的。

要数据的话:

    uv run -m vision_side.scripts.import_runs     # 把原型那几批转过来
    uv run -m vision_side.scripts.record --plan dirs   # 或者自己录一段

只走**逐帧日志**那一路:不载模型、不解码画面,整套跑完一两秒。视频那一路(重跑识别)
由 `uv run -m vision_side.scripts.replay --video` 手动跑,一次要几分钟,不放进自动测试。
"""

from __future__ import annotations

import pytest

from robot_side.config import load_demo
from vision_side.recordings import find_runs
from vision_side.replay import Report, replay_log, result_path

DEMO = load_demo()
RUNS = find_runs()


def ids(runs) -> list[str]:
    return [r.name for r in runs]


pytestmark = pytest.mark.skipif(
    not RUNS, reason="本机没有录制数据(`data/recordings/` 里是空的);这不算失败"
)


@pytest.fixture(scope="module", params=RUNS, ids=ids(RUNS))
def run(request):
    return request.param


@pytest.fixture(scope="module")
def reports():
    """每一份跑一遍,整个模块共用。跑两遍是为了验「同样的代码,结果一致」。"""
    return {r.name: replay_log(r, DEMO) for r in RUNS}


def test_跑得完并且认得出时间表(run, reports):
    report = reports[run.name]
    assert report.frames > 0, "一帧都没读到"
    assert report.reps, "时间表里一条提示都没有"


def test_同一份数据跑两遍逐条一致(run, reports):
    """回放测试的地基:同一段数据、同样的代码,结果应当**逐条**一致。

    不成立的话,「变差就报出来」报的就全是噪声,这一票也就白做了。
    """
    again = replay_log(run, DEMO)
    before = reports[run.name]
    assert again.totals() == before.totals()
    assert [r.verdict for r in again.reps] == [r.verdict for r in before.reps]


def test_和上一次比没有一项变差(run, reports):
    """**只要有一项变差就报出来,不设容差**(spec)。

    上一次的结果跟着数据走(录制那个文件夹里的 `replay-log.json`)。还没有基准时
    跳过——第一次跑本来就没得比,存基准是 `--save` 那一步的事。
    """
    path = result_path(run, "log")
    if not path.exists():
        pytest.skip(f"{run.name} 还没有上一次的结果,先跑 `... replay --save` 存一份")
    worse = reports[run.name].compare(Report.load(path))
    assert not worse, "变差的项:" + "、".join(c.line() for c in worse)


def test_逐条的结局也没变(run, reports):
    """总数一样、逐条不一样也要看得见:一次漏认换成一次方向判错会被总数盖住。"""
    path = result_path(run, "log")
    if not path.exists():
        pytest.skip(f"{run.name} 还没有上一次的结果")
    changed = reports[run.name].per_rep_changes(Report.load(path))
    assert not changed, "结局变了的:" + "、".join(c.name for c in changed)


def test_录的时候送的画面和回放要的对得上(run, reports):
    """对不上就是拿另一块画面的结果去套判定,等于凭空编数(`LogRecognizer`)。"""
    assert "送的画面和这次要的不一样" not in reports[run.name].note

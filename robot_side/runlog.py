"""日志:终端一份,本地文件一份。**写出去的活放在单独一条线程上。**

预览窗口是全屏置顶的,终端会被它挡住;出了已停止或者误触发,事后要能查出是什么
时候、因为什么停的(`docs/grill-decisions.md` 第一节第 4 组)。

文件落在 `data/logs/`,每次运行一个文件。整个 `data/` 已经在 `.gitignore` 里,
不会误提交。

**为什么要多一条线程**:`StreamHandler` 和 `FileHandler` 都是同步写,写在**调
`log.info()` 的那条线程上**。Windows 控制台写得慢的时候(输出多、窗口在重画),
谁记日志谁被按住——而记日志的正是主线程(状态逻辑、预览窗口)和执行线程(每段运动
报一次)。2026-09-22 站着验收时,后半程本该 1 秒的运动段变成 1–2 秒,卡顿正好落在
两条相邻日志之间:日志的时间戳是**建记录**时打的,所以上一条写得慢,下一条的时间
就往后推。

改法是标准库那一套:`QueueHandler` 把记录丢进队列就返回,`QueueListener` 在自己的
线程上慢慢写。时间戳不受影响(建记录时就定了),谁也不再被 I/O 按住。
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import sys
from datetime import datetime
from pathlib import Path

from robot_side.config import ROOT

LOG_DIR = ROOT / "data" / "logs"
"""运行日志放这里。`data/` 整个被 git 忽略。"""

FILE_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
TERMINAL_FORMAT = "%(asctime)s %(message)s"
TIME_FORMAT = "%H:%M:%S"


def start(name: str = "run", *, level: int = logging.INFO) -> tuple[logging.Logger, Path]:
    """开一份运行日志,返回日志器和落地的文件。

    重复调同一个名字不会叠加输出:每次先把上一次挂的东西摘掉。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{name}.log"

    log = logging.getLogger(f"robot_side.{name}")
    log.setLevel(level)
    log.propagate = False
    _detach(log)

    to_file = logging.FileHandler(path, encoding="utf-8")
    to_file.setFormatter(logging.Formatter(FILE_FORMAT, TIME_FORMAT))

    to_terminal = logging.StreamHandler(sys.stdout)
    to_terminal.setFormatter(logging.Formatter(TERMINAL_FORMAT, TIME_FORMAT))

    # 记日志的线程只往队列里放,真写出去的是下面这条线程(见模块说明)
    records: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    listener = logging.handlers.QueueListener(
        records, to_file, to_terminal, respect_handler_level=False
    )
    listener.start()
    handler = logging.handlers.QueueHandler(records)
    handler.listener = listener  # `close()` 要拿它来收尾
    log.addHandler(handler)

    log.info("日志:%s", path)
    return log, path


def close(log: logging.Logger) -> None:
    """收掉这份日志的输出。程序退出前调,好让写日志那条线程把剩下的写完、文件落盘。"""
    _detach(log)


def _detach(log: logging.Logger) -> None:
    """把挂着的输出都摘掉。挂过写日志线程的,先等它把队列里的写完。"""
    for handler in list(log.handlers):
        log.removeHandler(handler)
        listener = getattr(handler, "listener", None)
        if listener is not None:
            # 先停线程再关文件:反过来的话,队列里没写完的那几条会写进已经关掉的文件
            listener.stop()
            for downstream in listener.handlers:
                downstream.close()
        handler.close()

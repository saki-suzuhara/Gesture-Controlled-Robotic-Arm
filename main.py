"""正式程序的入口:站着做手势,机械臂做对应的预设动作。

    uv run python main.py                      # 空实现,不开 RoboDK
    uv run python main.py --arm jaka           # RoboDK 仿真
    uv run python main.py --arm nova5 --debug  # 再加上调试按键
    uv run python main.py --arm ../jaka-real/jaka.toml --real   # 连真机(配置副本放仓库外)

真东西在 `vision_side/app.py`(启动顺序和主循环都在那儿);这里只是个壳,好让
`python main.py` 一写就能跑。
"""

import sys

from vision_side.app import main

if __name__ == "__main__":
    sys.exit(main())

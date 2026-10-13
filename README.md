# 手势指挥机械臂

用笔记本摄像头识别手势,让机械臂做几个预设动作。术语见 `CONTEXT.md`,取舍见 `docs/adr/`,
每条决定的来龙去脉见 `docs/grill-decisions.md`,演示前要核对的见 `docs/demo-checklist.md`。

视觉一侧(摄像头、识别、触发判定、预览窗口)和机器人一侧(经 RoboDK 驱动机械臂)都已接上。
**默认只动仿真**,要连真机得在启动时明确加 `--real`。

## 跑起来

```bash
# 正式程序:站到操作位,把拳头举起来对准,然后做手势
uv run python main.py                                   # 空实现,不开 RoboDK
uv run python main.py --arm jaka                        # RoboDK 仿真(JAKA;--arm nova5 换另一台)
uv run python main.py --arm jaka --debug                # 再加调试按键:方向键、F、B、P、L、O、K
uv run python main.py --arm ../jaka-real/jaka.toml --real   # 连真机(IP 不进仓库,配置副本放仓库外)

# 只认手势、不接机械臂:屏幕上一件件带你做完全部触发手势加一次叫停
uv run -m vision_side.scripts.triggers

# 录一段手势、离线回放(录像只存本地 data/recordings/,不提交)
uv run -m vision_side.scripts.record --plan dirs        # 还有 depth、poses、stop、free
uv run -m vision_side.scripts.replay                    # 逐帧日志回放,和上一次逐项比

# 机器人一侧的小脚本
uv run python -m robot_side.scripts.build_station       # 建工作站(一次性,可以重复跑)
uv run python -m robot_side.scripts.action_demo --arm jaka   # 仿真里把预设动作跑一遍
uv run python -m robot_side.scripts.speed_check --arm ../jaka-real/jaka.toml --real   # 真机上核对速度受控
uv run python -m robot_side.scripts.set_standby --arm ../jaka-real/jaka.toml --real   # 读示教器上摆的姿态当待命位姿

# 测试(开着 RoboDK 时会多跑仿真那几份,慢;没开就跳过)
uv run pytest
```

## 打包

```bash
uv run python packaging/build_release.py     # 两份都打,放 dist/;只打一份加 --only source / portable
```

- **源码包** `RobotCameraCombination-<日期>-<提交>-source.zip`:`git archive` 打的,和仓库里一样,照上面用 uv 跑。
- **便携版** `RobotCameraCombination-<日期>-<提交>-win64.zip`:自带 Python 3.13 和全部依赖(版本照
  `uv.lock`),解压后双击 `.bat` 就跑,别的 Windows 电脑什么都不用装;连 RoboDK 仿真的入口要那台电脑
  另装 RoboDK。不给连真机的双击入口。包里的说明是 `packaging/使用说明.txt`。

两份都只打已提交的 HEAD,工作区有改动会先报出来。

## 手势

| 触发手势 | 怎么做 | 预设动作 |
|---|---|---|
| FIST MOVE LEFT / RIGHT | 握拳在起点区停住,朝左 / 右移过约 1.5 个拳头 | 左 / 右挪 150 mm |
| FIST MOVE UP / DOWN | 同上,朝上 / 下 | 上 / 下挪 100 mm |
| FIST MOVE FORWARD | 握拳停住,往镜头推出去约 20 cm 停住 | 远离你挪 100 mm |
| FIST MOVE BACK | 握拳停住,像敲桌子那样往下点两下 | 朝你挪 100 mm |
| HOLD PEACE(比 V) | 在起点区保持约 1 秒 | 画方块 |
| HOLD LIKE(竖拇指) | 同上 | 原地点头 |
| HOLD OK | 同上 | 鞠躬 |
| HOLD ROCK(食指加小指) | 同上 | 回待命(和按 H 一样) |

张开手掌朝镜头是叫停,机械臂在动时画面里任何人做都算;笔记本上按空格也是(窗口不在前台也行)。
对应关系在 `config/demo.toml` 里改,不用改代码。

## 配置

两份 TOML,每次启动读一遍并写进 RoboDK——**在 RoboDK 界面上改过的数,下次启动会被覆盖**。

- `config/arms/*.toml`:一台机械臂一份。工作站里这台机械臂的名字、连接方式、工作台坐标系、
  待命位姿、指针偏移、速度参数。换机械臂就换这一份。
- `config/demo.toml`:各台共用。预设动作的路点和速度、默认姿态、触发手势的对应、活动范围、
  视觉一侧的判定参数,以及和机械臂无关的运行参数。

单位:位置 mm、角度 °、速度 mm/s、停顿秒。

路点是**法兰**在工作台坐标系里的位姿,按相对出发点(停靠点)的偏移写(ADR-0013、ADR-0014);
控制器保持工具 0,指针偏移只作参考。

### 速度

每个预设动作一个速度:直线段法兰按它走,改姿态的那几段走关节运动,关节速度按指针尖端也走这么快
现算。仿真里就是这个速度。

**真机上要机械臂配置里开了 `[speed] program_sets_speed = true` 才生效**,实际速度 = 配置里的
速度 × 示教器上的速度倍率;没开时真机快慢全看示教器,改配置里的速度不起作用。

来龙去脉(ADR-0011 的两次更新):2026-09-20 第一次上 JAKA 时,经 RoboDK 设速度的四条通道全部
无效;2026-09-23 查出原因是 RoboDK 的 JAKA 驱动每收到一行指令都把速度重置成 100,于是包了一层
驱动(`robot_side/drivers/jaka_keep_speed.py`)。**这一层还没上真机验过**:上真机先跑
`speed_check`,受控了再在仓库外那份配置里开 `program_sets_speed`。

## 代码

机器人一侧(`robot_side/`):

- `arm.py` —— 机械臂接口,状态逻辑和机械臂之间唯一的缝;`robodk_arm.py` 是它的 RoboDK 实现
  (仿真和真机共用一份),`noarm.py` 是不开 RoboDK 时的空实现。
- `config.py` —— 两份 TOML 的读取和校验;写错的东西在启动时就报出来。
- `motion.py` —— 把路点算成一段段运动(纯计算)。
- `reach.py`、`startup.py`、`range_cache.py` —— 可达性检查、启动时的「按配置写入 → 可达性检查
  → 回待命」,连真机前把活动范围全查一遍并存下来。
- `state.py` —— 状态逻辑(空闲、冷却、执行中、回待命、已停止)。
- `executor.py` —— 逐段下发:一次只发一段,发之前先看「已叫停」标记。
- `stop.py` —— 叫停:按键线程 + 「已叫停」标记 + 机械臂接口的「停止」(ADR-0012)。
- `real.py`、`guard.py` —— 连真机:查上电使能、热身、每段等多久、下发前的护栏。
- `standby.py` —— 选待命位姿:只转 J6 对上默认姿态。
- `drivers/jaka_keep_speed.py` —— 包过的 JAKA 驱动,让程序设的速度传到控制柜。
- `preview.py` —— 预览窗口。`station.py` —— 按布置建出工作站。

视觉一侧(`vision_side/`):

- `frames.py`、`recognize.py`、`gestures.py` —— 取帧、识别结果两个缝,以及 `dynamic_gestures` 那一份实现。
- `align.py` —— 启动对准:放下检测区域和起点区、量手宽。
- `triggers.py` —— 触发判定:握拳平移(含往前推、敲两下)、姿势保持、停止手势。
- `loop.py`、`app.py` —— 识别线程和正式程序的启动顺序、主循环(`main.py` 只是个壳)。
- `overlay.py` —— 画面叠加和手势图例。
- `prompts.py`、`record.py`、`recordings.py`、`replay.py` —— 录制工具、提示时间表、回放测试。

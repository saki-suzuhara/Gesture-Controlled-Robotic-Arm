# 交接:RoboDK 叫停原型

写于 2026-09-19。用户用中文交流,请用中文回复。
项目根目录:`D:\Zetian\MyCourses\RobotCameraCombination`(下文路径都相对于它)。

同一天还交出了两份原型交接,由别的 session 做:两台共用路点原型、站着自由活动加叫停原型(同一目录下的 `handoff-waypoint-reach-prototype-2026-09-19.md`、`handoff-standing-free-activity-prototype-2026-09-19.md`)。**共用路点原型也要用 RoboDK,两个不要同时跑。**

## 本 session 只回答一个问题

**RoboDK 仿真里,运动用非阻塞方式发出、执行线程查 `Busy()` 时,从另一个线程叫停,能不能立刻打断?同一条连接叫停和另开一条连接叫停,哪种可靠?**

不需要用户做动作,只要用户同意启动 RoboDK、用他的许可证。

问题、做法和过关标准都已定,本文不重复,都在 `docs/grill-decisions.md` 里:

| 要看什么 | 在哪里 |
|---|---|
| 问题、为什么光靠对话定不了、做法、过关标准 | 第一节第 3 组「RoboDK 叫停原型(第三十五轮定下,还没做)」 |
| 为什么现在做、结论用在哪 | 第一节第 1 组「跑 to_spec、开写机器人一侧之前,先开新 session 做两个仿真原型……」;同组「机器人一侧的代码按这个顺序写」 |
| 叫停在 API 这一层的默认做法(两种都要试) | 第二节「叫停在 RoboDK API 这一层的做法」 |
| 0.5 秒从哪来 | 第一节第 3 组「仿真阶段怎样算跑通,也是 spec 的验收标准」 |
| 三线程、叫停从主线程调 | 第一节第 4 组「程序分三个线程」「叫停(手势或空格)时,主线程直接调用机械臂接口的「停止」……」 |
| RoboDK 的调研:连接的锁、多线程要各用各的连接、仿真速度默认 5 倍 | 第四节「RoboDK 调研(2026-09-19)」 |
| 代码放哪、结果文件名、分支名;不要和另一个 RoboDK 原型同时跑 | 第二节「三个原型的代码、结果和分支」「两个 RoboDK 原型不要同时跑」 |
| 都不过关时,回到拷问 | 第三节「第三十五轮之后」 |

**过关标准已定,不要按结果改。**真要改,先问用户。

## 先读这些

- `docs/grill-decisions.md`:见上表。开头的「当前状态速览」说明了整体进度。
- `docs/adr/0011-robodk-for-robot-side-jaka-first.md`:为什么机器人一侧改用 RoboDK。Consequences 里「叫停」一条就是本题。
- `CONTEXT.md`:术语和英文名。本次重点:仿真、叫停、执行中、工作站文件、机械臂配置。
- 之前的原型都是摄像头原型,和本题内容无关,但可以照它们的做法写结果、提交、交回拷问。结论在各 `prototype/` 分支的提交说明里,用 `git log --format=%B -1 <分支>` 查看。最近一个是 `prototype/pre-vision-checks`(8fdec4e),它交回拷问的文档是 `handsoff/handoff-grill-after-pre-vision-checks-2026-09-17.md`。

## 可以直接用的东西

- **RoboDK 5.9.0:**装在 `C:\RoboDK`,用户有许可证。
- **Python API:**RoboDK 自带的 `robodk` 包在 `C:\RoboDK\Python\robodk\`,纯 Python。把 `C:\RoboDK\Python` 加进 `sys.path` 就能用,不用装包。要用的调用都在 `robolink.py` 里:
  - `Robolink()`:RoboDK 没开时会自己启动;
  - `AddStation`、`AddFile`、`CloseStation`、`setSimulationSpeed`;
  - `Item.MoveL(..., blocking=False)`、`Item.Busy()`、`Item.Stop()`、`Item.Pose()`、`Item.Joints()`。
- **机械臂模型:**用户下载的 `model/JAKA-Zu20.robot`。只读,不要改、不要挪。
- **Python:**`.scratch/camera-prototype/.venv/Scripts/python.exe`(3.13)。

## 开工前必须先征得用户同意

- **启动 RoboDK、用许可证。**先问用户 RoboDK 现在开着没有,里面有没有没存的工作站。
- **下载、安装:**这次不需要。真要装(比如 PyPI 上的 `robodk` 6.0.1),先说明要装什么,再问。
- **写结果文件、提交代码:**见「做完以后」。

## 注意

- **只动自己新建的工作站。**
  - 用 `AddStation` 新建,不要动 RoboDK 里原有的工作站;
  - `CloseStation` 不提示保存,只在当前工作站是自己建的那个时调用;
  - 工作站不用存盘;要存,只存在本原型的目录下。
- **仿真速度设成 1 倍。**默认 5 倍,时间会全错。
- **计时:**从调 `Stop()` 那一刻算起,到 `Busy()` 变成 0 为止;同时记下停住时沿直线走了多远。停在终点,就算没打断。
- **先定好,再跑,不按结果调:**150 mm、50 mm/s、1 秒后叫停、每种做法 10 次、0.5 秒。
- **只需要一台机械臂:**本题不用桌子和指针,放不放不影响叫停。
- **不要在 main 的工作区里切到任何 `prototype/` 分支。**原型文件在 main 上被忽略、在分支里被跟踪,切回 main 时 git 会把它们从磁盘上删掉。读分支内容用 `git show`、`git log`。
- **不要改这几个文件:**`CONTEXT.md`、`docs/adr/`、`docs/grill-decisions.md`、`docs/demo-checklist.md`。
  - 结论带回拷问 session,由它吸收;
  - 这几个文件在 main 上有还没提交的改动,不要提交,也不要还原;
  - `model/` 在 main 上没被 git 跟踪,也别动。
- **正式程序的决定不适用于原型:**不要动 `pyproject.toml`、`main.py`、`.gitignore`,不要建 `robot` 分支。
- **结论要写清局限:**
  - 只在仿真里,不代表真机经驱动叫停要多久;
  - 只试了 RoboDK 5.9.0;
  - 只试了一种运动(直线)。
- **环境:**
  - Windows 11,主要用 PowerShell,也可以用 Git Bash;
  - Git Bash 里没有 `python` 命令,用上面那个 venv 的完整路径;
  - 在 Bash 里 `cd` 会改掉会话的工作目录,尽量用绝对路径。

## 做完以后

1. 出一张表:两种做法各 10 次,每次的停住用时和停下位置。按过关标准下结论。
2. 列出被证伪或得到证实的假设,比如「同一条连接上的 `Stop()` 会被卡住」。对相关决策给出「维持」或「调整」的建议,每条附数据。相关的是第二节「叫停在 RoboDK API 这一层的做法」,以及验收标准里的 0.5 秒。
3. 征得同意后,把完整结果写进 `notes/robodk-stop-prototype-results.md`。
4. 用户要求提交时,用临时 worktree,把代码提交到新分支 `prototype/robodk-stop`。只提交 `.scratch/robodk-prototype-1/` 下的代码,不提交工作站文件、模型和日志。
5. 写一份带回拷问 session 的交接文档,存到系统临时目录,写完把绝对路径单独打一行给用户。可以参考 `handsoff/handoff-grill-after-pre-vision-checks-2026-09-17.md` 的写法。

## 用户的习惯

- **提问方式:**给选项并标出推荐;用户只回答了部分题目时,没回答的题不要当作已确认。
- **原型要小:**一轮只回答一个问题。
- **结论要有数据,并写清局限。**
- **先问再动手:**写文件、提交、启动软件、装东西之前先问。
- **用户可能临时要求重测或改条件:**照做,但在结果里分开记,写清改了什么。

## 建议使用的技能

- `mattpocock-skills:prototype`:一次性原型,只回答一个问题。它分 LOGIC、UI 两条路,都不太对得上这种「跑一下,看 API 怎么表现」的原型;照它的通用规则做:用完即弃、一条命令能跑、把状态打印出来、做完提交到原型分支。
- `mattpocock-skills:diagnosing-bugs`:叫停表现异常、结果和预期对不上时用。
- `mattpocock-skills:research`:要查 RoboDK 文档或论坛、核实 API 行为时用。
- `mattpocock-skills:domain-modeling`:沿用 `CONTEXT.md` 的术语;结果如果要求改术语,带回拷问 session。
- `mattpocock-skills:grilling`:两种做法都不过关、需要回到拷问时用。
- `mattpocock-skills:handoff`:做完后写带回拷问 session 的交接文档。

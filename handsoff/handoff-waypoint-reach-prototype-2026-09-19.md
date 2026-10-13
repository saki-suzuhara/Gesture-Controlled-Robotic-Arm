# 交接:两台共用路点原型

写于 2026-09-19。用户用中文交流,请用中文回复。
项目根目录:`D:\Zetian\MyCourses\RobotCameraCombination`(下文路径都相对于它)。

同一天还交出了两份原型交接,由别的 session 做:RoboDK 叫停原型、站着自由活动加叫停原型(同一目录下的 `handoff-robodk-stop-prototype-2026-09-19.md`、`handoff-standing-free-activity-prototype-2026-09-19.md`)。**叫停原型也要用 RoboDK,两个不要同时跑。**

## 本 session 只回答一个问题

**同一套路点(同一个工作台坐标系、不加平移),JAKA Zu 20 和 Nova 5 是不是都够得着、每段直线都走得通?**

不需要用户做动作,只要用户同意启动 RoboDK、用他的许可证。

问题、做法和能定下的事都已定,本文不重复,都在 `docs/grill-decisions.md` 里:

| 要看什么 | 在哪里 |
|---|---|
| 问题、为什么光靠对话定不了、做法、能定下的事 | 第一节第 3 组「两台共用路点原型(第三十五轮定下,还没做)」 |
| 为什么现在做、结论用在哪 | 第一节第 1 组「跑 to_spec、开写机器人一侧之前,先开新 session 做两个仿真原型……」 |
| 六个预设动作的尺寸和速度 | 第一节第 3 组「第一版动作的初始尺寸」 |
| 默认姿态的初始角度 | 第二节「默认姿态的初始角度约为 (180°, 0°, 0°)……」 |
| 工作站的摆法、仿真用的占位数 | 第一节第 3 组「工作站里,机械臂放在离地面 0.5 m 高的桌子上」;第二节「工作站:两台各放一张……」「仿真用的占位数」 |
| 可达性检查加上直线段、两台都跑 | 第一节第 4 组「可达性检查加上直线段」「仿真跑通时两台都跑」 |
| 两台的臂展、型号更正(JAKA 是 Zu 20) | 第四节「RoboDK 调研(2026-09-19)」的「模型」一条;第一节第 4 组「(JAKA)型号是 Zu 20,不是 Zu 12」 |
| 代码放哪、结果文件名、分支名;不要和另一个 RoboDK 原型同时跑 | 第二节「三个原型的代码、结果和分支」「两个 RoboDK 原型不要同时跑」 |
| 怎么扫都不行时,回到拷问 | 第三节「第三十五轮之后」 |

## 先读这些

- `docs/grill-decisions.md`:见上表。开头的「当前状态速览」说明了整体进度。
- `docs/adr/0003-waypoints-in-worktable-frame.md`:本题检验的就是它的前提「换机械臂只重新标定、路点照常复用」。
- `docs/adr/0011-robodk-for-robot-side-jaka-first.md`:为什么机器人一侧改用 RoboDK。
- `CONTEXT.md`:术语和英文名。本次重点:路点、工作台坐标系(各轴方向)、默认姿态、待命位姿、指针尖端、指针偏移、预设动作、可达性检查、工作站文件、仿真。
- 之前的原型都是摄像头原型,和本题内容无关,但可以照它们的做法写结果、提交、交回拷问。结论在各 `prototype/` 分支的提交说明里,用 `git log --format=%B -1 <分支>` 查看。最近一个是 `prototype/pre-vision-checks`(8fdec4e),它交回拷问的文档是 `handsoff/handoff-grill-after-pre-vision-checks-2026-09-17.md`。

## 可以直接用的东西

- **RoboDK 5.9.0:**装在 `C:\RoboDK`,用户有许可证。
- **Python API:**RoboDK 自带的 `robodk` 包在 `C:\RoboDK\Python\robodk\`,纯 Python。把 `C:\RoboDK\Python` 加进 `sys.path` 就能用,不用装包。要用的调用都在 `robolink.py` 里:
  - `Robolink()`:RoboDK 没开时会自己启动;
  - `AddStation`、`AddFile`、`CloseStation`;
  - `Item.setPoseTool`、`Item.setPoseFrame`、`Item.SolveIK`(无解时返回空)、`Item.MoveL_Test`、`Item.Joints`、`Item.JointLimits`。
- **机械臂模型:**用户下载的 `model/JAKA-Zu20.robot`、`model/Dobot-Nova-5.robot`。只读,不要改、不要挪。
- **Python:**`.scratch/camera-prototype/.venv/Scripts/python.exe`(3.13)。

## 开工前必须先征得用户同意

- **启动 RoboDK、用许可证。**先问用户 RoboDK 现在开着没有,里面有没有没存的工作站。
- **下载、安装:**这次不需要。真要装,先说明要装什么,再问。
- **写结果文件、提交代码:**见「做完以后」。

## 注意

- **只动自己新建的工作站。**
  - 用 `AddStation` 新建,不要动 RoboDK 里原有的工作站;
  - `CloseStation` 不提示保存,只在当前工作站是自己建的那个时调用;
  - 工作站不用存盘;要存,只存在本原型的目录下。这不是正式的工作站文件,正式的开写时另建。
- **文档里没写死的,开跑前自己定好并写进结果,不按结果调:**
  - **原地点头绕哪根轴:**文档只写了「上下各转 15°、往复 2 次」。建议理解为以指针尖端为中心、绕工作台坐标系的 X 轴转,尖端不动;
  - **各段用直线还是关节运动:**建议预设动作里的移动都按直线段查;
  - **扫哪些待命点:**第一个待命点不行时,前后、高低各扫几档,范围和步长开跑前定好。
- **逆解要取同一组构型:**`SolveIK` 用待命位姿的关节角作初值,`MoveL_Test` 从上一个点的关节角出发,免得中途换构型。
- **故意穿过奇异点的那一段,**只用来确认 `MoveL_Test` 报得出来,不算进过 / 不过的表。
- **不要在 main 的工作区里切到任何 `prototype/` 分支。**原型文件在 main 上被忽略、在分支里被跟踪,切回 main 时 git 会把它们从磁盘上删掉。读分支内容用 `git show`、`git log`。
- **不要改这几个文件:**`CONTEXT.md`、`docs/adr/`、`docs/grill-decisions.md`、`docs/demo-checklist.md`。
  - 结论带回拷问 session,由它吸收;
  - 这几个文件在 main 上有还没提交的改动,不要提交,也不要还原;
  - `model/` 在 main 上没被 git 跟踪,也别动。
- **正式程序的决定不适用于原型:**不要动 `pyproject.toml`、`main.py`、`.gitignore`,不要建 `robot` 分支。
- **结论要写清局限:**
  - 用的是 RoboDK 机器人库的名义模型,真机的关节限位、安装角度可能不同;
  - 指针 200 mm、待命点都是占位数;
  - 不查碰撞。
- **环境:**
  - Windows 11,主要用 PowerShell,也可以用 Git Bash;
  - Git Bash 里没有 `python` 命令,用上面那个 venv 的完整路径;
  - 在 Bash 里 `cd` 会改掉会话的工作目录,尽量用绝对路径。

## 做完以后

1. 每台出一张表:每个待命点、每个预设动作,过没过;不过的写明是哪个路点、哪一段,原因是逆解无解还是直线走不通。另附故意穿过奇异点那一段的检查结果。
2. 按「能定下的事」下结论:共用一套路点、给某一台加平移、还是要改写法。列出被证伪或得到证实的假设,主要是 ADR-0003 的前提;对相关决策给出「维持」或「调整」的建议,每条附数据。
3. 征得同意后,把完整结果写进 `notes/waypoint-reach-prototype-results.md`。
4. 用户要求提交时,用临时 worktree,把代码提交到新分支 `prototype/waypoint-reach`。只提交 `.scratch/robodk-prototype-2/` 下的代码,不提交工作站文件、模型和日志。
5. 写一份带回拷问 session 的交接文档,存到系统临时目录,写完把绝对路径单独打一行给用户。可以参考 `handsoff/handoff-grill-after-pre-vision-checks-2026-09-17.md` 的写法。

## 用户的习惯

- **提问方式:**给选项并标出推荐;用户只回答了部分题目时,没回答的题不要当作已确认。
- **原型要小:**一轮只回答一个问题。
- **结论要有数据,并写清局限。**
- **先问再动手:**写文件、提交、启动软件、装东西之前先问。
- **用户可能临时要求重测或改条件:**照做,但在结果里分开记,写清改了什么。

## 建议使用的技能

- `mattpocock-skills:prototype`:一次性原型,只回答一个问题。它分 LOGIC、UI 两条路,都不太对得上这种「算一遍,看够不够得着」的原型;照它的通用规则做:用完即弃、一条命令能跑、把结果打印出来、做完提交到原型分支。
- `mattpocock-skills:domain-modeling`:路点、工作台坐标系、默认姿态都要照 `CONTEXT.md` 的定义;结果如果要求改定义或改 ADR-0003,带回拷问 session。
- `mattpocock-skills:diagnosing-bugs`:逆解结果和预期对不上时用。
- `mattpocock-skills:research`:要查 RoboDK 文档,核实 `SolveIK`、`MoveL_Test` 的用法时用。
- `mattpocock-skills:grilling`:怎么扫都不行、需要回到拷问时用。
- `mattpocock-skills:handoff`:做完后写带回拷问 session 的交接文档。

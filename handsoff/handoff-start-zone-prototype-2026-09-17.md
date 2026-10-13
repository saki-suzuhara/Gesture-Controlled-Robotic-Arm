# 交接:起点区原型

写于 2026-09-17。用户用中文交流,请用中文回复。
项目根目录:`D:\Zetian\MyCourses\RobotCameraCombination`(下文路径都相对于它)。

## 本 session 只回答一个问题

**起点区(3 个手宽)能不能挡住挪拳误触发?**只测最常见的两种情况:把拳头抬进起点区;进区停稳后不做了,直接放下。站着做。

问题、做法、时间上限和判定标准,用户都已确认,本文不重复,都在 `docs/grill-decisions.md` 里:

| 要看什么 | 在哪里 |
|---|---|
| 为什么选这个问题、为什么对话解决不了、这次不做什么 | 第一节第 4 组「再下一轮原型(起点区原型)回答……」 |
| 各步时间上限、每次的内容、判定表、细节、代码从哪里改 | 第一节第 3 组「起点区原型的做法、时间和判定标准」 |
| 时间上限的来历、准备段怎么提示、计数口径和顺序、代码和录像放哪 | 第二节里以「起点区原型」开头的几条 |
| 起点区怎么判定、READY、收手抑制、启动对准、「手宽」、均匀画面、录制前也要对准 | 第二节里的「起点区的判定」「READY 只亮字」「收手抑制仍然保留」「启动对准的细节」「手宽」「均匀画面的判定」「录制小工具在录制前也做启动对准」 |
| 握拳平移的判定参数 | 第一节第 3 组「握拳平移的判定改为……」「判方向的轴比改为 3」;第二节「握拳平移的判定细节(初值)」 |
| 这次结果会影响哪些问题 | 第三节「第十九轮之后」(使能手势),以及「必须跑起来才能定的」第一条 |

## 先读这些

- `docs/grill-decisions.md`:见上表。
- **之前三个原型的结论**都在 `prototype/*` 分支的提交说明里,用 `git log -1 --format=%B <提交>` 查看;完整结果在 `notes/` 下(git 忽略):

  | 原型 | 分支和提交 | 完整结果 |
  |---|---|---|
  | 第一轮(手势 → 动作映射) | `prototype/gesture-recognition` 2457941 | `notes/camera-prototype-results.md` |
  | 握拳平移 | `prototype/fist-move` 78f4023 | `notes/fist-move-prototype-results.md` |
  | 姿势保持 | `prototype/pose-hold` f1ec0f7 | `notes/pose-hold-prototype-results.md` |

  这次问题的由来(准备段误触发、抬手组、直接放下)和扫参表,都在握拳平移的结果里。
- `CONTEXT.md`:术语和英文名。本次重点:起点区、挪拳误触发、握拳平移、握拳类标签、收手抑制、启动对准、检测区域、摄像头占位图、准备段 / 「做!」段、取景检查、操作位。
- ADR:
  - `docs/adr/0009-fist-move-starts-from-start-zone.md`(本题);
  - `docs/adr/0008-detect-only-region-around-operator-spot.md`;
  - `docs/adr/0007-own-trigger-rules-fist-move-and-pose-hold.md`;
  - `docs/adr/0006-operator-distance-1-to-1.5m.md`。
- `docs/demo-checklist.md`:站着录之前的布置(地上的胶带、摄像头隐私开关、屏幕倾角),照着和用户核对。
- `handsoff/handoff-pose-hold-prototype-2026-09-17.md`:上一个原型的交接,可以参考写法。其中「坐着做」「下载示范图」不适用于本次。

## 可以直接复用的东西

- **`.scratch/camera-prototype/`:**
  - `.venv/`:Python 3.13,依赖已装好;
  - `vendor/dynamic_gestures/`:上游代码和模型;
  - `compat.py`:numpy 补丁。
- **`.scratch/camera-prototype-3/`(姿势保持,代码在提交 f1ec0f7):**
  - `pose_proto.py`:本次在它上面改。检测区域现在是写死的 `REGION`,要换成启动对准算出来的;
  - `analyze_pose.py`:录完出表的写法;
  - `refs/`:HaGRID 示范图的裁剪,有 `fist.png`。只存本地,不提交、不外传;界面上要显示署名,见 `pose_proto.py` 里的 `CREDIT`。
- **`.scratch/camera-prototype-2/`(握拳平移,代码在提交 78f4023):**
  - `fist_rule.py`:握拳平移的实时判定,`Params` 里能设标签集合和轴比;**里面没有收手抑制**;
  - `sweep2.py`:收手抑制(`R`)只在这里的离线扫参里有,要搬进实时判定;
  - `fist_proto.py`:「拳头 + 箭头」和「直接放下」的示意图和提示;
  - `analyze_fist.py`:逐次明细。
- 各文件开头都有用法说明。

## 开工前必须先征得用户同意

- **运行上游 `dynamic_gestures` 的代码:**之前几轮用户同意过,新 session 再确认一次。
- **录视频:**视频里有用户本人,只存本地,放在 `.scratch/camera-prototype-4/` 下。
- **判定标准:**用户已确认,原话是「A,但是安排的时间要控制在5分钟内」。开录前和用户核对一遍布置和流程;要改标准,先问用户。
- **下载、安装:**这次不需要。真要装,先说明要装什么,再问。
- **写结果文件、提交代码:**见「做完以后」。

## 注意

- **5 分钟是硬上限**,包括取景检查、启动对准、录制和出表。各步上限见决策汇总;屏幕上最好显示已用时间。
- **不要在 main 的工作区里切到任何 `prototype/` 分支。**原型文件在 main 上被忽略、在分支里被跟踪,切回 main 时 git 会把它们从磁盘上删掉。读分支内容用 `git show`、`git log`。
- **不要改这几个文件:**`CONTEXT.md`、ADR、`docs/grill-decisions.md`、`docs/demo-checklist.md`。结论带回拷问 session,由它吸收。
- **正式程序的决定不适用于原型。**`data/` 目录、pytest、`vision` 分支、空实现,都是给正式程序定的。不要动 `pyproject.toml`、`main.py`、`.gitignore`。
- **`.gitignore` 有还没提交的改动**(包括刚加的 `data/`),不要提交,也不要还原。
- **结论要写清局限:**单人、只录一轮、站着;离线扫参和判定用的是同一批数据,结果偏乐观。
- **环境:**
  - Windows 11,主要用 PowerShell,也可以用 Git Bash;
  - Git Bash 里 `python` 命令不可用,用 `.scratch/camera-prototype/.venv/Scripts/python.exe`;
  - 在 Bash 里 `cd` 会改掉会话的工作目录,尽量用绝对路径。

## 做完以后

1. 录完直接在屏幕上出表。按决策汇总里的判定表下结论,说明结果属于哪一种情况。
2. 列出被证伪或得到证实的假设;对相关决策给出「维持」或「调整」的建议,每条附数据。
3. 征得同意后,把完整结果写进 `notes/start-zone-prototype-results.md`。
4. 用户要求提交时,用临时 worktree,把原型代码提交到新的原型分支,比如 `prototype/start-zone`。只提交 `.scratch/camera-prototype-4/` 下的代码,不提交视频、日志和示范图。
5. 写一份带回拷问 session 的交接文档,存到系统临时目录,写完把绝对路径单独打一行给用户。可以参考 `handsoff/handoff-grill-after-pose-hold-2026-09-17.md` 的写法。

## 用户的习惯

- **提问方式:**给选项并标出推荐;用户只回答了部分题目时,没回答的题不要当作已确认。
- **每一步都要提示:**需要用户做动作的实测,每一步都在屏幕上提示,配示范图或示意图,开始前等用户按键。
- **原型要小:**一轮只回答一个问题,5 分钟内出结果。
- **结论要有数据。**
- **先问再动手:**写文件、提交之前先问。

## 建议使用的技能

- `mattpocock-skills:prototype`:本任务是一次性原型,只回答一个设计问题。
- `mattpocock-skills:domain-modeling`:沿用 `CONTEXT.md` 的术语和英文名;结果如果要求改术语,带回拷问 session。
- `mattpocock-skills:diagnosing-bugs`:启动对准、起点区或实时判定表现异常时用。
- `mattpocock-skills:grilling`:结果需要回到拷问,或用户想改方案时,和用户重新过一遍。
- `mattpocock-skills:handoff`:做完后写带回拷问 session 的交接文档。

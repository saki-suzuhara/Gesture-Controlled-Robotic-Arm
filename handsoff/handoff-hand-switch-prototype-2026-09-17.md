# 交接:换手误触发原型

写于 2026-09-17。用户用中文交流,请用中文回复。
项目根目录:`D:\Zetian\MyCourses\RobotCameraCombination`(下文路径都相对于它)。

## 本 session 只回答一个问题

**把「每帧只看最大的手」改成只跟住 READY 时那只拳头,能不能挡住另一只手造成的误触发(换手误触发),同时不让漏认变多?**站着做。

问题和做法都记在 `docs/grill-decisions.md` 里,本文不重复:

| 要看什么 | 在哪里 |
|---|---|
| 为什么选这个问题、为什么对话解决不了、这次不做什么 | 第一节第 4 组「再下一轮原型(换手误触发原型)回答……」 |
| 第 0 步(用旧录像离线筛候选)、第 1 步(录一轮)、时间上限、每次的内容、判定表、代码从哪里改 | 第一节第 3 组「换手误触发原型的做法、时间和判定标准」 |
| 候选 ② 的来历和门槛怎么定、屏幕上用哪条规则、左手动作怎么选、示意图、代码和录像放哪、分支名 | 第二节里以「换手误触发原型」「左手动作」开头的几条 |
| 沿用起点区原型的做法:计数口径、准备段怎么提示、打乱顺序 | 第二节里以「起点区原型」开头的几条 |
| 「离得太近」的提示和门槛 | 第一节第 4 组「启动对准时检查拳头手框的大小……」;第二节「「离得太近」的判断」 |
| 握拳平移的判定参数、收手抑制、起点区的判定、手宽 | 第一节第 3 组「握拳平移的判定改为……」「判方向的轴比改为 3」;第二节「握拳平移的判定细节(初值)」「起点区的判定」「收手抑制仍然保留」「手宽」 |
| 这题在前沿里的位置,以及这次结果会影响哪些题 | 第三节「起点区原型之后(第二十、二十一轮)」第 1、9 条 |
| 那次换手误触发的经过、被证伪的假设 | 第四节「起点区原型」要点 5,以及「起点区原型证伪或修正的假设」 |

**判定标准用户还没有明确确认。**开录前先把判定表讲给用户听,确认了再录;用户要改,就按用户说的改,并在结果里写明。

## 先读这些

- `docs/grill-decisions.md`:见上表。
- **之前原型的结论**在各 `prototype/` 分支的提交说明里,用 `git log --format=%B -<条数> <分支>` 查看;完整结果在 `notes/` 下(git 忽略):

  | 原型 | 分支和提交 | 完整结果 | 和本次的关系 |
  |---|---|---|---|
  | 起点区 | `prototype/start-zone` 4bab804、780d196、c4ca06f | `notes/start-zone-prototype-results.md`,文末「第二轮」一节 | 换手误触发是在第二轮发现的;代码从这里改 |
  | 握拳平移 | `prototype/fist-move` 78f4023 | `notes/fist-move-prototype-results.md` | 「最大的手换过跟踪编号 11/40 次」出自这里;`dirs` 组日志是第 0 步要用的旧录像 |
  | 姿势保持 | `prototype/pose-hold` f1ec0f7 | `notes/pose-hold-prototype-results.md` | 录制工具的框架、示范图 |
  | 第一轮 | `prototype/gesture-recognition` 2457941 | `notes/camera-prototype-results.md` | 环境和上游代码的修补 |

- `CONTEXT.md`:术语和英文名。本次重点:换手误触发、漏认、判反、手宽、起点区、启动对准、检测区域、握拳平移、握拳类标签、收手抑制、准备段 / 「做!」段。
- ADR:
  - `docs/adr/0007-own-trigger-rules-fist-move-and-pose-hold.md`:「把每帧最大的手连起来」就是在这里定的,后果里记了这次要解决的漏洞;
  - `docs/adr/0009-fist-move-starts-from-start-zone.md`;
  - `docs/adr/0008-detect-only-region-around-operator-spot.md`;
  - `docs/adr/0006-operator-distance-1-to-1.5m.md`。
- `docs/demo-checklist.md`:站着录之前的布置,包括新加的「离得太近」一项。
- `handsoff/handoff-start-zone-prototype-2026-09-17.md`、`handsoff/handoff-grill-after-start-zone-2026-09-17.md`:上一个原型的交接和回报,可以参考写法。

## 可以直接复用的东西

- **`.scratch/camera-prototype/`:**`.venv/`(Python 3.13,依赖已装好)、`vendor/dynamic_gestures/`(上游代码和模型)、`compat.py`。
- **`.scratch/camera-prototype-4/`(起点区原型):**`zone_proto.py`、`zone_rule.py`、`analyze_zone.py`、`diag_zone.py`、`selftest_zone.py`。
  - 各文件开头有用法说明;
  - 起点区大小用 `--zone-hw` 设,其余判定参数在 `zone_rule.py` 的 `Params` 里。
- **第 0 步要用的旧录像日志:**
  - `.scratch/camera-prototype-2/runs/20260917-081450-dirs/`:`frames.jsonl` 是实时记录(整幅画面)。另有 `replay-c800.jsonl` 等文件,按文件名推测是用录像换裁剪方式重跑的结果,用之前先对照 `.scratch/camera-prototype-2/replay_video.py` 确认。
  - `.scratch/camera-prototype-4/runs/20260917-114858-zone/`、`.scratch/camera-prototype-4/runs/20260917-120426-zone2-far/`:`frames.jsonl`;同目录的 `report.txt` 是逐次明细,`meta.json` 是参数。
  - 这些 `frames.jsonl` 每帧都有 `raw`(所有检测框)和 `trk`(跟踪结果)。从样例看,`trk` 每项的第一个数像是跟踪编号,第二个是标签;字段含义以各原型的代码为准。
- **示范图:**`.scratch/camera-prototype-3/refs/fist.png` 是拳头的示范图(HaGRID 裁剪)。只存本地,不提交、不外传,界面上要显示署名。左手动作没有现成的示范图,要画示意图。

## 开工前必须先征得用户同意

- **判定标准:**见上。
- **运行上游 `dynamic_gestures` 的代码:**之前几轮都同意过,新 session 再确认一次。
- **录视频:**视频里有用户本人,只存本地,放在 `.scratch/camera-prototype-5/` 下。
- **站位:**站在胶带标记上(约 1.2 m),对准时不能出现「离得太近」。开录前和用户核对一遍演示前检查清单里的布置。
- **下载、安装:**这次不需要。真要装,先说明要装什么,再问。
- **写结果文件、提交代码:**见「做完以后」。

## 注意

- **5 分钟是硬上限**,包括取景检查、启动对准、录制和出表。各步上限见决策汇总;屏幕上显示已用时间。
- **先做第 0 步**,不占用户的时间。候选 ② 的门槛在第 0 步之前定好,之后不按结果调。
- **不要在 main 的工作区里切到任何 `prototype/` 分支。**原型文件在 main 上被忽略、在分支里被跟踪,切回 main 时 git 会把它们从磁盘上删掉。读分支内容用 `git show`、`git log`。
- **不要改这几个文件:**`CONTEXT.md`、ADR、`docs/grill-decisions.md`、`docs/demo-checklist.md`。结论带回拷问 session,由它吸收。
- **不要改 `.scratch/camera-prototype-4/` 里的文件。**复制到 `.scratch/camera-prototype-5/` 再改;`-4` 要留作起点区原型的原样记录。
- **正式程序的决定不适用于原型。**`data/` 目录、pytest、`vision` 分支、空实现,都是给正式程序定的。不要动 `pyproject.toml`、`main.py`、`.gitignore`;`.gitignore` 有还没提交的改动,不要提交,也不要还原。
- **结论要写清局限:**单人、只录一轮;左手动作是照提示做的,不是自然发生的;离线重算和判定用的是同一批数据。
- **环境:**
  - Windows 11,主要用 PowerShell,也可以用 Git Bash;
  - Git Bash 里没有 `python` 命令,用 `.scratch/camera-prototype/.venv/Scripts/python.exe`;
  - 在 Bash 里 `cd` 会改掉会话的工作目录,尽量用绝对路径。

## 做完以后

1. 录完直接在屏幕上出表。按判定表下结论,说明结果属于哪一种情况。
2. 列出被证伪或得到证实的假设;对相关决策给出「维持」或「调整」的建议,每条附数据。
3. 征得同意后,把完整结果写进 `notes/hand-switch-prototype-results.md`。
4. 用户要求提交时,用临时 worktree,把原型代码提交到新分支 `prototype/hand-switch`。只提交 `.scratch/camera-prototype-5/` 下的代码,不提交视频、日志和示范图。
5. 写一份带回拷问 session 的交接文档,存到系统临时目录,写完把绝对路径单独打一行给用户。可以参考 `handsoff/handoff-grill-after-start-zone-2026-09-17.md` 的写法。

## 用户的习惯

- **提问方式:**给选项并标出推荐;用户只回答了部分题目时,没回答的题不要当作已确认。
- **每一步都要提示:**需要用户做动作的实测,每一步都在屏幕上提示,配示范图或示意图,开始前等用户按键。
- **原型要小:**一轮只回答一个问题,5 分钟内出结果。
- **结论要有数据,并写清局限。**
- **先问再动手:**写文件、提交之前先问。
- **用户可能临时要求重测或改条件**,上一个原型就是这样。照做,但在结果里分开记,写清改了什么。

## 建议使用的技能

- `mattpocock-skills:prototype`:本任务是一次性原型,只回答一个设计问题。
- `mattpocock-skills:domain-modeling`:沿用 `CONTEXT.md` 的术语和英文名;结果如果要求改术语,带回拷问 session。
- `mattpocock-skills:diagnosing-bugs`:跟踪编号、候选规则或实时判定表现异常时用。
- `mattpocock-skills:grilling`:结果需要回到拷问,或用户想改方案时,和用户重新过一遍。
- `mattpocock-skills:handoff`:做完后写带回拷问 session 的交接文档。

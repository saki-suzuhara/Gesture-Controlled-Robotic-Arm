# 交接:姿势保持候选原型

写于 2026-09-17。用户用中文交流,请用中文回复。
项目根目录:`D:\Zetian\MyCourses\RobotCameraCombination`(下文路径都相对于它)。

## 本 session 只回答一个问题

**姿势保持用哪几个姿势?**也就是:哪些候选姿势坐着做时能被稳定认出,好从中挑两个给「画方块」「原地点头」。

这次原型怎么做,拷问里已经定了,不在本文重复,见 `docs/grill-decisions.md`:
- 第一节第 4 组「下一轮原型回答「姿势保持用哪几个姿势」,坐着做」:为什么选这个问题、候选姿势、不判断起点区、这次不做的事;
- 第一节第 3 组「下一轮原型要在 5 分钟内出结果」「下一轮原型的做法和挑选标准」;
- 第一节第 2 组「程序不识别身体……」:用户要求「只认准手势,不认准手势相对身体位置」;
- 第二节:姿势保持的判定(1 秒里八成帧)、触发后要先松开、姿势保持不能用握拳类标签、录制时准备段 3 秒、示意图箭头要画在真实位置;
- 第三节:「被下一轮原型挡住的」三条,都等这次的结果。

## 先读这些(内容不在本文重复)

- `docs/grill-decisions.md`:全部已定决策、默认值和前沿问题。
- **上一个原型(握拳平移)的结论:**分支 `prototype/fist-move` 的提交 78f4023,用 `git log -1 --format=%B 78f4023` 查看。完整结果在 `notes/fist-move-prototype-results.md`(git 忽略)。
- **第一轮原型的结论:**分支 `prototype/gesture-recognition` 的提交 2457941,用 `git log -1 --format=%B 2457941` 查看。完整报告在 `notes/camera-prototype-results.md`;第 3 节有 12 个姿势在 1.2 m 处的逐帧正确率,可以当基线。
- `CONTEXT.md`:术语表,含英文名。本次重点:姿势保持、起点区、检测区域、握拳类标签、停止手势、示范姿势、提示时间表、摄像头占位图。
- ADR:`docs/adr/0007-own-trigger-rules-fist-move-and-pose-hold.md`、`docs/adr/0008-detect-only-region-around-operator-spot.md`、`docs/adr/0002-dynamic-gestures-for-recognition.md`。
- `handsoff/handoff-camera-prototype-2-2026-09-17.md`:第二轮原型的任务表,本次就是它的第 3 条。它的「项目现状」「第一轮留下的已核实事实」「环境」三节仍然有用;任务表里的其他条目本次不做。

## 可以直接复用的东西

- `.scratch/camera-prototype/`(第一轮):
  - `.venv/`:Python 3.13 环境,依赖已装好;
  - `vendor/dynamic_gestures/`:上游代码和模型,已核对哈希;
  - `compat.py`(numpy 补丁)、`overlay.py`(画中文、响提示音)。
- `.scratch/camera-prototype-2/`(握拳平移,代码在提交 78f4023 里):
  - `fist_proto.py`:全屏置顶、分步提示和示意图、提示音、占位图检测、取景检查、同时录视频和逐帧日志加提示时间表;
  - `analyze_fist.py`:逐次明细和汇总;
  - `replay_video.py`:用录像换预处理重跑,包括只送中间 800×600。
  - 各文件开头有用法说明。
- 标签名见 `vendor/dynamic_gestures/utils/enums.py` 的 `targets`。候选对应的标签是 ok、peace、like、rock、call、two_up。

## 最小要跑什么(对话里定的,别处没有)

- **程序:**在 `fist_proto.py` 的框架上改。
  - 加姿势保持的实时判定,规则见决策汇总第二节;
  - 只把画面中间 800×600 送检测器(坐标见决策汇总第一节第 3 组);不判断起点区。握拳平移原型里裁剪只在 `replay_video.py` 离线做过,实时采集循环里要自己加,手框坐标要换回整幅画面;
  - 录完自动出表,整轮从开录到出结果控制在约 5 分钟。
- **用户的动作:**坐着,离摄像头约 1.2 m。6 个候选各 5 次,每次在胸前摆出来、保持约 2 秒、放下,约 7 秒一次,合计约 4 分钟。
- **录完马上出的表,每个候选一行:**
  - 5 次里触发了几次;
  - 逐帧认对的比例;
  - 最常被认成什么;
  - 被认成握拳类标签、停止类标签的帧各占多少。

  用同一批日志,再离线比较保持时长 0.7–1.5 秒的效果,不占用户时间。
- **挑选标准:**见决策汇总第一节第 3 组。开工前和用户再确认一次。符合标准的不足两个时,带回拷问 session。

## 开工前必须先征得用户同意

- **下载示范姿势图:**决策是用官方示例图(HaGRID 仓库或 HaGRIDv2 论文),先核对许可证,再说明文件名、来源和大小,经用户同意后下载。许可证不允许时的退路,见决策汇总第二节。用户不同意下载的话,先商量提示图怎么画,不能只靠文字描述。
- **运行上游 `dynamic_gestures` 的代码:**之前两轮用户同意过,新 session 再确认一次。
- **录视频:**视频里有用户本人,只能存本地,放在被 git 忽略的目录,比如 `.scratch/` 下。
- **安装任何东西:**原则上不需要,现有环境够用。
- **gh CLI:**已登录,token 权限很大,只做只读操作。

## 注意

- **不要提交任何东西,除非用户明确要求。**文档决定不提交,`notes/`、`.scratch/` 被 git 忽略。用户要求提交原型代码时:
  - 用临时 worktree,提交到新的原型分支;
  - 只提交原型代码,不提交 `CONTEXT.md` 和 `docs/`。

  上一轮就是这么做的。
- **不要在 main 的工作区里切到任何 `prototype/` 分支再切回来。**原型文件在 main 上被忽略、在分支里被跟踪,切回 main 时 git 会把它们从磁盘上删掉。读分支内容,用 `git show`、`git log` 就够了。
- **修改 `CONTEXT.md`、ADR、`docs/grill-decisions.md`、`docs/demo-checklist.md` 之前,先问用户。**
- **结论要写清局限:**这次坐着做、单人,演示时是站着(决策汇总里已记下要在站姿下复核)。
- **环境:**
  - Windows 11,主要用 PowerShell,也可以用 Git Bash;
  - Git Bash 里 `python` 命令不可用,用 `.scratch/camera-prototype/.venv/Scripts/python.exe`;
  - 在 Bash 里 `cd` 会改掉会话的工作目录,尽量用绝对路径。

## 用户的习惯

- **提问方式:**给选项并标出推荐,用户多数选推荐项;只回答部分题目时,没回答的题不要默认已确认。
- **实测要短:**需要用户做动作的原型,约 5 分钟内出结果,一轮只回答一个问题。
- **每一步都要提示:**屏幕上告诉用户现在做什么,配示范图或示意图,开始前等用户按键。
- **结论要有数据:**每个结论附数据,并对相关决策给出「维持」或「调整」的建议。
- **结果文件:**写成 `notes/` 下的一个文件,写之前先征得同意。
- **交接文档:**写完后,把它的绝对路径单独打一行给用户。

## 建议使用的技能

- `mattpocock-skills:prototype`:本任务是一次性原型,只回答一个设计问题。
- `mattpocock-skills:research`:核对 HaGRID 示例图的许可证时,用一手资料。
- `mattpocock-skills:domain-modeling`:沿用 `CONTEXT.md` 的术语和英文名;结果如果要求改术语,先问用户。
- `mattpocock-skills:diagnosing-bugs`:实时裁剪或识别行为异常时用。
- `mattpocock-skills:grilling`:结果推翻已有决策时,比如合格的候选不足两个,和用户重新过一遍。

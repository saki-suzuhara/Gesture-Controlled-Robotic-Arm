# 交接：把握拳平移原型的结论带回拷问

写于 2026-09-17。用户用中文交流，请用中文回复。
项目根目录：`D:\Zetian\MyCourses\RobotCameraCombination`（下文路径都相对于它）。

## 要做的事

握拳平移原型已经做完，对应 `handsoff/handoff-fist-move-prototype-2026-09-17.md`，也就是第二轮交接的第 1、2 条。

结果**不算通过**。那份交接给了三种去向：通过、只有上下不行、左右也不行，这次哪一种都对不上。

请按 `mattpocock-skills:grilling` 的方式，和用户把受影响的决策重新过一遍，并请用户拍板。

结论不在本文重复，按下面的位置去读：

- **结论摘要：**分支 `prototype/fist-move` 上的提交 `78f40235a7f2bf04a76a552f3b48801a15c7632b`。
  - 看提交信息：`git log -1 --format=%B 78f4023`；
  - 这是这次结论的正式版本。
- **完整结果：**`notes/fist-move-prototype-results.md`（git 忽略）。
  - 内容：逐项数据、现场问题、扫参表（实时日志和 800×600 裁剪两张），以及「对相关决策的建议」「没回答的问题」两节。
  - **拷问从「对相关决策的建议」那张表开始。**
- **原型代码：**同一个提交里的 `.scratch/camera-prototype-2/`，共 7 个文件。
  - 本地另有 `runs/`：两次练习和 `dirs`、`raise`、`drop` 三组数据，含视频；视频里有用户本人。
  - 用 `analyze_fist.py <tag>`、`sweep2.py dirs raise drop` 可以重出数据。
  - `<tag>@c800` 读的是录像裁剪回放的结果，用法见各文件开头的说明。

## 背景（先读这些）

- `CONTEXT.md`：术语表，含各术语的英文名。
- `docs/grill-decisions.md`：
  - 第一、二节：现行决策和握拳平移的判定初值；
  - 第三节：前沿问题，含「被第二轮原型挡住的」三条。
- `docs/adr/0007-own-trigger-rules-fist-move-and-pose-hold.md`、`docs/adr/0002-dynamic-gestures-for-recognition.md`、`docs/adr/0006-operator-distance-1-to-1.5m.md`。
- `handsoff/handoff-camera-prototype-2-2026-09-17.md`：第二轮交接，第 3–10 条还没做。它顶部写明要等握拳平移通过后再做。

## 待拍板的题目

只列题目，数据和建议见结果文件。

1. **要不要加使能手势。**
   - 对应：决策汇总第一节第 2 组「不设使能手势」；第三节「被第二轮原型挡住的」第 2 条。
2. **上、下是否改用姿势保持，左右是否照旧。**
   - 对应：ADR-0007；fist-move 交接的「结果怎么用」。
3. **握拳平移的判定初值怎么改。**
   - 对应：决策汇总第二节；第三节「被挡住的」第 3 条。
   - 注意：如果把 `point` 算作握拳，就会影响第三节「被挡住的」第 1 条（画方块、点头用哪个保持姿势）。
4. **要不要只把操作位周围约 800×600 送检测器。**
   - 对应：ADR-0002 的「检测、跟踪和分类的结果照用」；决策汇总第一节第 3 组「摄像头用编号 0，分辨率 1280×720」；操作位（胶带标记）是否固定。
   - 这是一条新决策，可能要写 ADR。
5. **演示前检查清单要不要补两项**：摄像头隐私开关、屏幕倾角和取景。对应 `docs/demo-checklist.md`。
6. **下一步做什么**，取决于前几题：
   - 补测：自然抬手、站姿、裁剪方案下的自由活动误触发；
   - 或者直接做第二轮交接的第 3–10 条。
   - 如果要补测，照旧先写交接文档，再开新 session。

## 本对话里有、提交和结果文件里没写的

- **开工前用户同意的：**
  - 运行上游代码；
  - 录视频，只存本地；
  - 通过标准按 fist-move 交接里的提议。
- **测试过程中用户的回答：**
  - 第一次练习开头没有画面，是摄像头隐私开关当时关着；
  - 有一次没注意到「做!」提示；
  - 全程坐着，离摄像头约 1.2 m；
  - 抬手组动作和本意不符，用户选择**不重做**。
- **收尾时：**
  - 结果文件由用户同意后写入；
  - 用户起初选择代码不提交，后来明确要求提交到 `prototype/fist-move`，并且只提交原型代码、不提交 `CONTEXT.md` 和 `docs/`。
- **`docs/grill-decisions.md` 第三节已经过时：**
  - 它还写着第 1、2 条待做；
  - 交接文档指向 `C:\Users\zetia\AppData\Local\Temp\handoff-fist-move-prototype-2026-09-17.md`，实际上 `handsoff/` 下也有一份；
  - 拍板后要一起更新，改之前先问用户。

## 注意

- **不要提交任何东西，除非用户明确要求。**
  - `notes/`、`.scratch/` 都被 git 忽略，文档也决定不提交；
  - `.gitignore` 里这两行的改动本身也还没提交。
- **修改 `CONTEXT.md`、ADR、`docs/grill-decisions.md`、`docs/demo-checklist.md` 之前，先问用户。**
- **不要在 main 的工作区里切到 `prototype/fist-move` 或 `prototype/gesture-recognition` 再切回来。**
  - 原型文件在 main 上被忽略、在分支里被跟踪，切回 main 时 git 会把它们从磁盘上删掉；
  - 读分支内容用 `git show`、`git log` 就够了；
  - 真要再提交原型代码，用临时 worktree，这次就是这么做的。
- **拍板时要考虑这些局限**（详见提交信息最后两段和结果文件「没回答的问题」）：
  - 离线改法的参数是在同一批数据上调的；
  - 全程坐姿、单人；
  - 自然抬手没有测到。
- **用户的习惯：**
  - 提问给选项并标出推荐，用户多数选推荐项；
  - 每个结论附数据，并对相关决策给出「维持」或「调整」；
  - 需要用户做动作的实测，每一步都要屏幕提示并配示意图，开始前等用户按键；
  - 交接文档写完，把绝对路径单独打一行给用户。

## 建议使用的技能

- `mattpocock-skills:grilling`：主流程，逐题和用户拍板。
- `mattpocock-skills:domain-modeling`：拍板后更新 `CONTEXT.md`（比如握拳平移的定义要不要写进「握拳类标签」）和 ADR（ADR-0007 的后果；裁剪方案如果采纳，可能要新增一条 ADR）。
- `mattpocock-skills:writing-for-agents`：如果决定补测，写下一份原型交接时用。
- `mattpocock-skills:prototype`：只在补测的新 session 里用，本次拷问不需要。

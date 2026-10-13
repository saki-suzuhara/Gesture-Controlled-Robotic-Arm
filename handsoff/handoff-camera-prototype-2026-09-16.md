# 交接:只用笔记本和摄像头的手势识别原型

写于 2026-09-16。用户使用中文交流,请用中文回复。

## 范围

**本 session 只做一件事:**用笔记本自带的 RGB 摄像头做一个一次性原型,回答下面「待验证的问题」里的第 1–11 条。不连任何机械臂。

**不在本 session 范围内,不要碰:**
- 替身、假控制器、状态流转的正式实现:另一个 session 负责。
- 任何机械臂连接:Dobot 或 JAKA 的 SDK、机械臂配置和演示配置的正式实现。
- 待用户拍板的问题:Dobot 关节速度被「打两次折」。
- JAKA 相关的验证。

## 先读这些(内容不在本文重复)

- `CONTEXT.md`:项目术语表。交流和命名一律沿用这里的词,比如操作者、触发手势、停止手势、叫停、冷却、识别中断、替身。
- `docs/grill-decisions.md`:上一轮拷问定下的全部决策,以及「替用户填上的默认值」一节。和识别相关的有:
  - 停止手势用哪些标签、连续几帧才算;
  - 6 个触发手势及对应的预设动作;
  - 触发手势规则(起止姿势不能是张开手掌);
  - 触发手势只认画面里最大的那只手;
  - 画面先水平翻转;
  - 冷却时长、识别中断的判定时间;
  - 摄像头编号和分辨率;
  - 回放视频只存本地、不提交。
- `docs/adr/0002-dynamic-gestures-for-recognition.md`:为什么用 `dynamic_gestures`,以及复制进项目、适配 Python 3.13 的决定。
- `notes/gesture-controlled-cobot.md`:前期调研报告,第 1、2 章讲识别方案。该目录被 git 忽略。
- `CLAUDE.md`、`docs/agents/`:技能配置。问题记录用本地 markdown,放在 `.scratch/<feature>/`,该目录被 git 忽略。

## 项目现状

- uv 项目,Python 3.13(见 `.python-version`;`pyproject.toml` 要求 `>=3.13`)。还没有任何依赖,`main.py` 只是 hello world。
- git:除了用户的初始提交外没有别的提交。`.gitignore` 被用户改过,新增了 `.scratch/` 和 `notes/`。用户决定文档不提交。**不要提交任何东西,除非用户明确要求。**
- 原型代码建议放在 `.scratch/camera-prototype/`(已被 git 忽略),不要放进正式代码目录。开工时先和用户确认存放位置。

## 开工前必须先征得用户同意

- **下载代码和模型:**`ai-forever/dynamic_gestures`,commit `f073b2541ad53863ea2935498342cec0d3e1ce73`。仓库里带两个模型:`models/hand_detector.onnx`(1,219,959 字节)和 `models/crops_classifier.onnx`(411,683 字节)。
- **安装第三方包:**在临时的 Python 3.13 环境里装 onnxruntime、opencv、numpy、scipy、filterpy。注意 filterpy 会连带装上 matplotlib。
- **运行第三方代码:**比如它的 `run_demo.py`。
- **录制视频:**回放测试要录含用户本人的视频,只能存本地、放在被 git 忽略的目录。
- **gh CLI:**已登录,token 权限很大,只做只读操作。

## `dynamic_gestures` 的已核实事实(commit f073b25)

- **画面翻转:**`run_demo.py` 第 21 行 `cv2.flip(frame, 1)`,即先水平翻转画面。
- **跟踪器:**`MainController` 默认 `maxlen=30`、`min_frames=20`;跟踪用 OC-SORT,参数为 `max_age=30`、`min_hits=3`、IoU 阈值 0.3、`delta_t=3`,关联方式 GIoU。
- **模型输入:**检测器 320×240,分类器 128×128。
- **标签表:**`utils/enums.py` 的 `targets` 共 45 个标签,下面要用到的索引:

  | 索引 | 标签 |
  |---|---|
  | 0 | hand_down |
  | 1 | hand_right |
  | 2 | hand_left |
  | 4 | thumb_left |
  | 5 | thumb_right |
  | 16 | two_down |
  | 19 | point |
  | 25 | fist |
  | 31 | palm |
  | 35 | stop |
  | 36 | stop_inverted |
  | 39 | two_up |

- **本项目选定的触发手势及判定规则**(见 `utils/action_controller.py`):
  - 左滑 `SWIPE_LEFT`:hand_right → hand_left;检查滑动距离、至少 20 帧、水平方向。
  - 右滑 `SWIPE_RIGHT`:hand_left → hand_right;条件同上。
  - 双指上滑 `SWIPE_UP3`:two_down → two_up;至少 15 帧、竖直方向,不检查距离。
  - 双指下滑 `SWIPE_DOWN3`:two_up → two_down;条件同上。
  - 点击 `TAP`:fist 出现后,至少 8 帧再出现 point,并做水平、竖直两个方向的检查。两次都在 2–8 帧内完成算双击 `DOUBLE_TAP`,本项目不启用。
  - 拇指左滑 `SWIPE_LEFT2`:thumb_right → thumb_left;检查距离、至少 20 帧、水平方向。
- **停止手势用的标签:**palm(31)、stop(35)、stop_inverted(36)。这三个也是上滑 / 下滑的起止姿势、拖放的「放下」,所以那几个手势不启用。「逐帧静态标签可以直接取出来用」目前只是推断,需要验证(问题 5)。
- **依赖:**
  - 锁定的旧版本不能装在 Python 3.13 上:onnxruntime 1.13.1 最高只到 cp310,numpy 1.23.5 和 onnx 1.13.0 只到 cp311。filterpy 1.4.5 只有源码包;opencv-contrib-python 4.6.0.66 是 abi3,不受影响。
  - 代码里 import 了 scipy,但 requirements 里没写;onnx 包写了却没有代码用到。
  - 提供 Windows cp313 安装包的最早版本:onnxruntime 1.20.0 已核实;numpy 2.1.0、scipy 1.14.1 来自 agent 报告,未核实。
  - 仓库的 .py 文件里没有 `np.float` 这类 numpy 2 已删除的旧写法(已核实);filterpy 在 numpy 2 下能否正常用,尚未验证。
- **DirectML 问题:**`onnx_models.py` 里 `HandDetection` 会不带 providers 参数重新创建会话,DirectML 路径因此失效。只用 CPU 时没有影响。
- **许可证:**readme 自称是「Apache 2.0 的变体」,但 `license/LICENSE-2.0.txt` 与标准 Apache 2.0 文本逐字相同(agent 报告)。

## 待验证的问题(本 session 的全部任务)

| # | 要验证的决策或假设 | 最小验证 |
|---|---|---|
| 1 | Python 3.13 加新版依赖能跑通 `dynamic_gestures` | 建临时 3.13 环境并装好依赖,运行 `run_demo.py`,看能否打开摄像头、画面上能否出现事件 |
| 2 | 帧数阈值沿用默认值(20 / 15 / 8 帧)是否合适,取决于笔记本帧率 | 看 demo 显示的帧率;掐表做几次自然的左滑 |
| 3 | 6 个触发手势能认准、彼此不混(左滑 / 双指左滑 / 拇指左滑只差手型) | 每个手势做 10 次,相近手势也各做 10 次,记下实际触发的事件,整理成混淆表 |
| 4 | 画面翻转后,左滑对应操作者自己的左 | 开着翻转,向自己左边滑一次,看输出的事件名 |
| 5 | 停止手势能取逐帧静态标签,连续 3 帧才算 | 打印每只手每帧的静态标签和时间戳;张开手掌 10 次,统计凑满 3 帧用了多久;做其他手势时看有没有误报成停止手势 |
| 6 | 抬手准备叫停时不会触发已启用的手势 | 反复做「放下手 → 张开手掌」,记录有没有触发任何已启用的事件 |
| 7 | 进入空闲时清空手势记录、冷却 1 秒,能挡住新旧姿势拼出的误触发 | 见表下说明 |
| 8 | 识别中断的判定时间取 0.5 秒是否合适 | 开着预览窗口连续跑几分钟,统计相邻两次识别结果的时间间隔 |
| 9 | 摄像头用编号 0、分辨率 640×480 | 用 OpenCV 打开 0 号摄像头并设成 640×480,看实际分辨率;在实际站位上跑检测 |
| 10 | 只认画面里最大的那只手,就等于只认操作者 | 两个人同时入镜,或两只手一近一远,看事件来自哪只手 |
| 11 | 空闲时识别到手势就立即执行、不设使能手势,误触发不会太多 | 见表下说明 |

第 7、11 条原本写的是「用替身跑」,但替身归另一个 session。本 session 在原型里用一个一次性的「假忙碌」计时器模拟执行中,只需在这段时间里忽略触发手势、结束时清空记录再冷却。**不要实现替身或正式的状态逻辑。**
- 第 7 条:在「假忙碌」期间故意做半个左滑,结束后马上做另一半,看会不会触发;冷却时长分别试 0.5、1、2 秒。
- 第 11 条:连续跑 10 分钟,用户照常说话、比划、走动但不打算触发,数误触发次数。

## 产出要求

- 每个问题都给出结论和数据(帧率、混淆表、时间统计),并对相应决策提出「维持」或「调整」的建议。**不要自行修改决策**,需要改 `CONTEXT.md`、ADR 或 `docs/grill-decisions.md` 时先问用户。
- 结果建议写成 `notes/` 下的一个文件(被 git 忽略),写之前和用户确认。

## 环境

- 开发机 Windows 11。主要用 PowerShell,也可以用 Bash(Git Bash)。
- 网络:github.com、pypi.org、storage.googleapis.com 可以访问;ai.google.dev 访问不了。
- 之前用 WebFetch 读 PDF 时,工具会自动把文件缓存到会话目录,读 PDF 时要留意。

## 建议使用的技能

- `mattpocock-skills:prototype`:本任务就是一次性原型,用来回答设计上的问题。
- `mattpocock-skills:domain-modeling`:沿用 `CONTEXT.md` 的术语;实验结果如果要求修改术语定义,先问用户再改。
- `mattpocock-skills:diagnosing-bugs`:新版依赖跑不通,或识别行为异常时用。
- `mattpocock-skills:research`:需要查 onnxruntime、filterpy、opencv 兼容性等一手资料时用。
- `mattpocock-skills:grilling`:实验结果如果推翻了已有决策(比如阈值、手势选择),和用户重新过一遍。

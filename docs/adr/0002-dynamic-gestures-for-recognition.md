# 手势识别用 dynamic_gestures,不用 MediaPipe Gesture Recognizer

手势识别采用 HaGRID 团队的 dynamic_gestures(ONNX 模型 + onnxruntime)。它用同一套检测、跟踪和分类流水线,给出每只手逐帧的手框和姿势标签;触发手势和停止手势都从这些结果判定。

不用 MediaPipe Gesture Recognizer,因为它开箱只有 7 类静态手势,太少。停止手势直接取分类器逐帧输出的姿势标签,不另跑 HaGRID 的 33 类 PyTorch 检测器,以免多装一个运行时,也避免两个模型对同一帧给出互相矛盾的结果。

触发手势的判定规则见 ADR-0007;上游自带的动态手势事件不再使用。

## Consequences

- dynamic_gestures 不是 pip 包,锁定的依赖最高只支持 Python 3.10。项目把它的代码和模型复制进来(保留 Apache-2.0 许可声明),自行适配 Python 3.13 和新版依赖。
- 2026-09-17 的摄像头原型证实,新依赖下可以用,但移植时必须带上以下修补:
  - numpy ≥ 2.4 时,`ocsort/kalmanfilter.py` 里的 `w / float(h)` 要改成 `w / h`;
  - 关闭 onnxruntime 的线程自旋,否则有手时只有 9–15 fps;
  - `provider_options` 为空时不传。

  原型报告在 `notes/camera-prototype-results.md`,代码在分支 `prototype/gesture-recognition` 的提交 2457941。
- 同一原型显示,上游自带的 6 个触发手势即使在最好的条件下(操作距离 1.2 m、1280×720),也只有点击可靠。触发手势因此改用自写规则(ADR-0007),本 ADR 的识别方案不变。
- 握拳平移原型之后,只把操作位周围的检测区域送检测器(ADR-0008);检测和分类用的模型照旧。

"""上游 `dynamic_gestures`(commit f073b25,Apache-2.0)移植进来的那一份。

许可声明在 `license/LICENSE-2.0.txt`,和代码一起提交。除这个文件以外,目录里的东西
都是上游原样,只有三处移植改动(ADR-0002),每一处都在原地注明:

- `ocsort/kalmanfilter.py`:`w / float(h)` → `w / h`(numpy ≥ 2.4);
- `onnx_models.py`:关掉 onnxruntime 的线程自旋;`provider_options` 为空时不传;
  顺带删掉 `HandDetection` 里重建 session 的那一遍(它会把上面那条设置丢掉)。

另外 `main_controller.py` 的三行 import 改成了相对 import,好让它作为包的一部分被
import(上游是当脚本跑的)。这个 `__init__.py` 本身也不是上游的,只为把这一份标成包。

**上游自带的动态手势事件不用**(ADR-0007),我们只用它的检测、跟踪和姿势分类。
"""

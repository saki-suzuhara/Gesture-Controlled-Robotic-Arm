"""打两份发行包,放 `dist/`:

- **源码包** `RobotCameraCombination-<版本>-source.zip`:`git archive` 打的,和 GitHub 上那份
  一模一样。要自己装 uv,照 README 跑。
- **便携版** `RobotCameraCombination-<版本>-win64.zip`:自带 Python 3.13 和全部依赖,解压以后
  双击 `.bat` 就跑,别的 Windows 电脑什么都不用装。连 RoboDK 仿真的那几个要那台电脑另装 RoboDK
  (RoboDK 是商业软件,打不进包里)。

    uv run python packaging/build_release.py

两份都从 HEAD 打:工作区里有没提交的改动就先报出来、不打(`--allow-dirty` 硬打,但打进去的
仍是 HEAD,不是工作区)。便携版里的 Python 是本机 uv 装的那一份(python-build-standalone,
可以整个目录挪走),依赖版本照 `uv.lock`。
"""

from __future__ import annotations

import argparse
import datetime
import io
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
NAME = "RobotCameraCombination"

APP_PATHS = ["main.py", "README.md", "config", "model", "robot_side", "vision_side"]
"""便携版 `app/` 里放什么:跑起来要的就这些。测试、文档、交接在源码包里。"""

VC_RUNTIME = [
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "concrt140.dll",
    "vccorlib140.dll",
]
"""VC++ 运行库,从本机 System32 拷一份放在 `python.exe` 旁边(微软允许这样随程序分发)。

python-build-standalone 自带 `vcruntime140*.dll`,但 onnxruntime 还要 `msvcp140.dll`,
它的 wheel 没带;没装过 VC++ 运行库的电脑上会报 DLL load failed。"""

SMOKE_IMPORTS = "import cv2, numpy, onnxruntime, scipy, filterpy, robodk, tkinter, tomllib"

LAUNCHERS = {
    "1-手势识别(不接机械臂).bat": (
        "camera + gesture recognition only, no robot arm (any PC with a camera)",
        "main.py",
        False,
    ),
    "2-RoboDK仿真-JAKA.bat": ("RoboDK simulation, JAKA Zu 20", "main.py --arm jaka", True),
    "3-RoboDK仿真-Nova5.bat": ("RoboDK simulation, Dobot Nova 5", "main.py --arm nova5", True),
    "4-手势练习(逐个带你做).bat": (
        "walks you through every trigger gesture and one stop, no robot arm",
        "-m vision_side.scripts.triggers",
        False,
    ),
}
"""双击就跑的几个入口:文件名 → (说明, python 后面的参数, 要不要 RoboDK)。

**不放连真机的入口**:真机要仓库外那份填了 IP 的配置,上之前还要照演示清单过一遍,
不该是双击一下的事。使用说明里写了怎么从命令行连。"""

BAT_HEAD = r"""@echo off
rem {what}
setlocal
cd /d "%~dp0app"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
"""

BAT_NEEDS_ROBODK = r"""if exist "C:\RoboDK\bin\RoboDK.exe" goto run
reg query "HKLM\SOFTWARE\RoboDK" /v INSTDIR >nul 2>&1 && goto run
echo RoboDK is not installed on this PC ^(looked for C:\RoboDK\bin\RoboDK.exe^).
echo Install it from https://robodk.com/download into C:\RoboDK, or double-click
echo "1-..." to run the camera side without a robot arm.
pause
exit /b 1
:run
"""

BAT_RUN = r""""%~dp0python\python.exe" {args} %*
if errorlevel 1 pause
"""
"""出错退出时停住窗口,不然双击开的黑框一闪就没了,看不到报的是什么。正常退出不停。

bat 里只写 ASCII:`chcp 65001` 加中文在一些电脑上会把行切错,中文说明都放使用说明里。"""


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, cwd=ROOT, **kwargs)


def git(*args: str) -> str:
    return run(["git", *args], capture_output=True, text=True).stdout.strip()


def version_label() -> str:
    today = datetime.date.today().strftime("%Y%m%d")
    return f"{today}-{git('rev-parse', '--short', 'HEAD')}"


def check_clean(allow_dirty: bool) -> None:
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if not dirty:
        return
    print("工作区有没提交的改动(两份包都只打 HEAD,这些不会进去):")
    print(dirty)
    if not allow_dirty:
        sys.exit("先提交,或者加 --allow-dirty 照样打 HEAD")


def build_source(label: str) -> Path:
    out = DIST / f"{NAME}-{label}-source.zip"
    run(["git", "archive", "--format=zip", f"--prefix={NAME}/", "-o", str(out), "HEAD"])
    return out


def export_app(dest: Path) -> None:
    """把 HEAD 里跑起来要的那几样解到 `dest`(只要提交过的,不带 `__pycache__`、`data/`)。"""
    tar = run(["git", "archive", "--format=tar", "HEAD", *APP_PATHS], capture_output=True)
    with tarfile.open(fileobj=io.BytesIO(tar.stdout)) as archive:
        archive.extractall(dest, filter="data")


def build_python(dest: Path) -> None:
    base = Path(sys.base_prefix).resolve()
    print(f"  拷 Python:{base}")
    shutil.copytree(base, dest, ignore=shutil.ignore_patterns("__pycache__"))
    # uv 管的 Python 标了 EXTERNALLY-MANAGED,不让往里装;这是拷出来的一份,去掉
    (dest / "Lib" / "EXTERNALLY-MANAGED").unlink(missing_ok=True)
    for extra in ("include", "libs", "Scripts"):  # 编扩展用的头文件、带本机绝对路径的启动器
        shutil.rmtree(dest / extra, ignore_errors=True)
    site = dest / "Lib" / "site-packages"
    for item in site.iterdir():  # 只留照 uv.lock 装的那些
        if item.name != "README.txt":
            shutil.rmtree(item) if item.is_dir() else item.unlink()

    requirements = DIST / "build" / "requirements.txt"
    run(
        [
            "uv", "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project",
            "--format", "requirements-txt", "-q", "-o", str(requirements),
        ]
    )  # fmt: skip
    run(
        [
            "uv", "pip", "install", "--python", str(dest / "python.exe"),
            "--link-mode", "copy", "-r", str(requirements),
        ]
    )  # fmt: skip

    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for name in VC_RUNTIME:
        if (system32 / name).is_file() and not (dest / name).exists():
            shutil.copy2(system32 / name, dest / name)


def compile_bytecode(stage: Path) -> None:
    """把标准库、依赖和 app 全预编译成 .pyc,按源文件的哈希校验(checked-hash)。

    不编的话第一次运行要现编几千个文件,2026-10-12 实测把识别线程的启动拖过了启动检查的
    5 秒(VISION NOT READY),第二次才过。默认的按时间戳校验也不行:解压会改掉 .py 的
    修改时间,.pyc 全部作废,又回到现编。按哈希校验不看时间戳;app 里的代码改了照样重编。
    """
    compileall = [
        str(stage / "python" / "python.exe"), "-m", "compileall", "-q", "-j", "0",
        "--invalidation-mode", "checked-hash",
    ]  # fmt: skip
    # 依赖里万一有编不过的文件,只是少几个 .pyc,不拦着打包
    if subprocess.run([*compileall, str(stage / "python" / "Lib")], env=scrubbed_env()).returncode:
        print("  (依赖里有个别文件编不过,跳过)")
    # 自己的代码编不过就是写坏了,不打(2026-10-12 就这样撞出了 action_demo.py 里的一个真换行)
    if subprocess.run([*compileall, str(stage / "app")], env=scrubbed_env()).returncode:
        sys.exit("app 里有文件编不过,见上面")


def scrubbed_env() -> dict[str, str]:
    """模拟一台什么都没装的电脑:PATH 里只有 Windows 自己的目录,没有 uv、没有别的 Python。"""
    root = os.environ.get("SystemRoot", r"C:\Windows")
    keep = ("SystemRoot", "SystemDrive", "TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PATH"] = f"{root}\\System32;{root}"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    return env


def smoke_test(stage: Path) -> None:
    python = stage / "python" / "python.exe"
    run([str(python), "-c", SMOKE_IMPORTS], env=scrubbed_env())
    # 配置真读一遍、识别模型真跑一帧:import 过了不等于 TOML、ONNX 模型找得到、推理跑得动
    check = (
        "import numpy as np;"
        "from robot_side.config import load_demo; load_demo();"
        "from vision_side.frames import Frame;"
        "from vision_side.gestures import GestureRecognizer;"
        "GestureRecognizer().recognize(Frame(np.zeros((720, 1280, 3), np.uint8), 0.0, 0))"
    )
    subprocess.run([str(python), "-c", check], check=True, cwd=stage / "app", env=scrubbed_env())


def write_launchers(stage: Path, label: str) -> None:
    for filename, (what, args, needs_robodk) in LAUNCHERS.items():
        text = BAT_HEAD.format(what=what)
        if needs_robodk:
            text += BAT_NEEDS_ROBODK
        text += BAT_RUN.format(args=args)
        (stage / filename).write_bytes(text.replace("\n", "\r\n").encode("ascii"))
    guide = (ROOT / "packaging" / "使用说明.txt").read_text(encoding="utf-8")
    guide = guide.replace("{version}", label)
    # 带 BOM:老版本记事本没有 BOM 会按 GBK 猜,中文变乱码
    (stage / "使用说明.txt").write_text(
        guide.replace("\n", "\r\n"), encoding="utf-8-sig", newline=""
    )


def zip_dir(stage: Path, out: Path) -> None:
    """`zipfile` 遇到中文文件名会打上 UTF-8 标记,Windows 自带的解压认得。"""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(stage.rglob("*")):
            archive.write(path, Path(NAME) / path.relative_to(stage))


def build_portable(label: str) -> Path:
    stage = DIST / "build" / NAME
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    export_app(stage / "app")
    build_python(stage / "python")
    write_launchers(stage, label)
    compile_bytecode(stage)
    smoke_test(stage)
    out = DIST / f"{NAME}-{label}-win64.zip"
    print(f"  压缩 {out.name}(四百多 MB,要一两分钟)")
    zip_dir(stage, out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="打源码包和便携版,放 dist/")
    parser.add_argument("--allow-dirty", action="store_true", help="工作区有改动也照样打 HEAD")
    parser.add_argument("--only", choices=["source", "portable"], help="只打其中一份")
    args = parser.parse_args(argv)

    if sys.platform != "win32":
        sys.exit("便携版只能在 Windows 上打(拷的是本机的 Windows 版 Python)")
    check_clean(args.allow_dirty)
    DIST.mkdir(exist_ok=True)
    label = version_label()
    built = []
    if args.only in (None, "source"):
        print("源码包")
        built.append(build_source(label))
    if args.only in (None, "portable"):
        print("便携版")
        built.append(build_portable(label))
    for path in built:
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

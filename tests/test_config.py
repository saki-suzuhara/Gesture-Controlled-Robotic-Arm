"""配置读取与校验:配置里写错的东西要在启动时就报出来(第 02 票)。

只测外部行为:喂一份 TOML 文本进去,看读出来的值,或者看报出来的错说没说清
「哪个文件、哪个字段、为什么不对」。
"""

from __future__ import annotations

import textwrap

import pytest

from robot_side.config import ConfigError, load_arm_config, load_demo, load_demo_config

ARM_TOML = """
name = "JAKA Zu 20"
robot_name = "JAKA Zu20"

[connection]
driver = "robodk"

[worktable]
angle_deg = 0.0
origin_mm = [0.0, 0.0, 0.0]

[standby]
joints_deg = [118.09, 87.49, 137.05, 225.47, 90.0, 28.09]

[pointer]
offset_mm = [1.11, -1.26, 280.47]

[speed]
max_joint_degs = 30.0
max_linear_mms = 100.0
return_joint_percent = 33.0
"""

DEMO_TOML = """
default_orientation_deg = [180.0, 0.0, 0.0]

[run]
countdown_s = 3.0
cooldown_s = 2.0
vision_lost_after_s = 0.5
stop_gesture_frames = 3
lift_mm = 50.0
return_speed_mms = 50.0
range_mm = [300.0, 0.0, 200.0]

[vision]
detection_region_px = [800, 600]
hold_s = 1.2
measure_s = 0.8
still_hand_widths = 0.5
fist_percent = 80
start_zone_hand_widths = 3.0
too_close_px = 107.0
camera_index = 0
camera_px = [1280, 720]
settle_s = 0.3
settle_hand_widths = 0.5
settle_percent = 80
move_hand_widths = 1.5
move_max_s = 1.2
move_percent = 60
move_axis_ratio = 3.0
reverse_guard_s = 1.5
reverse_guard_hand_widths = 1.0
reverse_guard_axis_ratio = 2.0
lock_timeout_s = 1.2
settle_size_percent = 15
push_size_ratio = 1.4
push_height_ratio = 1.25
push_frames = 3
knock_hand_widths = 0.4
knock_return_hand_widths = 0.3
knock_window_s = 3.0
knock_percent = 50
pose_hold_s = 1.0
pose_percent = 80
release_s = 0.3

[[actions]]
name = "point_left"
display = "POINT LEFT"
speed_mms = 50.0
end = "stay"
waypoints = [{ dx_mm = -150.0 }]

[[actions]]
name = "nod"
display = "NOD"
speed_mms = 50.0
end = "back"
waypoints = [
  { nod_deg = 15.0 },
  { nod_deg = -15.0, pause_s = 0.2 },
]

[[triggers]]
kind = "fist_move"
direction = "left"
action = "point_left"

[[triggers]]
kind = "pose_hold"
pose = "like"
action = "nod"
"""


def write(tmp_path, text, name="arm.toml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def arm_without(tmp_path, line):
    """去掉某一行的机械臂配置,用来测「字段缺失」。"""
    kept = [ln for ln in ARM_TOML.splitlines() if not ln.startswith(line)]
    return write(tmp_path, "\n".join(kept))


# ---- 机械臂配置 --------------------------------------------------------


def test_读出机械臂配置(tmp_path):
    cfg = load_arm_config(write(tmp_path, ARM_TOML))
    assert cfg.name == "JAKA Zu 20"
    assert cfg.robot_name == "JAKA Zu20"
    assert cfg.worktable.angle_deg == 0.0
    assert cfg.worktable.origin_mm == (0.0, 0.0, 0.0)
    assert cfg.standby_joints_deg == (118.09, 87.49, 137.05, 225.47, 90.0, 28.09)
    assert cfg.pointer_offset_mm == (1.11, -1.26, 280.47)
    # 回待命的关节段按百分比写,换算成仿真要用的 °/s
    assert cfg.return_joint_degs == pytest.approx(9.9)


def test_工作台坐标系不写就是不转不平移(tmp_path):
    text = ARM_TOML.replace("angle_deg = 0.0\norigin_mm = [0.0, 0.0, 0.0]", "")
    cfg = load_arm_config(write(tmp_path, text))
    assert cfg.worktable.angle_deg == 0.0
    assert cfg.worktable.origin_mm == (0.0, 0.0, 0.0)


def test_缺字段时说清是哪个文件哪个字段(tmp_path):
    path = arm_without(tmp_path, "robot_name")
    with pytest.raises(ConfigError) as err:
        load_arm_config(path)
    assert "robot_name" in str(err.value)
    assert path.name in str(err.value)


def test_缺待命位姿也报错(tmp_path):
    with pytest.raises(ConfigError, match="standby.joints_deg"):
        load_arm_config(arm_without(tmp_path, "joints_deg"))


def test_待命位姿不是六个关节角就报错(tmp_path):
    text = ARM_TOML.replace("[118.09, 87.49, 137.05, 225.47, 90.0, 28.09]", "[0.0, 0.0, 0.0]")
    with pytest.raises(ConfigError, match="六"):
        load_arm_config(write(tmp_path, text))


def test_关节角超出正负360报错(tmp_path):
    text = ARM_TOML.replace("118.09", "999.0")
    with pytest.raises(ConfigError, match="standby.joints_deg"):
        load_arm_config(write(tmp_path, text))


def test_速度写成米每秒时报出单位不对(tmp_path):
    # 100 mm/s 写成 0.1(m/s):数值落在 mm/s 说不通的范围里
    text = ARM_TOML.replace("max_linear_mms = 100.0", "max_linear_mms = 0.1")
    with pytest.raises(ConfigError) as err:
        load_arm_config(write(tmp_path, text))
    assert "mm/s" in str(err.value)


def test_百分比写成小数时报出单位不对(tmp_path):
    text = ARM_TOML.replace("return_joint_percent = 33.0", "return_joint_percent = 0.33")
    with pytest.raises(ConfigError) as err:
        load_arm_config(write(tmp_path, text))
    assert "%" in str(err.value)


def test_多写的字段会被指出来(tmp_path):
    # 写错字段名(比如把 angle_deg 写成 angle)不能默默按默认值跑
    text = ARM_TOML.replace("angle_deg = 0.0", "angle = 0.0")
    with pytest.raises(ConfigError, match="angle"):
        load_arm_config(write(tmp_path, text))


# ---- 演示配置 ----------------------------------------------------------


def test_读出演示配置(tmp_path):
    cfg = load_demo_config(write(tmp_path, DEMO_TOML, "demo.toml"))
    assert cfg.default_orientation_deg == (180.0, 0.0, 0.0)
    assert cfg.run.lift_mm == 50.0
    assert cfg.run.return_speed_mms == 50.0
    assert [a.name for a in cfg.actions] == ["point_left", "nod"]
    left = cfg.action("point_left")
    assert left.display == "POINT LEFT"
    assert left.speed_mms == 50.0
    assert len(left.waypoints) == 1
    assert left.waypoints[0].dx_mm == -150.0
    assert left.waypoints[0].dy_mm == 0.0
    assert left.waypoints[0].nod_deg == 0.0
    nod = cfg.action("nod")
    assert [w.nod_deg for w in nod.waypoints] == [15.0, -15.0]
    assert nod.waypoints[1].pause_s == 0.2
    # 做完怎么停:左挪一步停在那儿,点头原地做完回原处(2026-09-23 起不回待命)
    assert left.end == "stay" and left.shift_mm == (-150.0, 0.0, 0.0)
    assert nod.end == "back" and nod.shift_mm == (0.0, 0.0, 0.0)
    assert cfg.run.range_mm == (300.0, 0.0, 200.0)


def test_触发手势对应到动作(tmp_path):
    cfg = load_demo_config(write(tmp_path, DEMO_TOML, "demo.toml"))
    assert cfg.action_for_trigger("fist_move", "left") == "point_left"
    assert cfg.action_for_trigger("pose_hold", "like") == "nod"


def test_触发手势用了会被认成停止手势的姿势就报错(tmp_path):
    text = DEMO_TOML.replace('pose = "like"', 'pose = "palm"')
    with pytest.raises(ConfigError) as err:
        load_demo_config(write(tmp_path, text, "demo.toml"))
    assert "palm" in str(err.value)
    assert "停止手势" in str(err.value)


@pytest.mark.parametrize("pose", ["stop", "stop_inverted", "three", "four"])
def test_停止手势的每个标签都挡住(tmp_path, pose):
    text = DEMO_TOML.replace('pose = "like"', f'pose = "{pose}"')
    with pytest.raises(ConfigError, match="停止手势"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_姿势保持不能用握拳类姿势(tmp_path):
    # 握拳要先停住,拳头停久一点就会先触发姿势保持
    text = DEMO_TOML.replace('pose = "like"', 'pose = "fist"')
    with pytest.raises(ConfigError, match="握拳"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_触发手势指向不存在的动作就报错(tmp_path):
    text = DEMO_TOML.replace('action = "nod"', 'action = "no_such_action"')
    with pytest.raises(ConfigError, match="no_such_action"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_两个触发手势指向同一个键就报错(tmp_path):
    text = DEMO_TOML + textwrap.dedent(
        """
        [[triggers]]
        kind = "pose_hold"
        pose = "like"
        action = "point_left"
        """
    )
    with pytest.raises(ConfigError, match="like"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_动作没有路点就报错(tmp_path):
    text = DEMO_TOML.replace("waypoints = [{ dx_mm = -150.0 }]", "waypoints = []")
    with pytest.raises(ConfigError, match="point_left"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_抬起高度写成米时报出单位不对(tmp_path):
    text = DEMO_TOML.replace("lift_mm = 50.0", "lift_mm = 0.05")
    with pytest.raises(ConfigError) as err:
        load_demo_config(write(tmp_path, text, "demo.toml"))
    assert "mm" in str(err.value)


def test_动作速度写成米每秒时报出单位不对(tmp_path):
    text = DEMO_TOML.replace("speed_mms = 50.0", "speed_mms = 0.05", 1)
    with pytest.raises(ConfigError) as err:
        load_demo_config(write(tmp_path, text, "demo.toml"))
    assert "mm/s" in str(err.value)


def test_点头角度超出正负180报错(tmp_path):
    text = DEMO_TOML.replace("nod_deg = 15.0", "nod_deg = 999.0")
    with pytest.raises(ConfigError, match="nod_deg"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


# ---- 启动对准那一节 ----------------------------------------------------


def test_启动对准的参数读得出来(tmp_path):
    cfg = load_demo_config(write(tmp_path, DEMO_TOML, "demo.toml"))
    assert cfg.vision.detection_region_px == (800, 600)
    assert cfg.vision.start_zone_hand_widths == 3.0
    assert cfg.vision.too_close_px == 107.0


def test_触发判定的参数读得出来(tmp_path):
    cfg = load_demo_config(write(tmp_path, DEMO_TOML, "demo.toml"))
    v = cfg.vision
    assert (v.settle_s, v.settle_hand_widths, v.settle_percent) == (0.3, 0.5, 80.0)
    assert (v.move_hand_widths, v.move_max_s, v.move_percent) == (1.5, 1.2, 60.0)
    assert v.move_axis_ratio == 3.0
    assert (v.reverse_guard_s, v.lock_timeout_s) == (1.5, 1.2)
    assert (v.pose_hold_s, v.pose_percent, v.release_s) == (1.0, 80.0, 0.3)
    # 前后两个方向(2026-09-24 录原型定的)
    assert v.settle_size_percent == 15.0
    assert (v.push_size_ratio, v.push_height_ratio, v.push_frames) == (1.4, 1.25, 3)
    assert (v.knock_hand_widths, v.knock_return_hand_widths) == (0.4, 0.3)
    assert (v.knock_window_s, v.knock_percent) == (3.0, 50.0)


def test_握拳平移的方向写错就报错(tmp_path):
    text = DEMO_TOML.replace('direction = "left"', 'direction = "backward"', 1)
    with pytest.raises(ConfigError, match="forward / back"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_摄像头编号和分辨率也在演示配置里(tmp_path):
    cfg = load_demo_config(write(tmp_path, DEMO_TOML, "demo.toml"))
    assert cfg.vision.camera_index == 0
    assert cfg.vision.camera_px == (1280, 720)


def test_轴比写成小于一就报错(tmp_path):
    # 主方向的位移至少是另一方向的几倍,小于 1 说不通
    text = DEMO_TOML.replace("move_axis_ratio = 3.0", "move_axis_ratio = 0.3")
    with pytest.raises(ConfigError, match="move_axis_ratio"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_保持时长写成毫秒时报出单位不对(tmp_path):
    text = DEMO_TOML.replace("pose_hold_s = 1.0", "pose_hold_s = 1000")
    with pytest.raises(ConfigError) as err:
        load_demo_config(write(tmp_path, text, "demo.toml"))
    assert "秒" in str(err.value)


def test_缺了启动对准那一节就报错(tmp_path):
    text = DEMO_TOML.replace("[vision]", "[vision_typo]")
    with pytest.raises(ConfigError, match="vision"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_对准参数少写一个就报出是哪个(tmp_path):
    text = DEMO_TOML.replace("too_close_px = 107.0", "")
    with pytest.raises(ConfigError, match="too_close_px"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_对准那一节多写一个字段也报出来(tmp_path):
    text = DEMO_TOML.replace("[vision]", "[vision]\nhold_sec = 1.2")
    with pytest.raises(ConfigError, match="hold_sec"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_检测区域写成三个数就报出来(tmp_path):
    text = DEMO_TOML.replace("detection_region_px = [800, 600]", "detection_region_px = [800, 600, 1]")
    with pytest.raises(ConfigError, match="detection_region_px"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_太近的门槛写成手宽倍数时报出单位不对(tmp_path):
    text = DEMO_TOML.replace("too_close_px = 107.0", "too_close_px = 1.25")
    with pytest.raises(ConfigError) as err:
        load_demo_config(write(tmp_path, text, "demo.toml"))
    assert "像素" in str(err.value)


def test_文件不存在时说清是哪个文件(tmp_path):
    missing = tmp_path / "没有这份.toml"
    with pytest.raises(ConfigError, match="没有这份.toml"):
        load_arm_config(missing)


# ---- 项目里那三份真配置 ------------------------------------------------


def test_项目里的三份配置都读得出来():
    from robot_side.config import CONFIG_DIR, HOME_ACTION, arm_config_path, load_demo

    demo = load_demo()
    assert {a.name for a in demo.actions} == {
        "point_left", "point_right", "raise", "lower", "move_away", "come_closer",
        "draw_square", "nod", "bow",
    }
    # 每个动作各有一个触发手势,另有一个手势对应回待命(保留名 home)
    assert {t.action for t in demo.triggers} == {a.name for a in demo.actions} | {HOME_ACTION}
    for arm_name in ("jaka", "nova5"):
        cfg = load_arm_config(arm_config_path(arm_name))
        assert len(cfg.standby_joints_deg) == 6
        assert cfg.robot_name
    assert (CONFIG_DIR / "demo.toml").is_file()


def test_路点可以自己写姿态(tmp_path):
    # 停在终点的动作最后一个路点不能改姿态,所以这里改成做完回原处
    text = DEMO_TOML.replace(
        "waypoints = [{ dx_mm = -150.0 }]",
        "waypoints = [{ dx_mm = -150.0, orientation_deg = [180.0, 0.0, 90.0] }]",
    ).replace('end = "stay"', 'end = "back"', 1)
    cfg = load_demo_config(write(tmp_path, text, "demo.toml"))
    assert cfg.action("point_left").waypoints[0].orientation_deg == (180.0, 0.0, 90.0)
    # 不写的路点还是 None,交给默认姿态
    assert cfg.action("nod").waypoints[0].orientation_deg is None


def test_路点姿态写错也报出来(tmp_path):
    text = DEMO_TOML.replace(
        "waypoints = [{ dx_mm = -150.0 }]",
        "waypoints = [{ dx_mm = -150.0, orientation_deg = [180.0, 0.0] }]",
    )
    with pytest.raises(ConfigError, match="orientation_deg"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_不写默认姿态就用法兰朝下(tmp_path):
    from robot_side.config import DEFAULT_ORIENTATION_DEG

    text = DEMO_TOML.replace("default_orientation_deg = [180.0, 0.0, 0.0]\n", "")
    cfg = load_demo_config(write(tmp_path, text, "demo.toml"))
    assert cfg.default_orientation_deg == DEFAULT_ORIENTATION_DEG


# ---- 做完停在哪儿、活动范围(2026-09-23 起动作做完不回待命) -----------


def test_动作没写做完怎么停就报错(tmp_path):
    text = DEMO_TOML.replace('end = "stay"\n', "", 1)
    with pytest.raises(ConfigError, match="end"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_做完怎么停只认stay和back(tmp_path):
    text = DEMO_TOML.replace('end = "stay"', 'end = "home"', 1)
    with pytest.raises(ConfigError, match="stay 或 back"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_停在终点的动作最后一个路点不能改姿态(tmp_path):
    # 停靠点一律是默认姿态,不然下一个动作一边走一边转回来
    text = DEMO_TOML.replace('end = "back"', 'end = "stay"', 1)
    with pytest.raises(ConfigError, match="不能改姿态"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_没写活动范围就报错(tmp_path):
    text = DEMO_TOML.replace("range_mm = [300.0, 0.0, 200.0]\n", "")
    with pytest.raises(ConfigError, match="range_mm"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_活动范围不能是负的(tmp_path):
    text = DEMO_TOML.replace("range_mm = [300.0, 0.0, 200.0]", "range_mm = [-300.0, 0.0, 200.0]")
    with pytest.raises(ConfigError, match="range_mm"):
        load_demo_config(write(tmp_path, text, "demo.toml"))


def test_正式的演示配置里左右上下前后停在终点画方块点头回原处():
    demo = load_demo()
    ends = {a.name: a.end for a in demo.actions}
    assert ends == {
        "point_left": "stay",
        "point_right": "stay",
        "raise": "stay",
        "lower": "stay",
        "move_away": "stay",
        "come_closer": "stay",
        "draw_square": "back",
        "nod": "back",
        "bow": "back",
    }
    # 左右各两步、上下各两步、前后各一步
    assert demo.run.range_mm == (300.0, 100.0, 200.0)
    # 往前(推)是远离操作者(+Y),往后(敲两下)是朝操作者
    assert demo.action("move_away").waypoints[-1].dy_mm == 100.0
    assert demo.action("come_closer").waypoints[-1].dy_mm == -100.0
    assert demo.action_for_trigger("fist_move", "forward") == "move_away"
    assert demo.action_for_trigger("fist_move", "back") == "come_closer"


def test_速度超过机械臂上限的那几个报得出来():
    # 超过的发出去会被截在上限上;2026-09-24 用户改成 200,看上去和 100 一样快
    import dataclasses

    from robot_side.config import arm_config_path, load_demo, over_linear_cap

    demo = load_demo()
    arm = load_arm_config(arm_config_path("jaka"))
    first, second, *rest = demo.actions
    demo = dataclasses.replace(
        demo,
        actions=(
            dataclasses.replace(first, speed_mms=arm.max_linear_mms * 2),
            dataclasses.replace(second, speed_mms=arm.max_linear_mms),
            *(dataclasses.replace(a, speed_mms=50.0) for a in rest),
        ),
    )
    assert over_linear_cap(demo, arm) == [(first.name, arm.max_linear_mms * 2)]

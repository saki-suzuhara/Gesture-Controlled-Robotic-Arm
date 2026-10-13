"""只读一次 JAKA 控制柜的上电 / 使能状态,吐一行 JSON。

    C:\\RoboDK\\Python-Embedded\\python.exe robot_side/scripts/read_jaka_status.py <控制柜IP>

**这一份跑在 RoboDK 自带的嵌入式 Python 3.10 上,不是项目的 3.13。** 所以它
**一行 `robot_side` 都不许 import**,也不用项目里的任何东西——只用标准库。
为什么非得这样:`jkrc.pyd`(2023-08 编译)在 3.12 起 `import` 就抛 `SystemError`,
而 RoboDK 的 JAKA 驱动本身又没有上电 / 使能这两个调用,状态位也不经它往外报
(`notes/jaka-first-connection-results.md`)。查上电使能只剩这一条路。

**只读。** 登录、读一次状态、登出,就这三步。`power_on` / `enable_robot` /
`clear_error` 一个都不碰:上电使能由人在示教器上做,程序只检查不执行(用户故事 27)。
也不发任何运动。

**「读成功」不等于「真连上了」**(2026-09-22 在没有控制柜的情况下实测):对着一个根本
不存在的 IP,`jkrc.login()` 卡 50 秒之后仍然返回 `(0,)`,`get_robot_status()` 也
返回 0,状态段是一串 0 或者未初始化的内存(见 `looks_live`)。所以这里读完还要看
**状态段本身像不像真的**,不像就报读不成——宁可说「读不出来」,也不能把一段垃圾
当成「没上电」报给人:那会让人跑去示教器上按半天,而问题其实在网线上。

吐出来的 JSON,成功时长这样:

    {"ok": true, "powered": 1, "enabled": 1, "status_head": [0, 1, 1, 1, 0.13, ...]}

读不成时 `{"ok": false, "error": "..."}`,退出码仍是 0——怎么处理由调用方
(`robot_side.real.read_power_state`)定,它一律当作「读不出来」,绝不猜成「有」。
"""

import json
import sys

JKRC_DIR = r"C:\RoboDK\api\Robot"
"""`jkrc.pyd` 在哪儿。RoboDK 的 JAKA 驱动用的也是这一份。"""

POWERED_INDEX = 2
ENABLED_INDEX = 3
RAPIDRATE_INDEX = 4
"""上电、使能、速度倍率在 `get_robot_status()[1]` 里的位置。

来源:第一次上 JAKA 实测(`notes/jaka-first-connection-results.md`)。
**别去读 `get_robot_state()` 的第三位**——那一位上电之后仍然是 0(推测是 servo
流模式),上一轮护栏就是在这里判错,在用户已经使能的情况下拒绝下发。

旁证(2026-09-22):同一版 SDK 吐出来的状态段里,下标 4 是一个浮点数(和笔记里读到
的「速度倍率 0.13」对得上),下标 19 是一个六元组(和笔记里「关节角读 `[1][19]`」
对得上)。整段的排布和笔记一致,所以这几个下标可信。

即便如此,它们只在那一版 SDK(V2.1.3_stable_win)上核实过,所以下面把状态头一起
吐出来:数对不上时,人能照着 `status_head` 自己看一眼。
"""

STATUS_HEAD = 8
"""吐多少个状态位出来给人对。只作诊断,不参与判定。"""

RAPIDRATE_FLOOR = 0.001
RAPIDRATE_CEILING = 1.0
"""速度倍率的合理范围,用来认出「这段状态不是真的」。

实测对照:真控制柜读到 **0.13**;对着不存在的 IP 读到 **0.0** 和
**2.1e-307**(未初始化的内存)。倍率是一个占比,真值落不到 0.001 以下,
所以这一条把两次假读都挡下了。
"""


def fail(why):
    print(json.dumps({"ok": False, "error": str(why)}, ensure_ascii=False))
    return 0


def looks_live(powered, enabled, rapidrate):
    """这段状态像不像从一台真控制柜上读来的。不像就别拿它当答案。

    三条一起看:上电位和使能位必须是 0 或 1(假读给过 7471201),速度倍率必须落在
    一个占比说得通的范围里(假读给过 0.0 和 2.1e-307)。
    """
    if powered not in (0, 1) or enabled not in (0, 1):
        return False
    if not isinstance(rapidrate, float):
        return False
    return RAPIDRATE_FLOOR <= rapidrate <= RAPIDRATE_CEILING


def main(argv):
    if len(argv) < 2 or not argv[1].strip():
        return fail("用法:read_jaka_status.py <控制柜IP>")
    ip = argv[1].strip()

    if JKRC_DIR not in sys.path:
        sys.path.insert(0, JKRC_DIR)
    try:
        import jkrc
    except Exception as exc:
        return fail(
            "import jkrc 失败(%s: %s);这一份要用 RoboDK 自带的 Python 3.10 跑"
            % (type(exc).__name__, exc)
        )

    rc = None
    try:
        rc = jkrc.RC(ip)
        ret = rc.login()
        if not (isinstance(ret, tuple) and ret and ret[0] == 0):
            return fail("jkrc login 返回 %r(0 才是成功);控制柜 IP 对不对?" % (ret,))
        status = rc.get_robot_status()
    except Exception as exc:
        return fail("读控制柜状态时出错(%s: %s)" % (type(exc).__name__, exc))
    finally:
        if rc is not None:
            try:
                rc.logout()
            except Exception:
                pass  # 读完就走,登不登得出去不影响这一次的答案

    if not (isinstance(status, tuple) and len(status) >= 2):
        return fail("get_robot_status 返回 %r,看不懂" % (status,))
    if status[0] != 0:
        return fail("get_robot_status 返回码 %r(0 才是成功)" % (status[0],))
    bits = status[1]
    try:
        head = [_plain(v) for v in list(bits)[:STATUS_HEAD]]
    except TypeError:
        return fail("get_robot_status 的状态段是 %r,不是一串数" % (bits,))
    if len(bits) <= RAPIDRATE_INDEX:
        return fail("状态段只有 %d 项,读不到上电使能;状态头 %r" % (len(bits), head))

    powered = bits[POWERED_INDEX]
    enabled = bits[ENABLED_INDEX]
    if not looks_live(powered, enabled, bits[RAPIDRATE_INDEX]):
        return fail(
            "读回来的状态段不像真的(状态头 %r);多半是根本没连上控制柜,"
            "而不是没上电使能——先查网线和 IP" % (head,)
        )

    print(
        json.dumps(
            {
                "ok": True,
                "powered": _plain(powered),
                "enabled": _plain(enabled),
                "status_head": head,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _plain(value):
    """把状态位变成 JSON 吐得出去的东西。吐不出去的原样转成字符串,由人去看。"""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return repr(value)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

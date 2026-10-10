# -*- coding: utf-8 -*-
"""EISeg 第二组定制补丁（幂等）：
1) config.yaml: 上一张 turn_prev 由 S 改为 A，下一张 turn_next 由 F 改为 D；
2) app.py completeLastMask: 去掉“是否完成最后一个目标”弹窗，保存时直接完成并保存。
"""
import io

CFG = r"D:\anaconda\envs\eiseg\lib\site-packages\eiseg\config\config.yaml"
APP = r"D:\anaconda\envs\eiseg\lib\site-packages\eiseg\app.py"


def rw(path, fn):
    with io.open(path, "r", encoding="utf-8") as f:
        s = f.read()
    s2 = fn(s)
    if s2 is None:
        return False
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(s2)
    return True


# ---- 1) 快捷键 A/D ----
def edit_cfg(s):
    changes = [("  turn_prev: S", "  turn_prev: A"),
               ("  turn_next: F", "  turn_next: D")]
    out = s
    for old, new in changes:
        if new in out:
            continue
        if out.count(old) != 1:
            print("[config] WARN anchor '%s' count=%d, skip this one" % (old, out.count(old)))
            continue
        out = out.replace(old, new)
    if out == s:
        print("[config] already set or unchanged")
        return None
    return out


# ---- 2) 去掉完成弹窗 ----
OLD_DIALOG = (
    "        res = self.warn(\n"
    "            self.tr(\"完成最后一个目标？\"),\n"
    "            self.tr(\"是否完成最后一个目标的标注，不完成不会进行保存。\"),\n"
    "            QMessageBox.Yes | QMessageBox.Cancel, )\n"
    "        if res == QMessageBox.Yes:\n"
    "            self.finishObject()\n"
    "            self.exportLabel()\n"
    "            self.setDirty(False)\n"
    "            return True\n"
    "        return False\n"
)
NEW_DIRECT = (
    "        # 定制：不弹窗，直接完成最后一个目标并保存\n"
    "        self.finishObject()\n"
    "        self.exportLabel()\n"
    "        self.setDirty(False)\n"
    "        return True\n"
)


def edit_app(s):
    if NEW_DIRECT in s:
        print("[app] already patched")
        return None
    if s.count(OLD_DIALOG) != 1:
        print("[app] ERROR dialog anchor count=%d, abort" % s.count(OLD_DIALOG))
        return None
    return s.replace(OLD_DIALOG, NEW_DIRECT)


if rw(CFG, edit_cfg):
    print("[config] shortcuts -> A(prev)/D(next) patched OK")
if rw(APP, edit_app):
    print("[app] completeLastMask dialog removed OK")
print("DONE")

# -*- coding: utf-8 -*-
"""给 EISeg 打两个补丁：
1) app.py: labelme 保存分支用 cv2.imread 读中文路径返回 None 导致崩溃 -> 改 imdecode；
2) widget/create.py: 滑块默认 pageStep=10 造成'点击可视化半径'只能取 0/10 -> 设 singleStep/pageStep=1。
幂等：已打过则跳过。
"""
import io

APP = r"D:\anaconda\envs\eiseg\lib\site-packages\eiseg\app.py"
CREATE = r"D:\anaconda\envs\eiseg\lib\site-packages\eiseg\widget\create.py"


def patch(path, old, new, tag):
    with io.open(path, "r", encoding="utf-8") as f:
        s = f.read()
    if new in s:
        print("[%s] already patched, skip" % tag)
        return
    cnt = s.count(old)
    if cnt != 1:
        print("[%s] ERROR: anchor found %d times, abort" % (tag, cnt))
        return
    s = s.replace(old, new)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(s)
    print("[%s] patched OK" % tag)


# --- patch 1: app.py 中文路径 imread ---
old1 = (
    "                img = cv2.imread(self.imagePath)\n"
    "                size = img.shape\n"
)
new1 = (
    "                # 中文路径 cv2.imread 返回 None，改用 imdecode 安全读取\n"
    "                img = cv2.imdecode(\n"
    "                    np.fromfile(self.imagePath, dtype=np.uint8),\n"
    "                    cv2.IMREAD_COLOR)\n"
    "                size = img.shape\n"
)
patch(APP, old1, new1, "app.py/imread")

# --- patch 2: create.py 滑块步长 ---
old2 = (
    "    sld.setOrientation(QtCore.Qt.Horizontal)\n"
)
new2 = (
    "    sld.setSingleStep(1)\n"
    "    sld.setPageStep(1)\n"
    "    sld.setOrientation(QtCore.Qt.Horizontal)\n"
)
patch(CREATE, old2, new2, "create.py/pageStep")

print("DONE")

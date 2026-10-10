# -*- coding: utf-8 -*-
"""把 label 目录下的灰度掩码(RGB_*.png) 转换为 EISeg 的 COCO 预标注，追加进 annotations.json。
- 保留已存在图片(手动 7 张)不动，只为尚未收录的图片新增；幂等按 file_name 去重。
- person(类别2) = 掩码 value==2 的外轮廓多边形(可多个)；
- ground(类别1) = 整幅矩形"减去 person 洞"的单环多边形(割线桥接)，与 person 共享同一边界，
  因此 ground 与 person 不重叠(符合手动标注约定)。
"""
import glob
import json
import os.path as osp
import shutil

import cv2
import numpy as np

LAB_DIR = r"D:\数据集\研磨\二次筛选\label"
COCO = osp.join(LAB_DIR, "annotations.json")
GROUND, PERSON = 1, 2
MIN_PERSON_AREA = 120
EPS = 1.5


def imread_gray(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def person_rings(mask):
    """person 掩码 -> 外轮廓多边形列表(Nx2 int32)。"""
    cnts, _ = cv2.findContours((mask == PERSON).astype(np.uint8) * 255,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_PERSON_AREA:
            continue
        c = cv2.approxPolyDP(c, EPS, True).reshape(-1, 2)
        if len(c) >= 3:
            rings.append(c.astype(np.int32))
    return rings


def _area2(ring):
    x = ring[:, 0].astype(np.float64)
    y = ring[:, 1].astype(np.float64)
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _proj_seg(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    if L == 0:
        return (ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L
    t = max(0.0, min(1.0, t))
    return (ax + t * dx, ay + t * dy)


def _bridge(poly, hole):
    """把 hole 用零宽垂直割线接到 poly 最近的边上(反向绕洞 -> 挖洞)。"""
    n = len(poly)
    hx, hy = hole[0]
    best = (1e18, 0, None)
    for k in range(n):
        pr = _proj_seg((hx, hy), poly[k], poly[(k + 1) % n])
        d = (pr[0] - hx) ** 2 + (pr[1] - hy) ** 2
        if d < best[0]:
            best = (d, k, pr)
    _, k, pr = best
    pr = (int(round(pr[0])), int(round(pr[1])))
    return poly[:k + 1] + [pr] + hole + [hole[0], pr] + poly[k + 1:]


def ground_flat(m, w, h):
    """ground 单环: 取 ground 掩码外轮廓(贴边 person 自动成凹口), 内部 person 作洞桥接。"""
    ground = (m == GROUND).astype(np.uint8) * 255
    cnts, hier = cv2.findContours(ground, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [0.0, 0.0, 0.0, float(h - 1), float(w - 1), float(h - 1), float(w - 1), 0.0]
    hier = hier[0]
    outers = [i for i in range(len(cnts)) if hier[i][3] == -1]
    oi = max(outers, key=lambda i: cv2.contourArea(cnts[i]))
    outer = cv2.approxPolyDP(cnts[oi], EPS, True).reshape(-1, 2)
    poly = [(int(x), int(y)) for x, y in outer]
    o_sgn = _area2(outer)
    for hi in range(len(cnts)):
        if hier[hi][3] != oi:
            continue
        hc = cv2.approxPolyDP(cnts[hi], EPS, True).reshape(-1, 2)
        if len(hc) < 3:
            continue
        hl = [(int(x), int(y)) for x, y in hc]
        if (_area2(hc) > 0) == (o_sgn > 0):  # 与外环同向 -> 翻成反向
            hl = hl[::-1]
        poly = _bridge(poly, hl)
    flat = []
    for x, y in poly:
        flat.append(float(x))
        flat.append(float(y))
    return flat


def main():
    data = json.load(open(COCO, "r", encoding="utf-8"))
    existing = {im["file_name"] for im in data["images"]}
    maxImgId = max([im["id"] for im in data["images"]], default=0)
    maxAnnId = max([a["id"] for a in data["annotations"]], default=0)

    masks = sorted(f for f in glob.glob(osp.join(LAB_DIR, "RGB_*.png"))
                   if not any(s in f for s in ("_cutout", "_pseudo")))
    added_img = added_ann = skipped = 0
    for mf in masks:
        base = osp.splitext(osp.basename(mf))[0]
        fname = base + ".jpg"
        if fname in existing:
            skipped += 1
            continue
        m = imread_gray(mf)
        if m is None:
            continue
        h, w = m.shape[:2]
        maxImgId += 1
        imgId = maxImgId
        data["images"].append({
            "id": imgId, "width": w, "height": h, "file_name": fname,
            "license": "", "flickr_url": "", "coco_url": "", "date_captured": "",
        })
        holes = person_rings(m)
        # ground: 整幅减 person 洞(稳健单环)
        maxAnnId += 1
        data["annotations"].append({
            "id": maxAnnId, "iscrowd": 0, "image_id": imgId, "category_id": GROUND,
            "segmentation": [ground_flat(m, w, h)],
            "area": float(w * h - sum(cv2.contourArea(hl) for hl in holes)),
            "bbox": [0.0, 0.0, float(w - 1), float(h - 1)],
        })
        added_ann += 1
        # person
        for ring in holes:
            xs, ys = ring[:, 0], ring[:, 1]
            flat = []
            for x, y in ring:
                flat.append(float(x))
                flat.append(float(y))
            maxAnnId += 1
            data["annotations"].append({
                "id": maxAnnId, "iscrowd": 0, "image_id": imgId, "category_id": PERSON,
                "segmentation": [flat],
                "area": float(cv2.contourArea(ring)),
                "bbox": [float(xs.min()), float(ys.min()),
                         float(xs.max() - xs.min()), float(ys.max() - ys.min())],
            })
            added_ann += 1
        added_img += 1

    shutil.copy(COCO, COCO + ".bak")
    with open(COCO, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print("added images=%d, added annotations=%d, skipped=%d" % (added_img, added_ann, skipped))
    print("total images=%d total annotations=%d" % (len(data["images"]), len(data["annotations"])))


if __name__ == "__main__":
    main()

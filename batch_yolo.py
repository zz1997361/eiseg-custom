# -*- coding: utf-8 -*-
"""批量自动标注 ground/person（YOLOv8-seg 人体实例分割）：
- person = YOLO 检出的人(多实例并集)；ground = 其余全部像素；
- 与人工真值约定一致(ground 铺满整帧减 person)；
- 写出灰度掩码 RGB_xxx.png(1=ground,2=person) + 伪彩 RGB_xxx_pseudo.png 到 label 目录；
- 跳过已存在掩码(你手动的 7 张)；可断点续跑。
"""
import glob
import os
import os.path as osp
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
from ultralytics import YOLO

IMG_DIR = r"D:\数据集\研磨\二次筛选"
LAB_DIR = r"D:\数据集\研磨\二次筛选\label"
MODEL = r"c:\Users\zhuliqiang\Favorites\yolov8n-seg.pt"
GROUND, PERSON = 1, 2
C_GROUND = (127, 255, 0)     # BGR，与现有伪彩一致
C_PERSON = (255, 0, 170)
CONF, IMGSZ = 0.30, 800


def imread(p, f=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), f)


def imwrite(path, img):
    cv2.imencode(osp.splitext(path)[1], img)[1].tofile(path)


def person_mask(res, h, w):
    pm = np.zeros((h, w), bool)
    if res.masks is None:
        return pm
    names = res.names
    mk = res.masks.data.cpu().numpy()
    for i, c in enumerate(res.boxes.cls.tolist()):
        if names[int(c)] == "person":
            pm |= cv2.resize(mk[i], (w, h)) > 0.5
    return pm


def main():
    os.makedirs(LAB_DIR, exist_ok=True)
    files = sorted(glob.glob(osp.join(IMG_DIR, "RGB_*.jpg")))
    total = len(files)
    todo = [f for f in files
            if not osp.exists(osp.join(LAB_DIR, osp.splitext(osp.basename(f))[0] + ".png"))]
    print("total=%d already_done=%d to_process=%d" % (total, total - len(todo), len(todo)), flush=True)

    model = YOLO(MODEL)
    t0 = time.time()
    n_person = n_ground = n_skip = n_bad = 0
    # 分批喂给 predict 以利用批量推理
    B = 16
    for start in range(0, len(todo), B):
        batch = todo[start:start + B]
        # 过滤已存在(断点续跑期间可能已写)
        batch = [p for p in batch
                 if not osp.exists(osp.join(LAB_DIR, osp.splitext(osp.basename(p))[0] + ".png"))]
        if not batch:
            continue
        results = model.predict(batch, conf=CONF, imgsz=IMGSZ, verbose=False)
        for path, res in zip(batch, results):
            base = osp.splitext(osp.basename(path))[0]
            img = imread(path)
            if img is None:
                n_bad += 1
                continue
            h, w = img.shape[:2]
            pm = person_mask(res, h, w)
            label = np.full((h, w), GROUND, np.uint8)
            label[pm] = PERSON
            imwrite(osp.join(LAB_DIR, base + ".png"), label)
            pseudo = np.zeros((h, w, 3), np.uint8)
            pseudo[label == GROUND] = C_GROUND
            pseudo[label == PERSON] = C_PERSON
            imwrite(osp.join(LAB_DIR, base + "_pseudo.png"), pseudo)
            if pm.any():
                n_person += 1
            else:
                n_ground += 1
        done = min(start + B, len(todo))
        if done % 160 < B or done == len(todo):
            el = time.time() - t0
            print("progress %d/%d  person=%d ground_only=%d  %.0fs  eta=%.0fs" % (
                done, len(todo), n_person, n_ground, el,
                el / done * (len(todo) - done)), flush=True)
    print("FINISHED person=%d ground_only=%d bad=%d processed=%d elapsed=%.0fs" % (
        n_person, n_ground, n_bad, n_person + n_ground + n_bad, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

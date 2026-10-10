# -*- coding: utf-8 -*-
"""EISeg 启动器：
1) 安装全局异常钩子，防止 PyQt5 因某个槽函数抛异常而直接终止进程（保证可连续标注）；
2) 启动 GUI 并自动加载研磨地面标注用的高精度通用模型。
"""
import os.path as osp
import sys
import traceback
from datetime import datetime

# ---- 全局异常钩子：捕获未处理异常写日志，但不退出程序 ----
ERROR_LOG = r"D:\eiseg_models\eiseg_error.log"


def _excepthook(exc_type, exc_value, tb):
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write("\n===== %s =====\n" % datetime.now())
            f.write("".join(traceback.format_exception(exc_type, exc_value, tb)))
    except Exception:
        pass
    # 关键：不调用 qFatal，吞掉异常，事件循环继续


sys.excepthook = _excepthook

from qtpy.QtWidgets import QApplication
from qtpy import QtCore
from eiseg import pjpath
from app import APP_EISeg

# 研磨地面标注模型：HRNet18_OCR64 高精度通用（RITM，COCO+LVIS 预训练）
MODEL_PARAM = r"D:\eiseg_models\hrnet18_ocr64\static_hrnet18_ocr64_cocolvis\static_hrnet18_ocr64_cocolvis.pdiparams"


def main():
    app = QApplication(sys.argv)
    settings = QtCore.QSettings(
        osp.join(pjpath, "config/setting.txt"), QtCore.QSettings.IniFormat)
    lang = settings.value("language")
    if lang != "中文":
        trans = QtCore.QTranslator(app)
        trans.load(osp.join(pjpath, "util/translate/%s" % lang))
        app.installTranslator(trans)

    window = APP_EISeg()
    window.currLanguage = lang
    window.showMaximized()
    QApplication.processEvents()

    # 自动加载模型（若文件存在）
    if osp.exists(MODEL_PARAM):
        try:
            window.setModelParam(MODEL_PARAM)
            window.statusbar.showMessage("HRNet18_OCR64 模型已自动加载", 8000)
        except Exception as e:
            print("auto load model failed:", e)
    else:
        print("model not found:", MODEL_PARAM)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

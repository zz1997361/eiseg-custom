# EISeg 定制版代码

对 [EISeg 1.1.1](https://github.com/PaddlePaddle/EISeg)（PaddleSeg 交互式分割标注工具）的一层定制：**启动器 + 补丁脚本 + 批量预标注脚本**。这些脚本不替换 EISeg 本体，而是在 pip 安装好的 EISeg 上做定点修改并封装启动。

> 免安装的绿色压缩包（自带 Python 环境 + HRNet18_OCR64 模型，双击即用）体积 520MB，超过 GitHub 单文件 100MB 限制，故不放在仓库里，改为在 **Releases** 提供下载，见本仓库右侧 Release 附件 `EISeg_Portable.zip`。

## 目录内容

| 文件 | 作用 |
|------|------|
| `eiseg_launch.py` | 启动器：安装全局异常钩子（防止 PyQt5 槽函数抛异常导致进程直接退出，可连续标注），启动 GUI 并自动加载模型 |
| `patch_eiseg.py` | 补丁①：`app.py` 中文路径 `cv2.imread` 返回 None 崩溃 → 改 `imdecode`；`widget/create.py` 可视化半径滑块步长 10 → 1 |
| `patch_eiseg2.py` | 补丁②：`config.yaml` 翻页快捷键改为 `A`(上一张)/`D`(下一张)；`app.py` 去掉保存时“是否完成最后一个目标”弹窗，直接完成并保存 |
| `masks_to_coco.py` | 把灰度分割掩码转成 COCO 多边形并追加进 `annotations.json`（EISeg 预标注只从 COCO 加载），ground 采用“整幅挖掉 person 洞”的单环，避免 ground 与 person 重叠 |
| `batch_yolo.py` | 用 YOLOv8-seg 对图片批量预标注人体，输出灰度掩码供 `masks_to_coco.py` 转换 |
| `使用说明.txt` | EISeg 交互标注使用说明（含 A/D 翻页、保存、微调修正顶点等操作） |

## 前置：安装 EISeg

```bash
conda create -n eiseg python=3.9
conda activate eiseg
pip install eiseg==1.1.1
# 需额外安装 paddlepaddle（CPU 或对应 CUDA 版）
```

## 打补丁（对已安装的 EISeg 生效）

脚本里硬编码了 Windows 下 `site-packages/eiseg` 的路径，请按你的环境改成对应路径后再运行：

```bash
python patch_eiseg.py
python patch_eiseg2.py
```

补丁均为**幂等**设计：已打过会自动跳过。

## 启动

编辑 `eiseg_launch.py` 里的 `MODEL_PARAM`（模型 `.pdiparams` 路径）与 `ERROR_LOG`，然后：

```bash
python eiseg_launch.py
```

## 批量预标注（可选）

1. 装依赖：`pip install ultralytics opencv-python`
2. 预标注人体：`python batch_yolo.py`（输出灰度掩码）
3. 转成 EISeg 可见的 COCO 预标注：`python masks_to_coco.py`（写入 `label/annotations.json`）

## 说明

- 脚本中的 `D:\...` 示例路径为原作者本机路径，复用前需改成你自己的路径。
- 本仓库仅含定制代码/脚本，不含 EISeg 本体、模型权重与数据集。
- 交互式标注的具体操作见 `使用说明.txt`。

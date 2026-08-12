# SODA-D 目标检测训练项目

这是一个基于 PyTorch 的 YOLO11 风格目标检测项目，使用项目内的 `CustomYOLO11`、Task-Aligned Assigner、IoU/DFL 损失，在 SODA-D 的 COCO 标注上训练。

## 1. 数据集

当前默认读取已经准备好的数据：

```text
/home/SODA-D/
├── Images/Images/                 # JPG 图片
└── Annotations/Annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

JSON 是 COCO detection 格式，标注框为 `[x, y, width, height]`。代码会自动转换为 `[x1, y1, x2, y2]`，类别编号也会自动映射为从 0 开始的连续编号。

注意：`Images.zip` 是分卷压缩包（同时存在 `Images.z01`、`Images.z02` 等文件）。必须让所有分卷都在同一目录后完整解压；当前目录若只有约 8,000 张图片，说明仍未解压完整。Linux 可使用 7-Zip 解压：

```bash
7z x /home/SODA-D/Images/Images.zip -o/home/SODA-D/Images/Images
```

训练代码会对 JSON 中找不到的图片打印警告并跳过，但正式训练不建议在缺图状态下开始。

## 2. 安装

建议使用 Python 3.10 或更高版本，并根据机器 CUDA 版本安装对应的 PyTorch：

```bash
cd /home/SOD
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果使用 GPU，请优先按照 PyTorch 官网命令安装匹配 CUDA 的 `torch` 和 `torchvision`。

## 3. 开始训练

默认配置为 640 输入尺寸、batch size 8、100 个 epoch、YOLO11-n 规模：

```bash
python train.py
```

常用覆盖参数：

```bash
python train.py --image-size 640 --batch-size 4 --epochs 100 --workers 4 --scale n
```

显存不足时优先减小 batch size；仍不足时可使用 `--scale s` 以外的更小规模只能选择 `n`，或降低输入尺寸为 512/416（必须是 32 的倍数）。

训练输出默认写入 `runs/soda_yolo11/`：

```text
best.pt   # 验证集 loss 最低的模型
last.pt   # 最近一轮模型
```

## 4. 项目结构

```text
.
├── train.py                    # 唯一推荐训练入口
├── config/dataset/config.py    # 训练配置
├── data/dataset_xjy.py         # COCO 检测数据集、增强、collate_fn
├── detection/model/model.py    # YOLO11 风格模型
├── detection/loss/             # Assigner、IoU、BBox、DFL 与总损失
├── detection/engine/train.py   # epoch、验证和 checkpoint 流程
└── runs/                       # 训练输出（自动创建）
```

## 5. 训练前快速检查

确认图片和 JSON 已解压后，可执行：

```bash
python -m py_compile train.py config/dataset/config.py data/dataset_xjy.py detection/engine/train.py
```

首次训练会打印训练/验证图片数量和类别名称。若提示某张图片不存在，说明图片压缩包尚未完整解压到 `/home/SODA-D/Images/Images`。

## 6. 说明

当前训练流程以验证集 loss 作为模型选择依据，尚未实现 mAP、Precision、Recall 统计和 NMS 推理脚本。模型的 `forward(..., decode=True)` 已提供原始输出解码，后续可在此基础上补充评估与推理模块。

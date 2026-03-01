## Purpose
给 AI 编码助手提供能快速上手本仓库的可操作指引：架构要点、常用命令、项目约定和调试提示。

## 快速一览
- **训练入口**: `train.py` — 负责数据加载、训练循环、早停与模型保存（输出 `best_model.pth`）。
- **模型定义**: `net.py` — U-Net 实现（`UNet(n_filters=32)` 默认）。
- **数据接口**: `data.py` — `COCOSegmentationDataset` 读取 COCO 格式标注并返回 `(image, mask)`。
- **推理脚本**: `predict.py` — 加载 `best_model.pth` 并生成 `predictions.png`。
- **下载/准备数据**: `download.py` + 解压到 `dataset/Brain_Tumor_Image_DataSet`（README 有示例命令）。
- **依赖**: `requirements.txt` 列出 `torch`, `torchvision`, `pycocotools`, `swanlab` 等。

## 大体架构与数据流
- 数据以 COCO 格式存放于 `dataset/Brain_Tumor_Image_DataSet/{train,valid,test}`，每组目录有 `_annotations.coco.json`。
- `train.py` 用 `pycocotools.coco.COCO` 加载元数据，并用 `COCOSegmentationDataset` 将图像与掩码送入 `DataLoader`。
- `net.UNet` 的前向输出为单通道概率图（Sigmoid），训练端用阈值 `0.5` 做二值化推断。

## 项目特有约定（重要）
- 数据预处理顺序（由代码决定）：`ToTensor()` → `Resize((256,256))` → `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`。
- 掩码处理：`COCOSegmentationDataset` 在 `__getitem__` 中构建 mask，并使用 `torch.nn.functional.interpolate(..., mode='nearest')` 统一到 `256x256`。
- 损失函数：训练使用 `combined_loss = 0.6 * dice + 0.4 * BCE`（见 `train.py`）。
- 早停与检查点：`patience=8`，当验证损失降低时保存为 `best_model.pth`。
- 日志/可视化：仓库集成 `swanlab`，训练中通过 `swanlab.init(...)` 与 `swanlab.log(...)` 记录指标与 matplotlib 图像。

## 典型开发/调试流程
1. 安装依赖：`pip install -r requirements.txt`（或在合适的 conda 环境里执行）。
2. 下载并解压数据：`python download.py`，然后 `unzip dataset/Brain_Tumor_Image_DataSet.zip -d dataset/`（见 README）。
3. 训练：`python train.py`（swanlab 配置在脚本内通过 `swanlab.init` 指定 `batch_size` / `learning_rate` / `num_epochs` / `device`）。
4. 推理：确保 `best_model.pth` 在项目根，运行 `python predict.py`，生成 `predictions.png`。

## 注意事项与已知细节
- `train.py` 会根据 `torch.cuda.is_available()` 选择 `device`，所以在无 GPU 机器上会自动回退到 CPU。
- `predict.py` 中 `load_model` 使用 `torch.load(..., weights_only=True)`：确保安装的 PyTorch 版本支持该参数，否则改为 `torch.load(model_path, map_location=device)`。
- `COCOSegmentationDataset` 返回的 `mask` 是 `float` 且带单通道；上游代码假设 `model` 输出与之可直接比较（同一 shape 与数据范围）。

## When editing code — actionable hints for AI agents
- 如果改动模型输入/输出形状，始终同步修改 `data.py` 的 `interpolate` 尺寸和 `predict.py` 的可视化尺寸（当前都为 256x256）。
- 若更改损失或训练超参，优先在 `swanlab.init(config=...)` 中暴露可变项以便实验记录一致。
- 当修改 `COCOSegmentationDataset`，保持返回 `(image_tensor, mask_tensor)` 的约定；许多训练/评估逻辑直接依赖此接口。

## 文件参考（快速链接）
- 训练脚本: [train.py](train.py)
- 模型定义: [net.py](net.py)
- 数据集加载: [data.py](data.py)
- 推理: [predict.py](predict.py)
- 数据目录: dataset/Brain_Tumor_Image_DataSet
- 依赖: [requirements.txt](requirements.txt)
- 使用说明: [README.md](README.md)

---
请确认是否需要我把 `predict.py` 的 `torch.load(..., weights_only=True)` 改成更兼容的调用，或把 `swanlab` 配置抽成 CLI/配置文件？我可以据此更新代码或文档。

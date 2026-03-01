# 🧠 Improved MK-UNet for Medical Image Segmentation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> 本项目为医学图像分割（以脑肿瘤 MRI 为例）提供了一套完整、高效且鲁棒的深度学习解决方案。基于经典的 U-Net 架构，我们融合了多核卷积、双重注意力机制与深度监督策略，显著提升了病灶边缘的分割精度与模型的泛化能力。

## ✨ 核心创新点 (Key Innovations)

1. **多核特征融合 (MKDC & MKIR)**：在编码器端引入多核深度可分离卷积（Multi-Kernel Depthwise Convolution）与倒置残差结构，自适应提取不同感受野下的多尺度病灶特征，极大丰富了深层语义信息。
2. **边缘感知与噪声抑制 (GAG & MKIRA)**：在解码阶段创新性地引入了分组注意力门（Group Attention Gate）与双重注意力机制，利用高层语义指导底层特征融合，有效过滤背景噪声，精准锐化肿瘤边界。
3. **深度监督机制 (Deep Supervision)**：在解码器中间层（Up2, Up3）增加辅助分类头，联合 Dice 与 BCE 构建多尺度损失函数（权重 `0.6 : 0.2 : 0.2`），有效缓解梯度弥散，加速模型收敛并提升微小病灶的敏感度。
4. **底层自适应抗噪 (CLAHE & Augmentation)**：在数据流底层内置了限制对比度自适应直方图均衡化（CLAHE）与空间完全同步的在线几何增强，从根源上解决了医学图像对比度低、泛化能力差的技术痛点。

## 📂 项目架构 (Project Structure)

```text
Medical-Segmentation/
├── dataset.py             # 核心数据流：COCO格式解析、CLAHE抗噪、同步几何增强
├── model_unet.py          # 基线模型：标准化 U-Net (带 BatchNorm)
├── model_mkunet.py        # 创新模型：Improved MK-UNet (带深度监督头)
├── train_unet.py          # Baseline 专属训练调度脚本
├── train_mkunet.py        # 创新模型专属训练调度脚本 (支持多尺度 Loss)
├── predict.py             # 全量测试集指标评估与定性大图生成
└── utils_model.py         # 模型轻量化与参数量评估工具
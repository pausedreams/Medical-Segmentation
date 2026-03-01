import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import random

# 导入你重命名后的新版模型
from model_unet import UNet
from model_mkunet import ImprovedUNet
from dataset import COCOSegmentationDataset

# ================= 配置区域 =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试集路径
TEST_IMG_DIR = './dataset/Brain_Tumor_Image_DataSet/test'
TEST_ANN_FILE = './dataset/Brain_Tumor_Image_DataSet/test/_annotations.coco.json'

# 🔥 选择你想预测和评估的模型 (修改这里切换)
# 选项: 'baseline' (标准U-Net) 或 'innovation' (改进的MK-UNet)
MODEL_TYPE = 'innovation' 

if MODEL_TYPE == 'baseline':
    MODEL_PATH = 'checkpoints/best_model_unet.pth'
    print("🚀 正在初始化基线模型 (Baseline U-Net)...")
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
else:
    MODEL_PATH = 'checkpoints/best_model_mkunet.pth'
    print("🚀 正在初始化创新模型 (Improved MK-UNet)...")
    model = ImprovedUNet(n_channels=3, n_classes=1).to(DEVICE)
# ===========================================

# --- 计算 Dice 和 IoU 的评估函数 ---
def calculate_metrics(pred, target, threshold=0.5):
    """计算单批次的 Dice 系数和 IoU"""
    pred_bin = (pred > threshold).float()
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    
    return dice.item(), iou.item()

def evaluate_and_visualize(model, dataset, device, num_samples=6):
    print(f"\n🔍 正在加载权重文件: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 找不到权重文件 {MODEL_PATH}！")
    
    # 加载权重并开启推理模式
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ 权重加载成功！\n")

    # =========================================
    # 1. 核心改进：全测试集定量评估 (计算 Test Dice/IoU)
    # =========================================
    print("📊 正在对整个测试集进行定量指标评估，请稍候...")
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    test_dice_sum = 0.0
    test_iou_sum = 0.0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            dice, iou = calculate_metrics(outputs, masks)
            test_dice_sum += dice
            test_iou_sum += iou
            
    avg_test_dice = test_dice_sum / len(test_loader)
    avg_test_iou = test_iou_sum / len(test_loader)
    
    print("=" * 45)
    print(f"🏆 【{MODEL_TYPE.upper()} 模型】最终测试集成绩单:")
    print(f"   Test - Dice: {avg_test_dice:.4f} | IoU: {avg_test_iou:.4f}")
    print("=" * 45)

    # =========================================
    # 2. 定性评估：生成 6张图 拼接的大矩阵图
    # =========================================
    print(f"\n📸 正在抽取 {num_samples} 张样本生成【统一可视化大图】...")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # 随机抽取索引
    indices = random.sample(range(len(dataset)), num_samples)

    # 🔥 创建一个大画布：num_samples 行，3 列
    # 动态调整高度：每行给 4 英寸的高度，保证图片不被挤压
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    with torch.no_grad():
        for row_idx, data_idx in enumerate(indices):
            image_tensor, mask_tensor = dataset[data_idx]
            input_tensor = image_tensor.unsqueeze(0).to(device)

            pred_tensor = model(input_tensor)
            pred_binary = (pred_tensor.squeeze().cpu() > 0.5).float().numpy()

            # 还原图像
            img_vis = image_tensor.cpu() * std + mean
            img_vis = torch.clamp(img_vis, 0, 1).permute(1, 2, 0).numpy()
            mask_vis = mask_tensor.squeeze().cpu().numpy()

            # 在对应的子图位置进行绘制
            # 第 1 列：原图
            axes[row_idx, 0].imshow(img_vis)
            axes[row_idx, 0].axis('off')
            if row_idx == 0:
                axes[row_idx, 0].set_title("Test Image (CLAHE)", fontsize=14, pad=10)

            # 第 2 列：金标准
            axes[row_idx, 1].imshow(mask_vis, cmap='gray')
            axes[row_idx, 1].axis('off')
            if row_idx == 0:
                axes[row_idx, 1].set_title("Ground Truth", fontsize=14, pad=10)

            # 第 3 列：模型预测
            axes[row_idx, 2].imshow(pred_binary, cmap='gray')
            axes[row_idx, 2].axis('off')
            if row_idx == 0:
                axes[row_idx, 2].set_title(f"Prediction ({MODEL_TYPE.upper()})", fontsize=14, pad=10)

    # 调整布局间距
    plt.tight_layout()
    
    # 🔥 自动保存为高清 PNG 图片 (分辨率300dpi)，可以直接用于写论文
    save_filename = f"{MODEL_TYPE}_predictions_grid.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"\n🎉 论文配图已成功保存为当前目录下的: 【{save_filename}】")
    
    # 弹出展示窗口
    plt.show()

def main():
    print("📊 正在初始化测试数据集 (自带 CLAHE 处理)...")
    test_coco = COCO(TEST_ANN_FILE)
    test_dataset = COCOSegmentationDataset(test_coco, TEST_IMG_DIR)
    
    # 执行评估与可视化，num_samples 改为 6
    evaluate_and_visualize(model, test_dataset, DEVICE, num_samples=6)

if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import swanlab
import numpy as np
import time

# --- 导入你的改进版模型 ---
# 确保你的模型文件名是 MK_net.py (注意下划线)
from MK_net import ImprovedUNet 
from data import COCOSegmentationDataset

# ================= 配置区域 =================
# 训练参数
BATCH_SIZE = 4
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据路径 (请确保这些路径是正确的)
TRAIN_IMG_DIR = './dataset/Brain_Tumor_Image_DataSet/train'
TRAIN_ANN_FILE = './dataset/Brain_Tumor_Image_DataSet/train/_annotations.coco.json'
VAL_IMG_DIR = './dataset/Brain_Tumor_Image_DataSet/valid'
VAL_ANN_FILE = './dataset/Brain_Tumor_Image_DataSet/valid/_annotations.coco.json'
# ===========================================

# --- 计算 Dice 和 IoU 的评估函数 ---
def calculate_metrics(pred, target, threshold=0.5):
    """
    计算 Dice 系数和 IoU
    """
    # 将概率图转换为二值图 (0或1)
    pred_bin = (pred > threshold).float()
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Dice Coefficient
    # 公式: 2 * Intersection / (Sum of pixels)
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    
    # IoU (Intersection over Union)
    # 公式: Intersection / (Union - Intersection)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    
    return dice.item(), iou.item()

def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    
    # Dice Loss
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_loss = 1 - ((2. * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6))
    
    return 0.4 * bce + 0.6 * dice_loss

def main():
    print(f"🚀 使用设备: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. 初始化 SwanLab 实验跟踪 (MK-UNet 专用)
    # 尝试连接云端，如果网络有问题，它通常会报错或者回退
    # 请确保终端已经运行过 `swanlab login`
    try:
        run = swanlab.init(
            project="Medical-Image-Segmentation-Graduation", 
            experiment_name="Train-MK-UNet-Improved-Cloud", # 实验名标明是改进版
            config={
                "model": "MK-UNet",
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "device": str(DEVICE)
            },
            # mode="local" # 注释掉这一行以尝试连接云端。如果报错 SSL，请取消注释开启离线模式
        )
    except Exception as e:
        print(f"⚠️ SwanLab 云端连接失败: {e}")
        print("   切换到本地离线模式...")
        run = swanlab.init(
            project="Medical-Image-Segmentation-Graduation", 
            experiment_name="Train-MK-UNet-Improved-Local",
            config={
                "model": "MK-UNet",
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "device": str(DEVICE)
            },
            mode="local"
        )

    # 2. 准备数据集和加载器
    # 定义图像预处理：调整大小至 256x256 并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("📊 正在加载数据集...")
    # 先初始化 COCO 对象
    train_coco = COCO(TRAIN_ANN_FILE)
    val_coco = COCO(VAL_ANN_FILE)

    # 再创建数据集
    train_dataset = COCOSegmentationDataset(train_coco, TRAIN_IMG_DIR, transform=transform)
    val_dataset = COCOSegmentationDataset(val_coco, VAL_IMG_DIR, transform=transform)
    
    # Windows 下通常建议 num_workers=0，如果有性能问题可以尝试设置为 2 或 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 初始化模型 (直接加载 ImprovedUNet)
    print(f"🚀 正在加载改进版 Improved MK-UNet...")
    model = ImprovedUNet(n_channels=3, n_classes=1).to(DEVICE)
    save_path = 'best_model_MK_UNet.pth' # 专门保存为 MK-UNet 的权重

    # 打印参数量，确认轻量化效果
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔥 当前 MK-UNet 参数量: {total_params/1e6:.4f} M (百万)")

    # 4. 定义损失函数和优化器
    criterion = combined_loss # 使用组合损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 训练循环
    best_val_dice = 0.0 # 修改为监控 Dice
    
    print("🏁 开始训练...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        train_dice_sum = 0
        train_iou_sum = 0
        
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 计算训练集的指标
            dice, iou = calculate_metrics(outputs, masks)
            train_dice_sum += dice
            train_iou_sum += iou
            
            # 打印进度 (每10个batch)
            if (i + 1) % 10 == 0:
                print(f"   Batch {i+1}/{len(train_loader)} Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = train_dice_sum / len(train_loader)
        avg_train_iou = train_iou_sum / len(train_loader)
        
        # 6. 验证循环
        model.eval()
        val_loss = 0
        val_dice_sum = 0
        val_iou_sum = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # 计算验证集的 Dice 和 IoU
                dice, iou = calculate_metrics(outputs, masks)
                val_dice_sum += dice
                val_iou_sum += iou

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice_sum / len(val_loader)
        avg_val_iou = val_iou_sum / len(val_loader)
        
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        
        # 打印详细日志
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] (耗时: {epoch_duration:.2f}s)")
        print(f"  Train - Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f} | IoU: {avg_train_iou:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}   | Dice: {avg_val_dice:.4f}   | IoU: {avg_val_iou:.4f}")

        # 记录到 SwanLab (所有指标)
        swanlab.log({
            "Train/Loss": avg_train_loss,
            "Train/Dice": avg_train_dice,
            "Train/IoU": avg_train_iou,
            "Val/Loss": avg_val_loss,
            "Val/Dice": avg_val_dice,
            "Val/IoU": avg_val_iou
        })

        # 7. 保存最佳模型 (依据: Val Dice 越高越好)
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), save_path)
            print(f"✅ 模型性能提升 (Dice: {avg_val_dice:.4f})，已保存至 {save_path}")

    total_duration = time.time() - start_time
    print(f"🎉 训练结束！总耗时: {total_duration/60:.2f} 分钟")

if __name__ == '__main__':
    main()
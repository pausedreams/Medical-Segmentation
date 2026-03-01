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
from net import UNet
# 如果您使用的是改进版，请确保这里导入的是正确的模型类，例如:
from MK_net import ImprovedUNet 
from data import COCOSegmentationDataset


# ================= 配置区域 =================
# 数据路径设置
train_dir = './dataset/Brain_Tumor_Image_DataSet/train'
val_dir = './dataset/Brain_Tumor_Image_DataSet/valid'
test_dir = './dataset/Brain_Tumor_Image_DataSet/test'

train_annotation_file = './dataset/Brain_Tumor_Image_DataSet/train/_annotations.coco.json'
test_annotation_file = './dataset/Brain_Tumor_Image_DataSet/test/_annotations.coco.json'
val_annotation_file = './dataset/Brain_Tumor_Image_DataSet/valid/_annotations.coco.json'
# ===========================================

# 定义损失函数
def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def combined_loss(pred, target):
    dice = dice_loss(pred, target)
    bce = nn.BCELoss()(pred, target)
    return 0.6 * dice + 0.4 * bce

# --- 新增：计算 Dice 和 IoU 的评估函数 ---
def calculate_metrics(pred, target, threshold=0.5):
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

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_dice = 0.0 # 修改为根据 Dice 保存最佳模型
    patience = 8
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        train_dice = 0
        train_iou = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # 计算指标
            acc = (outputs.round() == masks).float().mean().item()
            dice, iou = calculate_metrics(outputs, masks)
            
            train_acc += acc
            train_dice += dice
            train_iou += iou

        # 计算平均值
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        
        # 验证循环
        model.eval()
        val_loss = 0
        val_acc = 0
        val_dice = 0
        val_iou = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                acc = (outputs.round() == masks).float().mean().item()
                dice, iou = calculate_metrics(outputs, masks)
                
                val_acc += acc
                val_dice += dice
                val_iou += iou
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        # 记录到 SwanLab
        swanlab.log(
            {
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/dice": train_dice, # 新增
                "train/iou": train_iou,   # 新增
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/dice": val_dice,     # 新增
                "val/iou": val_iou,       # 新增
            },
            step=epoch+1
        )
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')
        
        # 早停策略 (修改为监控 Dice，Dice 越高越好)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✅ New best model saved! (Val Dice: {val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def main():
    # 初始化 SwanLab
    swanlab.init(
        project="Unet-Medical-Segmentation",
        experiment_name="Improved-Training-Metrics",
        config={
            "batch_size": 4,  # 根据您的显存调整
            "learning_rate": 1e-4,
            "num_epochs": 40,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
    )
    
    device = torch.device(swanlab.config["device"])
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载COCO对象 (需确保json路径正确)
    train_coco = COCO(train_annotation_file)
    val_coco = COCO(val_annotation_file)
    test_coco = COCO(test_annotation_file)
    
    # 创建数据集
    train_dataset = COCOSegmentationDataset(train_coco, train_dir, transform=transform)
    val_dataset = COCOSegmentationDataset(val_coco, val_dir, transform=transform)
    test_dataset = COCOSegmentationDataset(test_coco, test_dir, transform=transform)
    
    # 创建数据加载器
    BATCH_SIZE = swanlab.config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型 (这里默认用原始 UNet，如果你想跑改进版，请替换为 ImprovedUNet)
    #model = UNet(n_filters=32).to(device) 
    model = ImprovedUNet(n_channels=3, n_classes=1).to(device) # 如果你要跑改进版，取消这行注释
    
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🔥 当前模型参数量: {total_params/1e6:.4f} M (百万)\n")
    optimizer = optim.Adam(model.parameters(), lr=swanlab.config["learning_rate"])
    
    # 开始训练
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=combined_loss,
        optimizer=optimizer,
        num_epochs=swanlab.config["num_epochs"],
        device=device,
    )
    
    # ... (后续的可视化代码可以保留) ...

# ... (visualize_predictions 函数保持不变) ...

if __name__ == '__main__':
    main()
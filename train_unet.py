import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import swanlab

# 仅导入基线模型和数据集
from model_unet import UNet
from dataset import COCOSegmentationDataset

# ================= 配置区域 =================
# 数据路径设置
train_dir = './dataset/Brain_Tumor_Image_DataSet/train'
val_dir = './dataset/Brain_Tumor_Image_DataSet/valid'
test_dir = './dataset/Brain_Tumor_Image_DataSet/test'

train_annotation_file = './dataset/Brain_Tumor_Image_DataSet/train/_annotations.coco.json'
test_annotation_file = './dataset/Brain_Tumor_Image_DataSet/test/_annotations.coco.json'
val_annotation_file = './dataset/Brain_Tumor_Image_DataSet/valid/_annotations.coco.json'
# ===========================================

# 定义损失函数 (Dice + BCE 联合损失，很棒的设计)
def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def combined_loss(pred, target):
    dice = dice_loss(pred, target)
    bce = nn.BCELoss()(pred, target)
    return 0.6 * dice + 0.4 * bce

# 计算 Dice 和 IoU 的评估函数
def calculate_metrics(pred, target, threshold=0.5):
    # 将概率图转换为二值图 (0或1)
    pred_bin = (pred > threshold).float()
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Dice Coefficient
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    
    # IoU (Intersection over Union)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    
    return dice.item(), iou.item()

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_dice = 0.0  # 根据 Dice 保存最佳模型
    patience = 8
    patience_counter = 0

    # 确保保存权重的文件夹存在
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/best_model_unet.pth'

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
                "train/dice": train_dice,
                "train/iou": train_iou,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/dice": val_dice,
                "val/iou": val_iou,
            },
            step=epoch+1
        )
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')
        
        # 早停策略 (监控 Val Dice)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved to {save_path}! (Val Dice: {val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

def main():
    # 初始化 SwanLab (修改为 Baseline 的专属实验名称)
    swanlab.init(
        project="Unet-Medical-Segmentation",
        experiment_name="Baseline-UNet-Training",
        config={
            "batch_size": 4, 
            "learning_rate": 1e-4,
            "num_epochs": 40,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_type": "Standard_UNet"
        },
    )
    
    device = torch.device(swanlab.config["device"])
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        # 注意：此处使用的均值和方差是 ImageNet 的标准值。
        # 如果是针对脑部 MRI，后续我们可以在 dataset.py 中引入专门的医疗图像归一化或 CLAHE 预处理。
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载COCO对象
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
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化基线模型
    print("🚀 Initializing Baseline UNet...")
    model = UNet(n_channels=3, n_classes=1).to(device) 
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔥 当前 Baseline 模型参数量: {total_params/1e6:.4f} M\n")
    
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

if __name__ == '__main__':
    main()
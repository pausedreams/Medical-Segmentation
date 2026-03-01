import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import swanlab

# --- 导入你的改进版模型和数据集 ---
from model_mkunet import ImprovedUNet 
from dataset import COCOSegmentationDataset

# ================= 配置区域 =================
# 训练参数
BATCH_SIZE = 4
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据路径
TRAIN_IMG_DIR = './dataset/Brain_Tumor_Image_DataSet/train'
TRAIN_ANN_FILE = './dataset/Brain_Tumor_Image_DataSet/train/_annotations.coco.json'
VAL_IMG_DIR = './dataset/Brain_Tumor_Image_DataSet/valid'
VAL_ANN_FILE = './dataset/Brain_Tumor_Image_DataSet/valid/_annotations.coco.json'
# ===========================================

# --- 计算 Dice 和 IoU 的评估函数 ---
def calculate_metrics(pred, target, threshold=0.5):
    """计算 Dice 系数和 IoU"""
    pred_bin = (pred > threshold).float()
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    
    return dice.item(), iou.item()

# 定义联合损失函数 (BCE + Dice)
def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_loss = 1 - ((2. * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6))
    
    return 0.4 * bce + 0.6 * dice_loss

# ==========================================
# 新增：SwanLab 随机验证集可视化函数
# ==========================================
def log_predictions_to_swanlab(model, dataset, device, num_samples=4):
    """
    在训练结束后，随机抽取验证集样本进行可视化预测，并上传至 SwanLab
    """
    print(f"\n📸 正在生成 {num_samples} 张随机预测可视化图并上传 SwanLab...")
    model.eval() # 确保模型处于推理模式
    
    # 随机抽取样本索引
    indices = random.sample(range(len(dataset)), num_samples)
    swan_images = []
    
    # 定义反归一化参数 (还原为肉眼可看的 RGB 图像)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    with torch.no_grad():
        for idx in indices:
            image_tensor, mask_tensor = dataset[idx]
            
            # 送入显卡并增加 batch 维度
            input_tensor = image_tensor.unsqueeze(0).to(device)
            
            # 模型预测 (推理模式下只返回主输出 out_main)
            pred_tensor = model(input_tensor)
            pred_binary = (pred_tensor.squeeze().cpu() > 0.5).float().numpy()
            
            # 还原原始图像用于展示
            img_vis = image_tensor.cpu() * std + mean
            img_vis = torch.clamp(img_vis, 0, 1).permute(1, 2, 0).numpy()
            mask_vis = mask_tensor.squeeze().cpu().numpy()
            
            # --- 使用 Matplotlib 拼接对比图 ---
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img_vis)
            axes[0].set_title(f"Original Image (Sample {idx})")
            axes[0].axis('off')
            
            axes[1].imshow(mask_vis, cmap='gray')
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')
            
            axes[2].imshow(pred_binary, cmap='gray')
            axes[2].set_title("Predicted Mask (MK-UNet)")
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # 将 Matplotlib 图像转换为 SwanLab 格式
            swan_images.append(swanlab.Image(fig, caption=f"Validation Sample {idx}"))
            plt.close(fig)
            
    # 一次性上传所有对比图到 SwanLab
    swanlab.log({"Final_Visualization": swan_images})
    print("✅ 可视化结果已成功同步至 SwanLab 面板！")

# ==========================================
# 独立的训练核心函数 (已整合深度监督 Deep Supervision)
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    best_val_dice = 0.0
    patience = 8
    patience_counter = 0
    
    print("🏁 开始训练 Improved MK-UNet (开启深度监督 Deep Supervision)...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        train_dice_sum = 0
        train_iou_sum = 0
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # 🔥 1. 深度监督：接收主分支和两个辅助分支的预测结果
            outputs_main, outputs_up2, outputs_up3 = model(images)
            
            # 🔥 2. 将辅助分支的特征图上采样(放大)到真实 Mask 的尺寸 (256x256)
            outputs_up2 = F.interpolate(outputs_up2, size=masks.shape[2:], mode='bilinear', align_corners=False)
            outputs_up3 = F.interpolate(outputs_up3, size=masks.shape[2:], mode='bilinear', align_corners=False)
            
            # 🔥 3. 计算多尺度联合损失 (权重分配：主输出0.6, 辅助输出各0.2)
            loss_main = criterion(outputs_main, masks)
            loss_up2 = criterion(outputs_up2, masks)
            loss_up3 = criterion(outputs_up3, masks)
            loss = 0.6 * loss_main + 0.2 * loss_up2 + 0.2 * loss_up3
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 注意：计算准确率指标时，只看主分支的表现
            dice, iou = calculate_metrics(outputs_main, masks)
            train_dice_sum += dice
            train_iou_sum += iou
            
            if (i + 1) % 10 == 0:
                print(f"   Batch {i+1}/{len(train_loader)} Total Loss: {loss.item():.4f} (Main: {loss_main.item():.4f})")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice_sum / len(train_loader)
        avg_train_iou = train_iou_sum / len(train_loader)
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        val_dice_sum = 0
        val_iou_sum = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                # 🔥 验证模式下，模型只返回主分支输出
                outputs_main = model(images)
                loss = criterion(outputs_main, masks)
                
                val_loss += loss.item()
                dice, iou = calculate_metrics(outputs_main, masks)
                val_dice_sum += dice
                val_iou_sum += iou

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice_sum / len(val_loader)
        avg_val_iou = val_iou_sum / len(val_loader)
        
        epoch_duration = time.time() - epoch_start
        
        # 打印详细日志
        print(f"\nEpoch [{epoch+1}/{num_epochs}] (耗时: {epoch_duration:.2f}s)")
        print(f"  Train - Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f} | IoU: {avg_train_iou:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} | IoU: {avg_val_iou:.4f}")

        # 记录到 SwanLab
        swanlab.log({
            "Train/Loss": avg_train_loss,
            "Train/Dice": avg_train_dice,
            "Train/IoU": avg_train_iou,
            "Val/Loss": avg_val_loss,
            "Val/Dice": avg_val_dice,
            "Val/IoU": avg_val_iou
        }, step=epoch+1)

        # 早停与最佳模型保存策略
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ 模型性能提升 (Dice: {avg_val_dice:.4f})，已保存至 {save_path}")
        else:
            patience_counter += 1
            print(f"⚠️ 验证集 Dice 未提升 (Patience: {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"🛑 触发早停机制，训练在第 {epoch+1} 轮提前结束。")
                break

    total_duration = time.time() - start_time
    print(f"\n🎉 训练完全结束！总耗时: {total_duration/60:.2f} 分钟")


def main():
    print(f"🚀 使用设备: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. 初始化 SwanLab 实验跟踪
    try:
        run = swanlab.init(
            project="Medical-Image-Segmentation-Graduation", 
            experiment_name="Innovation-MKUNet-DeepSupervision",
            config={
                "model": "MK-UNet (Deep Supervision)",
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "device": str(DEVICE)
            }
        )
    except Exception as e:
        print(f"⚠️ SwanLab 云端连接失败: {e}\n   切换到本地离线模式...")
        run = swanlab.init(
            project="Medical-Image-Segmentation-Graduation", 
            experiment_name="Innovation-MKUNet-DeepSupervision-Local",
            config={
                "model": "MK-UNet (Deep Supervision)",
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "device": str(DEVICE)
            },
            mode="local"
        )

    # 2. 准备数据集和加载器 (Transform已内置于dataset.py)
    print("\n📊 正在加载数据集 (自带 CLAHE 与同步几何增强)...")
    train_coco = COCO(TRAIN_ANN_FILE)
    val_coco = COCO(VAL_ANN_FILE)

    train_dataset = COCOSegmentationDataset(train_coco, TRAIN_IMG_DIR)
    val_dataset = COCOSegmentationDataset(val_coco, VAL_IMG_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 初始化模型
    print("🚀 正在加载创新版 Improved MK-UNet (带辅助输出头)...")
    model = ImprovedUNet(n_channels=3, n_classes=1).to(DEVICE)
    
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/best_model_mkunet.pth' 

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔥 当前 MK-UNet 参数量: {total_params/1e6:.4f} M\n")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 调用训练核心函数
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=combined_loss,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        save_path=save_path
    )

    # ==========================================
    # 🔥 5. 训练结束：加载最佳权重并进行可视化检查
    # ==========================================
    print("\n🔍 正在加载最佳模型权重用于最终定性评估...")
    # 安全加载最佳权重
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    else:
        print("⚠️ 未找到最佳权重文件，将使用最后一代权重进行可视化。")
        
    # 调用可视化函数，从验证集中随机抽 4 张图
    log_predictions_to_swanlab(model, val_dataset, DEVICE, num_samples=4)

if __name__ == '__main__':
    main()
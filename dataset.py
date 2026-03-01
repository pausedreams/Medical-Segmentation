import os
import cv2
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class COCOSegmentationDataset(Dataset):
    def __init__(self, coco, image_dir, transform=None):
        """
        学术级医学图像数据集加载器
        :param coco: pycocotools 初始化的 COCO 对象
        :param image_dir: 图像所在文件夹路径
        :param transform: (遗留参数，内部已采用更安全的同步变换替代)
        """
        self.coco = coco
        self.image_dir = image_dir
        self.image_ids = coco.getImgIds()
        
        # 💡 导师黑科技：智能判断当前是否为训练集 (无需修改外部训练脚本)
        # 只有训练集才会开启随机翻转和旋转，验证集/测试集只做 Resize 和 CLAHE
        self.is_train = 'train' in image_dir.lower()

    def __len__(self):
        return len(self.image_ids)

    def apply_clahe(self, image_np):
        """
        像素级抗噪增强：对比度受限自适应直方图均衡化 (CLAHE)
        有效解决医学图像边缘模糊、对比度低的问题
        """
        # 创建 CLAHE 对象 (clipLimit 控制对比度阈值，tileGridSize 决定局部窗口大小)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 将 RGB 图像转换到 LAB 颜色空间，仅对亮度通道(L)进行增强
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return image_clahe
        elif len(image_np.shape) == 2 or image_np.shape[2] == 1:
            return clahe.apply(image_np)
        return image_np

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        # 1. 加载图像并进行 CLAHE 增强
        image_np = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
        image_np = self.apply_clahe(image_np)  # 注入像素级抗噪灵魂
        image_pil = Image.fromarray(image_np)

        # 2. 创建精准的二值化掩码 (Mask)
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        for ann in anns:
            mask_np = np.maximum(mask_np, self.coco.annToMask(ann))
        mask_pil = Image.fromarray(mask_np)

        # ==========================================
        # 3. 同步空间变换 (Synchronized Spatial Transforms)
        # ==========================================
        
        # 首先，统一将 Image 和 Mask Resize 到 256x256
        # Image 用双线性插值(BILINEAR)，Mask 必须用最近邻(NEAREST)防止产生0.5这种非法类别值
        image_pil = TF.resize(image_pil, (256, 256), interpolation=Image.BILINEAR)
        mask_pil = TF.resize(mask_pil, (256, 256), interpolation=Image.NEAREST)

        # 如果是训练集，则进行严格对齐的几何数据增强
        if self.is_train:
            # 50% 概率水平翻转
            if random.random() > 0.5:
                image_pil = TF.hflip(image_pil)
                mask_pil = TF.hflip(mask_pil)
            
            # 50% 概率垂直翻转
            if random.random() > 0.5:
                image_pil = TF.vflip(image_pil)
                mask_pil = TF.vflip(mask_pil)
            
            # 50% 概率随机旋转 (-15度 到 15度)
            if random.random() > 0.5:
                angle = random.randint(-15, 15)
                image_pil = TF.rotate(image_pil, angle, interpolation=Image.BILINEAR)
                mask_pil = TF.rotate(mask_pil, angle, interpolation=Image.NEAREST)

        # ==========================================
        # 4. 张量转换与标准化
        # ==========================================
        
        # 将 PIL 转换为 Tensor (会自动将像素值缩放到 0~1 之间)
        image_tensor = TF.to_tensor(image_pil)
        
        # Mask 转换为张量，并增加通道维度 [1, H, W]
        mask_tensor = torch.as_tensor(np.array(mask_pil), dtype=torch.float32).unsqueeze(0)

        # 对 Image 进行标准化 (医疗图像推荐使用 ImageNet 均值方差作为预训练基础)
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image_tensor, mask_tensor
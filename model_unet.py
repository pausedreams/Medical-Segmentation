import torch
import torch.nn as nn

# ==========================================
# 标准 U-Net (Baseline)
# 融入了 BatchNorm2d 以确保公平且强劲的基线性能
# ==========================================

# 定义 U-Net 模型的下采样块 (Encoder Block)
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0, max_pooling=True):
        super(DownBlock, self).__init__()
        # Conv -> BN -> ReLU 是目前最标准的学术配置
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2) if max_pooling else None
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dropout:
            x = self.dropout(x)
        # 保存用于跳跃连接的特征图 (Skip Connection)
        skip = x
        if self.maxpool:
            x = self.maxpool(x)
        return x, skip

# 定义 U-Net 模型的上采样块 (Decoder Block)
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # 使用反卷积进行上采样
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # 拼接后的通道数为 out_channels * 2
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        # 在通道维度(dim=1)进行特征拼接
        x = torch.cat([x, skip], dim=1)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# 定义完整的 U-Net 模型
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, n_filters=32):
        super(UNet, self).__init__()
        
        # 编码器路径 (Encoder)
        self.down1 = DownBlock(n_channels, n_filters)                  # 32
        self.down2 = DownBlock(n_filters, n_filters * 2)               # 64
        self.down3 = DownBlock(n_filters * 2, n_filters * 4)           # 128
        self.down4 = DownBlock(n_filters * 4, n_filters * 8)           # 256
        self.down5 = DownBlock(n_filters * 8, n_filters * 16)          # 512
        
        # 瓶颈层 (Bottleneck) - 移除 maxpooling，加入 dropout 防止过拟合
        self.bottleneck = DownBlock(n_filters * 16, n_filters * 32, dropout_prob=0.4, max_pooling=False) # 1024
        
        # 解码器路径 (Decoder)
        self.up1 = UpBlock(n_filters * 32, n_filters * 16)             # 512
        self.up2 = UpBlock(n_filters * 16, n_filters * 8)              # 256
        self.up3 = UpBlock(n_filters * 8, n_filters * 4)               # 128
        self.up4 = UpBlock(n_filters * 4, n_filters * 2)               # 64
        self.up5 = UpBlock(n_filters * 2, n_filters)                   # 32
        
        # 输出层 (Output)
        self.outc = nn.Conv2d(n_filters, n_classes, 1)

    def forward(self, x):
        # 编码器路径
        x1, skip1 = self.down1(x)      
        x2, skip2 = self.down2(x1)     
        x3, skip3 = self.down3(x2)     
        x4, skip4 = self.down4(x3)     
        x5, skip5 = self.down5(x4)     
        
        # 瓶颈层
        x6, _ = self.bottleneck(x5)    
        
        # 解码器路径
        x = self.up1(x6, skip5)    
        x = self.up2(x, skip4)     
        x = self.up3(x, skip3)     
        x = self.up4(x, skip2)     
        x = self.up5(x, skip1)     
        
        # 输出与激活
        x = self.outc(x)
        # 注意：因为你的 train_unet.py 中使用了 BCELoss，所以这里需要 sigmoid。
        # 如果未来换成 BCEWithLogitsLoss，则需要去掉此处的 sigmoid。
        return torch.sigmoid(x)

if __name__ == '__main__':
    # 简单的本地测试，确保网络张量流转正常
    model = UNet(n_channels=3, n_classes=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("✅ Baseline U-Net 测试通过！输出形状:", y.shape)
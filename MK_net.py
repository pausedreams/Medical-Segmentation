import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 基础工具函数与核心卷积模块
# ==========================================

# 通道洗牌：确保多路径特征充分融合
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    return x.view(batchsize, -1, height, width)

# MKDC 模块：多尺度特征提取核心
class MKDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1, 3, 5]):
        super(MKDC, self).__init__()
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                # 深度可分离卷积 (groups=in_channels)
                # 保持输入输出通道一致，方便后续拼接
                nn.Conv2d(in_channels, in_channels, k, padding=k//2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True)
            ) for k in kernel_sizes
        ])

    def forward(self, x):
        # 1. 并行计算并拼接 (Concat)
        # 这里的输出通道数 = 输入通道数 * 卷积核数量(3)
        out = torch.cat([dw(x) for dw in self.dwconvs], dim=1)
        
        # 2. 通道洗牌
        # groups 必须等于分支的数量 (这里是 3)
        return channel_shuffle(out, groups=len(self.dwconvs)) 

# MKIR 模块：多核倒置残差块
class MKIR(nn.Module):
    def __init__(self, in_c, out_c, expansion_factor=2, kernel_sizes=[1, 3, 5]):
        super(MKIR, self).__init__()
        # 中间层的通道数 (升维)
        ex_c = in_c * expansion_factor
        
        # 1. 升维 (1x1 Conv)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, ex_c, 1, bias=False), 
            nn.BatchNorm2d(ex_c), 
            nn.ReLU6(inplace=True)
        )
        
        # 2. 多核深度卷积 (MKDC)
        self.mkdc = MKDC(ex_c, kernel_sizes=kernel_sizes)
        
        # 🔥【关键修复点】计算 MKDC 输出后的总通道数
        # 因为 MKDC 使用了 concat，所以通道数翻了 len(kernel_sizes) 倍
        mkdc_out_c = ex_c * len(kernel_sizes)
        
        # 3. 降维 (1x1 Conv)
        # 输入必须匹配 MKDC 的输出通道数 (mkdc_out_c)
        self.pconv2 = nn.Sequential(
            nn.Conv2d(mkdc_out_c, out_c, 1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        
        # 4. 残差连接
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        # 残差结构: F(x) + x
        return self.pconv2(self.mkdc(self.pconv1(x))) + self.skip(x)

# ==========================================
# 2. 注意力机制模块 (Attention Blocks)
# ==========================================

# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 这里的 ratio=16 可能会让通道数变得太小，加上 max(1, ...) 保护
        reduced_planes = max(1, in_planes // ratio)
        
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 使用7x7大核卷积
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

# MKIRA: 多核倒置残差注意力模块 (用于解码器)
class MKIRA(nn.Module):
    def __init__(self, in_c, out_c):
        super(MKIRA, self).__init__()
        self.ca = ChannelAttention(in_c)
        self.sa = SpatialAttention()
        self.mkir = MKIR(in_c, out_c) 

    def forward(self, x):
        x = self.ca(x) * x # 通道加权
        x = self.sa(x) * x # 空间加权
        x = self.mkir(x)   # 特征提取与降维
        return x

# GAG: 分组注意力门 (用于跳跃连接)
class GAG(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(GAG, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, padding=1, groups=2, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, padding=1, groups=2, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi 

# ==========================================
# 3. U-Net 改进版组件与主类
# ==========================================

# 改进后的下采样块
class ImprovedDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImprovedDownBlock, self).__init__()
        self.mkir = MKIR(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.mkir(x)
        skip = x
        x = self.maxpool(x)
        return x, skip

# 改进后的上采样块
class ImprovedUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImprovedUpBlock, self).__init__()
        # 双线性插值
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # GAG模块
        self.gag = GAG(F_g=in_channels, F_l=out_channels, F_int=out_channels // 2)
        
        # MKIRA模块
        # 输入通道 = 上层输出(in_channels) + 跳跃连接(out_channels)
        self.mkira = MKIRA(in_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.gag(g=x, x=skip) 
        x = torch.cat([x, skip], dim=1)
        x = self.mkira(x)
        return x

# 最终模型类
class ImprovedUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(ImprovedUNet, self).__init__()
        # 轻量化通道配置
        filters = [16, 32, 64, 96, 160] 

        # 编码器
        self.down1 = ImprovedDownBlock(n_channels, filters[0])
        self.down2 = ImprovedDownBlock(filters[0], filters[1])
        self.down3 = ImprovedDownBlock(filters[1], filters[2])
        self.down4 = ImprovedDownBlock(filters[2], filters[3])

        # 瓶颈层
        self.bottleneck = MKIR(filters[3], filters[4])

        # 解码器
        self.up1 = ImprovedUpBlock(filters[4], filters[3])
        self.up2 = ImprovedUpBlock(filters[3], filters[2])
        self.up3 = ImprovedUpBlock(filters[2], filters[1])
        self.up4 = ImprovedUpBlock(filters[1], filters[0])

        # 输出层
        self.outc = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, x):
        # 编码
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)

        # 瓶颈
        x5 = self.bottleneck(x4)

        # 解码
        x = self.up1(x5, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return torch.sigmoid(self.outc(x))

if __name__ == '__main__':
    # 测试代码
    try:
        model = ImprovedUNet(n_channels=3, n_classes=1)
        x = torch.randn(4, 3, 256, 256)
        y = model(x)
        print("✅ 模型测试通过！")
        print(f"输入: {x.shape}")
        print(f"输出: {y.shape}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {total_params/1e6:.4f}M")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
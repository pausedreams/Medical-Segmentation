import torch
from net import UNet
# 注意：请确保您的文件名是 MK_net.py (下划线)，否则 import 会报错
# 如果您的文件名是 MK-net.py，请先重命名为 MK_net.py
from MK_net import ImprovedUNet 

def count_parameters(model):
    """
    计算模型的可训练参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # 1. 实例化原始 U-Net (Baseline)
    baseline_model = UNet(n_channels=3, n_classes=1)
    baseline_params = count_parameters(baseline_model)
    
    # 2. 实例化改进版 MK-UNet (Improved)
    improved_model = ImprovedUNet(n_channels=3, n_classes=1)
    improved_params = count_parameters(improved_model)

    # 3. 打印结果
    print("="*40)
    print(f"📊 模型参数量对比 (Model Parameters Comparison)")
    print("="*40)
    
    print(f"🔵 原始 U-Net (Baseline):")
    print(f"   - 总参数: {baseline_params:,}")
    print(f"   - 约合: {baseline_params/1e6:.2f} M (百万)")
    
    print("-" * 40)
    
    print(f"🔴 改进版 MK-UNet (Improved):")
    print(f"   - 总参数: {improved_params:,}")
    print(f"   - 约合: {improved_params/1e6:.2f} M (百万)")
    
    print("="*40)
    
    # 计算压缩率
    ratio = baseline_params / improved_params
    reduction = (1 - improved_params / baseline_params) * 100
    
    print(f"🚀 优化成果:")
    print(f"   - 参数量减少了: {reduction:.2f}%")
    print(f"   - 模型体积缩小了约: {ratio:.1f} 倍")
    print("="*40)

if __name__ == '__main__':
    main()
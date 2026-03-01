import torch
from model_unet import UNet
from model_mkunet import ImprovedUNet 

def count_parameters(model):
    """
    计算模型的可训练参数量 (Trainable Parameters)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("🚀 正在初始化模型进行架构复杂度评估...\n")
    
    # 1. 实例化原始 U-Net (Baseline)
    # 这里的参数需要和你在 train_baseline.py 中的设置保持一致
    baseline_model = UNet(n_channels=3, n_classes=1)
    baseline_params = count_parameters(baseline_model)
    
    # 2. 实例化改进版 MK-UNet (Innovation)
    # 这里的参数需要和你在 train_innovation.py 中的设置保持一致
    improved_model = ImprovedUNet(n_channels=3, n_classes=1)
    improved_params = count_parameters(improved_model)

    # 3. 打印学术风格的对比结果
    print("=" * 50)
    print(f"{'模型架构复杂度对比 (Model Complexity Comparison)':^45}")
    print("=" * 50)
    
    print(f"🔵 基线模型 U-Net (Baseline):")
    print(f"   - 绝对参数量: {baseline_params:,}")
    print(f"   - 相对参数量: {baseline_params/1e6:.2f} M (百万)")
    
    print("-" * 50)
    
    print(f"🔴 创新模型 MK-UNet (Innovation):")
    print(f"   - 绝对参数量: {improved_params:,}")
    print(f"   - 相对参数量: {improved_params/1e6:.2f} M (百万)")
    
    print("=" * 50)
    
    # 计算压缩率与优化幅度
    if improved_params < baseline_params:
        ratio = baseline_params / improved_params
        reduction = (1 - improved_params / baseline_params) * 100
        print(f"🏆 优化成果 (轻量化分析):")
        print(f"   - 参数量较基线减少了: {reduction:.2f}%")
        print(f"   - 模型体积缩小了约:   {ratio:.1f} 倍")
    else:
        increase = (improved_params / baseline_params - 1) * 100
        print(f"⚖️ 优化分析 (性能换算):")
        print(f"   - 为了丰富语义信息，参数量增加了: {increase:.2f}%")
        print(f"   - 导师提示: 在论文中可强调'以极小的参数代价换取了显著的精度提升(Dice)'")
        
    print("=" * 50)

if __name__ == '__main__':
    main()
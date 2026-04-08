import sys
import os
import argparse
import torch
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.infer import infer_single

def main():
    parser = argparse.ArgumentParser(description='对X光图像进行分类预测')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='计算设备')
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型文件不存在: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        sys.exit(1)
    
    print("=" * 50)
    print("胸部X光分类推理")
    print("=" * 50)
    
    # 推理
    result = infer_single(args.checkpoint, args.image, device=args.device)
    
    print("=" * 50)


if __name__ == '__main__':
    main()
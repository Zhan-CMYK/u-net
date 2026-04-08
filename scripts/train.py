import sys
import os
import yaml
import torch
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trainers.train import train_from_config

def main():
    parser = argparse.ArgumentParser(description='训练胸部X光分类模型')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='计算设备')
    
    args = parser.parse_args()
    
    # 加载配置
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=" * 50)
    print("胸部X光二分类模型训练")
    print("=" * 50)
    print(f"配置文件: {args.config}")
    print(f"设备: {args.device}")
    print(f"数据路径:")
    print(f"  训练集: {config['data']['train_dir']}")
    print(f"  测试集: {config['data']['test_dir']}")
    print(f"  验证集比例: {config['data']['validation_split']}")
    print("=" * 50)
    
    # 开始训练
    trainer, train_loader, val_loader, test_loader = train_from_config(config, device=args.device)
    
    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)
    print(f"模型已保存到: {trainer.checkpoint_dir}/best_model.pth")
    print(f"日志已保存到: {trainer.log_dir}")
    print(f"\n使用 TensorBoard 查看训练过程:")
    print(f"  tensorboard --logdir {trainer.log_dir}")


if __name__ == '__main__':
    main()
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard.compat.tensorflow_stub import io as tb_io
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.datasets.xray_dataset import get_data_loaders
from src.models.unet_classifier import create_model
from src.utils.metrics import MetricsCalculator


class Trainer:
    """训练器"""
    
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # 创建输出目录
        self.log_dir = Path(config['logging']['log_dir'])
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        self.model = create_model(config).to(device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training'].get('learning_rate', 1e-4)
        )
        
        # 初始化学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['training'].get('scheduler_factor', 0.5),
            patience=config['training'].get('scheduler_patience', 5),
            verbose=True
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 早停
        self.early_stopping_patience = config['early_stopping'].get('patience', 10)
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"设备: {device}")
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        metrics_calc = MetricsCalculator()
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录指标
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)
            metrics_calc.update(preds, labels, probs)
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        metrics = metrics_calc.compute()
        
        return avg_loss, metrics
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        metrics_calc = MetricsCalculator()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)
                metrics_calc.update(preds, labels, probs)
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        metrics = metrics_calc.compute()
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader):
        """完整训练流程"""
        epochs = self.config['training'].get('epochs', 100)
        
        print("\n开始训练...")
        print(f"总轮数: {epochs}")
        print(f"早停耐心: {self.early_stopping_patience}\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            
            # 训练
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_metrics = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics.get('accuracy', 0), epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics.get('accuracy', 0), epoch)
            self.writer.add_scalar('F1/train', train_metrics.get('f1', 0), epoch)
            self.writer.add_scalar('F1/val', val_metrics.get('f1', 0), epoch)
            
            # 打印指标
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_metrics.get('accuracy', 0):.4f} | Val Acc: {val_metrics.get('accuracy', 0):.4f}")
            print(f"Train F1: {train_metrics.get('f1', 0):.4f} | Val F1: {val_metrics.get('f1', 0):.4f}")
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"✓ 最佳模型已保存 (Val Loss: {val_loss:.4f})")
            else:
                self.early_stopping_counter += 1
                print(f"早停计数: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"\n早停触发！在第 {self.best_epoch+1} 个epoch达到最佳性能")
                    break
        
        self.writer.close()
        print(f"\n训练完成！最佳模型保存在 {self.checkpoint_dir}/best_model.pth")
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if is_best:
            save_path = self.checkpoint_dir / 'best_model.pth'
        else:
            save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型: {checkpoint_path}")


def train_from_config(config, device='cpu'):
    """从配置文件训练"""
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(config, num_workers=0)
    
    # 创建训练器
    trainer = Trainer(config, device=device)
    
    # 训练
    trainer.train(train_loader, val_loader)
    
    # 测试
    print("\n\n开始测试...")
    trainer.model.load_state_dict(torch.load(trainer.checkpoint_dir / 'best_model.pth', map_location=device)['model_state_dict'])
    test_loss, test_metrics = trainer.validate(test_loader)
    
    print("\n测试集结果:")
    print(f"  准确率: {test_metrics.get('accuracy', 0):.4f}")
    print(f"  精确率: {test_metrics.get('precision', 0):.4f}")
    print(f"  召回率: {test_metrics.get('recall', 0):.4f}")
    print(f"  F1 分数: {test_metrics.get('f1', 0):.4f}")
    if 'auc' in test_metrics:
        print(f"  AUC: {test_metrics.get('auc', 0):.4f}")
    if 'sensitivity' in test_metrics:
        print(f"  灵敏度: {test_metrics.get('sensitivity', 0):.4f}")
        print(f"  特异性: {test_metrics.get('specificity', 0):.4f}")
    
    return trainer, train_loader, val_loader, test_loader
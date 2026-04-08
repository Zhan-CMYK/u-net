import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷积块：Conv -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetClassifier(nn.Module):
    """U-Net 编码器 + 分类头用于二分类"""
    
    def __init__(self, in_channels=1, base_channels=32, num_layers=4, num_classes=2):
        """
        Args:
            in_channels: 输入通道数（灰度图为1）
            base_channels: 基础通道数
            num_layers: U-Net编码器层数
            num_classes: 分类类别数
        """
        super(UNetClassifier, self).__init__()
        
        self.num_layers = num_layers
        self.base_channels = base_channels
        
        # 编码器（下采样）
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(ConvBlock(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
        
        # 底部特征
        self.bottleneck = ConvBlock(in_ch, in_ch * 2)
        
        # 全局平均池化
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        classifier_in_channels = in_ch * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        # 编码器前向传播
        encoder_features = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            encoder_features.append(x)
            x = pool(x)
        
        # 底部特征
        x = self.bottleneck(x)
        
        # 全局平均池化
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        
        return x


def create_model(config):
    """创建模型"""
    model = UNetClassifier(
        in_channels=1,
        base_channels=config['model'].get('base_channels', 32),
        num_layers=config['model'].get('num_layers', 4),
        num_classes=config['data'].get('num_classes', 2)
    )
    return model
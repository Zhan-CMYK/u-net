import os
import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChestXrayDataset(Dataset):
    """胸部X光数据集加载器"""
    
    def __init__(self, image_dir, transform=None, is_training=True):
        """
        Args:
            image_dir: 图像目录（包含 NORMAL/ 和 PNEUMONIA/ 子目录）
            transform: 数据增强变换
            is_training: 是否为训练集（影响是否应用增强）
        """
        self.image_dir = image_dir
        self.transform = transform
        self.is_training = is_training
        self.images = []
        self.labels = []
        
        # 加载 NORMAL 类（标签 0）
        normal_dir = os.path.join(image_dir, 'NORMAL')
        if os.path.exists(normal_dir):
            for img_name in os.listdir(normal_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG')):
                    self.images.append(os.path.join(normal_dir, img_name))
                    self.labels.append(0)
        
        # 加载 PNEUMONIA 类（标签 1）
        pneumonia_dir = os.path.join(image_dir, 'PNEUMONIA')
        if os.path.exists(pneumonia_dir):
            for img_name in os.listdir(pneumonia_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG')):
                    self.images.append(os.path.join(pneumonia_dir, img_name))
                    self.labels.append(1)
        
        print(f"加载了 {len(self.images)} 张图像，NORMAL: {self.labels.count(0)}, PNEUMONIA: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图像（灰度）
        image = Image.open(img_path).convert('L')
        image = np.array(image, dtype=np.float32) / 255.0
        
        # 应用变换
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # 默认转换为张量
            image = torch.from_numpy(image).unsqueeze(0)
        
        return image, torch.tensor(label, dtype=torch.long)


def get_data_loaders(config, num_workers=0):
    """
    获取训练、验证和测试数据加载器
    
    Args:
        config: 配置字典
        num_workers: 数据加载线程数（CPU建议为0）
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # 定义训练集变换（含数据增强）
    train_transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ], is_check_shapes=False)
    
    # 定义验证/测试集变换（无增强，仅标准化）
    val_test_transform = A.Compose([
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ], is_check_shapes=False)
    
    # 加载完整训练集
    full_train_dataset = ChestXrayDataset(
        config['data']['train_dir'],
        transform=train_transform,
        is_training=True
    )
    
    # 按 80/20 比例划分为训练集和验证集
    val_split = config['data'].get('validation_split', 0.2)
    train_size = int(len(full_train_dataset) * (1 - val_split))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )
    
    # 更新验证集的变换（移除增强）
    val_dataset.dataset.transform = val_test_transform
    val_dataset.dataset.is_training = False
    
    # 加载测试集
    test_dataset = ChestXrayDataset(
        config['data']['test_dir'],
        transform=val_test_transform,
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training'].get('batch_size', 4),
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training'].get('batch_size', 4),
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training'].get('batch_size', 4),
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"\n数据划分完成:")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
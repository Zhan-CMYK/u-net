import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Inferencer:
    """推理器"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model_path = model_path
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        
        # 这里需要动态导入模型类
        from src.models.unet_classifier import UNetClassifier
        self.model = UNetClassifier(in_channels=1, base_channels=32, num_layers=4, num_classes=2)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # 定义变换
        self.transform = A.Compose([
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ], is_check_shapes=False)
        
        self.class_names = ['NORMAL', 'PNEUMONIA']
        print(f"已加载模型: {model_path}")
    
    def predict(self, image_path, return_prob=True):
        """对单张图像进行预测"""
        # 加载图像
        image = Image.open(image_path).convert('L')
        image = np.array(image, dtype=np.float32) / 255.0
        
        # 应用变换
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(image_tensor)
            prob = F.softmax(output, dim=1)
            pred_class = torch.argmax(prob, dim=1).item()
        
        result = {
            'class': self.class_names[pred_class],
            'class_id': pred_class,
            'probabilities': {
                self.class_names[i]: prob[0, i].item() 
                for i in range(len(self.class_names))
            }
        }
        
        return result
    
    def predict_batch(self, image_paths):
        """批量预测"""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        
        return results


def infer_single(model_path, image_path, device='cpu'):
    """推理单张图像"""
    inferencer = Inferencer(model_path, device=device)
    result = inferencer.predict(image_path)
    
    print(f"\n图像: {image_path}")
    print(f"预测类别: {result['class']}")
    print(f"置信度:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
    
    return result

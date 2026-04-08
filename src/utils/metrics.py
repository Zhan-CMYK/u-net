import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, preds, targets, probs=None):
        """更新指标"""
        self.predictions.extend(preds.cpu().numpy().tolist())
        self.targets.extend(targets.cpu().numpy().tolist())
        if probs is not None:
            self.probabilities.extend(probs.cpu().numpy().tolist())
    
    def compute(self):
        """计算所有指标"""
        if len(self.predictions) == 0:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='weighted', zero_division=0),
            'recall': recall_score(targets, predictions, average='weighted', zero_division=0),
            'f1': f1_score(targets, predictions, average='weighted', zero_division=0),
        }
        
        # 计算二分类特定指标
        if len(np.unique(targets)) == 2:
            if len(self.probabilities) > 0:
                probabilities = np.array(self.probabilities)
                if probabilities.ndim > 1:
                    probs_positive = probabilities[:, 1]
                else:
                    probs_positive = probabilities
                
                try:
                    metrics['auc'] = roc_auc_score(targets, probs_positive)
                except:
                    metrics['auc'] = 0.0
            
            # 混淆矩阵
            cm = confusion_matrix(targets, predictions)
            metrics['confusion_matrix'] = cm
            
            # 计算 TN, FP, FN, TP
            tn, fp, fn, tp = cm.ravel()
            metrics['tn'] = tn
            metrics['fp'] = fp
            metrics['fn'] = fn
            metrics['tp'] = tp
            
            # 灵敏度和特异性
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def get_confusion_matrix(self):
        """获取混淆矩阵"""
        if len(self.predictions) == 0:
            return None
        targets = np.array(self.targets)
        predictions = np.array(self.predictions)
        return confusion_matrix(targets, predictions)
    
    def plot_confusion_matrix(self, save_path=None, class_names=None):
        """绘制混淆矩阵"""
        cm = self.get_confusion_matrix()
        if cm is None:
            return
        
        if class_names is None:
            class_names = ['NORMAL', 'PNEUMONIA']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        plt.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               ylabel='True label', xlabel='Predicted label')
        
        # 添加数值标签
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, save_path=None, class_names=None):
        """绘制ROC曲线"""
        if len(self.probabilities) == 0:
            return None
        
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        if probabilities.ndim > 1:
            probs_positive = probabilities[:, 1]
        else:
            probs_positive = probabilities
        
        try:
            fpr, tpr, _ = roc_curve(targets, probs_positive)
            auc = roc_auc_score(targets, probs_positive)
        except:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
               xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax.legend(loc="lower right")
        plt.title('ROC Curve')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
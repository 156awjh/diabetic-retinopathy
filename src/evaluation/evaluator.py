"""
模型评估器
==========

提供统一的模型评估接口。

使用示例:
---------
>>> from src.evaluation.evaluator import ModelEvaluator
>>> 
>>> evaluator = ModelEvaluator(model, test_data)
>>> report = evaluator.evaluate()
>>> evaluator.save_report('trained_models/resnet50')
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    模型评估器
    
    属性:
    -----
    model : BaseModel
        要评估的模型
    test_data : tf.data.Dataset
        测试数据集
    """
    
    CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    CLASS_NAMES_CN = ['无病变', '轻度', '中度', '重度', '增殖性']
    
    def __init__(self, model, test_data):
        """
        初始化评估器
        
        参数:
        -----
        model : BaseModel
            要评估的模型
        test_data : tf.data.Dataset
            测试数据集
        """
        self.model = model
        self.test_data = test_data
        self.y_true: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None
        self.y_pred_proba: Optional[np.ndarray] = None
        self.report: Optional[Dict[str, Any]] = None
    
    def evaluate(self) -> Dict[str, Any]:
        """
        执行评估
        
        返回:
        -----
        dict
            评估报告
        """
        print(f"\n{'='*60}")
        print(f"评估模型: {self.model.model_name}")
        print(f"{'='*60}\n")
        
        # 获取预测结果
        print("正在预测...")
        self.y_pred_proba = self.model.predict(self.test_data)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        
        # 获取真实标签
        print("获取真实标签...")
        y_true_list = []
        for _, labels in self.test_data:
            y_true_list.append(np.argmax(labels.numpy(), axis=1))
        self.y_true = np.concatenate(y_true_list)
        
        # 计算指标
        print("计算评估指标...")
        self.report = self._compute_metrics()
        
        # 打印报告
        self._print_report()
        
        return self.report

    def _compute_metrics(self) -> Dict[str, Any]:
        """计算评估指标"""
        report = {
            'model_name': self.model.model_name,
            'total_samples': len(self.y_true),
            
            # 整体指标
            'accuracy': float(accuracy_score(self.y_true, self.y_pred)),
            
            # Macro 平均
            'precision_macro': float(precision_score(
                self.y_true, self.y_pred, average='macro', zero_division=0
            )),
            'recall_macro': float(recall_score(
                self.y_true, self.y_pred, average='macro', zero_division=0
            )),
            'f1_macro': float(f1_score(
                self.y_true, self.y_pred, average='macro', zero_division=0
            )),
            
            # Weighted 平均
            'precision_weighted': float(precision_score(
                self.y_true, self.y_pred, average='weighted', zero_division=0
            )),
            'recall_weighted': float(recall_score(
                self.y_true, self.y_pred, average='weighted', zero_division=0
            )),
            'f1_weighted': float(f1_score(
                self.y_true, self.y_pred, average='weighted', zero_division=0
            )),
            
            # 每个类别的指标
            'per_class': {}
        }
        
        # 计算每个类别的指标
        precision_per_class = precision_score(
            self.y_true, self.y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            self.y_true, self.y_pred, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            self.y_true, self.y_pred, average=None, zero_division=0
        )
        
        for i, class_name in enumerate(self.CLASS_NAMES):
            support = int(np.sum(self.y_true == i))
            report['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i]),
                'support': support
            }
        
        # 混淆矩阵
        report['confusion_matrix'] = confusion_matrix(
            self.y_true, self.y_pred
        ).tolist()
        
        return report
    
    def _print_report(self):
        """打印评估报告"""
        print(f"\n{'='*60}")
        print("评估结果")
        print(f"{'='*60}")
        print(f"总样本数: {self.report['total_samples']}")
        print(f"\n整体指标:")
        print(f"  准确率 (Accuracy): {self.report['accuracy']:.4f}")
        print(f"  精确率 (Precision, macro): {self.report['precision_macro']:.4f}")
        print(f"  召回率 (Recall, macro): {self.report['recall_macro']:.4f}")
        print(f"  F1 分数 (F1, macro): {self.report['f1_macro']:.4f}")
        
        print(f"\n各类别指标:")
        print(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1':<10} {'样本数':<10}")
        print("-" * 55)
        for i, (class_name, metrics) in enumerate(self.report['per_class'].items()):
            cn_name = self.CLASS_NAMES_CN[i]
            print(f"{cn_name:<12} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} "
                  f"{metrics['support']:<10}")
        print(f"{'='*60}\n")
    
    def plot_confusion_matrix(self, save_path: str = None, figsize: tuple = (10, 8)):
        """
        绘制混淆矩阵
        
        参数:
        -----
        save_path : str
            保存路径
        figsize : tuple
            图像大小
        """
        cm = np.array(self.report['confusion_matrix'])
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.CLASS_NAMES_CN,
            yticklabels=self.CLASS_NAMES_CN
        )
        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('真实类别', fontsize=12)
        plt.title(f'混淆矩阵 - {self.model.model_name}', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")
        
        plt.close()
    
    def save_report(self, save_dir: str):
        """
        保存评估报告
        
        参数:
        -----
        save_dir : str
            保存目录
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSON 报告
        report_file = save_path / 'evaluation_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        print(f"评估报告已保存: {report_file}")
        
        # 保存混淆矩阵图
        cm_file = save_path / 'confusion_matrix.png'
        self.plot_confusion_matrix(str(cm_file))
        
        print(f"\n所有评估结果已保存到: {save_path}")

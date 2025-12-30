"""
集成学习模型
============

整合多个训练好的模型进行集成预测。

使用示例:
---------
>>> ensemble = EnsembleModel()
>>> ensemble.load_models('trained_models')
>>> predictions = ensemble.predict(test_data, strategy='soft_voting')
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from tensorflow import keras

from .voting import hard_voting, soft_voting, weighted_voting, get_confidence_scores


class EnsembleModel:
    """
    集成学习模型
    
    属性:
    -----
    models : dict
        已加载的模型字典 {model_name: keras.Model}
    weights : dict
        模型权重字典 {model_name: weight}
    """
    
    MODEL_NAMES = [
        'resnet50', 'efficientnet_b0', 'vgg16', 'mobilenetv2',
        'se_resnet', 'resnext50', 'densenet121', 'inceptionv3'
    ]
    
    def __init__(self):
        """初始化集成模型"""
        self.models: Dict[str, keras.Model] = {}
        self.weights: Dict[str, float] = {}
        self.model_accuracies: Dict[str, float] = {}
    
    def load_models(self, models_dir: str, model_names: List[str] = None):
        """
        加载训练好的模型
        
        参数:
        -----
        models_dir : str
            模型目录
        model_names : list
            要加载的模型名称列表，默认加载所有模型
        """
        models_path = Path(models_dir)
        names_to_load = model_names or self.MODEL_NAMES
        
        print(f"\n{'='*60}")
        print("加载模型")
        print(f"{'='*60}")
        
        for model_name in names_to_load:
            model_file = models_path / model_name / f"{model_name}_best.keras"
            
            if model_file.exists():
                try:
                    self.models[model_name] = keras.models.load_model(str(model_file))
                    self.weights[model_name] = 1.0  # 默认权重
                    print(f"  ✓ 已加载: {model_name}")
                    
                    # 尝试加载评估报告获取准确率
                    eval_file = models_path / model_name / 'evaluation_report.json'
                    if eval_file.exists():
                        with open(eval_file, 'r') as f:
                            report = json.load(f)
                            self.model_accuracies[model_name] = report.get('accuracy', 0)
                except Exception as e:
                    print(f"  ✗ 加载失败: {model_name} - {e}")
            else:
                print(f"  ✗ 未找到: {model_name}")
        
        print(f"\n共加载 {len(self.models)} 个模型")
        print(f"{'='*60}\n")
    
    def set_weights(self, weights: Dict[str, float]):
        """
        设置模型权重
        
        参数:
        -----
        weights : dict
            模型权重字典 {model_name: weight}
        """
        self.weights.update(weights)
        print("模型权重已更新:")
        for name, weight in self.weights.items():
            if name in self.models:
                print(f"  {name}: {weight:.4f}")
    
    def set_weights_from_accuracy(self):
        """根据模型准确率自动设置权重"""
        if not self.model_accuracies:
            print("警告: 没有模型准确率信息，使用默认权重")
            return
        
        for name in self.models:
            if name in self.model_accuracies:
                self.weights[name] = self.model_accuracies[name]
        
        print("已根据准确率设置权重:")
        for name, weight in self.weights.items():
            if name in self.models:
                print(f"  {name}: {weight:.4f}")

    def predict(self, data, strategy: str = 'soft_voting') -> np.ndarray:
        """
        集成预测
        
        参数:
        -----
        data : tf.data.Dataset
            输入数据
        strategy : str
            投票策略: 'hard_voting', 'soft_voting', 'weighted_voting'
        
        返回:
        -----
        np.ndarray
            预测类别
        """
        if not self.models:
            raise ValueError("没有加载任何模型")
        
        print(f"使用 {strategy} 策略进行集成预测...")
        
        # 获取所有模型的预测
        all_probas = []
        all_preds = []
        
        for name, model in self.models.items():
            proba = model.predict(data, verbose=0)
            all_probas.append(proba)
            all_preds.append(np.argmax(proba, axis=1))
        
        # 根据策略进行投票
        if strategy == 'hard_voting':
            return hard_voting(all_preds)
        elif strategy == 'soft_voting':
            return soft_voting(all_probas)
        elif strategy == 'weighted_voting':
            weights = [self.weights.get(name, 1.0) for name in self.models.keys()]
            return weighted_voting(all_probas, weights)
        else:
            raise ValueError(f"未知策略: {strategy}")
    
    def predict_proba(self, data, strategy: str = 'soft_voting') -> np.ndarray:
        """
        获取集成预测概率
        
        参数:
        -----
        data : tf.data.Dataset
            输入数据
        strategy : str
            投票策略
        
        返回:
        -----
        np.ndarray
            预测概率，形状为 (n_samples, n_classes)
        """
        if not self.models:
            raise ValueError("没有加载任何模型")
        
        all_probas = []
        for name, model in self.models.items():
            proba = model.predict(data, verbose=0)
            all_probas.append(proba)
        
        if strategy == 'soft_voting':
            return np.mean(all_probas, axis=0)
        elif strategy == 'weighted_voting':
            weights = np.array([self.weights.get(name, 1.0) for name in self.models.keys()])
            weights = weights / weights.sum()
            weighted_proba = np.zeros_like(all_probas[0])
            for proba, weight in zip(all_probas, weights):
                weighted_proba += proba * weight
            return weighted_proba
        else:
            return np.mean(all_probas, axis=0)
    
    def generate_comparison_report(self, 
                                   test_data, 
                                   y_true: np.ndarray,
                                   save_path: str) -> Dict:
        """
        生成对比报告
        
        参数:
        -----
        test_data : tf.data.Dataset
            测试数据
        y_true : np.ndarray
            真实标签
        save_path : str
            保存路径
        
        返回:
        -----
        dict
            对比报告
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        print(f"\n{'='*60}")
        print("生成对比报告")
        print(f"{'='*60}\n")
        
        results = {
            'individual_models': {},
            'ensemble': {}
        }
        
        # 评估每个单独模型
        print("评估单个模型...")
        for name, model in self.models.items():
            proba = model.predict(test_data, verbose=0)
            preds = np.argmax(proba, axis=1)
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, average='macro')
            results['individual_models'][name] = {
                'accuracy': float(acc),
                'f1_macro': float(f1)
            }
            print(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}")
        
        # 评估集成模型
        print("\n评估集成模型...")
        for strategy in ['hard_voting', 'soft_voting', 'weighted_voting']:
            preds = self.predict(test_data, strategy=strategy)
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, average='macro')
            results['ensemble'][strategy] = {
                'accuracy': float(acc),
                'f1_macro': float(f1)
            }
            print(f"  {strategy}: Acc={acc:.4f}, F1={f1:.4f}")
        
        # 找出最佳策略
        best_strategy = max(results['ensemble'].items(), 
                           key=lambda x: x[1]['accuracy'])
        results['best_strategy'] = best_strategy[0]
        results['best_accuracy'] = best_strategy[1]['accuracy']
        
        # 找出最佳单模型
        best_model = max(results['individual_models'].items(),
                        key=lambda x: x[1]['accuracy'])
        results['best_individual_model'] = best_model[0]
        results['best_individual_accuracy'] = best_model[1]['accuracy']
        
        # 计算提升
        improvement = results['best_accuracy'] - results['best_individual_accuracy']
        results['improvement'] = float(improvement)
        
        print(f"\n{'='*60}")
        print(f"最佳单模型: {results['best_individual_model']} "
              f"(Acc={results['best_individual_accuracy']:.4f})")
        print(f"最佳集成策略: {results['best_strategy']} "
              f"(Acc={results['best_accuracy']:.4f})")
        print(f"性能提升: {improvement:+.4f}")
        print(f"{'='*60}\n")
        
        # 保存报告
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"对比报告已保存: {save_file}")
        
        return results

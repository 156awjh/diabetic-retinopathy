"""
模型训练器
==========

提供统一的模型训练接口。

使用示例:
---------
>>> from src.models.resnet50 import ResNet50Model
>>> from src.training.trainer import ModelTrainer
>>> 
>>> model = ResNet50Model()
>>> trainer = ModelTrainer(model)
>>> trainer.train(train_data, val_data, epochs=50)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from .callbacks import get_callbacks


class ModelTrainer:
    """
    通用模型训练器
    
    属性:
    -----
    model : BaseModel
        要训练的模型
    save_dir : Path
        模型保存目录
    """
    
    def __init__(self, 
                 model,
                 save_dir: str = 'trained_models'):
        """
        初始化训练器
        
        参数:
        -----
        model : BaseModel
            要训练的模型实例
        save_dir : str
            模型保存目录
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.model_save_dir = self.save_dir / model.model_name
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self,
              train_data,
              val_data,
              epochs: int = 50,
              learning_rate: float = 0.001,
              class_weight: Dict[int, float] = None,
              custom_callbacks: list = None):
        """
        执行训练
        
        参数:
        -----
        train_data : tf.data.Dataset
            训练数据集
        val_data : tf.data.Dataset
            验证数据集
        epochs : int
            训练轮数
        learning_rate : float
            学习率
        class_weight : dict
            类别权重
        custom_callbacks : list
            自定义回调函数
        
        返回:
        -----
        keras.callbacks.History
            训练历史
        """
        print(f"\n{'='*60}")
        print(f"开始训练: {self.model.model_name}")
        print(f"{'='*60}")
        print(f"训练轮数: {epochs}")
        print(f"学习率: {learning_rate}")
        print(f"保存目录: {self.model_save_dir}")
        print(f"{'='*60}\n")
        
        # 编译模型
        self.model.compile(learning_rate=learning_rate)
        
        # 获取回调函数
        callbacks = custom_callbacks or get_callbacks(
            self.model.model_name,
            str(self.save_dir)
        )
        
        # 训练
        history = self.model.train(
            train_data,
            val_data,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight
        )
        
        # 保存模型和历史
        self.model.save(str(self.save_dir))
        
        # 保存训练元数据
        self._save_training_metadata(epochs, learning_rate)
        
        print(f"\n{'='*60}")
        print(f"训练完成: {self.model.model_name}")
        print(f"模型已保存到: {self.model_save_dir}")
        print(f"{'='*60}\n")
        
        return history
    
    def _save_training_metadata(self, epochs: int, learning_rate: float):
        """保存训练元数据"""
        metadata = {
            'model_name': self.model.model_name,
            'training_date': datetime.now().isoformat(),
            'epochs': epochs,
            'learning_rate': learning_rate,
            'input_shape': list(self.model.input_shape),
            'num_classes': self.model.num_classes
        }
        
        # 如果有训练历史，添加最终指标
        if self.model.history is not None:
            history = self.model.history.history
            metadata['final_metrics'] = {
                'train_loss': float(history['loss'][-1]),
                'train_accuracy': float(history['accuracy'][-1]),
                'val_loss': float(history['val_loss'][-1]),
                'val_accuracy': float(history['val_accuracy'][-1])
            }
        
        metadata_file = self.model_save_dir / 'training_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

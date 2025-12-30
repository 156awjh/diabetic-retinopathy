"""
模型基类
========

所有团队成员的模型必须继承此基类，确保接口统一。

使用示例:
---------
>>> class MyModel(BaseModel):
...     def __init__(self):
...         super().__init__(model_name='my_model')
...     
...     def build(self):
...         # 构建你的模型
...         return model
"""

import numpy as np
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from tensorflow import keras


class BaseModel(ABC):
    """
    模型基类
    
    所有团队成员实现的模型类都应继承此类。
    
    属性:
    -----
    model_name : str
        模型名称，用于保存文件
    num_classes : int
        分类类别数，默认 5
    input_shape : tuple
        输入图像形状，默认 (224, 224, 3)
    model : keras.Model
        Keras 模型实例
    history : keras.callbacks.History
        训练历史
    """
    
    def __init__(self, 
                 model_name: str,
                 num_classes: int = 5,
                 input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """
        初始化基类
        
        参数:
        -----
        model_name : str
            模型名称（如 'resnet50', 'vgg16'）
        num_classes : int
            分类类别数，默认 5
        input_shape : tuple
            输入图像形状，默认 (224, 224, 3)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model: Optional[keras.Model] = None
        self.history = None
    
    @abstractmethod
    def build(self) -> keras.Model:
        """
        构建模型
        
        子类必须实现此方法！
        
        返回:
        -----
        keras.Model
            构建好的 Keras 模型
        """
        pass
    
    def compile(self, 
                learning_rate: float = 0.001,
                optimizer: str = 'adam',
                loss: str = 'categorical_crossentropy',
                metrics: list = None):
        """
        编译模型
        
        参数:
        -----
        learning_rate : float
            学习率，默认 0.001
        optimizer : str
            优化器，默认 'adam'
        loss : str
            损失函数，默认 'categorical_crossentropy'
        metrics : list
            评估指标，默认 ['accuracy']
        """
        if self.model is None:
            self.model = self.build()
        
        if metrics is None:
            metrics = ['accuracy']
        
        # 创建优化器
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        return self

    def train(self, 
              train_data,
              val_data,
              epochs: int = 50,
              callbacks: list = None,
              class_weight: dict = None) -> keras.callbacks.History:
        """
        训练模型
        
        参数:
        -----
        train_data : tf.data.Dataset
            训练数据集
        val_data : tf.data.Dataset
            验证数据集
        epochs : int
            训练轮数，默认 50
        callbacks : list
            回调函数列表
        class_weight : dict
            类别权重
        
        返回:
        -----
        keras.callbacks.History
            训练历史
        """
        if self.model is None:
            raise ValueError("模型未构建，请先调用 build() 或 compile()")
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight
        )
        
        return self.history
    
    def predict(self, data) -> np.ndarray:
        """
        预测
        
        参数:
        -----
        data : tf.data.Dataset or np.ndarray
            输入数据
        
        返回:
        -----
        np.ndarray
            预测概率，形状为 (n_samples, num_classes)
        """
        if self.model is None:
            raise ValueError("模型未加载或构建")
        
        return self.model.predict(data)
    
    def predict_classes(self, data) -> np.ndarray:
        """
        预测类别
        
        参数:
        -----
        data : tf.data.Dataset or np.ndarray
            输入数据
        
        返回:
        -----
        np.ndarray
            预测类别，形状为 (n_samples,)
        """
        proba = self.predict(data)
        return np.argmax(proba, axis=1)
    
    def save(self, save_dir: str):
        """
        保存模型和训练历史
        
        参数:
        -----
        save_dir : str
            保存目录
        """
        save_path = Path(save_dir) / self.model_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model_file = save_path / f"{self.model_name}_best.keras"
        self.model.save(str(model_file))
        print(f"模型已保存: {model_file}")
        
        # 保存训练历史
        if self.history is not None:
            history_file = save_path / "training_history.json"
            history_dict = {k: [float(v) for v in vals] 
                          for k, vals in self.history.history.items()}
            with open(history_file, 'w') as f:
                json.dump(history_dict, f, indent=2)
            print(f"训练历史已保存: {history_file}")
        
        # 保存元数据
        metadata = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'input_shape': list(self.input_shape)
        }
        metadata_file = save_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"元数据已保存: {metadata_file}")
    
    def load(self, model_path: str):
        """
        加载模型
        
        参数:
        -----
        model_path : str
            模型文件路径（.keras 文件）
        
        返回:
        -----
        self
            返回自身以支持链式调用
        """
        self.model = keras.models.load_model(model_path)
        print(f"模型已加载: {model_path}")
        return self
    
    def summary(self):
        """打印模型摘要"""
        if self.model is not None:
            self.model.summary()
        else:
            print("模型未构建")
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取模型配置
        
        返回:
        -----
        dict
            模型配置字典
        """
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"

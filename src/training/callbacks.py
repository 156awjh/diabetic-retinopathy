"""
训练回调函数
============

提供标准的训练回调函数配置。
"""

from tensorflow import keras
from pathlib import Path


def get_callbacks(model_name: str, 
                  save_dir: str = 'trained_models',
                  patience_early_stop: int = 10,
                  patience_lr: int = 5) -> list:
    """
    获取标准训练回调函数
    
    参数:
    -----
    model_name : str
        模型名称
    save_dir : str
        保存目录
    patience_early_stop : int
        早停耐心值
    patience_lr : int
        学习率衰减耐心值
    
    返回:
    -----
    list
        回调函数列表
    """
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # 早停
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1
        ),
        
        # 学习率衰减
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1
        ),
        
        # 模型检查点
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_path / f"{model_name}_best_auto.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard 日志
        keras.callbacks.TensorBoard(
            log_dir=str(save_path / 'logs'),
            histogram_freq=1
        )
    ]
    
    return callbacks

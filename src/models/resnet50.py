"""
ResNet-50 模型
==============

组长负责

使用示例:
---------
>>> from src.models.resnet50 import ResNet50Model
>>> model = ResNet50Model()
>>> model.compile()
>>> model.summary()
"""

from tensorflow import keras
from tensorflow.keras import layers
from .base_model import BaseModel


class ResNet50Model(BaseModel):
    """
    ResNet-50 模型
    
    使用 ImageNet 预训练权重，添加自定义分类头。
    
    参数:
    -----
    num_classes : int
        分类类别数，默认 5
    input_shape : tuple
        输入图像形状，默认 (224, 224, 3)
    pretrained : bool
        是否使用预训练权重，默认 True
    freeze_base : bool
        是否冻结预训练层，默认 True
    """
    
    def __init__(self, 
                 num_classes: int = 5,
                 input_shape: tuple = (224, 224, 3),
                 pretrained: bool = True,
                 freeze_base: bool = True):
        super().__init__(
            model_name='resnet50',
            num_classes=num_classes,
            input_shape=input_shape
        )
        self.pretrained = pretrained
        self.freeze_base = freeze_base
    
    def build(self) -> keras.Model:
        """
        构建 ResNet-50 模型
        
        返回:
        -----
        keras.Model
            构建好的模型
        """
        # 加载预训练的 ResNet50
        base_model = keras.applications.ResNet50(
            weights='imagenet' if self.pretrained else None,
            include_top=False,
            input_shape=self.input_shape
        )
        
        # 是否冻结预训练层
        base_model.trainable = not self.freeze_base
        
        # 构建模型
        inputs = keras.Input(shape=self.input_shape)
        
        # 预处理（ResNet50 需要特定的预处理）
        x = keras.applications.resnet50.preprocess_input(inputs)
        
        # 特征提取
        x = base_model(x, training=False)
        
        # 分类头
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs, name='resnet50_dr')
        
        return self.model
    
    def unfreeze_layers(self, num_layers: int = 50):
        """
        解冻部分预训练层进行微调
        
        参数:
        -----
        num_layers : int
            要解冻的层数（从顶部开始）
        """
        if self.model is None:
            raise ValueError("模型未构建")
        
        # 找到 ResNet50 基础模型
        for layer in self.model.layers:
            if isinstance(layer, keras.Model):
                # 解冻最后 num_layers 层
                for i, sub_layer in enumerate(layer.layers):
                    if i >= len(layer.layers) - num_layers:
                        sub_layer.trainable = True
                    else:
                        sub_layer.trainable = False
                break
        
        print(f"已解冻最后 {num_layers} 层")


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("测试 ResNet-50 模型...")
    
    model = ResNet50Model()
    model.compile()
    model.summary()
    
    print("\n模型构建成功！")

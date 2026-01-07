"""
MobileNetV2 模型（优化版 - 修复预处理问题）
==========================================

轻量级模型，适合移动设备和资源受限环境。

修复内容：
- 正确处理[0,1]归一化输入
- 降低学习率至0.0001（更适合微调）
- 优化分类头结构

使用示例:
---------
>>> from src.models.MobileNetV2 import MobileNetV2Model
>>> model = MobileNetV2Model()
>>> model.compile()
>>> model.summary()
"""

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from .base_model import BaseModel


class MobileNetV2Model(BaseModel):
    """
    MobileNetV2 模型
    
    使用 ImageNet 预训练权重，添加自定义分类头。
    MobileNetV2 是一个轻量级模型，参数量少，推理速度快。
    
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
    alpha : float
        宽度乘数，控制网络的宽度，默认 1.0
        较小的值（如 0.5, 0.75）可以减少参数量和计算量
    """
    
    def __init__(self, 
                 num_classes: int = 5,
                 input_shape: tuple = (224, 224, 3),
                 pretrained: bool = True,
                 freeze_base: bool = False,
                 alpha: float = 1.0):
        super().__init__(
            model_name='mobilenetv2',
            num_classes=num_classes,
            input_shape=input_shape
        )
        self.pretrained = pretrained
        self.freeze_base = freeze_base
        self.alpha = alpha
    
    def build(self) -> keras.Model:
        """
        构建 MobileNetV2 模型
        
        注意：
        - DataLoader输出[0,1]归一化图像
        - MobileNetV2需要[-1,1]范围输入
        - 这里进行简单的线性变换：x*2-1
        
        返回:
        -----
        keras.Model
            构建好的模型
        """
        # 加载预训练的 MobileNetV2
        base_model = keras.applications.MobileNetV2(
            weights='imagenet' if self.pretrained else None,
            include_top=False,
            input_shape=self.input_shape,
            alpha=self.alpha
        )
        
        # 冻结策略：微调模式下只冻结前100层（共154层），保留高层特征学习能力
        if self.freeze_base:
            base_model.trainable = False
        else:
            # 微调：冻结底层，训练高层
            for layer in base_model.layers[:100]:
                layer.trainable = False
            for layer in base_model.layers[100:]:
                layer.trainable = True
        
        # 构建模型
        inputs = keras.Input(shape=self.input_shape)
        
        # 预处理：DataLoader已归一化到[0,1]，MobileNetV2需要[-1,1]
        # 将[0,1]转换为[-1,1]
        x = inputs * 2.0 - 1.0
        
        # 特征提取（微调时允许BN更新）
        x = base_model(x, training=(not self.freeze_base))
        
        # 分类头：轻量级但有效
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Dropout(0.5, name='drop1')(x)
        x = layers.Dense(512, activation='relu', name='fc512')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Dropout(0.4, name='drop2')(x)
        x = layers.Dense(256, activation='relu', name='fc256')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.Dropout(0.3, name='drop3')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = keras.Model(inputs, outputs, name='mobilenetv2_dr')
        
        return self.model
    
    def compile(self, learning_rate: float = 0.0001, optimizer: str = 'adam', loss: str = 'categorical_crossentropy', metrics: list = None):
        """
        编译模型
        
        参数:
        -----
        learning_rate : float
            学习率，默认 0.0001（微调模式用更小的学习率）
        optimizer : str
            优化器，默认 'adam'
        loss : str
            损失函数，默认 'categorical_crossentropy'
        metrics : list
            评估指标，默认 ['accuracy']
        
        注意:
        -----
        类不均衡问题建议通过以下方式处理：
        1. 数据层面：使用过采样后的数据集
        2. 训练时：在 train() 方法中传入 class_weight 参数
        """
        if self.model is None:
            self.model = self.build()

        if metrics is None:
            metrics = ['accuracy']

        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, clipnorm=1.0)
        else:
            opt = optimizer

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
        return self
    
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
        
        # 找到 MobileNetV2 基础模型
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
        
    def make_weighted_dataset(self, images, labels, batch_size=32):
        """
        构建支持加权采样的训练数据集（自动模拟 WeightedRandomSampler）
        """
        import numpy as np

        if labels.ndim > 1:
            labels_int = np.argmax(labels, axis=1)
        else:
            labels_int = labels

        # 计算类别权重
        class_counts = np.bincount(labels_int, minlength=self.num_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[labels_int]
        sample_weights /= np.sum(sample_weights)

        # 加权随机采样
        sampled_indices = np.random.choice(
            np.arange(len(images)),
            size=len(images),
            replace=True,
            p=sample_weights
        )

        # 根据采样结果生成数据集
        sampled_images = images[sampled_indices]
        sampled_labels = labels[sampled_indices]

        import tensorflow as tf
        dataset = tf.data.Dataset.from_tensor_slices((sampled_images, sampled_labels))
        dataset = dataset.shuffle(buffer_size=len(images))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset


    def compile(self, learning_rate: float = 0.0001, optimizer: str = 'adam', loss: str = 'categorical_crossentropy', metrics: list = None):
        """
        编译模型（自动启用 Weighted Sampling）
        """
        from tensorflow import keras
        import numpy as np
        import types
        import tensorflow as tf

        if self.model is None:
            self.model = self.build()

        if metrics is None:
            metrics = ['accuracy']

        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, clipnorm=1.0)
        else:
            opt = optimizer

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)

        # 自动拦截 fit() 调用，注入加权采样逻辑
        original_fit = self.model.fit

        def weighted_fit(model_self, x=None, y=None, *args, **kwargs):
            # 若传入 numpy 数据，则自动构建加权采样数据集
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                print("⚖️ [mobilenetv2] 自动启用加权采样 Weighted Sampling")
                batch_size = kwargs.get("batch_size", 32)
                dataset = self.make_weighted_dataset(x, y, batch_size=batch_size)
                return original_fit(dataset, *args, **kwargs)
            # 否则直接调用原始 fit
            return original_fit(x, y, *args, **kwargs)

        self.model.fit = types.MethodType(weighted_fit, self.model)
        return self



# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("测试 MobileNetV2 模型...")
    
    model = MobileNetV2Model()
    model.compile()
    model.summary()
    
    print("\n模型构建成功！")



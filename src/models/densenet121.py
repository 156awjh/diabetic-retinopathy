"""
DenseNet-121 模型
=================

使用示例:
---------
>>> from src.models.densenet121 import DenseNet121Model
>>> model = DenseNet121Model()
>>> model.compile()
>>> model.summary()
"""

from tensorflow import keras
from tensorflow.keras import layers
from .base_model import BaseModel


class DenseNet121Model(BaseModel):
    """
    DenseNet-121 模型

    使用 ImageNet 预训练权重，添加自定义分类头，兼容糖尿病视网膜病变分类任务。

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
        """初始化 DenseNet121 模型，调用父类 BaseModel 的初始化方法"""
        super().__init__(
            model_name='densenet121',  # 模型名称，用于保存和标识
            num_classes=num_classes,
            input_shape=input_shape
        )
        # 新增 DenseNet121 专属属性，与 ResNet50 保持一致范式
        self.pretrained = pretrained
        self.freeze_base = freeze_base

    def build(self) -> keras.Model:
        """
        构建 DenseNet-121 模型（实现 BaseModel 的抽象方法）

        返回:
        -----
        keras.Model
            构建好的 DenseNet121 分类模型
        """
        # 加载预训练的 DenseNet121 基础模型
        base_model = keras.applications.DenseNet121(
            weights='imagenet' if self.pretrained else None,
            include_top=False,  # 不包含原始分类头
            input_shape=self.input_shape
        )

        # 是否冻结预训练层（冻结后仅训练自定义分类头）
        base_model.trainable = not self.freeze_base

        # 构建完整模型流程
        inputs = keras.Input(shape=self.input_shape)

        # DenseNet121 专属预处理（必须对应，保证模型性能）
        x = keras.applications.densenet.preprocess_input(inputs)

        # 特征提取（通过基础模型获取图像特征）
        x = base_model(x, training=False)

        # 自定义分类头（与 ResNet50 保持一致结构，保证任务兼容性）
        x = layers.GlobalAveragePooling2D()(x)  # 全局平均池化，压缩特征维度
        x = layers.BatchNormalization()(x)      # 批量归一化，防止过拟合
        x = layers.Dropout(0.5)(x)              # 随机失活，防止过拟合
        x = layers.Dense(256, activation='relu')(x)  # 全连接层，进一步提取特征
        x = layers.BatchNormalization()(x)      # 批量归一化
        x = layers.Dropout(0.3)(x)              # 随机失活
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)  # 最终分类头

        # 赋值给实例属性，同时返回模型
        self.model = keras.Model(inputs, outputs, name='densenet121_dr')

        return self.model

    def unfreeze_layers(self, num_layers: int = 121):
        """
        解冻部分预训练层进行微调（与 ResNet50 保持一致接口，提升扩展性）

        参数:
        -----
        num_layers : int
            要解冻的层数（从顶部开始，默认 121 层即解冻全部）
        """
        if self.model is None:
            raise ValueError("模型未构建，请先调用 compile() 或 build() 方法")

        # 找到 DenseNet121 基础模型并解冻指定层数
        for layer in self.model.layers:
            if isinstance(layer, keras.Model):
                # 解冻最后 num_layers 层，其余保持冻结
                for i, sub_layer in enumerate(layer.layers):
                    if i >= len(layer.layers) - num_layers:
                        sub_layer.trainable = True
                    else:
                        sub_layer.trainable = False
                break

        print(f"已解冻 DenseNet121 最后 {num_layers} 层，可进行模型微调")


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("测试 DenseNet-121 模型...")

    model = DenseNet121Model()
    model.compile()
    model.summary()

    print("\nDenseNet-121 模型构建成功！")

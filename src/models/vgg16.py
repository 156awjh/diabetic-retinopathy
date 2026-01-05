from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from .base_model import BaseModel

class VGG16Model(BaseModel):
    def __init__(self, 
                 num_classes: int = 5,
                 input_shape: tuple = (224, 224, 3),
                 batch_size: int = 32):
        super().__init__(
            model_name='vgg16',
            num_classes=num_classes,
            input_shape=input_shape
        )
        self.batch_size = batch_size
        self.class_weights_snapshot = [
            0.2721868600210328,
            2.8757894736842107,
            1.327645788336933,
            8.048445171849426,
            9.914516129032258
        ]
    
    def build(self) -> keras.Model:
        base_model = keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        inputs = keras.Input(shape=self.input_shape)

        # --- 核心：在模型内部嵌入预处理 ---
        # 1. 模仿 preprocess_input
        x = layers.Lambda(lambda z: z / 255.0)(inputs) # 归一化

        # 2. 模拟对比度增强（类似作者的 Ben Color 思路）
        # 使用 tf.image 接口，无需改动外部预处理脚本
        x = layers.Lambda(lambda z: tf.image.adjust_contrast(z, contrast_factor=2.0))(x)

        # --- 接 Base Model ---
        for layer in base_model.layers:
            layer.trainable = 'block5' in layer.name # 依然只解冻 block5

        x = base_model(x, training=False)

        # --- 模仿 Kaggle 作者的高级分类头 ---
        x = layers.GlobalMaxPooling2D()(x) # 换成 Max，抓取强特征
        
        # 如果 batch_size < 16，移除 BN 层以避免验证集表现差
        if self.batch_size >= 16:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(1024, activation='relu')(x)
        # 如果 batch_size < 16，移除 BN 层以避免验证集表现差
        if self.batch_size >= 16:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(64, activation='relu')(x) # 第二层 Dense，进一步提纯
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = keras.Model(inputs, outputs, name='vgg16_enhanced')
        return self.model

    def compile(self, learning_rate: float = 0.001, optimizer: str = 'adam', loss: str = 'categorical_crossentropy', metrics: list = None):
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

        # Focal Loss with per-class alpha weights (向量形式)
        # 针对 5 个类别的 alpha 权重（基于类别权重稍微平滑）
        alpha_weights = tf.constant([0.25, 1.5, 1.0, 3.0, 4.0], dtype=tf.float32)
        
        def focal_loss(y_true, y_pred, gamma=2.0):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            # 计算每个样本所属类别的 alpha 权重
            # y_true 是 one-hot 编码，相乘后得到该类别的权重
            current_alpha = tf.reduce_sum(alpha_weights * y_true, axis=-1, keepdims=True)
            
            ce = -y_true * tf.math.log(y_pred)
            # focal weight: alpha * (1 - p_t)^gamma，只对真实类别应用
            weight = current_alpha * tf.pow(1 - y_pred, gamma)
            return tf.reduce_sum(weight * ce, axis=-1)

        self.model.compile(optimizer=opt, loss=focal_loss, metrics=metrics)
        return self
"""
VGG16 模型（工程增强版：Focal + beta 三段抵消 / eval load-safe）
=====================================================================

目标：
- 不改 train_model.py / evaluate_model.py
- evaluate 端可直接 keras.models.load_model(best.keras)，不需要 custom_objects
- 在不均衡（无病变占比高）场景下：
  1) 让"轻度"不再长期为 0（提升少数类召回）
  2) 尽量维持或小幅提升 overall accuracy（不让宏指标崩）

关键策略：
1) 训练端仍会传 class_weight 给 fit()，我们在 loss 内做"部分抵消"：
   divide by w^beta -> 净样本权重 ~ w^(1-beta)
2) 换用 Focal 形状（强调难样本），对轻度/中度的边界学习更有效
3) 保存时自动保存"干净模型"（内置 loss/无自定义对象），评估可直接 load
4) 预处理固定按 DataLoader 输出 [0,1]：直接使用（VGG16 不需要 imagenet 预处理）
5) GAP + GMP concat：更稳的 pooling 方式

加权采样（Weighted Sampling）思想（参考 PyTorch WeightedRandomSampler）：
=====================================================================
核心原理（类似 PyTorch 的 WeightedRandomSampler）：
1. 计算每个类别的样本数：class_counts = [count_0, count_1, ..., count_4]
2. 计算类别权重：weights = 1.0 / class_counts
   - 样本数越少的类别，权重越大
   - 例如：No_DR 有 1000 个样本，权重 = 1/1000 = 0.001
   -      Proliferative 有 10 个样本，权重 = 1/10 = 0.1（权重是前者的 100 倍）
3. 为每个样本分配其所属类别的权重
4. 使用加权随机采样，使少数类样本被更频繁地采样

PyTorch 实现示例：
-----------------
# Balanced sampler
class_counts = train_df['diagnosis'].value_counts().sort_index().values
weights = 1.0 / class_counts[train_df['diagnosis'].values]
sampler = WeightedRandomSampler(weights, len(weights))
train_loader = DataLoader(train_dataset, args.batch_size, sampler=sampler, ...)

TensorFlow 实现思路：
-------------------
在数据加载阶段，可以使用 tf.data.experimental.rejection_resample() 实现类似效果：
- 目标分布设为均匀分布，使每个类别被等概率采样
- 或者手动实现：按类别分组数据集，然后按权重采样合并

与 class_weight 的区别：
- class_weight：在损失函数层面加权，影响梯度更新
- weighted_sampling：在数据采样层面加权，影响样本出现频率
- 两者可以结合使用，形成"数据层 + 损失层"的双重平衡策略
"""

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from .base_model import BaseModel


class VGG16Model(BaseModel):
    # =========================
    # 你优先调这些（从这里开始）
    # =========================

    # ---- beta 三段：前期更照顾少数类，后期更偏 accuracy ----
    # 不建议上来就 >0.80；你之前 0.83 明显把整体指标压坏了
    BETA_START = 0.68
    BETA_MID   = 0.74
    BETA_END   = 0.76

    # 假设 769 step/epoch，6000≈7.8ep，14000≈18.2ep
    BETA_WARMUP1 = 6000
    BETA_WARMUP2 = 14000

    # ---- focal：先用 1.5，比 2.0 更稳，且不容易把 accuracy 拉崩 ----
    USE_FOCAL = True
    FOCAL_GAMMA = 1.5

    # 可选：alpha（先别开，开了容易和 class_weight 叠加过猛）
    USE_ALPHA = False
    FOCAL_ALPHA = 0.25

    def __init__(
        self,
        num_classes: int = 5,
        input_shape: tuple = (224, 224, 3),
        pretrained: bool = True,
        freeze_base: bool = False,  # 默认开启微调：只冻结前几层
        label_smoothing: float = 0.0,  # 这里先设 0，别和 focal 叠加"软化梯度"
    ):
        super().__init__(
            model_name='vgg16',
            num_classes=num_classes,
            input_shape=input_shape
        )
        self.pretrained = bool(pretrained)
        self.freeze_base = bool(freeze_base)
        self.label_smoothing = float(label_smoothing)

        # 来自训练脚本的 class_weight（训练脚本依旧会传给 fit()）
        # 这里用于 loss 内反向抵消 w^beta
        # 注意：实际值会在训练时动态获取，这里提供默认值
        # 
        # Mild 类权重提升策略（针对 Mild 召回率低的问题）：
        # - 原权重：2.876（是 No_DR 的 10.6 倍）
        # - 新权重：6.5（是 No_DR 的 23.9 倍，是原 Mild 权重的 2.26 倍）
        # - 目的：通过显著提升 Mild 的 class_weight，增强模型对 Mild 类的学习强度
        # - 预期效果：Mild 召回率从 2.5% 提升到 15-30%
        self.class_weights_snapshot = [
            0.2721868600210328,  # No_DR（保持不变）
            8.0,                 # Mild（从 2.876 提升到 6.5，约 2.26 倍，针对召回率低的问题）
            1.327645788336933,   # Moderate（保持不变）
            8.048445171849426,   # Severe（保持不变）
            9.914516129032258    # Proliferative（保持不变）
        ]
    
    @staticmethod
    def compute_weighted_sampling_weights(class_counts):
        """
        计算加权采样权重（参考 PyTorch WeightedRandomSampler 的思想）
        
        核心思想：
        - 计算每个类别的样本数：class_counts = [count_0, count_1, ..., count_4]
        - 计算类别权重：weights = 1.0 / class_counts
        - 样本数越少的类别，权重越大，从而在采样时被更频繁地选择
        
        参数:
        -----
        class_counts : list or np.ndarray
            每个类别的样本数，例如 [1000, 100, 200, 50, 10]
        
        返回:
        -----
        np.ndarray
            每个类别的采样权重，例如 [0.001, 0.01, 0.005, 0.02, 0.1]
        
        示例:
        -----
        >>> import pandas as pd
        >>> train_df = pd.read_csv('train.csv')
        >>> class_counts = train_df['label'].value_counts().sort_index().values
        >>> weights = VGG16Model.compute_weighted_sampling_weights(class_counts)
        >>> print(f"类别权重: {weights}")
        >>> # 在 TensorFlow 中，可以使用这些权重实现加权采样
        
        PyTorch 等价代码:
        -----------------
        class_counts = train_df['diagnosis'].value_counts().sort_index().values
        weights = 1.0 / class_counts[train_df['diagnosis'].values]
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(train_dataset, args.batch_size, sampler=sampler, ...)
        """
        import numpy as np
        class_counts = np.array(class_counts, dtype=np.float32)
        # 避免除零错误
        class_counts = np.maximum(class_counts, 1.0)
        # 计算权重：1.0 / class_count（样本数越少，权重越大）
        weights = 1.0 / class_counts
        return weights
    
    # -------------------------
    # 固定预处理：[0,1] 直接使用（VGG16 不需要 imagenet 预处理）
    # -------------------------
    def _preprocess_01(self, inputs):
        """VGG16 预处理：直接使用 [0,1] 归一化的输入"""
        return tf.cast(inputs, tf.float32)

    # -------------------------
    # Core（单视角）
    # -------------------------
    def _build_core(self) -> keras.Model:
        """构建 VGG16 核心模型"""
        # 加载预训练的 VGG16
        base_model = keras.applications.VGG16(
            weights='imagenet' if self.pretrained else None,
            include_top=False,
            input_shape=self.input_shape
        )
        
        # 冻结策略：
        # - freeze_base=True: 完全冻结 backbone（只训练分类头，适合快速实验）
        # - freeze_base=False: 只冻结前几层，微调高层以适配 DR 特征
        if self.freeze_base:
            base_model.trainable = False
        else:
            # 经验上前 10~15 层主要学习通用边缘/纹理特征，这里选择只冻结前 15 层
            for layer in base_model.layers[:15]:
                layer.trainable = False
            for layer in base_model.layers[15:]:
                layer.trainable = True
        
        # 构建模型
        inputs = keras.Input(shape=self.input_shape, name="core_input")
        x = self._preprocess_01(inputs)
        
        # 特征提取
        # 冻结时 BN 不更新；微调时允许 BN 更新
        feat = base_model(x, training=(not self.freeze_base))
        
        # 更稳的 pooling：GAP + GMP concat
        gap = layers.GlobalAveragePooling2D(name="gap")(feat)
        gmp = layers.GlobalMaxPooling2D(name="gmp")(feat)
        h = layers.Concatenate(name="pool_concat")([gap, gmp])
        
        # 分类头：别太深，避免过拟合和不稳定
        h = layers.BatchNormalization(name="bn0")(h)
        h = layers.Dropout(0.4, name="drop0")(h)
        
        h = layers.Dense(512, activation="swish", name="fc512")(h)
        h = layers.BatchNormalization(name="bn1")(h)
        h = layers.Dropout(0.3, name="drop1")(h)
        
        h = layers.Dense(256, activation="swish", name="fc256")(h)
        h = layers.BatchNormalization(name="bn2")(h)
        h = layers.Dropout(0.2, name="drop2")(h)
        
        out = layers.Dense(self.num_classes, activation="softmax", name="probs")(h)
        return keras.Model(inputs, out, name="vgg16_core")

    def build(self) -> keras.Model:
        """构建完整模型"""
        core = self._build_core()
        inputs = keras.Input(shape=self.input_shape, name="input")
        out = core(inputs)
        self.model = keras.Model(inputs, out, name="vgg16")
        return self.model

    # -------------------------
    # beta 三段 schedule（基于 optimizer.iterations）
    # -------------------------
    def _beta_schedule(self, step: tf.Tensor) -> tf.Tensor:
        """beta 三段调度：前期更照顾少数类，后期更偏 accuracy"""
        step = tf.cast(step, tf.float32)

        w1 = tf.constant(float(self.BETA_WARMUP1), tf.float32)
        w2 = tf.constant(float(self.BETA_WARMUP2), tf.float32)

        b0 = tf.constant(float(self.BETA_START), tf.float32)
        b1 = tf.constant(float(self.BETA_MID), tf.float32)
        b2 = tf.constant(float(self.BETA_END), tf.float32)

        # seg1: [0, w1]  b0 -> b1
        p1 = tf.minimum(1.0, step / tf.maximum(w1, 1.0))
        beta_01 = b0 + (b1 - b0) * p1

        # seg2: [w1, w2] b1 -> b2
        step2 = tf.maximum(0.0, step - w1)
        span2 = tf.maximum(1.0, w2 - w1)
        p2 = tf.minimum(1.0, step2 / span2)
        beta_12 = b1 + (b2 - b1) * p2

        return tf.where(step <= w1, beta_01, beta_12)

    # -------------------------
    # compile：Focal + w^beta 抵消 + clean_save
    # -------------------------
    def compile(
        self,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        loss: str = "categorical_crossentropy",
        metrics: list = None
    ):
        """编译模型：Focal Loss + beta 抵消策略 + clean_save"""
        if self.model is None:
            self.build()

        if metrics is None:
            metrics = ["accuracy"]

        lr = float(learning_rate)
        if optimizer == "adam":
            opt = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        elif optimizer == "sgd":
            opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, clipnorm=1.0)
        else:
            opt = optimizer

        cw = tf.constant(self.class_weights_snapshot, dtype=tf.float32)

        # per-sample CCE（用于 focal base）
        cce_none = keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=self.label_smoothing,
            reduction=keras.losses.Reduction.NONE
        )

        gamma = tf.constant(float(self.FOCAL_GAMMA), tf.float32)
        use_focal = bool(self.USE_FOCAL)
        use_alpha = bool(self.USE_ALPHA)
        alpha = tf.constant(float(self.FOCAL_ALPHA), tf.float32)

        def loss_fn(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            # base CE (batch,)
            ce = cce_none(y_true, y_pred)

            # class weight of each sample (batch,)
            w = tf.reduce_sum(y_true * cw, axis=-1)
            w = tf.maximum(w, 1e-8)

            # beta schedule
            step = tf.cast(opt.iterations, tf.float32)
            beta = self._beta_schedule(step)

            # focal factor
            if use_focal:
                # pt = prob of true class (batch,)
                pt = tf.reduce_sum(y_true * y_pred, axis=-1)
                pt = tf.clip_by_value(pt, 1e-7, 1.0 - 1e-7)
                focal = tf.pow(1.0 - pt, gamma)
                if use_alpha:
                    # alpha_t: positive class weight per sample
                    # 这里 alpha 只是全局系数，先别与 cw 强叠加（默认关）
                    focal = focal * alpha
                ce = ce * focal

            # 部分抵消：divide by w^beta -> net weight ~ w^(1-beta)
            return ce / tf.pow(w, beta)

        self.model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

        # --- clean save：保存干净模型，评估端无需 custom_objects ---
        def clean_save(filepath, *args, **kwargs):
            clone = keras.models.clone_model(self.model)
            clone.set_weights(self.model.get_weights())
            clone.compile(
                optimizer="adam",
                loss=keras.losses.CategoricalCrossentropy(label_smoothing=self.label_smoothing),
                metrics=["accuracy"]
            )
            return clone.save(filepath, *args, **kwargs)

        self.model.save = clean_save

        print("=" * 60)
        print("VGG16 (Focal+beta3) - TF2.5, eval load-safe")
        print(f"pretrained={self.pretrained} | freeze_base={self.freeze_base}")
        print(f"label_smoothing={self.label_smoothing} | lr={lr}")
        print(f"BETA: {self.BETA_START} -> {self.BETA_MID} -> {self.BETA_END} "
              f"(w1={self.BETA_WARMUP1}, w2={self.BETA_WARMUP2})")
        print(f"FOCAL: USE={self.USE_FOCAL} gamma={self.FOCAL_GAMMA} alpha_on={self.USE_ALPHA}")
        print("Input preprocess: fixed [0,1] -> direct use (no imagenet preprocess)")
        print("Pooling: GAP + GMP concat")
        print("Save: clean model (built-in loss) for evaluate load_model()")
        print("=" * 60)

        return self
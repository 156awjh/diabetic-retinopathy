"""
ResNeXt-50 模型（工程增强版：Focal + beta 三段抵消 / eval load-safe）
=====================================================================

目标：
- 不改 train_model.py / evaluate_model.py
- evaluate 端可直接 keras.models.load_model(best.keras)，不需要 custom_objects
- 在不均衡（无病变占比高）场景下：
  1) 让“轻度”不再长期为 0（提升少数类召回）
  2) 尽量维持或小幅提升 overall accuracy（不让宏指标崩）

关键策略：
1) 训练端仍会传 class_weight 给 fit()，我们在 loss 内做“部分抵消”：
   divide by w^beta -> 净样本权重 ~ w^(1-beta)
2) 换用 Focal 形状（强调难样本），对轻度/中度的边界学习更有效
3) 保存时自动保存“干净模型”（内置 loss/无自定义对象），评估可直接 load
4) 预处理固定按 DataLoader 输出 [0,1]：*255 -> resnet_v2.preprocess_input
   （避免 tf.cond 构图导致 KerasTensor 报错）
"""

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from .base_model import BaseModel


class ResNeXt50Model(BaseModel):
    # =========================
    # 你优先调这些（从这里开始）
    # =========================

    # ---- beta 三段：前期更照顾少数类，后期更偏 accuracy ----
    # 不建议上来就 >0.80；你之前 0.83 明显把整体指标压坏了
    BETA_START = 0.68
    BETA_MID   = 0.74
    BETA_END   = 0.76

    # 769 step/epoch（你日志），6000≈7.8ep，14000≈18.2ep
    BETA_WARMUP1 = 6000
    BETA_WARMUP2 = 14000

    # ---- focal：先用 1.5，比 2.0 更稳，且不容易把 accuracy 拉崩 ----
    USE_FOCAL = True
    FOCAL_GAMMA = 1.5

    # 可选：alpha（先别开，开了容易和 class_weight 叠加过猛）
    USE_ALPHA = False
    FOCAL_ALPHA = 0.25

    # TTA：先关。你已经验证 TTA 会拖慢并不一定提升 overall。
    # 如果后续你要“挤 0.2~0.5 个点”，再开。
    USE_TTA = False

    def __init__(
        self,
        num_classes: int = 5,
        input_shape: tuple = (224, 224, 3),
        pretrained: bool = True,
        freeze_base: bool = True,
        label_smoothing: float = 0.0,  # 这里先设 0，别和 focal 叠加“软化梯度”
    ):
        super().__init__(
            model_name="resnext50",
            num_classes=num_classes,
            input_shape=input_shape
        )
        self.pretrained = bool(pretrained)
        self.freeze_base = bool(freeze_base)
        self.label_smoothing = float(label_smoothing)

        # 来自你日志的 class_weight（训练脚本依旧会传给 fit()）
        # 这里用于 loss 内反向抵消 w^beta
        self.class_weights_snapshot = [
            0.2721868600210328,
            2.8757894736842107,
            1.327645788336933,
            8.048445171849426,
            9.914516129032258
        ]

    # -------------------------
    # 固定预处理：[0,1] -> *255 -> resnet_v2 preprocess
    # -------------------------
    def _preprocess_01(self, inputs):
        x = tf.cast(inputs, tf.float32) * 255.0
        x = keras.applications.resnet_v2.preprocess_input(x)
        return x

    # -------------------------
    # Core（单视角）
    # -------------------------
    def _build_core(self) -> keras.Model:
        base_model = keras.applications.ResNet50V2(
            weights="imagenet" if self.pretrained else None,
            include_top=False,
            input_shape=self.input_shape
        )
        base_model.trainable = (not self.freeze_base)

        inputs = keras.Input(shape=self.input_shape, name="core_input")
        x = self._preprocess_01(inputs)

        # 冻结时 BN 不更新；微调时允许 BN 更新
        feat = base_model(x, training=(not self.freeze_base))

        # 更稳的 pooling：GAP + GMP concat（你之前这步是对的）
        gap = layers.GlobalAveragePooling2D(name="gap")(feat)
        gmp = layers.GlobalMaxPooling2D(name="gmp")(feat)
        h = layers.Concatenate(name="pool_concat")([gap, gmp])

        # 头部：别太深，避免过拟合和不稳定
        h = layers.BatchNormalization(name="bn0")(h)
        h = layers.Dropout(0.4, name="drop0")(h)

        h = layers.Dense(512, activation="swish", name="fc512")(h)
        h = layers.BatchNormalization(name="bn1")(h)
        h = layers.Dropout(0.3, name="drop1")(h)

        h = layers.Dense(256, activation="swish", name="fc256")(h)
        h = layers.BatchNormalization(name="bn2")(h)
        h = layers.Dropout(0.2, name="drop2")(h)

        out = layers.Dense(self.num_classes, activation="softmax", name="probs")(h)
        return keras.Model(inputs, out, name="resnext50_core")

    # -------------------------
    # build：默认不启用 TTA（更稳、更快、更利于比较实验）
    # -------------------------
    def build(self) -> keras.Model:
        core = self._build_core()
        inputs = keras.Input(shape=self.input_shape, name="input")

        if not self.USE_TTA:
            out = core(inputs)
            self.model = keras.Model(inputs, out, name="resnext50")
            return self.model

        # 若你后续要开 TTA，再补（你前一版那套 deterministic crop 是可用的）
        out = core(inputs)
        self.model = keras.Model(inputs, out, name="resnext50")
        return self.model

    # -------------------------
    # beta 三段 schedule（基于 optimizer.iterations）
    # -------------------------
    def _beta_schedule(self, step: tf.Tensor) -> tf.Tensor:
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
        if self.model is None:
            self.build()

        if metrics is None:
            metrics = ["accuracy"]

        lr = float(learning_rate)
        if optimizer == "adam":
            opt = keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == "sgd":
            opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
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
        print("ResNeXt-50 (Focal+beta3) - TF2.5, eval load-safe")
        print(f"pretrained={self.pretrained} | freeze_base={self.freeze_base} | USE_TTA={self.USE_TTA}")
        print(f"label_smoothing={self.label_smoothing} | lr={lr}")
        print(f"BETA: {self.BETA_START} -> {self.BETA_MID} -> {self.BETA_END} "
              f"(w1={self.BETA_WARMUP1}, w2={self.BETA_WARMUP2})")
        print(f"FOCAL: USE={self.USE_FOCAL} gamma={self.FOCAL_GAMMA} alpha_on={self.USE_ALPHA}")
        print("Input preprocess: fixed [0,1] -> *255 -> resnet_v2.preprocess_input")
        print("Save: clean model (built-in loss) for evaluate load_model()")
        print("=" * 60)

        return self


if __name__ == "__main__":
    m = ResNeXt50Model()
    m.compile()
    m.model.summary()

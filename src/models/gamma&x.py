from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from .base_model import BaseModel

class VGG16Model(BaseModel):
    def __init__(self, 
                 num_classes: int = 5,
                 input_shape: tuple = (224, 224, 3)):
        super().__init__(
            model_name='vgg16',
            num_classes=num_classes,
            input_shape=input_shape
        )
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
        for layer in base_model.layers:
            if 'block4' in layer.name or 'block5' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False

        inputs = keras.Input(shape=self.input_shape)
        x = keras.applications.vgg16.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = keras.Model(inputs, outputs, name='vgg16')
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

        # Focal Loss
        def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            ce = -y_true * tf.math.log(y_pred)
            weight = y_true * tf.pow(1 - y_pred, gamma) * alpha
            return tf.reduce_sum(weight * ce, axis=-1)

        self.model.compile(optimizer=opt, loss=focal_loss, metrics=metrics)
        return self

        def cw_sqrt_cce(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            per = cce(y_true, y_pred)
            w = tf.reduce_sum(y_true * cw, axis=-1)
            w = tf.maximum(w, 1e-8)
            return per / tf.sqrt(w)

        self.model.compile(optimizer=opt, loss=cw_sqrt_cce, metrics=metrics)
        return self
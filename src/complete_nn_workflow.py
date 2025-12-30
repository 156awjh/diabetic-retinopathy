"""

本程序演示一个完整的神经网络训练流程，为后续 X-ray 图像识别项目打基础。
使用 MNIST 手写数字数据集作为示例。

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ============================================================================
# 【第一部分】数据准备
# ============================================================================
# 说明：这里加载和预处理数据
# 对于 X-ray 识别项目，你需要：
#   - 使用 tf.keras.preprocessing.image_dataset_from_directory() 加载图像
#   - 或使用 tf.data.Dataset 构建数据管道

def load_and_preprocess_data():
    """
    数据加载与预处理
    
    对于 X-ray 项目，替换为：
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'path/to/xray/train',
        image_size=(224, 224),
        batch_size=32
    )
    """
    
    # 加载 MNIST 数据集（示例）
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # 数据预处理：归一化到 [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # 添加通道维度 (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"训练集形状: {x_train.shape}")
    print(f"测试集形状: {x_test.shape}")
    print(f"类别数量: {len(np.unique(y_train))}")
    
    return (x_train, y_train), (x_test, y_test)


# 
# 常用层类型：
#   - Conv2D: 卷积层，用于提取图像特征
#   - MaxPooling2D: 池化层，降低特征图尺寸
#   - Flatten: 展平层，将多维特征转为一维
#   - Dense: 全连接层，用于分类
#   - Dropout: 正则化层，防止过拟合
#   - BatchNormalization: 批归一化，加速训练

def build_cnn_model(input_shape, num_classes):
    """
    构建卷积神经网络模型
    
    参数:
        input_shape: 输入图像形状，如 (224, 224, 3) 用于彩色图像
        num_classes: 分类类别数量
    
    对于 X-ray 识别，可以：
    1. 使用预训练模型（迁移学习）：ResNet50, VGG16, EfficientNet
    2. 自定义 CNN 架构
    """
    
    model = keras.Sequential([
        # -------- 输入层 --------
        keras.Input(shape=input_shape),
        
        # -------- 特征提取层（卷积块1）--------
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        
        # -------- 特征提取层（卷积块2）--------
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        
        # -------- 分类层 --------
        layers.Flatten(name="flatten"),
        layers.Dropout(0.5, name="dropout"),  # 防止过拟合
        layers.Dense(128, activation="relu", name="dense1"),
        layers.Dense(num_classes, activation="softmax", name="output")
    ], name="CNN_Classifier")
    
    # 打印模型结构
    print("\n模型架构:")
    model.summary()
    
    return model


def build_transfer_learning_model(input_shape, num_classes):
    """
    使用迁移学习构建模型（推荐用于 X-ray 识别）
    
    迁移学习优势：
    - 使用在大规模数据集上预训练的权重
    - 需要更少的训练数据
    - 训练更快，效果更好
    """
    print("\n【迁移学习模型示例】")
    
    # 加载预训练的 ResNet50（不包含顶层分类器）
    base_model = keras.applications.ResNet50(
        weights="imagenet",      # 使用 ImageNet 预训练权重
        include_top=False,       # 不包含原始分类层
        input_shape=input_shape
    )
    
    # 冻结预训练层（可选，微调时解冻）
    base_model.trainable = False
    
    # 添加自定义分类头
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    return model



def compile_model(model):
    """
    编译模型 - 配置训练参数
    """
    
    # -------- 学习率配置 --------
    # 固定学习率
    learning_rate = 0.001
    
    # -------- 优化器选择 --------
    # 常用优化器：
    # - Adam: 自适应学习率，最常用
    # - SGD: 随机梯度下降，配合动量使用
    # - RMSprop: 适合 RNN
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # -------- 损失函数选择 --------
    # 分类问题：
    # - sparse_categorical_crossentropy: 标签为整数 (0, 1, 2, ...)
    # - categorical_crossentropy: 标签为 one-hot 编码
    # - binary_crossentropy: 二分类问题
    loss_function = "sparse_categorical_crossentropy"
    
    # -------- 编译模型 --------
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=["accuracy"]  # 训练时监控的指标
    )
    
    print(f"\n训练配置:")
    print(f"  优化器: Adam")
    print(f"  学习率: {learning_rate}")
    print(f"  损失函数: {loss_function}")
    print(f"  监控指标: accuracy")
    
    return model


# ============================================================================
# 【第四部分】模型训练 ⭐⭐⭐
# ============================================================================
# 说明：这里执行实际的训练过程
#
# 重要参数：
#   - epochs: 训练轮数
#   - batch_size: 批次大小
#   - validation_split: 验证集比例
#   - callbacks: 回调函数（早停、模型保存、学习率调整等）

def train_model(model, x_train, y_train, x_val=None, y_val=None):
    """
    训练模型
    """
    
    # -------- 训练超参数 --------
    EPOCHS = 5           # 训练轮数（实际项目中通常 50-100）
    BATCH_SIZE = 64      # 批次大小
    VALIDATION_SPLIT = 0.1  # 验证集比例
    
    # -------- 回调函数配置 --------
    callbacks = [
        # 早停：验证损失不再下降时停止训练
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        
        # 模型检查点：保存最佳模型
        keras.callbacks.ModelCheckpoint(
            filepath="best_model.keras",
            monitor="val_accuracy",
            save_best_only=True
        ),
        
        # 学习率调整：验证损失停滞时降低学习率
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    print(f"\n训练参数:")
    print(f"  训练轮数 (epochs): {EPOCHS}")
    print(f"  批次大小 (batch_size): {BATCH_SIZE}")
    print(f"  验证集比例: {VALIDATION_SPLIT}")
    print(f"\n回调函数:")
    print(f"  - EarlyStopping: 早停防止过拟合")
    print(f"  - ModelCheckpoint: 保存最佳模型")
    print(f"  - ReduceLROnPlateau: 自动调整学习率")
    
    print("\n开始训练...")
    print("-" * 40)
    
    # -------- 执行训练 --------
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    return history




def evaluate_model(model, x_test, y_test, history):
    """
    评估模型性能
    """
    print("\n" + "=" * 60)
    print("【第五部分】评估指标")
    print("=" * 60)
    
    # -------- 基本评估 --------
    print("\n1. 基本评估指标:")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"   测试集损失 (Loss): {test_loss:.4f}")
    print(f"   测试集准确率 (Accuracy): {test_accuracy:.4f}")
    
    # -------- 预测结果 --------
    print("\n2. 预测示例:")
    predictions = model.predict(x_test[:5], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    print(f"   真实标签: {y_test[:5]}")
    print(f"   预测标签: {predicted_classes}")
    
    # -------- 详细分类报告 --------
    print("\n3. 详细评估（需要 sklearn）:")
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        
        y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
        
        print("\n   分类报告:")
        print(classification_report(y_test, y_pred, digits=4))
        
        print("   混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
    except ImportError:
        print("   (安装 sklearn 以获取详细报告: pip install scikit-learn)")
    
    # -------- 训练历史 --------
    print("\n4. 训练历史:")
    print(f"   最终训练准确率: {history.history['accuracy'][-1]:.4f}")
    print(f"   最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
    print(f"   最终训练损失: {history.history['loss'][-1]:.4f}")
    print(f"   最终验证损失: {history.history['val_loss'][-1]:.4f}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("TensorFlow 完整神经网络工作流程演示")
    print("=" * 60)
    print(f"TensorFlow 版本: {tf.__version__}")
    
    # 【第一部分】数据准备
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # 【第二部分】构造神经网络结构
    input_shape = (28, 28, 1)  # MNIST 图像尺寸
    num_classes = 10           # 0-9 共 10 个类别
    model = build_cnn_model(input_shape, num_classes)
    
    # 【第三部分】训练配置
    model = compile_model(model)
    
    # 【第四部分】模型训练
    history = train_model(model, x_train, y_train)
    
    # 【第五部分】评估指标
    evaluate_model(model, x_test, y_test, history)



if __name__ == "__main__":
    main()

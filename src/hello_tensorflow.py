"""
TensorFlow 入门演示程序
========================

本程序演示 TensorFlow 的基本工作流程，包括：
1. 张量 (Tensor) 的创建
2. 基本数学运算
3. 简单的神经网络层操作

TensorFlow 工作流程说明：
-----------------------
TensorFlow 2.x 默认使用 Eager Execution（即时执行）模式，
这意味着操作会立即执行并返回结果，便于调试和学习。

工作流程：定义数据/张量 -> 定义操作 -> 执行计算 -> 获取结果
"""

import tensorflow as tf
import numpy as np

def main():
    # ========================================
    # 第一部分：显示 TensorFlow 版本信息
    # ========================================
    print("=" * 50)
    print("TensorFlow 入门演示程序")
    print("=" * 50)
    print(f"\nTensorFlow 版本: {tf.__version__}")
    print(f"Eager Execution 模式: {tf.executing_eagerly()}")
    
    # ========================================
    # 第二部分：张量 (Tensor) 基础
    # ========================================
    # 张量是 TensorFlow 中的核心数据结构，本质上是多维数组
    
    print("\n" + "-" * 50)
    print("第一部分：张量 (Tensor) 基础")
    print("-" * 50)
    
    # 1. 标量 (0维张量) - 单个数值
    scalar = tf.constant(3.14)
    print(f"\n1. 标量 (0维张量):")
    print(f"   值: {scalar}")
    print(f"   形状: {scalar.shape}")
    print(f"   数据类型: {scalar.dtype}")
    
    # 2. 向量 (1维张量) - 一维数组
    vector = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"\n2. 向量 (1维张量):")
    print(f"   值: {vector}")
    print(f"   形状: {vector.shape}")

    # 3. 矩阵 (2维张量) - 二维数组
    matrix = tf.constant([[1, 2, 3],
                          [4, 5, 6]])
    print(f"\n3. 矩阵 (2维张量):")
    print(f"   值:\n{matrix}")
    print(f"   形状: {matrix.shape}")
    
    # ========================================
    # 第三部分：基本张量运算
    # ========================================
    print("\n" + "-" * 50)
    print("第二部分：基本张量运算")
    print("-" * 50)
    
    # 创建两个张量用于运算演示
    a = tf.constant([1, 2, 3, 4])
    b = tf.constant([5, 6, 7, 8])
    
    print(f"\n张量 a: {a.numpy()}")
    print(f"张量 b: {b.numpy()}")
    
    # 1. 加法运算
    add_result = tf.add(a, b)  # 或者使用 a + b
    print(f"\n1. 加法 (a + b): {add_result.numpy()}")
    
    # 2. 乘法运算 (逐元素)
    mul_result = tf.multiply(a, b)  # 或者使用 a * b
    print(f"2. 乘法 (a * b): {mul_result.numpy()}")
    
    # 3. 矩阵乘法
    matrix_a = tf.constant([[1, 2], [3, 4]])
    matrix_b = tf.constant([[5, 6], [7, 8]])
    matmul_result = tf.matmul(matrix_a, matrix_b)
    print(f"\n3. 矩阵乘法:")
    print(f"   矩阵 A:\n{matrix_a.numpy()}")
    print(f"   矩阵 B:\n{matrix_b.numpy()}")
    print(f"   A @ B:\n{matmul_result.numpy()}")
    
    # 4. 常用数学函数
    x = tf.constant([1.0, 4.0, 9.0, 16.0])
    print(f"\n4. 常用数学函数:")
    print(f"   原始值: {x.numpy()}")
    print(f"   平方根: {tf.sqrt(x).numpy()}")
    print(f"   求和: {tf.reduce_sum(x).numpy()}")
    print(f"   平均值: {tf.reduce_mean(x).numpy()}")
    
    # ========================================
    # 第四部分：简单神经网络层演示
    # ========================================
    print("\n" + "-" * 50)
    print("第三部分：简单神经网络层演示")
    print("-" * 50)
    
    # 使用 Keras Dense 层演示神经网络的基本单元
    # Dense 层是全连接层，每个输入节点都连接到每个输出节点
    
    # 创建一个简单的 Dense 层：4个输入，3个输出
    dense_layer = tf.keras.layers.Dense(
        units=3,           # 输出维度
        activation='relu', # 激活函数
        name='demo_dense'
    )
    
    # 准备输入数据：batch_size=2, features=4
    input_data = tf.constant([[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0]])
    
    # 前向传播
    output = dense_layer(input_data)
    
    print(f"\nDense 层配置:")
    print(f"   输入形状: {input_data.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"\n输入数据:\n{input_data.numpy()}")
    print(f"\n输出结果:\n{output.numpy()}")
    
    # 查看层的权重
    print(f"\n层的权重 (weights):")
    print(f"   权重矩阵形状: {dense_layer.weights[0].shape}")
    print(f"   偏置向量形状: {dense_layer.weights[1].shape}")
    
    # ========================================
    # 第五部分：Eager vs Graph 执行模式
    # ========================================
    print("\n" + "-" * 50)
    print("第四部分：Eager vs Graph 执行模式")
    print("-" * 50)
    
    # Eager Execution（即时执行）- TensorFlow 2.x 默认模式
    # 操作立即执行，便于调试
    print("\n1. Eager Execution（即时执行）:")
    eager_result = tf.add(1, 2)
    print(f"   tf.add(1, 2) = {eager_result.numpy()}")
    print(f"   结果立即可用，无需 session")
    
    # Graph Execution（图执行）- 使用 @tf.function 装饰器
    # 将 Python 函数编译为 TensorFlow 图，提高性能
    @tf.function
    def graph_add(x, y):
        return tf.add(x, y)
    
    print("\n2. Graph Execution（图执行）:")
    graph_result = graph_add(tf.constant(1), tf.constant(2))
    print(f"   使用 @tf.function 装饰器")
    print(f"   graph_add(1, 2) = {graph_result.numpy()}")
    print(f"   函数被编译为计算图，执行更高效")
    
    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
    print("""
TensorFlow 工作流程总结：
1. 创建张量 (Tensor) - 数据的基本单位
2. 定义操作 - 加法、乘法、矩阵运算等
3. 执行计算 - Eager 模式下立即执行
4. 获取结果 - 使用 .numpy() 转换为 NumPy 数组

下一步学习建议：
- 学习 tf.keras 构建神经网络模型
- 了解 tf.data 进行数据处理
- 探索 tf.GradientTape 进行自动微分
""")

if __name__ == "__main__":
    main()

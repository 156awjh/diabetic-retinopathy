"""
糖尿病视网膜病变数据预处理主程序
================================

本程序执行完整的数据预处理流程：
1. 数据集扫描与划分
2. 类别平衡处理（过采样 + 类别权重）
3. 数据可视化
4. 生成统计报告

作者: [你的名字]
日期: 2024

使用方法:
---------
python src/main_preprocessing.py
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

import pandas as pd
from datetime import datetime

# 导入自定义模块
from config import (
    DATA_DIR, PREPROCESSING_OUTPUT_DIR, CLASS_NAMES_CN,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    OVERSAMPLE_STRATEGY, CLASS_WEIGHT_STRATEGY,
    TRAIN_CSV_FILE, VAL_CSV_FILE, TEST_CSV_FILE,
    OVERSAMPLED_TRAIN_CSV, CLASS_WEIGHTS_FILE,
    DISTRIBUTION_PLOT, BALANCE_COMPARISON_PLOT
)
from preprocessing.dataset_splitter import DatasetSplitter
from preprocessing.class_balancer import ClassBalancer
from preprocessing.visualization import DataVisualizer


def print_header(title: str) -> None:
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str) -> None:
    """打印章节标题"""
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def main():
    """主函数"""
    start_time = datetime.now()
    
    print_header("糖尿病视网膜病变数据预处理")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {PREPROCESSING_OUTPUT_DIR}")
    
    # ================================================================
    # 第一步：数据集划分
    # ================================================================
    print_header("第一步：数据集划分")
    
    # 创建划分器
    splitter = DatasetSplitter(
        data_dir=str(DATA_DIR),
        output_dir=str(PREPROCESSING_OUTPUT_DIR),
        random_seed=RANDOM_SEED
    )
    
    # 扫描数据集
    dataset_df = splitter.scan_dataset()
    
    # 划分数据集
    splits = splitter.split(
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    
    # 保存划分结果
    splitter.save_splits(splits)
    
    # 打印统计
    splitter.print_statistics(splits)
    
    # ================================================================
    # 第二步：类别平衡处理
    # ================================================================
    print_header("第二步：类别平衡处理")
    
    # 获取训练集类别分布
    train_df = splits['train']
    class_counts = train_df['label'].value_counts().to_dict()
    
    # 创建平衡器
    balancer = ClassBalancer(class_counts, random_seed=RANDOM_SEED)
    
    # 过采样
    print_section("2.1 随机过采样")
    balanced_train_df = balancer.random_oversample(
        train_df,
        strategy=OVERSAMPLE_STRATEGY
    )
    balancer.save_oversampled_data(balanced_train_df, str(OVERSAMPLED_TRAIN_CSV))
    
    # 计算类别权重
    print_section("2.2 计算类别权重")
    class_weights = balancer.calculate_class_weights(strategy=CLASS_WEIGHT_STRATEGY)
    balancer.save_weights(class_weights, str(CLASS_WEIGHTS_FILE))
    
    # ================================================================
    # 第三步：数据可视化
    # ================================================================
    print_header("第三步：数据可视化")
    
    # 创建可视化器
    visualizer = DataVisualizer(output_dir=str(PREPROCESSING_OUTPUT_DIR))
    
    # 原始分布图
    print_section("3.1 绘制原始类别分布")
    visualizer.plot_class_distribution(
        class_counts,
        title="训练集类别分布（原始）",
        save_path=str(DISTRIBUTION_PLOT)
    )
    
    # 平衡前后对比图
    print_section("3.2 绘制平衡前后对比")
    after_counts = balanced_train_df['label'].value_counts().to_dict()
    visualizer.plot_balance_comparison(
        class_counts,
        after_counts,
        save_path=str(BALANCE_COMPARISON_PLOT)
    )
    
    # 类别权重图
    print_section("3.3 绘制类别权重")
    visualizer.plot_class_weights(
        class_weights,
        save_path=str(PREPROCESSING_OUTPUT_DIR / "class_weights.png")
    )
    
    # 数据集划分分布图
    print_section("3.4 绘制数据集划分分布")
    visualizer.plot_split_distribution(
        splits,
        save_path=str(PREPROCESSING_OUTPUT_DIR / "split_distribution.png")
    )
    
    # ================================================================
    # 第四步：生成统计报告
    # ================================================================
    print_header("第四步：生成统计报告")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 生成报告
    report = f"""
================================================================================
                    糖尿病视网膜病变数据预处理报告
================================================================================

生成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
处理耗时: {duration:.2f} 秒

--------------------------------------------------------------------------------
1. 数据集概况
--------------------------------------------------------------------------------
数据目录: {DATA_DIR}
总样本数: {len(dataset_df):,}

类别分布:
"""
    
    for label in range(5):
        count = len(dataset_df[dataset_df['label'] == label])
        pct = count / len(dataset_df) * 100
        report += f"  类别 {label} ({CLASS_NAMES_CN[label]}): {count:,} ({pct:.1f}%)\n"
    
    report += f"""
--------------------------------------------------------------------------------
2. 数据集划分
--------------------------------------------------------------------------------
划分比例: 训练集 {TRAIN_RATIO:.0%} / 验证集 {VAL_RATIO:.0%} / 测试集 {TEST_RATIO:.0%}
随机种子: {RANDOM_SEED}

训练集: {len(splits['train']):,} 样本
验证集: {len(splits['val']):,} 样本
测试集: {len(splits['test']):,} 样本

--------------------------------------------------------------------------------
3. 类别平衡处理
--------------------------------------------------------------------------------
过采样策略: {OVERSAMPLE_STRATEGY}
过采样前训练集: {len(train_df):,} 样本
过采样后训练集: {len(balanced_train_df):,} 样本

类别权重策略: {CLASS_WEIGHT_STRATEGY}
类别权重:
"""
    
    for label, weight in sorted(class_weights.items()):
        report += f"  类别 {label} ({CLASS_NAMES_CN[label]}): {weight:.4f}\n"
    
    report += f"""
--------------------------------------------------------------------------------
4. 输出文件
--------------------------------------------------------------------------------
训练集: {TRAIN_CSV_FILE}
验证集: {VAL_CSV_FILE}
测试集: {TEST_CSV_FILE}
过采样训练集: {OVERSAMPLED_TRAIN_CSV}
类别权重: {CLASS_WEIGHTS_FILE}
类别分布图: {DISTRIBUTION_PLOT}
平衡对比图: {BALANCE_COMPARISON_PLOT}

================================================================================
                              预处理完成！
================================================================================
"""
    
    print(report)
    
    # 保存报告
    report_file = PREPROCESSING_OUTPUT_DIR / "preprocessing_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"报告已保存: {report_file}")
    
    return splits, balanced_train_df, class_weights


if __name__ == "__main__":
    splits, balanced_train_df, class_weights = main()

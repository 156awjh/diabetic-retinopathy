"""
数据可视化模块
==============

提供数据分布可视化功能，包括：
1. 类别分布柱状图
2. 平衡前后对比图
3. 数据集划分饼图

作者: [你的名字]
日期: 2024
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CLASS_NAMES_CN, NUM_CLASSES, PREPROCESSING_OUTPUT_DIR,
    DISTRIBUTION_PLOT, BALANCE_COMPARISON_PLOT
)


class DataVisualizer:
    """
    数据可视化器
    
    功能:
    -----
    1. 绘制类别分布柱状图
    2. 绘制平衡前后对比图
    3. 保存图片到指定目录
    """
    
    def __init__(self, output_dir: str = None):
        """
        初始化可视化器
        
        参数:
        -----
        output_dir : str, optional
            输出目录，默认使用配置文件中的 PREPROCESSING_OUTPUT_DIR
        """
        self.output_dir = Path(output_dir) if output_dir else PREPROCESSING_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 颜色方案
        self.colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    
    def plot_class_distribution(
        self,
        class_counts: Dict[int, int],
        title: str = "类别分布",
        save_path: str = None,
        show: bool = False
    ) -> None:
        """
        绘制类别分布柱状图
        
        参数:
        -----
        class_counts : dict
            各类别样本数量
        title : str
            图表标题
        save_path : str, optional
            保存路径
        show : bool
            是否显示图表
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = [f"{i}\n{CLASS_NAMES_CN[i]}" for i in sorted(class_counts.keys())]
        counts = [class_counts[i] for i in sorted(class_counts.keys())]
        total = sum(counts)
        
        bars = ax.bar(labels, counts, color=self.colors, edgecolor='white', linewidth=1.5)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = count / total * 100
            ax.annotate(
                f'{count:,}\n({pct:.1f}%)',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold'
            )
        
        ax.set_xlabel('类别', fontsize=12)
        ax.set_ylabel('样本数量', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(counts) * 1.15)
        
        # 添加网格线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存图表: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_balance_comparison(
        self,
        before_counts: Dict[int, int],
        after_counts: Dict[int, int],
        title: str = "类别平衡前后对比",
        save_path: str = None,
        show: bool = False
    ) -> None:
        """
        绘制平衡前后对比图
        
        参数:
        -----
        before_counts : dict
            平衡前各类别样本数量
        after_counts : dict
            平衡后各类别样本数量
        title : str
            图表标题
        save_path : str, optional
            保存路径
        show : bool
            是否显示图表
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        labels = [f"{i}\n{CLASS_NAMES_CN[i]}" for i in range(NUM_CLASSES)]
        
        # 左图：平衡前
        before_values = [before_counts.get(i, 0) for i in range(NUM_CLASSES)]
        bars1 = axes[0].bar(labels, before_values, color=self.colors, edgecolor='white', linewidth=1.5)
        axes[0].set_title('平衡前', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('类别', fontsize=10)
        axes[0].set_ylabel('样本数量', fontsize=10)
        
        for bar, count in zip(bars1, before_values):
            height = bar.get_height()
            axes[0].annotate(
                f'{count:,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9
            )
        
        # 右图：平衡后
        after_values = [after_counts.get(i, 0) for i in range(NUM_CLASSES)]
        bars2 = axes[1].bar(labels, after_values, color=self.colors, edgecolor='white', linewidth=1.5)
        axes[1].set_title('平衡后 (过采样)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('类别', fontsize=10)
        axes[1].set_ylabel('样本数量', fontsize=10)
        
        for bar, count in zip(bars2, after_values):
            height = bar.get_height()
            axes[1].annotate(
                f'{count:,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9
            )
        
        # 统一 Y 轴范围
        max_val = max(max(before_values), max(after_values))
        for ax in axes:
            ax.set_ylim(0, max_val * 1.15)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存图表: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_split_distribution(
        self,
        splits: Dict[str, pd.DataFrame],
        save_path: str = None,
        show: bool = False
    ) -> None:
        """
        绘制数据集划分分布图
        
        参数:
        -----
        splits : dict
            划分结果 {'train': df, 'val': df, 'test': df}
        save_path : str, optional
            保存路径
        show : bool
            是否显示图表
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        split_names = {'train': '训练集', 'val': '验证集', 'test': '测试集'}
        
        for idx, (split_name, df) in enumerate(splits.items()):
            ax = axes[idx]
            
            counts = df['label'].value_counts().sort_index()
            labels = [f"{i}\n{CLASS_NAMES_CN[i]}" for i in counts.index]
            
            bars = ax.bar(labels, counts.values, color=self.colors, edgecolor='white', linewidth=1.5)
            
            ax.set_title(f'{split_names[split_name]} ({len(df):,} 样本)', fontsize=12, fontweight='bold')
            ax.set_xlabel('类别', fontsize=10)
            ax.set_ylabel('样本数量', fontsize=10)
            
            for bar, count in zip(bars, counts.values):
                height = bar.get_height()
                ax.annotate(
                    f'{count:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8
                )
            
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
        
        fig.suptitle('数据集划分分布', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存图表: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_class_weights(
        self,
        weights: Dict[int, float],
        save_path: str = None,
        show: bool = False
    ) -> None:
        """
        绘制类别权重图
        
        参数:
        -----
        weights : dict
            类别权重
        save_path : str, optional
            保存路径
        show : bool
            是否显示图表
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = [f"{i}\n{CLASS_NAMES_CN[i]}" for i in sorted(weights.keys())]
        values = [weights[i] for i in sorted(weights.keys())]
        
        bars = ax.bar(labels, values, color=self.colors, edgecolor='white', linewidth=1.5)
        
        for bar, weight in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{weight:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold'
            )
        
        ax.set_xlabel('类别', fontsize=12)
        ax.set_ylabel('权重', fontsize=12)
        ax.set_title('类别权重分布', fontsize=14, fontweight='bold')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='基准权重 (1.0)')
        ax.legend()
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存图表: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    from config import TRAIN_CSV_FILE, OVERSAMPLED_TRAIN_CSV, CLASS_WEIGHTS_FILE
    import json
    
    # 创建可视化器
    visualizer = DataVisualizer()
    
    # 加载数据
    train_df = pd.read_csv(TRAIN_CSV_FILE)
    oversampled_df = pd.read_csv(OVERSAMPLED_TRAIN_CSV)
    
    # 获取类别分布
    before_counts = train_df['label'].value_counts().to_dict()
    after_counts = oversampled_df['label'].value_counts().to_dict()
    
    # 绘制原始分布
    visualizer.plot_class_distribution(
        before_counts,
        title="训练集类别分布（原始）",
        save_path=DISTRIBUTION_PLOT
    )
    
    # 绘制平衡前后对比
    visualizer.plot_balance_comparison(
        before_counts,
        after_counts,
        save_path=BALANCE_COMPARISON_PLOT
    )
    
    # 加载并绘制类别权重
    with open(CLASS_WEIGHTS_FILE, 'r') as f:
        weights = {int(k): v for k, v in json.load(f).items()}
    
    visualizer.plot_class_weights(
        weights,
        save_path=PREPROCESSING_OUTPUT_DIR / "class_weights.png"
    )
    
    print("\n可视化完成!")

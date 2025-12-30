"""
数据集划分模块
==============

负责将原始数据集划分为训练集、验证集和测试集。
使用分层采样确保各子集中类别分布与原始数据一致。

作者: [你的名字]
日期: 2024
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split

# 导入配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, PREPROCESSING_OUTPUT_DIR, CLASS_NAMES, CLASS_NAMES_CN,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, SUPPORTED_FORMATS,
    SPLIT_CSV_FILE, TRAIN_CSV_FILE, VAL_CSV_FILE, TEST_CSV_FILE
)


class DatasetSplitter:
    """
    数据集划分器
    
    功能:
    -----
    1. 扫描数据目录，收集所有图片路径和标签
    2. 使用分层采样划分训练集、验证集、测试集
    3. 保存划分结果到 CSV 文件
    4. 打印详细的统计信息
    
    使用示例:
    ---------
    >>> splitter = DatasetSplitter(data_dir='data/data2')
    >>> splits = splitter.split()
    >>> splitter.save_splits(splits)
    >>> splitter.print_statistics(splits)
    
    属性:
    -----
    data_dir : Path
        原始数据目录路径
    output_dir : Path
        输出目录路径
    random_seed : int
        随机种子，确保可复现
    """
    
    def __init__(
        self,
        data_dir: str = None,
        output_dir: str = None,
        random_seed: int = None
    ):
        """
        初始化数据集划分器
        
        参数:
        -----
        data_dir : str, optional
            原始数据目录路径，默认使用配置文件中的 DATA_DIR
        output_dir : str, optional
            输出目录路径，默认使用配置文件中的 PREPROCESSING_OUTPUT_DIR
        random_seed : int, optional
            随机种子，默认使用配置文件中的 RANDOM_SEED
        """
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else PREPROCESSING_OUTPUT_DIR
        self.random_seed = random_seed if random_seed is not None else RANDOM_SEED
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集 DataFrame
        self.dataset_df = None
    
    def scan_dataset(self) -> pd.DataFrame:
        """
        扫描数据集目录，收集所有图片信息
        
        返回:
        -----
        pd.DataFrame
            包含以下列的 DataFrame:
            - filepath: 图片完整路径
            - filename: 图片文件名
            - label: 类别标签 (0-4)
            - label_name: 类别英文名称
            - label_name_cn: 类别中文名称
        """
        print("\n" + "=" * 60)
        print("【扫描数据集】")
        print("=" * 60)
        print(f"数据目录: {self.data_dir}")
        
        data_records = []
        
        # 遍历每个类别目录
        for label in range(5):
            class_dir = self.data_dir / str(label)
            
            if not class_dir.exists():
                print(f"  警告: 类别 {label} 目录不存在: {class_dir}")
                continue
            
            # 收集该类别下的所有图片
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in SUPPORTED_FORMATS:
                    data_records.append({
                        'filepath': str(img_file),
                        'filename': img_file.name,
                        'label': label,
                        'label_name': CLASS_NAMES[label],
                        'label_name_cn': CLASS_NAMES_CN[label]
                    })
        
        # 创建 DataFrame
        self.dataset_df = pd.DataFrame(data_records)
        
        # 打印扫描结果
        print(f"\n扫描完成!")
        print(f"总样本数: {len(self.dataset_df)}")
        print(f"\n各类别样本数:")
        for label in range(5):
            count = len(self.dataset_df[self.dataset_df['label'] == label])
            pct = count / len(self.dataset_df) * 100
            print(f"  类别 {label} ({CLASS_NAMES_CN[label]}): {count:,} ({pct:.1f}%)")
        
        return self.dataset_df
    
    def split(
        self,
        train_ratio: float = None,
        val_ratio: float = None,
        test_ratio: float = None
    ) -> Dict[str, pd.DataFrame]:
        """
        分层划分数据集
        
        使用分层采样确保每个子集中的类别分布与原始数据一致。
        
        参数:
        -----
        train_ratio : float, optional
            训练集比例，默认 0.70
        val_ratio : float, optional
            验证集比例，默认 0.15
        test_ratio : float, optional
            测试集比例，默认 0.15
        
        返回:
        -----
        dict
            {'train': DataFrame, 'val': DataFrame, 'test': DataFrame}
        """
        # 使用默认值或配置值
        train_ratio = train_ratio or TRAIN_RATIO
        val_ratio = val_ratio or VAL_RATIO
        test_ratio = test_ratio or TEST_RATIO
        
        # 验证比例
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"划分比例之和必须为 1.0，当前为 {total_ratio}")
        
        # 如果还没有扫描数据集，先扫描
        if self.dataset_df is None:
            self.scan_dataset()
        
        print("\n" + "=" * 60)
        print("【数据集划分】")
        print("=" * 60)
        print(f"划分比例: 训练集 {train_ratio:.0%} / 验证集 {val_ratio:.0%} / 测试集 {test_ratio:.0%}")
        print(f"随机种子: {self.random_seed}")
        
        # 第一次划分: 分出测试集
        train_val_df, test_df = train_test_split(
            self.dataset_df,
            test_size=test_ratio,
            stratify=self.dataset_df['label'],
            random_state=self.random_seed
        )
        
        # 第二次划分: 从剩余数据中分出验证集
        # 计算验证集在剩余数据中的比例
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            stratify=train_val_df['label'],
            random_state=self.random_seed
        )
        
        # 添加 split 列标记
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        print(f"\n划分完成!")
        print(f"  训练集: {len(train_df):,} 样本")
        print(f"  验证集: {len(val_df):,} 样本")
        print(f"  测试集: {len(test_df):,} 样本")
        
        return splits
    
    def save_splits(self, splits: Dict[str, pd.DataFrame]) -> None:
        """
        保存划分结果到 CSV 文件
        
        参数:
        -----
        splits : dict
            划分结果字典
        """
        print("\n" + "=" * 60)
        print("【保存划分结果】")
        print("=" * 60)
        
        # 保存各子集
        splits['train'].to_csv(TRAIN_CSV_FILE, index=False, encoding='utf-8')
        splits['val'].to_csv(VAL_CSV_FILE, index=False, encoding='utf-8')
        splits['test'].to_csv(TEST_CSV_FILE, index=False, encoding='utf-8')
        
        # 保存合并的完整划分信息
        all_splits = pd.concat([splits['train'], splits['val'], splits['test']])
        all_splits.to_csv(SPLIT_CSV_FILE, index=False, encoding='utf-8')
        
        print(f"已保存:")
        print(f"  训练集: {TRAIN_CSV_FILE}")
        print(f"  验证集: {VAL_CSV_FILE}")
        print(f"  测试集: {TEST_CSV_FILE}")
        print(f"  完整划分: {SPLIT_CSV_FILE}")
    
    def print_statistics(self, splits: Dict[str, pd.DataFrame]) -> None:
        """
        打印详细的划分统计信息
        
        参数:
        -----
        splits : dict
            划分结果字典
        """
        print("\n" + "=" * 60)
        print("【划分统计详情】")
        print("=" * 60)
        
        # 创建统计表格
        stats_data = []
        
        for split_name, df in splits.items():
            for label in range(5):
                count = len(df[df['label'] == label])
                pct = count / len(df) * 100 if len(df) > 0 else 0
                stats_data.append({
                    '数据集': split_name,
                    '类别': label,
                    '类别名称': CLASS_NAMES_CN[label],
                    '样本数': count,
                    '占比': f"{pct:.1f}%"
                })
        
        stats_df = pd.DataFrame(stats_data)
        
        # 按数据集分组打印
        for split_name in ['train', 'val', 'test']:
            split_stats = stats_df[stats_df['数据集'] == split_name]
            total = split_stats['样本数'].sum()
            
            print(f"\n{split_name.upper()} 集 (共 {total:,} 样本):")
            print("-" * 40)
            for _, row in split_stats.iterrows():
                print(f"  类别 {row['类别']} ({row['类别名称']}): {row['样本数']:,} ({row['占比']})")
        
        # 验证分层采样效果
        print("\n" + "-" * 60)
        print("【分层采样验证】")
        print("-" * 60)
        print("各类别在不同数据集中的比例应该接近:")
        
        for label in range(5):
            original_pct = len(self.dataset_df[self.dataset_df['label'] == label]) / len(self.dataset_df) * 100
            train_pct = len(splits['train'][splits['train']['label'] == label]) / len(splits['train']) * 100
            val_pct = len(splits['val'][splits['val']['label'] == label]) / len(splits['val']) * 100
            test_pct = len(splits['test'][splits['test']['label'] == label]) / len(splits['test']) * 100
            
            print(f"  类别 {label}: 原始 {original_pct:.1f}% | 训练 {train_pct:.1f}% | 验证 {val_pct:.1f}% | 测试 {test_pct:.1f}%")


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 创建划分器
    splitter = DatasetSplitter()
    
    # 扫描数据集
    splitter.scan_dataset()
    
    # 划分数据集
    splits = splitter.split()
    
    # 保存结果
    splitter.save_splits(splits)
    
    # 打印统计
    splitter.print_statistics(splits)

"""
类别平衡模块
============

处理类别不均衡问题，提供两种策略：
1. 过采样 (Oversampling): 增加少数类样本数量
2. 类别权重 (Class Weights): 在损失函数中给少数类更高权重

作者: [你的名字]
日期: 2024
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Literal
from collections import Counter

# 导入配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CLASS_NAMES_CN, NUM_CLASSES, RANDOM_SEED,
    OVERSAMPLED_TRAIN_CSV, CLASS_WEIGHTS_FILE
)


class ClassBalancer:
    """
    类别平衡器
    
    功能:
    -----
    1. 随机过采样：复制少数类样本以平衡数据集
    2. 类别权重计算：为损失函数计算类别权重
    
    使用示例:
    ---------
    >>> # 从 DataFrame 获取类别分布
    >>> class_counts = train_df['label'].value_counts().to_dict()
    >>> balancer = ClassBalancer(class_counts)
    >>> 
    >>> # 过采样
    >>> balanced_df = balancer.random_oversample(train_df, strategy='median')
    >>> 
    >>> # 计算类别权重
    >>> weights = balancer.calculate_class_weights(strategy='balanced')
    
    属性:
    -----
    class_counts : dict
        各类别样本数量 {0: 25810, 1: 2443, ...}
    total_samples : int
        总样本数
    """
    
    def __init__(self, class_counts: Dict[int, int], random_seed: int = None):
        """
        初始化类别平衡器
        
        参数:
        -----
        class_counts : dict
            各类别样本数量，如 {0: 25810, 1: 2443, 2: 5292, 3: 873, 4: 708}
        random_seed : int, optional
            随机种子，默认使用配置文件中的 RANDOM_SEED
        """
        self.class_counts = dict(sorted(class_counts.items()))
        self.total_samples = sum(class_counts.values())
        self.num_classes = len(class_counts)
        self.random_seed = random_seed if random_seed is not None else RANDOM_SEED
        
        # 设置随机种子
        np.random.seed(self.random_seed)
        
        # 打印初始分布
        self._print_distribution("原始类别分布", self.class_counts)
    
    def _print_distribution(self, title: str, counts: Dict[int, int]) -> None:
        """打印类别分布"""
        total = sum(counts.values())
        print(f"\n{title}:")
        print("-" * 50)
        for label, count in sorted(counts.items()):
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  类别 {label} ({CLASS_NAMES_CN.get(label, 'Unknown'):4s}): {count:6,} ({pct:5.1f}%) {bar}")
        print(f"  {'总计':14s}: {total:6,}")
    
    def random_oversample(
        self,
        df: pd.DataFrame,
        strategy: Literal['median', 'max', 'custom'] = 'median',
        target_count: int = None
    ) -> pd.DataFrame:
        """
        随机过采样
        
        通过随机复制少数类样本来平衡数据集。
        
        参数:
        -----
        df : pd.DataFrame
            原始数据 DataFrame，必须包含 'label' 列
        strategy : str
            过采样策略:
            - 'median': 将所有类别采样到中位数数量
            - 'max': 将所有类别采样到最大类数量
            - 'custom': 使用自定义目标数量
        target_count : int, optional
            当 strategy='custom' 时，指定目标样本数
        
        返回:
        -----
        pd.DataFrame
            过采样后的 DataFrame
        """
        print("\n" + "=" * 60)
        print("【随机过采样】")
        print("=" * 60)
        print(f"策略: {strategy}")
        
        # 确定目标数量
        counts = list(self.class_counts.values())
        
        if strategy == 'median':
            target = int(np.median(counts))
            print(f"目标数量: 中位数 = {target:,}")
        elif strategy == 'max':
            target = max(counts)
            print(f"目标数量: 最大值 = {target:,}")
        elif strategy == 'custom':
            if target_count is None:
                raise ValueError("strategy='custom' 时必须指定 target_count")
            target = target_count
            print(f"目标数量: 自定义 = {target:,}")
        else:
            raise ValueError(f"未知策略: {strategy}")
        
        # 执行过采样
        oversampled_dfs = []
        new_counts = {}
        
        for label in sorted(self.class_counts.keys()):
            class_df = df[df['label'] == label].copy()
            current_count = len(class_df)
            
            if current_count < target:
                # 需要过采样
                samples_needed = target - current_count
                
                # 随机选择要复制的样本索引
                oversample_indices = np.random.choice(
                    class_df.index,
                    size=samples_needed,
                    replace=True
                )
                
                # 复制样本
                oversampled = df.loc[oversample_indices].copy()
                
                # 合并原始样本和过采样样本
                class_df = pd.concat([class_df, oversampled], ignore_index=True)
                
                print(f"  类别 {label}: {current_count:,} -> {len(class_df):,} (+{samples_needed:,})")
            else:
                print(f"  类别 {label}: {current_count:,} (无需过采样)")
            
            oversampled_dfs.append(class_df)
            new_counts[label] = len(class_df)
        
        # 合并所有类别
        result_df = pd.concat(oversampled_dfs, ignore_index=True)
        
        # 打乱顺序
        result_df = result_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # 打印过采样后的分布
        self._print_distribution("过采样后类别分布", new_counts)
        
        return result_df
    
    def calculate_class_weights(
        self,
        strategy: Literal['balanced', 'sqrt_balanced', 'custom'] = 'balanced',
        custom_weights: Dict[int, float] = None
    ) -> Dict[int, float]:
        """
        计算类别权重
        
        用于在损失函数中给少数类更高的权重，使模型更关注少数类。
        
        参数:
        -----
        strategy : str
            权重计算策略:
            - 'balanced': 标准平衡权重，weight = total / (n_classes * count)
            - 'sqrt_balanced': 平方根平衡，更温和的权重
            - 'custom': 使用自定义权重
        custom_weights : dict, optional
            当 strategy='custom' 时，指定自定义权重
        
        返回:
        -----
        dict
            类别权重字典 {0: weight0, 1: weight1, ...}
        
        公式说明:
        ---------
        balanced:
            weight[i] = total_samples / (num_classes * class_count[i])
            
        sqrt_balanced:
            weight[i] = sqrt(total_samples / (num_classes * class_count[i]))
        """
        print("\n" + "=" * 60)
        print("【计算类别权重】")
        print("=" * 60)
        print(f"策略: {strategy}")
        
        weights = {}
        
        if strategy == 'balanced':
            # 标准平衡权重
            for label, count in self.class_counts.items():
                weights[label] = self.total_samples / (self.num_classes * count)
        
        elif strategy == 'sqrt_balanced':
            # 平方根平衡权重（更温和）
            for label, count in self.class_counts.items():
                weights[label] = np.sqrt(self.total_samples / (self.num_classes * count))
        
        elif strategy == 'custom':
            if custom_weights is None:
                raise ValueError("strategy='custom' 时必须指定 custom_weights")
            weights = custom_weights.copy()
        
        else:
            raise ValueError(f"未知策略: {strategy}")
        
        # 打印权重
        print("\n类别权重:")
        print("-" * 50)
        for label, weight in sorted(weights.items()):
            count = self.class_counts[label]
            effective = weight * count
            print(f"  类别 {label} ({CLASS_NAMES_CN.get(label, 'Unknown'):4s}): "
                  f"权重 = {weight:.4f}, 样本数 = {count:,}, 有效样本 = {effective:.1f}")
        
        # 验证：权重 * 样本数 应该近似相等
        effective_samples = [weights[l] * self.class_counts[l] for l in weights]
        print(f"\n验证: 有效样本数范围 [{min(effective_samples):.1f}, {max(effective_samples):.1f}]")
        
        return weights
    
    def save_oversampled_data(self, df: pd.DataFrame, filepath: str = None) -> None:
        """
        保存过采样后的数据到 CSV 文件
        
        参数:
        -----
        df : pd.DataFrame
            过采样后的 DataFrame
        filepath : str, optional
            保存路径，默认使用配置文件中的 OVERSAMPLED_TRAIN_CSV
        """
        filepath = filepath or OVERSAMPLED_TRAIN_CSV
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"\n已保存过采样数据: {filepath}")
    
    def save_weights(self, weights: Dict[int, float], filepath: str = None) -> None:
        """
        保存类别权重到 JSON 文件
        
        参数:
        -----
        weights : dict
            类别权重字典
        filepath : str, optional
            保存路径，默认使用配置文件中的 CLASS_WEIGHTS_FILE
        """
        filepath = filepath or CLASS_WEIGHTS_FILE
        
        # 转换键为字符串（JSON 要求）
        weights_str_keys = {str(k): v for k, v in weights.items()}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(weights_str_keys, f, indent=2, ensure_ascii=False)
        
        print(f"\n已保存类别权重: {filepath}")
    
    @staticmethod
    def load_weights(filepath: str = None) -> Dict[int, float]:
        """
        从 JSON 文件加载类别权重
        
        参数:
        -----
        filepath : str, optional
            文件路径，默认使用配置文件中的 CLASS_WEIGHTS_FILE
        
        返回:
        -----
        dict
            类别权重字典
        """
        filepath = filepath or CLASS_WEIGHTS_FILE
        
        with open(filepath, 'r', encoding='utf-8') as f:
            weights_str_keys = json.load(f)
        
        # 转换键回整数
        return {int(k): v for k, v in weights_str_keys.items()}


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    from config import TRAIN_CSV_FILE
    
    # 加载训练集
    print("加载训练集...")
    train_df = pd.read_csv(TRAIN_CSV_FILE)
    
    # 获取类别分布
    class_counts = train_df['label'].value_counts().to_dict()
    
    # 创建平衡器
    balancer = ClassBalancer(class_counts)
    
    # 测试过采样
    print("\n" + "=" * 60)
    print("测试过采样 (median 策略)")
    print("=" * 60)
    balanced_df = balancer.random_oversample(train_df, strategy='median')
    balancer.save_oversampled_data(balanced_df)
    
    # 测试类别权重计算
    print("\n" + "=" * 60)
    print("测试类别权重计算")
    print("=" * 60)
    
    print("\n--- balanced 策略 ---")
    weights_balanced = balancer.calculate_class_weights(strategy='balanced')
    
    print("\n--- sqrt_balanced 策略 ---")
    weights_sqrt = balancer.calculate_class_weights(strategy='sqrt_balanced')
    
    # 保存权重
    balancer.save_weights(weights_balanced)
    
    print("\n" + "=" * 60)
    print("类别平衡处理完成!")
    print("=" * 60)

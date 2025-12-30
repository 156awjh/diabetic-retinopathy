"""
投票策略
========

提供集成学习的投票策略实现。
"""

import numpy as np
from typing import List, Dict


def hard_voting(predictions: List[np.ndarray]) -> np.ndarray:
    """
    硬投票（多数表决）
    
    参数:
    -----
    predictions : list
        各模型的预测类别列表，每个元素形状为 (n_samples,)
    
    返回:
    -----
    np.ndarray
        最终预测类别，形状为 (n_samples,)
    """
    # 转换为数组 (n_models, n_samples)
    all_preds = np.array(predictions)
    
    # 对每个样本进行多数表决
    n_samples = all_preds.shape[1]
    final_preds = np.zeros(n_samples, dtype=np.int32)
    
    for i in range(n_samples):
        votes = all_preds[:, i]
        # 统计每个类别的票数，选择票数最多的
        final_preds[i] = np.bincount(votes).argmax()
    
    return final_preds


def soft_voting(probabilities: List[np.ndarray]) -> np.ndarray:
    """
    软投票（概率平均）
    
    参数:
    -----
    probabilities : list
        各模型的预测概率列表，每个元素形状为 (n_samples, n_classes)
    
    返回:
    -----
    np.ndarray
        最终预测类别，形状为 (n_samples,)
    """
    # 计算平均概率
    avg_proba = np.mean(probabilities, axis=0)
    
    # 返回概率最大的类别
    return np.argmax(avg_proba, axis=1)


def weighted_voting(probabilities: List[np.ndarray], 
                    weights: List[float]) -> np.ndarray:
    """
    加权投票
    
    参数:
    -----
    probabilities : list
        各模型的预测概率列表，每个元素形状为 (n_samples, n_classes)
    weights : list
        各模型的权重
    
    返回:
    -----
    np.ndarray
        最终预测类别，形状为 (n_samples,)
    """
    # 归一化权重
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # 计算加权平均概率
    weighted_proba = np.zeros_like(probabilities[0])
    for proba, weight in zip(probabilities, weights):
        weighted_proba += proba * weight
    
    # 返回概率最大的类别
    return np.argmax(weighted_proba, axis=1)


def get_confidence_scores(probabilities: List[np.ndarray],
                          strategy: str = 'soft',
                          weights: List[float] = None) -> np.ndarray:
    """
    获取置信度分数
    
    参数:
    -----
    probabilities : list
        各模型的预测概率列表
    strategy : str
        投票策略 ('soft' 或 'weighted')
    weights : list
        权重（仅用于 weighted 策略）
    
    返回:
    -----
    np.ndarray
        置信度分数，形状为 (n_samples,)
    """
    if strategy == 'soft':
        avg_proba = np.mean(probabilities, axis=0)
    elif strategy == 'weighted' and weights is not None:
        weights = np.array(weights)
        weights = weights / weights.sum()
        avg_proba = np.zeros_like(probabilities[0])
        for proba, weight in zip(probabilities, weights):
            avg_proba += proba * weight
    else:
        avg_proba = np.mean(probabilities, axis=0)
    
    # 返回最大概率作为置信度
    return np.max(avg_proba, axis=1)

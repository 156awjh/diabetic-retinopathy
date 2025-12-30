"""
集成学习脚本
============

使用方法:
---------
python scripts/run_ensemble.py
python scripts/run_ensemble.py --strategy weighted_voting
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DRDataLoader
from src.ensemble.ensemble_model import EnsembleModel


def main():
    parser = argparse.ArgumentParser(description='运行集成学习')
    parser.add_argument('--models_dir', type=str, default='trained_models',
                        help='模型目录 (默认: trained_models)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小 (默认: 32)')
    parser.add_argument('--strategy', type=str, default='all',
                        choices=['hard_voting', 'soft_voting', 'weighted_voting', 'all'],
                        help='投票策略 (默认: all)')
    parser.add_argument('--output', type=str, default='output/evaluation/ensemble_comparison.json',
                        help='输出报告路径')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"集成学习配置")
    print(f"{'='*60}")
    print(f"模型目录: {args.models_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"投票策略: {args.strategy}")
    print(f"输出路径: {args.output}")
    print(f"{'='*60}\n")
    
    # 加载测试数据
    print("加载测试数据...")
    loader = DRDataLoader(batch_size=args.batch_size)
    test_data = loader.load_test_data()
    
    # 获取真实标签
    print("获取真实标签...")
    y_true_list = []
    for _, labels in test_data:
        y_true_list.append(np.argmax(labels.numpy(), axis=1))
    y_true = np.concatenate(y_true_list)
    
    # 重新加载测试数据（因为迭代器已耗尽）
    test_data = loader.load_test_data()
    
    # 加载集成模型
    ensemble = EnsembleModel()
    ensemble.load_models(args.models_dir)
    
    # 根据准确率设置权重
    ensemble.set_weights_from_accuracy()
    
    # 生成对比报告
    results = ensemble.generate_comparison_report(
        test_data,
        y_true,
        args.output
    )
    
    print("\n集成学习完成！")
    print(f"最佳策略: {results['best_strategy']}")
    print(f"最佳准确率: {results['best_accuracy']:.4f}")
    print(f"相比最佳单模型提升: {results['improvement']:+.4f}")


if __name__ == '__main__':
    main()

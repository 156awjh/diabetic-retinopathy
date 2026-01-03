"""
模型评估脚本
============

使用方法:
---------
python scripts/evaluate_model.py --model resnet50
python scripts/evaluate_model.py --model vgg16 --batch_size 16
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DRDataLoader
from src.evaluation.evaluator import ModelEvaluator
from src.models.base_model import BaseModel
from tensorflow import keras


class LoadedModel(BaseModel):
    """用于加载已训练模型的包装类"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
    
    def build(self):
        # 不需要构建，直接加载
        pass


def main():
    parser = argparse.ArgumentParser(description='评估模型')
    parser.add_argument('--model', type=str, required=True,
                        help='模型名称')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小 (默认: 32)')
    parser.add_argument('--models_dir', type=str, default='trained_models',
                        help='模型目录 (默认: trained_models)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"评估配置")
    print(f"{'='*60}")
    print(f"模型: {args.model}")
    print(f"批次大小: {args.batch_size}")
    print(f"模型目录: {args.models_dir}")
    print(f"{'='*60}\n")
    
    # 加载测试数据
    print("加载测试数据...")
    if(args.model == 'inceptionv3'):
        loader = DRDataLoader(image_size=(299, 299),batch_size=args.batch_size)
    else:
        loader = DRDataLoader(batch_size=args.batch_size)
    test_data = loader.load_test_data()
    
    # 加载模型
    model_path = Path(args.models_dir) / args.model / f"{args.model}_best.keras"
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)
    
    print(f"加载模型: {model_path}")
    model = LoadedModel(args.model)
    model.load(str(model_path))
    
    # 评估
    evaluator = ModelEvaluator(model, test_data)
    report = evaluator.evaluate()
    
    # 保存报告
    save_dir = Path(args.models_dir) / args.model
    evaluator.save_report(str(save_dir))
    
    print("\n评估完成！")


if __name__ == '__main__':
    main()

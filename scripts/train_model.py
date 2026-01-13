"""
模型训练脚本
============

使用方法:
---------
python scripts/train_model.py --model resnet50
python scripts/train_model.py --model vgg16 --epochs 30 --batch_size 16
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DRDataLoader
from src.training.trainer import ModelTrainer


def get_model(model_name: str):
    """
    根据名称获取模型实例
    
    参数:
    -----
    model_name : str
        模型名称
    
    返回:
    -----
    BaseModel
        模型实例
    """
    # 动态导入模型
    if model_name == 'resnet50':
        from src.models.resnet50 import ResNet50Model
        return ResNet50Model()
    elif model_name == 'efficientnet_b0':
        from src.models.efficientnet_b0 import EfficientNetB0Model
        return EfficientNetB0Model()
    elif model_name == 'vgg16':
        from src.models.vgg16 import VGG16Model
        return VGG16Model()
    elif model_name == 'mobilenetv2':
        from src.models.mobilenetv2 import MobileNetV2Model
        return MobileNetV2Model()
    elif model_name == 'se_resnet':
        from src.models.se_resnet import SEResNetModel
        return SEResNetModel()
    elif model_name == 'resnext50':
        from src.models.resnext50 import ResNeXt50Model
        return ResNeXt50Model()
    elif model_name == 'densenet121':
        from src.models.densenet121 import DenseNet121Model
        return DenseNet121Model()
    elif model_name == 'inceptionv3':
        from src.models.inceptionv3 import InceptionV3Model
        return InceptionV3Model()
    else:
        raise ValueError(f"未知模型: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='训练模型')
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet50', 'efficientnet_b0', 'vgg16', 
                                'mobilenetv2', 'se_resnet', 'resnext50',
                                'densenet121', 'inceptionv3'],
                        help='模型名称')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数 (默认: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小 (默认: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率 (默认: 0.001)')
    parser.add_argument('--save_dir', type=str, default='trained_models',
                        help='模型保存目录 (默认: trained_models)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"训练配置")
    print(f"{'='*60}")
    print(f"模型: {args.model}")
    print(f"轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"保存目录: {args.save_dir}")
    print(f"{'='*60}\n")
    
    # 加载数据
    print("加载数据...")
    loader = DRDataLoader(batch_size=args.batch_size)
    train_data = loader.load_train_data(augment=True)
    val_data = loader.load_val_data()
    
    # 获取类别权重
    class_weights = loader.get_class_weights()
    print(f"类别权重: {class_weights}")
    
    # 创建模型
    print(f"\n创建模型: {args.model}")
    model = get_model(args.model)
    
    # 训练
    trainer = ModelTrainer(model, save_dir=args.save_dir)
    trainer.train(
        train_data,
        val_data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        class_weight=class_weights
    )
    # 在 trainer.train() 之后添加
    from sklearn.metrics import classification_report
    import numpy as np

    val_gen.reset()
    y_pred_prob = model.model.predict(val_gen, steps=np.ceil(val_gen.n / val_gen.batch_size))
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = val_gen.labels[:len(y_pred)]

    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    print("=" * 50)
    print("每类别详细分析")
    print("=" * 50)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    for name in class_names:
        m = report[name]
        print(f"\n{name}:")
        print(f"  精确率: {m['precision']:.4f}")
        print(f"  召回率: {m['recall']:.4f}")
        print(f"  F1: {m['f1-score']:.4f}")
        print(f"  样本数: {int(m['support'])}")
        print("\n训练完成！")


if __name__ == '__main__':
    main()

import numpy as np
import cv2  # 冗余导入，可删除
from PIL import Image
import os, random, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import torchmetrics
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.utils.data import Subset

# ===================== 核心修复：所有逻辑包裹在main函数中 =====================
def main():
    torch.cuda.empty_cache()
    # 修复路径字符串终止问题（Windows路径）
    path_train = r'D:\ml\data\data2'  # 移除末尾反斜杠，避免转义错误

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 输出目录（适配Windows路径）
    output_dir = r'D:\ml\src\models\VIT'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 随机种子设置
    def set_random_seed(seed: int) -> None:
        print(f"Setting seeds: {seed} ...... ")
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    set_random_seed(123)

    # 平衡类别权重函数
    def make_weights_for_balanced_classes(labels):
        count = torch.bincount(labels).to(device)
        print('Count:', count.cpu().detach().numpy())

        weight = 1. / count.cpu().detach().numpy()
        print('Data sampling weight:', weight)
        samples_weight = np.array([weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    # 超参数配置
    batch_size = 32  # 若显存不足可改为32/64
    IMAGE_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # 数据增强（训练集）和预处理（测试集）
    train_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 3.3)),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # 加载数据集
    all_dataset = ImageFolder(path_train, transform=train_transform)
    dataset_test = ImageFolder(path_train, transform=test_transform)
    targets = all_dataset.targets

    # 划分训练/测试集
    train_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        shuffle=True,
        stratify=targets)
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    train_dataset = Subset(all_dataset, train_idx)
    val_dataset = Subset(dataset_test, test_idx)
    test_dataset = Subset(dataset_test, test_idx)

    # 打印类别分布
    labels = torch.tensor(all_dataset.targets)[train_dataset.indices]
    count = torch.bincount(labels).to(device)
    print('Train Count:', count.cpu().detach().numpy())

    labels = torch.tensor(all_dataset.targets)[val_dataset.indices]
    count = torch.bincount(labels).to(device)
    print('Validation Count:', count.cpu().detach().numpy())

    labels = torch.tensor(all_dataset.targets)[test_dataset.indices]
    count = torch.bincount(labels).to(device)
    print('Test Count:', count.cpu().detach().numpy())

    # 类别映射
    label_mapping = {
        0: "healthy",
        1: "mild npdr",
        2: "moderate npdr",
        3: "severe npdr",
        4: "pdr"
    }

    # 图像反归一化函数
    def inverse_normalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    # 图像可视化函数
    def imshow(img):
        img = img / 2 + 0.5
        inverse_normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # 临时加载一批数据可视化（可选）
    # 修复：Windows下先使用num_workers=0加载可视化数据
    train_loader_temp = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    try:
        images, labels = next(iter(train_loader_temp))
        imshow(torchvision.utils.make_grid(images))
    except:
        print("Skipping image visualization (optional)")

    # 生成平衡采样权重
    weights = make_weights_for_balanced_classes(torch.tensor(all_dataset.targets)[train_dataset.indices])
    weighted_sampler = sampler.WeightedRandomSampler(weights, len(weights))

    # ===================== 关键修复：Windows下num_workers设为0 =====================
    # Windows系统多进程易出错，num_workers=0 避免进程冲突
    num_workers = 0 if os.name == 'nt' else 4

    # 构建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,  # Windows下设为0
        sampler=weighted_sampler
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    # 初始化训练参数
    best_test_acc = 0
    best_epoch = 0
    num_classes = 5
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    logs = ''

    # 加载ViT模型
    print("Model: ViT")
    model = models.vit_b_16(weights='IMAGENET1K_V1').to(device)
    # 替换分类头
    model.heads = nn.Sequential(
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 5)
    )
    model = model.to(device)
    print(model)

    # 计算可训练参数
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of trainable parameters in the model: {num_params}")

    # 类别权重（平衡损失）
    labels = torch.tensor(all_dataset.targets)[train_dataset.indices]
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()  # 归一化
    class_weights = class_weights.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode='min', verbose=True)

    # 评估指标
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='weighted').to(device)
    confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='true').to(device)
    class_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average=None).to(device)

    # 训练循环
    num_epochs = 30
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        t = tqdm(enumerate(train_loader, 0), total=len(train_loader),
                 smoothing=0.9, position=0, leave=True,
                 desc=f"Train: Epoch: {epoch+1}/{num_epochs}")
        
        for i, (inputs, labels) in t:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            outputs = F.softmax(outputs, dim=-1)
            train_accuracy = accuracy(outputs, labels)

        # 记录训练损失和精度
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = accuracy.compute()
        train_accuracies.append(float(train_accuracy))
        accuracy.reset()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            t = tqdm(enumerate(val_loader, 0), total=len(val_loader),
                     smoothing=0.9, position=0, leave=True,
                     desc=f"Val: Epoch: {epoch+1}/{num_epochs}")
            for i, (inputs, labels) in t:
                inputs = inputs.to(device).float()
                labels = labels.to(device).long()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                outputs = F.softmax(outputs, dim=-1)
                val_accuracy = accuracy(outputs, labels)
                confmat.update(outputs, labels)
                val_class_accuracy = class_accuracy(outputs, labels)

        # 记录验证损失和精度
        val_class_accuracy = class_accuracy.compute()
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = accuracy.compute()
        val_accuracies.append(float(val_accuracy))
        test_loss = val_loss
        test_accuracy = val_accuracy

        # 更新学习率
        scheduler.step(val_loss)
        lr_log = f"LR: {optimizer.param_groups[0]['lr']}"
        print(lr_log)
        logs += lr_log + '\n'

        # 打印训练结果
        train_results = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        print(train_results)
        logs += train_results + '\n'

        # 保存最后一轮模型
        torch.save(model.state_dict(), os.path.join(output_dir, 'last.pth'))

        # 每5轮保存一次模型并绘制混淆矩阵
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f'epoch{epoch+1}.pth'))
            # 绘制混淆矩阵
            fig, ax = plt.subplots(figsize=(8, 6))
            confmat_vals = np.around(confmat.compute().cpu().detach().numpy(), 3)
            im = ax.imshow(confmat_vals, cmap='Blues')
            ax.set_xticks(np.arange(num_classes))
            ax.set_yticks(np.arange(num_classes))
            ax.set_xticklabels([label_mapping[i] for i in range(num_classes)], rotation=45, ha='right')
            ax.set_yticklabels([label_mapping[i] for i in range(num_classes)])
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('True class')
            # 标注数值
            for i in range(num_classes):
                for j in range(num_classes):
                    text = ax.text(j, i, confmat_vals[i, j], ha="center", va="center", color="black", fontsize=10)
            ax.set_title(f"Confusion Matrix (Epoch {epoch+1})")
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"conf_mat_epoch{epoch+1}.png"))
            plt.close()

        # 保存最优模型
        if best_test_acc <= test_accuracy:
            best_epoch = epoch+1
            log = f"Improve accuracy from {best_test_acc:.4f} to {test_accuracy:.4f}"
            print(log)
            logs += log + "\n"
            best_test_acc = test_accuracy
            torch.save(model.state_dict(), os.path.join(output_dir, 'best.pth'))
            # 绘制最优模型混淆矩阵
            fig, ax = plt.subplots(figsize=(8, 6))
            confmat_vals = np.around(confmat.compute().cpu().detach().numpy(), 3)
            im = ax.imshow(confmat_vals, cmap='Blues')
            ax.set_xticks(np.arange(num_classes))
            ax.set_yticks(np.arange(num_classes))
            ax.set_xticklabels([label_mapping[i] for i in range(num_classes)], rotation=45, ha='right')
            ax.set_yticklabels([label_mapping[i] for i in range(num_classes)])
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('True class')
            for i in range(num_classes):
                for j in range(num_classes):
                    text = ax.text(j, i, confmat_vals[i, j], ha="center", va="center", color="black", fontsize=10)
            ax.set_title("Confusion Matrix (Best Model)")
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "conf_mat_best.png"))
            plt.close()

        # 重置评估指标
        accuracy.reset()
        class_accuracy.reset()
        confmat.reset()

    # 保存日志
    with open(os.path.join(output_dir, 'log.txt'), 'w') as log_file:
        log_file.write(logs)
        log_file.write(f'\nBest val accuracy: {best_test_acc:.4f} in epoch {best_epoch}')

    # 绘制损失和精度曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train and Validation Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_accuracy_graph.png'))
    plt.close()

    # 保存模型结构和权重
    torch.save(model, os.path.join(output_dir, 'model_architecture.pth'))
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.pth'))

    # 测试阶段
    model.eval()
    test_loss = 0
    test_acc = 0
    accuracy.reset()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            test_loss += loss.item()
            test_acc += accuracy(pred, y)

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f'Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}')

    # 单批次测试（打印logits和预测结果）
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(test_loader))
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        print("Logits (raw outputs):", outputs[:5])  # 只打印前5个样本
        print("Predicted classes:", outputs.argmax(-1)[:5])
        print("Actual classes:", labels[:5])

    print("Model training and testing complete!")

# ===================== 核心修复：Windows多进程保护 =====================
if __name__ == '__main__':
    # Windows下多进程必需的freeze_support（可选，不加也能运行）
    try:
        from torch.multiprocessing import freeze_support
        freeze_support()
    except:
        pass
    main()

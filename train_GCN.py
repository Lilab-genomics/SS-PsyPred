import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from functools import partial
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, recall_score, precision_score, f1_score
)
import warnings

warnings.filterwarnings('ignore')

# 添加 torch_geometric 的全局池化函数导入
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

# 导入您的现有模块
from datasets.func import FuncDataset, collate_fn_struct_only
from models.build_models import build_model


def extract_mutation_window_features(features, batch, mut_positions, window_size=11):
    """
    提取突变位点周围的窗口特征

    Args:
        features: [N, 128] 每个残基的特征
        batch: [N] 每个残基所属的蛋白质batch索引
        mut_positions: list of int 每个蛋白质的突变位置索引（在原始序列中的位置）
        window_size: 窗口大小（左右各window_size//2个残基）

    Returns:
        window_mean: [B, 128] 窗口内特征的均值
        window_max: [B, 128] 窗口内特征的最大值
    """
    device = features.device
    batch_size = batch.max().item() + 1

    # 初始化结果张量
    window_mean = torch.zeros(batch_size, features.shape[1], device=device)
    window_max = torch.zeros(batch_size, features.shape[1], device=device)

    # 为每个蛋白质提取窗口特征
    for b in range(batch_size):
        # 获取当前蛋白质的所有残基索引
        mask = (batch == b)
        protein_features = features[mask]  # [L, 128]

        # 获取当前蛋白质的突变位置
        if b < len(mut_positions):
            mut_pos = mut_positions[b]
        else:
            mut_pos = -1

        # 如果突变位置无效或超出范围，使用整个蛋白质的均值和最大值
        if mut_pos < 0 or mut_pos >= protein_features.shape[0]:
            window_mean[b] = protein_features.mean(dim=0)
            window_max[b] = protein_features.max(dim=0)[0]  # 修复：只取最大值，不要索引
            continue

        # 计算窗口边界
        half_window = window_size // 2
        start = max(0, mut_pos - half_window)
        end = min(protein_features.shape[0], mut_pos + half_window + 1)

        # 提取窗口内特征
        window_features = protein_features[start:end]  # [window_len, 128]

        # 计算均值和最大值
        window_mean[b] = window_features.mean(dim=0)
        window_max[b] = window_features.max(dim=0)[0]  # 修复：只取最大值，不要索引

    return window_mean, window_max

def compute_metrics(y_true, y_pred, y_score=None):
    """
    计算评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签（二值化）
        y_score: 预测概率分数

    Returns:
        metrics_dict: 包含各项指标的字典
    """
    metrics = {}

    # 准确率
    metrics['acc'] = accuracy_score(y_true, y_pred)

    # 召回率
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)

    # 精确率
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)

    # F1分数
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # AUC (需要概率分数)
    if y_score is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_score)
        except:
            metrics['auc'] = 0.5

        # AUPR
        try:
            metrics['aupr'] = average_precision_score(y_true, y_score)
        except:
            metrics['aupr'] = 0.0

    return metrics


def train_epoch(model, loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []

    pbar = tqdm(loader, desc="Training")
    for batch_idx, data in enumerate(pbar):
        # 移动数据到设备
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)

        # 前向传播获取残基特征
        features = model(data)  # [N, 128]
        batch = data['batch']

        # 1. 全局平均池化
        global_pooled = global_mean_pool(features, batch)  # [B, 128]

        # 2. 突变位点窗口池化
        mut_positions = data.get('mut_position_npz', [-1] * (batch.max().item() + 1))
        if isinstance(mut_positions, list):
            pass  # 已经是列表
        else:
            mut_positions = [mut_positions]  # 转换为列表

        window_mean, window_max = extract_mutation_window_features(
            features, batch, mut_positions, window_size=11
        )

        # 3. 拼接三类特征: 全局平均(128) + 窗口均值(128) + 窗口最大值(128) = 384
        pooled_features = torch.cat([global_pooled, window_mean, window_max], dim=1)  # [B, 384]

        # 获取特征维度
        feature_dim = pooled_features.shape[1]

        # 根据实际维度创建或调整分类头
        if not hasattr(model, 'classifier') or model.classifier[0].in_features != feature_dim:
            model.classifier = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            ).to(device)
            if batch_idx == 0:
                print(f"创建分类头，输入维度: {feature_dim}")

        logits = model.classifier(pooled_features).squeeze(-1)

        # 计算损失
        labels = data['label'].squeeze(-1).float().to(device)
        loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_scores.extend(probs.cpu().detach().numpy())

        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})

    # 计算指标
    metrics = compute_metrics(all_labels, all_preds, all_scores)
    metrics['loss'] = total_loss / len(loader)

    return metrics


def evaluate(model, loader, criterion, device, return_predictions=False):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []
    all_protein_ids = []

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating"):
            # 移动数据到设备
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)

            # 前向传播获取残基特征
            features = model(data)
            batch = data['batch']

            # 1. 全局平均池化
            global_pooled = global_mean_pool(features, batch)  # [B, 128]

            # 2. 突变位点窗口池化
            mut_positions = data.get('mut_position_npz', [-1] * (batch.max().item() + 1))
            if isinstance(mut_positions, list):
                pass  # 已经是列表
            else:
                mut_positions = [mut_positions]

            window_mean, window_max = extract_mutation_window_features(
                features, batch, mut_positions, window_size=11
            )

            # 3. 拼接三类特征
            pooled_features = torch.cat([global_pooled, window_mean, window_max], dim=1)  # [B, 384]

            # 分类
            if hasattr(model, 'classifier'):
                logits = model.classifier(pooled_features).squeeze(-1)
            else:
                # 如果没有分类头，创建临时分类头
                temp_classifier = nn.Sequential(
                    nn.Linear(384, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 1)
                ).to(device)
                logits = temp_classifier(pooled_features).squeeze(-1)

            # 计算损失
            labels = data['label'].squeeze(-1).float().to(device)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 预测
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(probs.cpu().detach().numpy())

            # 收集蛋白质ID
            if 'uniprot_id' in data:
                protein_ids = data['uniprot_id']
                if isinstance(protein_ids, list):
                    all_protein_ids.extend(protein_ids)
                else:
                    all_protein_ids.append(protein_ids)

    # 计算指标
    metrics = compute_metrics(all_labels, all_preds, all_scores)
    metrics['loss'] = total_loss / len(loader)

    if return_predictions:
        return metrics, all_labels, all_preds, all_scores, all_protein_ids

    return metrics


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Protein Structure Feature Extraction and Training')

    # 运行模式
    parser.add_argument('--mode', type=str, default='train', choices=['extract', 'train'],
                        help='运行模式: extract-只提取特征, train-训练模型')

    # 基础配置
    parser.add_argument('--model', default='f3s', type=str)
    parser.add_argument('--dataset', default='func', type=str)
    parser.add_argument('--structure', default=True, type=bool)
    parser.add_argument('--surface', default=False, type=bool)
    parser.add_argument('--sequence', default=False, type=bool)
    parser.add_argument('--acc_iter', default=1, type=int, metavar='N', help='number of grad acc iter')

    # 数据路径
    parser.add_argument('--data_dir', default='/data2/yanmengxiang/haiyun/GCN/datasets', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)

    # 训练配置
    parser.add_argument('--epochs', default=100, type=int, help='最大训练轮数')
    parser.add_argument('--lr', default=1e-3, type=float, help='学习率')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='权重衰减')
    parser.add_argument('--patience', default=20, type=int, help='早停耐心值')
    parser.add_argument('--val_ratio', default=0.2, type=float, help='验证集比例')

    # 输出路径
    parser.add_argument('--output_dir', default='/data2/yanmengxiang/haiyun/GCN/outputs', type=str)
    parser.add_argument('--checkpoint_dir', default='/data2/yanmengxiang/haiyun/GCN/checkpoints', type=str)

    # 结构配置
    parser.add_argument('--geometric-radius', default=4.0, type=float)
    parser.add_argument('--sequential-kernel-size', default=21, type=int)
    parser.add_argument('--kernel-channels', nargs='+', default=[24], type=int)
    parser.add_argument('--base-width', default=64, type=float)
    parser.add_argument('--channels', nargs='+', default=[32, 64, 128], type=int)

    # 极性特征配置
    parser.add_argument('--use-polarity', action='store_true', default=True,
                        help='是否使用极性特征')

    # 多尺度特征融合
    parser.add_argument('--use-multiscale', action='store_true', default=False,
                        help='是否使用多尺度特征融合')

    # 卷积类型参数
    parser.add_argument('--conv-type', default='gnn', type=str,
                        choices=['gnn'], help='卷积类型: 只支持GNN')
    parser.add_argument('--gnn-heads', default=4, type=int, help='GNN注意力头数')
    parser.add_argument('--gnn-dropout', default=0.1, type=float, help='GNN dropout率')

    args = parser.parse_args()

    # 根据conv-type设置其他参数
    args.kernel_channels = [32, 64]

    return args


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, is_best=False):
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    # 保存最新检查点
    latest_path = osp.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)

    # 保存最佳模型
    if is_best:
        best_path = osp.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✅ 保存最佳模型到: {best_path}")


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """加载检查点"""
    checkpoint_path = osp.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 获取模型状态字典
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint['model_state_dict']

        # 只加载匹配的权重
        matched_keys = []
        unmatched_keys = []

        for key, value in checkpoint_state_dict.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                matched_keys.append(key)
                model_state_dict[key] = value
            else:
                unmatched_keys.append(key)

        # 加载匹配的权重
        model.load_state_dict(model_state_dict, strict=False)

        if matched_keys:
            print(f"✅ 加载了 {len(matched_keys)} 个匹配的层")
        if unmatched_keys:
            print(f"⚠️ 跳过了 {len(unmatched_keys)} 个不匹配的层")

        # 加载优化器状态
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ 加载优化器状态")
        except:
            print(f"⚠️ 优化器状态不匹配，使用新的优化器")

        start_epoch = checkpoint['epoch'] + 1
        best_val_auc = checkpoint['metrics'].get('val_auc', 0)
        print(f"✅ 加载检查点: epoch {checkpoint['epoch']}, val_auc: {best_val_auc:.4f}")
        return start_epoch, best_val_auc
    return 0, 0


def main():
    args = parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"运行模式: {args.mode}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 加载数据集
    print("\n" + "=" * 60)
    print("加载数据集...")
    print("=" * 60)
    full_dataset = FuncDataset(args, random_seed=args.seed, split='training', rotation=True)

    # 分割训练集和验证集
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # 测试集
    test_dataset = FuncDataset(args, random_seed=args.seed, split='testing', rotation=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=partial(collate_fn_struct_only, config=args),
        drop_last=False
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=partial(collate_fn_struct_only, config=args),
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=partial(collate_fn_struct_only, config=args),
        drop_last=False
    )

    # 构建模型
    print("\n" + "=" * 60)
    print("构建模型...")
    print("=" * 60)
    model = build_model(args).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"池化方式: 全局平均池化 + 突变位点窗口均值池化 + 突变位点窗口最大值池化")
    print(f"窗口大小: 11 (左右各5个残基)")
    print(f"最终特征维度: 384 (128 + 128 + 128)")

    # 如果只是提取特征
    if args.mode == 'extract':
        print("\n" + "=" * 60)
        print("特征提取模式...")
        print("=" * 60)
        print("注意: 特征提取模式只提取残基级别特征，不进行池化")

        # 创建保存目录
        train_output_dir = osp.join(args.output_dir, 'train_struct')
        test_output_dir = osp.join(args.output_dir, 'test_struct')
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        # 提取训练集特征
        train_features, train_protein_ids = extract_struct_features(
            train_loader, model, device, full_dataset, 'train'
        )

        # 保存特征
        def save_features(features, protein_ids, output_dir, split):
            os.makedirs(output_dir, exist_ok=True)
            npz_filename = f"{split}_features.npz"
            output_path = osp.join(output_dir, npz_filename)
            save_dict = {}
            for feature, protein_id in zip(features, protein_ids):
                save_dict[protein_id] = feature
            np.savez(output_path, **save_dict)
            print(f"保存 {split} 特征到 {output_path}")

        save_features(train_features, train_protein_ids, train_output_dir, 'train')

        # 提取测试集特征
        test_features, test_protein_ids = extract_struct_features(
            test_loader, model, device, test_dataset, 'test'
        )
        save_features(test_features, test_protein_ids, test_output_dir, 'test')

        print("\n特征提取完成！")
        return

    # 训练模式
    print("\n" + "=" * 60)
    print("训练模式...")
    print("=" * 60)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    criterion = nn.BCEWithLogitsLoss()

    # 尝试加载检查点
    start_epoch, best_val_auc = load_checkpoint(model, optimizer, args.checkpoint_dir, device)

    # 早停变量
    patience_counter = 0
    best_epoch = 0

    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_auc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_aupr': [],
        'val_f1': []
    }

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    print(f"初始学习率: {args.lr}")
    print(f"最大epochs: {args.epochs}")
    print(f"早停耐心值: {args.patience}")
    print(f"验证集比例: {args.val_ratio}")
    print("=" * 60 + "\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)

        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['train_auc'].append(train_metrics.get('auc', 0))
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_auc'].append(val_metrics.get('auc', 0))
        history['val_aupr'].append(val_metrics.get('aupr', 0))
        history['val_f1'].append(val_metrics['f1'])

        # 打印指标
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['acc']:.4f}, "
              f"AUC: {train_metrics.get('auc', 0):.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['acc']:.4f}, "
              f"AUC: {val_metrics.get('auc', 0):.4f}, "
              f"AUPR: {val_metrics.get('aupr', 0):.4f}, "
              f"F1: {val_metrics['f1']:.4f}")

        # 学习率调度
        current_val_auc = val_metrics.get('auc', 0)
        scheduler.step(current_val_auc)

        # 检查是否为最佳模型
        is_best = current_val_auc > best_val_auc
        if is_best:
            best_val_auc = current_val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"🎉 新的最佳模型! Val AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            print(f"验证集AUC未提升，耐心计数: {patience_counter}/{args.patience}")

        # 保存检查点
        save_checkpoint(
            model, optimizer, epoch + 1,
            {'val_auc': current_val_auc, 'epoch': epoch + 1},
            args.checkpoint_dir, is_best
        )

        # 早停检查
        if patience_counter >= args.patience:
            print(f"\n⚠️ 早停触发! 在 {epoch + 1} 个epoch后停止训练")
            print(f"最佳模型在 epoch {best_epoch}, Val AUC: {best_val_auc:.4f}")
            break

    # 加载最佳模型进行测试
    print("\n" + "=" * 60)
    print("加载最佳模型进行测试...")
    print("=" * 60)

    best_checkpoint_path = osp.join(args.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载最佳模型 (epoch {checkpoint['epoch']})")
    else:
        print("⚠️ 未找到最佳模型，使用最后一个epoch的模型")

    # 测试集评估
    print("\n" + "=" * 60)
    print("测试集评估...")
    print("=" * 60)

    test_metrics, test_labels, test_preds, test_scores, test_ids = evaluate(
        model, test_loader, criterion, device, return_predictions=True
    )

    print(f"\n测试集结果:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['acc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  AUPR: {test_metrics['aupr']:.4f}")

    # 保存测试结果
    results_file = osp.join(args.output_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Val AUC: {best_val_auc:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n\n")
        f.write("Test Metrics:\n")
        f.write(f"  Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"  Accuracy: {test_metrics['acc']:.4f}\n")
        f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"  F1 Score: {test_metrics['f1']:.4f}\n")
        f.write(f"  AUC: {test_metrics['auc']:.4f}\n")
        f.write(f"  AUPR: {test_metrics['aupr']:.4f}\n\n")

        f.write("Individual Predictions:\n")
        f.write("Protein ID\tTrue Label\tPred Label\tScore\n")
        for pid, true, pred, score in zip(test_ids, test_labels, test_preds, test_scores):
            f.write(f"{pid}\t{true}\t{pred}\t{score:.4f}\n")

    print(f"\n✅ 测试结果已保存到: {results_file}")

    # 保存训练历史
    history_file = osp.join(args.output_dir, 'training_history.npy')
    np.save(history_file, history)
    print(f"✅ 训练历史已保存到: {history_file}")

    # 打印最佳验证指标摘要
    print("\n" + "=" * 60)
    print("训练摘要")
    print("=" * 60)
    print(f"最佳验证集 AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"最佳验证集 AUPR: {max(history['val_aupr']):.4f}")
    print(f"最佳验证集 F1: {max(history['val_f1']):.4f}")
    print(f"最终测试集 AUC: {test_metrics['auc']:.4f}")
    print(f"最终测试集 AUPR: {test_metrics['aupr']:.4f}")
    print(f"最终测试集 F1: {test_metrics['f1']:.4f}")


def extract_struct_features(dataloader, model, device, dataset, split):
    """提取结构特征（用于特征提取模式）"""
    model.eval()
    all_features = []
    all_protein_ids = []

    with torch.no_grad():
        protein_names = dataset.protein_names

        for batch_idx, data in tqdm(enumerate(dataloader), desc=f"Extracting {split} features"):
            # 移动数据到设备
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)

            # 前向传播
            features = model(data)

            # 获取蛋白质ID
            if batch_idx < len(protein_names):
                protein_id = protein_names[batch_idx]
            else:
                protein_id = f"protein_{batch_idx}"

            # 保存特征和蛋白质ID
            all_features.append(features.cpu().numpy())
            all_protein_ids.append(protein_id)

            # 每处理50个蛋白质打印一次进度
            if (batch_idx + 1) % 50 == 0:
                print(f"已处理 {batch_idx + 1} 个蛋白质，特征形状: {features.shape}")

    print(f"\n{split} 特征提取完成:")
    print(f"  总共提取了 {len(all_features)} 个蛋白质的特征")
    print(f"  每个蛋白质的特征形状示例: {all_features[0].shape if all_features else '无特征'}")

    return all_features, all_protein_ids


if __name__ == '__main__':
    main()
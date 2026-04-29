import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.nn import global_mean_pool
from datasets.func import FuncDataset, collate_fn_struct_only
from models.build_models import build_model

# ===================== 路径配置 =====================
DATA_DIR = "/datasets"
CHECKPOINT_PATH = "/data2/yanmengxiang/haiyun/GCN/checkpoints/best_model.pth"
OUTPUT_DIR = "/outputs"

# 结构文件路径（用于读取 mut_position）
TRAIN_STRUCT_DIR = "/data2/yanmengxiang/haiyun/GCN/datasets/structure_train"
TEST_STRUCT_DIR  = "/data2/yanmengxiang/haiyun/GCN/datasets/structure_test_544"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ======================================================

# ===================== 384维池化函数（完全和你代码一致） =====================
def pool_with_mut_window(feats, mut_pos, window=11):
    L = feats.shape[0]
    left = max(0, mut_pos - window)
    right = min(L, mut_pos + window + 1)
    window_feats = feats[left:right]

    global_mean = feats.mean(axis=0)
    window_mean = window_feats.mean(axis=0)
    window_max  = window_feats.max(axis=0)

    return np.concatenate(
        [global_mean, window_mean, window_max],
        axis=0
    )

# ===================== 特征提取 + 立即池化 =====================
def extract_and_pool_features(dataloader, model, device, dataset, struct_dir):
    """提取GCN特征 → 立即池化为384维"""
    model.eval()
    feat_dict = {}
    protein_names = dataset.protein_names

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(dataloader), desc="Extracting & pooling"):
            # 数据加载到设备
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)

            # 残基级特征 [L, dim]
            res_feats = model(data).cpu().numpy()

            # 蛋白ID
            protein_id = protein_names[batch_idx]

            # 读取突变位置
            struct_path = osp.join(struct_dir, f"{protein_id}.npz")
            struct_data = np.load(struct_path)
            mut_pos = int(struct_data["mut_position"])

            # 池化 → 384维
            pooled_feat = pool_with_mut_window(res_feats, mut_pos, window=11)
            feat_dict[protein_id] = pooled_feat

    return feat_dict

# ===================== 保存最终384维特征 =====================
def save_final_features(feat_dict, output_path):
    np.savez(output_path, features=feat_dict)
    print(f"✅ 已保存384维池化特征: {output_path}")
    print(f"   样本数量: {len(feat_dict)}")
    print(f"   特征维度: {list(feat_dict.values())[0].shape[0]}\n")

# ===================== 主函数 =====================
def main():
    class Args:
        mode = "extract"
        model = "f3s"
        dataset = "func"
        structure = True
        surface = False
        sequence = False
        data_dir = DATA_DIR
        batch_size = 1
        workers = 1
        seed = 42
        val_ratio = 0.2
        geometric_radius = 4.0
        sequential_kernel_size = 21
        kernel_channels = [32, 64]
        base_width = 64
        channels = [32, 64, 128]
        use_polarity = True
        use_multiscale = False
        conv_type = "gnn"
        gnn_heads = 4
        gnn_dropout = 0.1
        output_dir = OUTPUT_DIR

    args = Args()

    # 加载数据集
    full_dataset = FuncDataset(args, random_seed=args.seed, split='training', rotation=False)
    test_dataset = FuncDataset(args, random_seed=args.seed, split='testing', rotation=False)

    # DataLoader
    train_loader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=1,
                              collate_fn=partial(collate_fn_struct_only, config=args))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                             collate_fn=partial(collate_fn_struct_only, config=args))

    # 模型 + 加载权重
    model = build_model(args).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("✅ 成功加载预训练GCN模型\n")

    # ===================== 提取 + 池化 =====================
    print("===== 处理训练集 =====")
    train_feat_dict = extract_and_pool_features(train_loader, model, DEVICE, full_dataset, TRAIN_STRUCT_DIR)
    save_final_features(train_feat_dict, osp.join(OUTPUT_DIR, "train_struct_features_384.npz"))

    print("===== 处理测试集 =====")
    test_feat_dict = extract_and_pool_features(test_loader, model, DEVICE, test_dataset, TEST_STRUCT_DIR)
    save_final_features(test_feat_dict, osp.join(OUTPUT_DIR, "test_struct_features_384.npz"))

    print("🎉 全部完成！仅保留384维池化特征")

if __name__ == '__main__':
    main()
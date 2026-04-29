# func.py - 完整修改版本

import numpy as np
import torch
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from scipy.spatial.transform import Rotation
import torch.nn.functional as F


def orientation(pos):
    u = normalize(X=pos[1:, :] - pos[:-1, :], norm='l2', axis=1)
    u1 = u[1:, :]
    u2 = u[:-1, :]
    b = normalize(X=u2 - u1, norm='l2', axis=1)
    n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
    o = normalize(X=np.cross(b, n), norm='l2', axis=1)
    ori = np.stack([b, n, o], axis=1)
    return np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0)


def get_polarity_features(amino_ids, device='cpu'):
    """
    根据氨基酸索引计算极性特征

    极性分类：
    - 非极性 (class 0): G, A, V, L, I, P, M, F, W
    - 极性不带电 (class 1): S, T, C, N, Q, Y
    - 带正电 (class 2): K, R, H
    - 带负电 (class 3): D, E

    Args:
        amino_ids: 氨基酸索引张量，形状 [N]，值范围0-19
        device: 设备

    Returns:
        polarity_features: 极性特征张量，形状 [N, 4]
    """
    # 极性映射字典
    polarity_map = {
        0: 1,  # G - 甘氨酸，极性不带电
        1: 0,  # A - 丙氨酸，非极性
        2: 1,  # S - 丝氨酸，极性不带电
        3: 0,  # P - 脯氨酸，非极性
        4: 0,  # V - 缬氨酸，非极性
        5: 1,  # T - 苏氨酸，极性不带电
        6: 0,  # C - 半胱氨酸，非极性
        7: 0,  # I - 异亮氨酸，非极性
        8: 0,  # L - 亮氨酸，非极性
        9: 1,  # N - 天冬酰胺，极性不带电
        10: 3,  # D - 天冬氨酸，负电
        11: 1,  # Q - 谷氨酰胺，极性不带电
        12: 2,  # K - 赖氨酸，正电
        13: 3,  # E - 谷氨酸，负电
        14: 0,  # M - 甲硫氨酸，非极性
        15: 1,  # H - 组氨酸，极性不带电
        16: 0,  # F - 苯丙氨酸，非极性
        17: 2,  # R - 精氨酸，正电
        18: 0,  # Y - 酪氨酸，非极性
        19: 0  # W - 色氨酸，非极性
    }

    # 创建极性特征张量
    polarity_features = torch.zeros(amino_ids.shape[0], 4, device=device)

    for i in range(amino_ids.shape[0]):
        aa_idx = amino_ids[i].item()
        polarity_class = polarity_map.get(aa_idx, 0)
        polarity_features[i, polarity_class] = 1.0

    return polarity_features


def get_mutation_onehot(amino_ids, mutant_str, mut_position_npz, device='cpu'):
    """
    创建突变one-hot编码

    Args:
        amino_ids: 氨基酸索引张量，形状 [N]
        mutant_str: 突变字符串，如 "F7L"
        mut_position_npz: npz文件中的突变位置索引
        device: 设备

    Returns:
        mutation_onehot: 突变one-hot编码，形状 [N, 20]
    """
    # 氨基酸映射
    idx_to_amino_acid = {
        0: "G", 1: "A", 2: "S", 3: "P", 4: "V",
        5: "T", 6: "C", 7: "I", 8: "L", 9: "N",
        10: "D", 11: "Q", 12: "K", 13: "E",
        14: "M", 15: "H", 16: "F", 17: "R",
        18: "Y", 19: "W"
    }

    amino_acid_to_idx = {v: k for k, v in idx_to_amino_acid.items()}

    # 初始化突变one-hot为全零
    mutation_onehot = torch.zeros(amino_ids.shape[0], 20, device=device)

    # 如果没有突变信息，直接返回全零
    if not mutant_str or mut_position_npz < 0 or mut_position_npz >= amino_ids.shape[0]:
        return mutation_onehot

    # 解析突变字符串
    wild_aa = mutant_str[0]  # 野生型氨基酸
    mut_aa = mutant_str[-1]  # 突变后氨基酸

    # 转换为索引
    wild_aa_idx = amino_acid_to_idx.get(wild_aa, -1)
    mut_aa_idx = amino_acid_to_idx.get(mut_aa, -1)

    if wild_aa_idx == -1 or mut_aa_idx == -1:
        return mutation_onehot

    # 验证npz中的突变位置是否正确
    original_aa_idx = amino_ids[mut_position_npz].item()
    original_aa_letter = idx_to_amino_acid.get(original_aa_idx, '?')

    if original_aa_letter == wild_aa:
        # 创建突变氨基酸的one-hot向量
        mut_onehot_vec = torch.zeros(20, device=device)
        mut_onehot_vec[mut_aa_idx] = 1.0
        mutation_onehot[mut_position_npz] = mut_onehot_vec
    else:
        # 尝试在整个序列中寻找匹配的氨基酸
        for i in range(amino_ids.shape[0]):
            if amino_ids[i].item() == wild_aa_idx:
                mut_onehot_vec = torch.zeros(20, device=device)
                mut_onehot_vec[mut_aa_idx] = 1.0
                mutation_onehot[i] = mut_onehot_vec
                break

    return mutation_onehot


def build_struct_features(amino_ids, mutant_str, mut_position_npz, use_polarity=True, device='cpu'):
    """
    构建结构特征：野生型one-hot + 突变one-hot + 极性特征

    Args:
        amino_ids: 氨基酸索引张量，形状 [N]
        mutant_str: 突变字符串
        mut_position_npz: 突变位置
        use_polarity: 是否使用极性特征
        device: 设备

    Returns:
        features: 合并后的特征张量
    """
    # 1. 野生型one-hot
    base_onehot = F.one_hot(amino_ids, num_classes=20).float()

    # 2. 突变one-hot
    mutation_onehot = get_mutation_onehot(amino_ids, mutant_str, mut_position_npz, device)

    # 3. 极性特征（可选）
    if use_polarity:
        polarity_features = get_polarity_features(amino_ids, device)
        # 合并所有特征: [20维one-hot] + [20维突变] + [4维极性] = 44维
        features = torch.cat([base_onehot, mutation_onehot, polarity_features], dim=1)
    else:
        # 只合并one-hot特征: [20维one-hot] + [20维突变] = 40维
        features = torch.cat([base_onehot, mutation_onehot], dim=1)

    return features


class FuncDataset(Dataset):
    def __init__(self, cfg, random_seed=0, split='training', rotation=True):
        super(FuncDataset, self).__init__()

        self.random_state = np.random.RandomState(random_seed)
        self.split = split
        self.rotation = rotation
        self.cfg = cfg

        self.surface = self.cfg.surface
        self.structure = self.cfg.structure

        # 极性特征配置
        self.use_polarity = getattr(cfg, 'use_polarity', True)

        # 设置数据目录
        if split == 'training':
            self.surface_dir = "/root/autodl-tmp/datasets/surface_train"
            self.structure_dir = "/data2/yanmengxiang/haiyun/GCN/datasets/structure_train"
        else:
            self.surface_dir = "/root/autodl-tmp/datasets/surface_test"
            self.structure_dir = "/data2/yanmengxiang/haiyun/GCN/datasets/structure_test"

        # 收集结构数据文件作为蛋白质名称的基础
        self.structure_files = []
        if os.path.exists(self.structure_dir):
            print(f"扫描结构数据目录: {self.structure_dir}")
            for file in os.listdir(self.structure_dir):
                if file.endswith('.npz'):
                    protein_name = file.replace('.npz', '')
                    self.structure_files.append(protein_name)

        print(f"找到结构数据文件: {len(self.structure_files)} 个")

        # 蛋白质名称直接从结构文件获取
        self.protein_names = self.structure_files

        # 标签处理
        self.num_classes = 1

    def __len__(self):
        return len(self.protein_names)

    def __getitem__(self, idx):
        protein_name = self.protein_names[idx]  # 例如: Q99574_F7L
        output = {}

        # 解析蛋白质名称获取基本信息
        parts = protein_name.split('_')

        # 新格式: uniprotID_mutant_label
        # 例: Q99574_F7L_1
        if len(parts) >= 3:
            uniprot_id = parts[0]  # Q99574
            mutant_str = parts[1]  # F7L
            label = int(parts[2])  # 1
        elif len(parts) == 2:
            # 兼容旧格式
            uniprot_id = parts[0]
            mutant_str = parts[1]
            label = 0
        else:
            uniprot_id = protein_name
            mutant_str = ""
            label = 0

        output['uniprot_id'] = uniprot_id
        output['mutant_str'] = mutant_str
        output['label'] = torch.tensor([label], dtype=torch.long)

        rot = self.gen_rot()
        center = None

        # 结构数据部分
        if self.structure:
            structure_file_path = os.path.join(self.structure_dir, protein_name + '.npz')

            if os.path.exists(structure_file_path):
                try:
                    data = np.load(structure_file_path)
                    pos = data['coords']
                    amino_ids = data['amino_ids']

                    # 读取突变位置（如果有）
                    if 'mut_position' in data:
                        mut_position_npz = int(data['mut_position'])  # 在已处理数据中的索引
                        output['mut_position_npz'] = mut_position_npz
                    else:
                        output['mut_position_npz'] = -1  # 表示没有突变位置信息

                    # 计算中心
                    center = np.sum(a=pos, axis=0, keepdims=True) / pos.shape[0]
                    pos = pos - center

                    if self.rotation:
                        pos = np.matmul(pos, rot.numpy())

                    ori = orientation(pos)
                    amino = amino_ids.astype(int)

                    if self.split == "training":
                        pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

                    pos = pos.astype(dtype=np.float32)
                    ori = ori.astype(dtype=np.float32)
                    seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

                    x = torch.from_numpy(amino)
                    ori = torch.from_numpy(ori)
                    seq = torch.from_numpy(seq)
                    pos = torch.from_numpy(pos)

                    # 注意：这里不构建特征，只保存原始数据
                    # 特征构建将在模型forward中进行
                    output['x'] = x
                    output['ori'] = ori
                    output['seq'] = seq
                    output['pos'] = pos

                except Exception as e:
                    print(f"❌ 加载结构数据失败 {structure_file_path}: {e}")
                    output['x'] = torch.empty(0, dtype=torch.long)
                    output['ori'] = torch.empty(0, 3, 3, dtype=torch.float32)
                    output['seq'] = torch.empty(0, 1, dtype=torch.float32)
                    output['pos'] = torch.empty(0, 3, dtype=torch.float32)
                    output['mut_position_npz'] = -1
            else:
                output['x'] = torch.empty(0, dtype=torch.long)
                output['ori'] = torch.empty(0, 3, 3, dtype=torch.float32)
                output['seq'] = torch.empty(0, 1, dtype=torch.float32)
                output['pos'] = torch.empty(0, 3, dtype=torch.float32)
                output['mut_position_npz'] = -1

        return output

    def gen_rot(self):
        R = torch.FloatTensor(Rotation.random().as_matrix())
        return R


# func.py - 修改 collate_fn_struct_only 函数

def collate_fn_struct_only(list_data, config):
    """只处理结构数据的简化collate函数，并预构建特征"""
    dict_inputs = {}
    batch_labels_list = []

    batched_x = []
    batched_ori = []
    batched_seq = []
    batched_pos = []
    batched_index = []

    # 存储预构建的特征
    batched_features = []

    # 获取配置
    use_polarity = getattr(config, 'use_polarity', True)

    for ind, batch_i in enumerate(list_data):
        label = batch_i['label']
        batch_labels_list.append(label)

        # 获取原始数据
        amino_ids = batch_i['x']
        pos = batch_i['pos']
        ori = batch_i['ori']
        seq = batch_i['seq']

        # 获取突变信息
        mutant_str = batch_i.get('mutant_str', '')
        mut_position_npz = batch_i.get('mut_position_npz', -1)

        # 在这里预构建特征，而不是在模型中
        features = build_struct_features(
            amino_ids=amino_ids,
            mutant_str=mutant_str,
            mut_position_npz=mut_position_npz,
            use_polarity=use_polarity,
            device='cpu'  # 先在CPU上构建，后续会移动到GPU
        )

        batched_features.append(features)
        batched_ori.append(ori)
        batched_seq.append(seq)
        batched_pos.append(pos)
        batched_index.append((ind * torch.ones(len(amino_ids), dtype=torch.int64)))

    # 合并所有数据
    dict_inputs['label'] = torch.cat(batch_labels_list)
    dict_inputs['features'] = torch.cat(batched_features, dim=0)  # 使用预构建的特征
    dict_inputs['ori'] = torch.cat(batched_ori, dim=0)
    dict_inputs['seq'] = torch.cat(batched_seq, dim=0)
    dict_inputs['pos'] = torch.cat(batched_pos, dim=0)
    dict_inputs['batch'] = torch.cat(batched_index, dim=0)

    # 传递其他信息
    dict_inputs['mutant_str'] = [b.get('mutant_str', '') for b in list_data]
    dict_inputs['mut_position_npz'] = [b.get('mut_position_npz', -1) for b in list_data]
    dict_inputs['uniprot_id'] = [b.get('uniprot_id', f'protein_{i}') for i, b in enumerate(list_data)]

    return dict_inputs


if __name__ == '__main__':
    print()
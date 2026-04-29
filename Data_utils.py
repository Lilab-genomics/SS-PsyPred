import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle


class ProtT5H5Dataset(Dataset):
    def __init__(self, h5_path, max_len=128):
        self.h5 = h5py.File(h5_path, "r")
        self.keys = list(self.h5.keys())
        self.max_len = max_len

    def __len__(self):
        return len(self.keys)

    def center_pad(self, x):
        L, D = x.shape
        mask = np.ones(self.max_len, dtype=np.bool_)

        if L > self.max_len:
            start = (L - self.max_len) // 2
            x = x[start:start + self.max_len]
        elif L < self.max_len:
            pad = self.max_len - L
            left = pad // 2
            out = np.zeros((self.max_len, D), dtype=np.float32)
            out[left:left + L] = x
            mask[:] = False
            mask[left:left + L] = True
            x = out
        return x, mask

    def __getitem__(self, idx):
        key = self.keys[idx]
        x, mask = self.center_pad(self.h5[key][:])
        y = int(key[-1])
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(y, dtype=torch.float32)
        )


class FeatureDataset(Dataset):
    def __init__(self, h5_path, max_len=128):
        self.h5 = h5py.File(h5_path, "r")
        self.keys = list(self.h5.keys())
        self.max_len = max_len

    def __len__(self):
        return len(self.keys)

    def center_pad(self, x):
        L, D = x.shape
        mask = np.ones(self.max_len, dtype=np.bool_)
        if L > self.max_len:
            start = (L - self.max_len) // 2
            x = x[start:start + self.max_len]
        elif L < self.max_len:
            pad = self.max_len - L
            left = pad // 2
            out = np.zeros((self.max_len, D), dtype=np.float32)
            out[left:left + L] = x
            mask[:] = False
            mask[left:left + L] = True
            x = out
        return x, mask

    def __getitem__(self, idx):
        key = self.keys[idx]
        x, mask = self.center_pad(self.h5[key][:])
        return key, torch.tensor(x, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


def load_three_modalities_with_keys(pth_ref, pth_alt, npz_struct, return_keys=False):
    ref_dict = torch.load(pth_ref, map_location="cpu", weights_only=False)
    alt_dict = torch.load(pth_alt, map_location="cpu", weights_only=False)
    struct_data = np.load(npz_struct, allow_pickle=True)
    if "arr_0" in struct_data:
        struct_dict = struct_data["arr_0"].item()
    elif "features" in struct_data:
        struct_dict = struct_data["features"].item()
    else:
        first_key = list(struct_data.keys())[0]
        struct_dict = struct_data[first_key].item()

    common_keys = sorted(
        set(ref_dict.keys()) & set(alt_dict.keys()) & set(struct_dict.keys())
    )
    print(f"[INFO] sample number: {len(common_keys)}")
    X1, X2, X3, Y = [], [], [], []
    for key in common_keys:
        x1 = ref_dict[key].cpu().numpy() if isinstance(ref_dict[key], torch.Tensor) else ref_dict[key]
        x2 = alt_dict[key].cpu().numpy() if isinstance(alt_dict[key], torch.Tensor) else alt_dict[key]
        x3 = struct_dict[key]

        y = 1 if str(key).endswith("_1") else 0

        X1.append(x1)
        X2.append(x2)
        X3.append(x3)
        Y.append(y)

    X1 = torch.tensor(np.array(X1), dtype=torch.float32)
    X2 = torch.tensor(np.array(X2), dtype=torch.float32)
    X3 = torch.tensor(np.array(X3), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    if return_keys:
        return X1, X2, X3, Y, common_keys
    else:
        return X1, X2, X3, Y
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from Data_utils import FeatureDataset
from models import PositionalEncoding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 特征保存
# =========================
def save_features(residue_feats, sequence_feats, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(residue_feats, os.path.join(out_dir, f"{prefix}_enhanced_residue.pth"))
    torch.save(sequence_feats, os.path.join(out_dir, f"{prefix}_enhanced_sequence.pth"))
    if "ref" in prefix:
        print("The features of the reference sequence have been saved！")
    elif "alt" in prefix:
        print("The alternate sequence features have been saved！")



class FeatureExtractor(nn.Module):
    def __init__(self, dim=768, max_len=128):
        super().__init__()
        self.pos_enc = PositionalEncoding(dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=8, dim_feedforward=1024, dropout=0.3, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x, mask):
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=~mask)
        mask_f = mask.unsqueeze(-1)
        seq_feat = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return x, seq_feat

def extract_features(
    trained_model,
    h5_path,
    out_dir,
    prefix,
    max_len=128,
    batch_size=8
):
    dataset = FeatureDataset(h5_path, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    extractor = FeatureExtractor().to(DEVICE)
    extractor.load_state_dict(trained_model.state_dict(), strict=False)
    extractor.eval()

    res_dict = {}
    seq_dict = {}

    with torch.no_grad():
        for keys, x, mask in loader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            res_feat, seq_feat = extractor(x, mask)

            res_feat = res_feat.cpu().numpy()
            seq_feat = seq_feat.cpu().numpy()
            mask = mask.cpu().numpy()

            for i, key in enumerate(keys):
                valid_len = mask[i].sum()
                res_dict[key] = res_feat[i][:valid_len]
                seq_dict[key] = seq_feat[i]

    save_features(res_dict, seq_dict, out_dir, prefix)
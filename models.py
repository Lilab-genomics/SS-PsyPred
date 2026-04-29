import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        x: (B, L, D)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, dim=768, max_len=128):
        super().__init__()

        self.pos_enc = PositionalEncoding(dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.3,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # 🔥 输出 1 维
        )

    def forward(self, x, mask):
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=~mask)

        # masked mean pooling
        mask = mask.unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.mlp(x).squeeze(-1)  # 🔥 去掉二维输出


class GatedCompetitiveFusion(nn.Module):
    def __init__(self, dim1=768, dim2=768, dim3=384, hidden=256):
        super().__init__()
        self.proj1 = nn.Linear(dim1, hidden)
        self.proj2 = nn.Linear(dim2, hidden)
        self.proj3 = nn.Linear(dim3, hidden)

        self.gate = nn.Sequential(
            nn.Linear(hidden * 3, 3),
            nn.Softmax(dim=-1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x1, x2, x3):
        h1 = self.proj1(x1)
        h2 = self.proj2(x2)
        h3 = self.proj3(x3)

        concat = torch.cat([h1, h2, h3], dim=-1)
        weights = self.gate(concat)
        w1, w2, w3 = weights.unbind(dim=-1)

        fused = h1 * w1.unsqueeze(-1) + h2 * w2.unsqueeze(-1) + h3 * w3.unsqueeze(-1)
        return self.classifier(fused)
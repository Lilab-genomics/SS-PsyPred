import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from models import TransformerClassifier
from Data_utils import ProtT5H5Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)

from models import GatedCompetitiveFusion
from Data_utils import load_three_modalities_with_keys
#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)

def train_transformer(
    epochs,
    h5_path: str,
    save_path: str,
    max_len=128,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4
):

    dataset = ProtT5H5Dataset(h5_path, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerClassifier(dim=768, max_len=max_len).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        all_probs = []
        all_labels = []

        for x, mask, y in loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(prob)
            all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(loader)
        auc = compute_auc(all_labels, all_probs)
        print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} ")

    torch.save(model.state_dict(), save_path)
    print(f"model have been save to: {save_path}")
    return model

def train_fusion_model(
        train_tf, train_lstm, train_struct,
        test_tf, test_lstm, test_struct,
        dim1=768, dim2=128, dim3=384, hidden=256,
        epochs=50, lr=1e-4, patience=8,
        save_path="./checkpoints/best_fusion_model.pt"
):
    print("\n[INFO] loading train-set...")
    X1_tr, X2_tr, X3_tr, Y_tr = load_three_modalities_with_keys(train_tf, train_lstm, train_struct)
    print("\n[INFO] loading test-set...")
    X1_te, X2_te, X3_te, Y_te, test_keys = load_three_modalities_with_keys(test_tf, test_lstm, test_struct,
                                                                           return_keys=True)

    X1_tr, X1_val, X2_tr, X2_val, X3_tr, X3_val, Y_tr, Y_val = train_test_split(
        X1_tr, X2_tr, X3_tr, Y_tr, test_size=0.2, stratify=Y_tr, random_state=42
    )

    X1_tr, X2_tr, X3_tr, Y_tr = X1_tr.to(DEVICE), X2_tr.to(DEVICE), X3_tr.to(DEVICE), Y_tr.to(DEVICE)
    X1_val, X2_val, X3_val, Y_val = X1_val.to(DEVICE), X2_val.to(DEVICE), X3_val.to(DEVICE), Y_val.to(DEVICE)
    X1_te, X2_te, X3_te, Y_te = X1_te.to(DEVICE), X2_te.to(DEVICE), X3_te.to(DEVICE), Y_te.to(DEVICE)

    model = GatedCompetitiveFusion(
        dim1=dim1, dim2=dim2, dim3=dim3, hidden=hidden
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_val_auc = 0
    es_count = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X1_tr, X2_tr, X3_tr).squeeze()
        loss = criterion(logits, Y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X1_val, X2_val, X3_val).squeeze()
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(Y_val.cpu().numpy(), val_prob)

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch + 1:2d} | Val AUC: {val_auc:.4f} | LR: {current_lr:.6f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            es_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"save best checkpoints | Best Val AUC: {best_val_auc:.4f}")
        else:
            es_count += 1
            if es_count >= patience:
                print(f"\n[INFO] early stop!")
                break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    model.eval()
    with torch.no_grad():
        h1 = model.proj1(X1_te)
        h2 = model.proj2(X2_te)
        h3 = model.proj3(X3_te)
        concat = torch.cat([h1, h2, h3], dim=-1)
        weights = model.gate(concat)
        w1, w2, w3 = weights.unbind(dim=-1)
        fused_feature = h1 * w1.unsqueeze(-1) + h2 * w2.unsqueeze(-1) + h3 * w3.unsqueeze(-1)

        logits = model.classifier(fused_feature).squeeze()
        prob = torch.sigmoid(logits).cpu().numpy()
        pred = (prob >= 0.5).astype(float)
        y_true = Y_te.cpu().numpy()

    fused_feature_np = fused_feature.cpu().numpy()
    print(f"fusion feature shape: {fused_feature_np.shape}")

    np.savez(
        "./enhanced_features/test_fused_feature.npz",
        keys=test_keys,
        fused_feature=fused_feature_np,
        y_true=y_true,
        prob=prob
    )
    print("fusion feature been save to ./enhanced_features/test_fused_feature.npz")

    print(f"ACC        : {accuracy_score(y_true, pred):.4f}")
    print(f"AUC        : {roc_auc_score(y_true, prob):.4f}")
    print(f"AUPR       : {average_precision_score(y_true, prob):.4f}")
    print(f"Precision  : {precision_score(y_true, pred):.4f}")
    print(f"Recall     : {recall_score(y_true, pred):.4f}")
    print(f"F1         : {f1_score(y_true, pred):.4f}")

    return test_keys, y_true, prob, pred
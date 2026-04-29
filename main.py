from train import train_transformer, train_fusion_model
from feature_extract import extract_features
import random
import numpy as np
import torch
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
SEED = 42
MAX_LEN = 128
FEAT_DIR = "./enhanced_features"

TRAIN_REF_H5 = "train_ref.h5"
TEST_REF_H5 = "test_544_ref.h5"
TRAIN_ALT_H5 = "train_alt.h5"
TEST_ALT_H5 = "test_544_alt.h5"

TRAIN_STRUCT = "train_struct_pooled.npz"
TEST_STRUCT = "test_544_struct.npz"

MODEL_PATH = "./checkpoints/best_transformer.pt"
FUSION_SAVE_PATH = "./checkpoints/best_fusion_model.pt"

TRAIN_REF_AUG = f"{FEAT_DIR}/train_ref_enhanced_sequence.pth"
TEST_REF_AUG  = f"{FEAT_DIR}/test_ref_enhanced_sequence.pth"
TRAIN_ALT_AUG = f"{FEAT_DIR}/train_alt_enhanced_sequence.pth"
TEST_ALT_AUG  = f"{FEAT_DIR}/test_alt_enhanced_sequence.pth"

DIM1 = 768
DIM2 = 768
DIM3 = 384
HIDDEN = 256

EPOCHS_FUSION = 50
LR_FUSION = 1e-4
PATIENCE_FUSION = 8
SAVE_RESULT = True

# ==============================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# ==============================================
if __name__ == "__main__":
    model_ref = train_transformer(
        epochs=3,
        h5_path=TRAIN_REF_H5,
        save_path=MODEL_PATH,
        max_len=MAX_LEN
    )

    extract_features(
        model_ref,
        TRAIN_REF_H5,
        FEAT_DIR,
        "train_ref",
        max_len=MAX_LEN
    )

    extract_features(
        model_ref,
        TEST_REF_H5,
        FEAT_DIR,
        "test_ref",
        max_len=MAX_LEN
    )


    model_alt = train_transformer(
        epochs=4,
        h5_path=TRAIN_ALT_H5,
        save_path=MODEL_PATH,
        max_len=MAX_LEN
    )

    extract_features(
        model_alt,
        TRAIN_ALT_H5,
        FEAT_DIR,
        "train_alt",
        max_len=MAX_LEN
    )

    extract_features(
        model_alt,
        TEST_ALT_H5,
        FEAT_DIR,
        "test_alt",
        max_len=MAX_LEN
    )


    test_keys, y_true, prob, pred = train_fusion_model(

        train_tf=TRAIN_REF_AUG,
        test_tf=TEST_REF_AUG,

        train_lstm=TRAIN_ALT_AUG,
        test_lstm=TEST_ALT_AUG,

        train_struct=TRAIN_STRUCT,
        test_struct=TEST_STRUCT,

        dim1=DIM1,
        dim2=DIM2,
        dim3=DIM3,
        hidden=HIDDEN,
        epochs=EPOCHS_FUSION,
        lr=LR_FUSION,
        patience=PATIENCE_FUSION,
        save_path=FUSION_SAVE_PATH
    )

    # 保存预测结果
    if SAVE_RESULT:
        result_df = pd.DataFrame({
            "sample_key": test_keys,
            "true_label": y_true,
            "predict_score": prob,
            "predict_class": pred
        })
        result_df.to_csv("test_predictions.csv", index=False, encoding="utf-8")

import torch
from typing import Literal

from models.deepfm import DeepFM
from models.dcnv3 import DCNv3
from models.mlp import MLP


MODEL_PATHS = {
    "DeepFM": "./checkpoints/deepfm.pt",
    "DCNv3": "./checkpoints/dcnv3.pt",
    "MLP": "./checkpoints/mlp.pt",
}

def load_model_mlp(model_path = MODEL_PATHS["MLP"], device=None):
    # Example
    model = MLP(
        embed_dim=16,
        hidden_layers=[64, 32],
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()
    return model, device

def load_model_dcnv3(dataset, model_path=MODEL_PATHS["DCNv3"], device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DCNv3(
        field_dims=dataset.field_dims,
        dense_dim=dataset.dense_dim,
        embed_dim=32,
        num_deep_cross_layers=5,
        num_shallow_cross_layers=2,
        deep_net_dropout=0,
        shallow_net_dropout=0,
        layer_norm=False,
        batch_norm=False
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def load_model_deepfm(model_path = MODEL_PATHS["DeepFM"], device=None):
    # Example
    model = DeepFM()
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()
    return model,device

# def load_model(model_name: Literal["DeepFM", "DCNv3", "MLP"], device=None):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if model_name == "MLP":
#         return load_model_mlp(device=device)
#     elif model_name == "DeepFM":
#         return load_model_deepfm(device=device)
#     elif model_name == "DCNv3":
#         return load_model_dcnv3(device=device)


if __name__ == "__main__":
    import numpy as np
    import torch
    from src.data_utils import load_dataset

    print("🔹 Loading dataset...")
    dataset = load_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔹 Using device:", device)

    # =========================
    # Test DCNv3
    # =========================
    print("\n🚀 Testing DCNv3")
    model, device = load_model_dcnv3(
        dataset,
        r"src\checkpoints\dcnv3.pth",
        device=device
    )


    # Lấy 1 sample bất kỳ
    sparse, dense, target = dataset[0]

    sparse = sparse.unsqueeze(0).to(device)   # (1, num_sparse)
    dense = dense.unsqueeze(0).to(device)     # (1, dense_dim)

    with torch.no_grad():
        output = model(sparse, dense)

    print("DCNv3 output:", output)
    print("y_pred shape:", output["y_pred"].shape)
    print("y_d shape:", output["y_d"].shape)
    print("y_s shape:", output["y_s"].shape)


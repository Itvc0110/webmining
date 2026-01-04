import torch
from typing import Literal

from models.deepfm import DeepFM
from models.dcnv3 import DCNv3
from models.mlp import MLP


MODEL_PATHS = {
    "DeepFM": "./checkpoints/deepfm.pt",
    "DCNv3": "./checkpoints/deepfm_autofis.pt",
    "MLP": "./checkpoints/mlp.pt",
}

def load_model_mlp(model_path = MODEL_PATHS["MLP"], device="cpu"):
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
    return model

def load_model_dcnv3(model_path = MODEL_PATHS["DCNv3"], device="cpu"):
    # Example
    model = DCNv3()
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()
    return model

def load_model_deepfm(model_path = MODEL_PATHS["DeepFM"], device="cpu"):
    # Example
    model = DeepFM()
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()
    return model

def load_model(model_name: Literal["DeepFM", "DCNv3", "MLP"], device='cpu'):
    if model_name == "MLP":
        return load_model_mlp(device=device)
    elif model_name == "DeepFM":
        return load_model_deepfm(device=device)
    elif model_name == "DCNv3":
        return load_model_dcnv3(device=device)

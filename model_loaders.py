import torch
import mlflow.pytorch
from models import ModelConfiguration
import inject
import os

from models.base_pong_model import BasePongModel


def save_mlflow_model(model, path):
    mlflow.pytorch.log_model(model, path)

def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)


@inject.params(device="device")
def load_model(path, device: torch.device):
    if path.startswith("runs"):
        dst_path = os.path.dirname(path.replace("runs:", "./artifacts"))
        os.makedirs(dst_path, exist_ok=True)
        model = mlflow.pytorch.load_model(path, dst_path, map_location=device)
    else:
        model = mlflow.pytorch.load_model(path, map_location=device)
    return model

import torch
import mlflow.pytorch
from runtime_configuration import Model
from models import ModelConfiguration

config = ModelConfiguration()


def save_mlflow_model(model, path):
    mlflow.pytorch.log_model(model, path)

def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)


def load_mlflow_model(path):
    model = mlflow.pytorch.load_model(path).to(device=config.device)
    return model

def load_pytorch_model(path):
    """Load weights into a pytorch model from the specified path to .pth file"""
    model = Model().to(device=config.device)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device(config.device)))
    return model
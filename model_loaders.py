import torch
import mlflow.pytorch
from models import ModelConfiguration
import inject


def save_mlflow_model(model, path):
    mlflow.pytorch.log_model(model, path)

def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)


@inject.params(config=ModelConfiguration)
def load_mlflow_model(path, config: ModelConfiguration):
    model = mlflow.pytorch.load_model(path, map_location=torch.device(config.device))
    return model

@inject.params(config=ModelConfiguration)
def load_pytorch_model(path, config: ModelConfiguration):
    """Load weights into a pytorch model from the specified path to .pth file"""
    model = config.model().to(device=config.device)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device(config.device)))
    return model
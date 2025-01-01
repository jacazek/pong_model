from models import TransformerModel, RNNModel
from models import FlashAttentionTransformer

mlflow_server_url = "http://localhost:8080"

# model to use during training and inference
Model = FlashAttentionTransformer

# path to mlflow model
mlflow_model_path = f"runs:/732539a5e0f049559d6dba2d02e7577a/model_e69"

# path to pytorch model weights
model_path = f"{Model.__name__}_weights.pth"


classification_threshold = 0.5
temperature_variance = 0.0

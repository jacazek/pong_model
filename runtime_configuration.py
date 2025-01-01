from models import TransformerModel, RNNModel
from models import FlashAttentionTransformer

# model to use during training and inference
Model = RNNModel
model_path = f"{Model.__name__}_weights.pth"
classification_threshold = 0.5
temperature_variance = 0.0

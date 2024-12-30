from model import TransformerModel, RNNModel
from Transformer import AlternateTransformerModel

# model to use during training and inference
Model = TransformerModel
model_path = f"{Model.__name__}_weights.pth"
classification_threshold = 0.5
temperature_variance = 0.0

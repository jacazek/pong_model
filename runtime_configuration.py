from models import Transformermodel, RNNModel
from models import FlashAttentionTransformer

# model to use during training and inference
Model = Transformermodel
model_path = f"{Model.__name__}_weights.pth"
classification_threshold = 0.5
temperature_variance = 0.0

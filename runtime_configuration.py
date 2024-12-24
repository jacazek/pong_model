from model import TransformerModel, RNNModel

# model to use during training and inference
Model = TransformerModel
model_path = f"{type(Model).__name__}.pth"
classification_threshold = 0.99

from models import TransformerModel, RNNModel
# from models import FlashAttentionTransformer

mlflow_server_url = "http://localhost:8080"

# model to use during training and inference
Model = RNNModel

# path to mlflow model
# use a run that corresponds with the desired model type (rnn/transformer/flashtransformer)
mlflow_model_path = f"runs:/62a7d1ead3564c379cbacbff4ef7ac55/model_e99"

# path to pytorch model weights
model_path = f"{Model.__name__}_weights.pth"


classification_threshold = 0.5
temperature_variance = 0.0

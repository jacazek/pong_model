from dataclasses import dataclass
import argparse
import os
import subprocess
from models.rnn import RNNModel
from models.transformer import TransformerModel
# from models.transformer_flashattn import FlashAttentionTransformer
from models import ModelConfiguration

model_dictionary = {RNNModel.__name__: RNNModel,
          TransformerModel.__name__: TransformerModel}
          # FlashAttentionTransformer.__name__: FlashAttentionTransformer}
model_names = list(model_dictionary.keys())

@dataclass
class TrainArguments(ModelConfiguration):
    mlflow_server_url: str = "https://localhost:8080"
    epochs: int = 100
    batch_size: int = 1024

    learning_rate: float = 0.0001
    gamma: float = 0.95
    model_type: str = RNNModel.__name__
    model = None

    train_data_set_steps: int = 3200000
    validate_dataset_steps: int = 10000
    # keep this parameter last
    command: str = ""

    @staticmethod
    def get_arguments():
        parser = argparse.ArgumentParser(description="Train configuration")

        parser.add_argument("--mlflow_server_url", type=str, default="http://localhost:8080", help="mlflow server url")
        parser.add_argument("--epochs", type=int, default=100, help="The number of epochs to train")
        parser.add_argument("--batch_size", type=int, default=1024,
                            help="The size of each batch for training and validation")
        parser.add_argument("--learning_rate", type=float, default=0.0001,
                            help="Initial learning rate for optimizer")
        parser.add_argument("--gamma", type=float, default=0.95,
                            help="The learning rate gamma for the scheduler")

        parser.add_argument("--model_type", type=str, default=model_names[0],help="The model type to train", choices=model_names)
        parser.add_argument("--input_size", type=int, default=16, help="The input size of the model")
        parser.add_argument("--hidden_size", type=int, default=128, help="The hidden size of the model")
        parser.add_argument("--num_layers", type=int, default=2, help="The number of layers of the model")
        parser.add_argument("--number_heads", type=int, default=16, help="The number of heads of the model (transformer model only)")
        parser.add_argument("--input_sequence_length", type=int, default=10,help="The length of the input sequence")

        args = parser.parse_args()
        train_arguments = TrainArguments()
        for key, value in vars(args).items():
            setattr(train_arguments, key, value)

        train_arguments.model = model_dictionary[args.__dict__["model_type"]]

        train_arguments.command = str(subprocess.run(["ps", "-p", f"{os.getpid()}", "-o", "args", "--no-headers"], capture_output=True,
               text=True).stdout)

        return train_arguments

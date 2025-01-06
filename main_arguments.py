from dataclasses import dataclass
import argparse
import os
import subprocess
from models.rnn import RNNModel
from models.transformer import TransformerModel
from models.transformer_flashattn import FlashAttentionTransformer
from models import ModelConfiguration
from exact_engine import generate_pong_states
from fuzzy_engine import generate_fuzzy_states

generators = {
    "exact": generate_pong_states,
    "fuzzy": generate_fuzzy_states,
}

model_dictionary = {RNNModel.__name__: RNNModel,
          TransformerModel.__name__: TransformerModel,
          FlashAttentionTransformer.__name__: FlashAttentionTransformer}
model_names = list(model_dictionary.keys())

@dataclass
class MainArguments(ModelConfiguration):
    mlflow_server_url: str = "https://localhost:8080"

    model_type: str = RNNModel.__name__

    generator_type: str = list(generators.keys())[1]
    generator = list(generators.values())[1]

    # keep this parameter last
    command: str = ""

    @staticmethod
    def get_arguments():
        parser = argparse.ArgumentParser(description="Main configuration")

        parser.add_argument("--mlflow_server_url", type=str, default="http://localhost:8080", help="mlflow server url")
        parser.add_argument("--model_type", type=str, default=model_names[0],help="The model type to train", choices=model_names)
        parser.add_argument("--generator_type", type=str, default=list(generators.keys())[1],help="The generator type to train", choices=list(generators.keys()))
        parser.add_argument("--input_size", type=int, default=16, help="The input size of the model")
        parser.add_argument("--hidden_size", type=int, default=128, help="The hidden size of the model")
        parser.add_argument("--num_layers", type=int, default=2, help="The number of layers of the model")
        parser.add_argument("--number_heads", type=int, default=16, help="The number of heads of the model (transformer model only)")
        parser.add_argument("--input_sequence_length", type=int, default=10,help="The length of the input sequence")

        args = parser.parse_args()
        main_arguments = MainArguments()
        for key, value in vars(args).items():
            setattr(main_arguments, key, value)

        main_arguments.generator = generators.get(args.__dict__["generator_type"])

        main_arguments.command = str(subprocess.run(["ps", "-p", f"{os.getpid()}", "-o", "args", "--no-headers"], capture_output=True,
               text=True).stdout)

        return main_arguments

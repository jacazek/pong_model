from dataclasses import dataclass
import argparse
import os
import subprocess
from models import ModelConfiguration

generators = [
    "exact",
    "fuzzy"
]

model_names = ["RNNModel", "TransformerModel", "FlashAttentionTransformer"]

@dataclass
class MainArguments(ModelConfiguration):
    mlflow_server_url: str = "https://localhost:8080"
    model_path: str = None
    model_type: str = model_names[0]
    generator_type: str = generators[1]

    @staticmethod
    def get_arguments():
        parser = argparse.ArgumentParser(description="Main configuration")

        parser.add_argument("--mlflow_server_url", type=str, default="http://localhost:8080", help="mlflow server url")
        parser.add_argument("--generator_type", type=str, default=generators[1],help="The generator type to train. Exact calculates the pong game. Fuzzy runs a trained model.", choices=generators)
        parser.add_argument("--input_sequence_length", type=int, default=20,help="The length of the input sequence.")
        parser.add_argument("--model_path", type=str, required=True, help="Path to the model to use for fuzzy engine. Either an mlflow path (runs:/000fc0c95642447899b50e9104b7f6a0/model_e44) or local path (artifacts/000fc0c95642447899b50e9104b7f6a0/model_e44).")
        parser.add_argument("--device", type=str, default="cuda", help="The device to use", choices=["cpu", "cuda"])

        args = parser.parse_args()
        main_arguments = MainArguments()
        for key, value in vars(args).items():
            setattr(main_arguments, key, value)

        print(main_arguments.__dict__)

        return main_arguments

# general parameters
import torch
import dataclasses


@dataclasses.dataclass
class ModelConfiguration:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model parameters
    input_size = 16
    hidden_size = 128
    output_size = 4
    discrete_output_size = 6
    num_layers = 2
    number_heads = 16

    # training parameters
    input_sequence_length=20

    # def get_model_path(self):
    #     return f"{self.model.__name__}_weights.pth"
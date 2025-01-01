# general parameters
import torch


class ModelConfiguration:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # model parameters
        self.input_size = 16
        self.hidden_size = 128
        self.output_size = 4
        self.discrete_output_size = 6
        self.num_layers = 2
        self.number_heads = 8

        # training parameters
        self.input_sequence_length=20

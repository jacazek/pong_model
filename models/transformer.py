import torch
from torch import nn as nn

from .base_pong_model import BasePongModel
from . import ModelConfiguration

config = ModelConfiguration()


class TransformerModel(BasePongModel):
    def __init__(self):
        super(TransformerModel, self).__init__()
        # Consider using decoder only with flash attention
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, config.hidden_size))

        # self.transformer = nn.TransformerEncoder(
        self.transformer_list = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.number_heads,
            dim_feedforward=config.hidden_size,
            batch_first=True,
        ) for _ in range(config.num_layers)])
        #     num_layers=num_layers,
        # )
        # self.transformer = nn.TransformerDecoderLayer(
        #     d_model=hidden_size,
        #     nhead=number_heads,
        #     dim_feedforward=hidden_size,
        #     # dropout=dropout
        # )

    def _forward(self, x):
        seq_len = x.size(1)
        positions = self.positional_encoding[:, :seq_len, :]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = x + positions
        # x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, hidden_dim)
        for transformer in self.transformer_list:
            x = transformer(x, src_mask=causal_mask, is_causal=True)

        # Get the output of the transformer for each sequence by
        # Averaging over the sequence dimension to reduce dimensions for use in predicting next state
        # (similar to using last hidden state of LSTM)
        return x.mean(dim=1)

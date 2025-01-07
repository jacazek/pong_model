import torch
from torch import nn as nn

from .base_pong_model import BasePongModel

class TransformerModel(BasePongModel):
    def __init__(self):
        super(TransformerModel, self).__init__()
        # Consider using decoder only with flash attention
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, self.config.hidden_size))

        self.summary_token = nn.Parameter(torch.randn((1, 1, self.config.hidden_size)))

        self.transformer_list = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size,
            nhead=self.config.number_heads,
            dim_feedforward=self.config.hidden_size,
            batch_first=True,
        ) for _ in range(self.config.num_layers)])


    def _forward(self, x):
        batch_size = x.size(0)
        summaries = self.summary_token.expand(batch_size, -1, -1)
        x = torch.cat([x, summaries], dim=1)
        seq_len = x.size(1)
        positions = self.positional_encoding[:, :seq_len, :]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = x + positions
        for transformer in self.transformer_list:
            x = transformer(x, src_mask=causal_mask, is_causal=True)

        # Get the summary token for each example
        # (similar to using last hidden state of LSTM)
        return x[:, -1, :]

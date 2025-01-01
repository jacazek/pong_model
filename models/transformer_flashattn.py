import torch
from torch import nn as nn
import math

from flash_attn.modules.mha import MHA
from .base_pong_model import BasePongModel
from . import ModelConfiguration

config = ModelConfiguration()


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        positions = self.positional_encoding[:seq_len, :].to(x.device)
        x = x + positions
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        logits = self.fc_out(x)
        return logits

    @staticmethod
    def _generate_positional_encoding(max_seq_len, embed_dim):
        pe = torch.zeros(max_seq_len, embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_dim)))
                if i + 1 < embed_dim:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / embed_dim)))
        return pe.unsqueeze(0)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.mha = MHA(embed_dim, num_heads, causal=False)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output = self.mha(x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.mha = MHA(embed_dim, num_heads)
#         self.mha.fo
#         assert embed_dim % num_heads == 0
#         self.head_dim = embed_dim // num_heads
#         self.num_heads = num_heads
#
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.fc_out = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, q, k, v, mask=None):
#         batch_size = q.size(0)
#         q = self.query(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         k = self.key(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.value(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#
#         # Scaled dot-product attention
#         # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         # if mask is not None:
#         #     scores = scores.masked_fill(mask == 0, float('-inf'))
#         # attn_weights = torch.softmax(scores, dim=-1)
#         # attn_output = torch.matmul(attn_weights, v)
#         attn_output = self.mha(q, k, v, mask)
#
#         # Concatenate heads
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
#         return self.fc_out(attn_output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class FlashAttentionTransformer(BasePongModel):
    def _forward(self, x: torch.Tensor):
        x = self.transformer(x)
        return x.mean(dim=1)

    def __init__(self):
        super(FlashAttentionTransformer, self).__init__()
        self.transformer = Transformer(config.hidden_size, config.number_heads, config.num_layers, config.hidden_size, config.input_sequence_length, 0.2)


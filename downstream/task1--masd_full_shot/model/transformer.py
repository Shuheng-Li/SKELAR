import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import math
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class MatchingNet(nn.Module):
    def __init__(self, hidden_dim, weights):
        super(MatchingNet, self).__init__()
        self.weights = nn.Parameter(weights, requires_grad = False)
        self.linear = nn.Linear(hidden_dim, weights.size(-1))
    def forward(self, x):
        x = self.linear(x)
        prob = torch.einsum('bn,ln->bl', x, self.weights)
        return prob 

class AttentionNet(nn.Module):
    def __init__(self, hidden_dim, weights, out_dim = 32):
        super(AttentionNet, self).__init__()
        self.weights = nn.Parameter(weights, requires_grad = False)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.q = nn.Linear(weights.size(-1), out_dim)
        self.k = nn.Linear(weights.size(-1), out_dim)
        self.v = nn.Linear(weights.size(-1), out_dim)
        
    
    def forward(self, x):
        x = self.linear(x)
        q, k, v = self.q(self.weights), self.k(self.weights), self.v(self.weights)
        out = scaled_dot_product_attention(q, k, v).mean(1)
        #print(out.size())
        prob = torch.einsum('bn,ln->bl', x, out)
        return prob 


class IMUTransformerEncoder(nn.Module):
    def __init__(self, window_size, input_channel, weights = None, dropout = 0.1):
        super().__init__()

        self.transformer_dim = 256

        self.input_proj = nn.Sequential(nn.Conv1d(input_channel, self.transformer_dim, 1), nn.GELU(),
        )

        self.window_size = window_size
        self.encode_position = False
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = 8,
                                       dim_feedforward = 512,
                                       dropout = dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = 8,
                                              norm = nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        

        hidden_dim = self.transformer_dim #//4
        self.classifier = AttentionNet(hidden_dim, weights)


    def forward(self, src):
        #src = data.get('imu')  # Shape N x S x C with S = sequence length, N = batch size, C = channels

        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = self.input_proj(src.transpose(1, 2)).permute(2, 0, 1)
        #print(src.size())
        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        if self.encode_position:
            src += self.position_embed

        # Transformer Encoder pass
        target = self.transformer_encoder(src)[0]
        # Class probability
        #target = self.imu_head(target)
        return self.classifier(target)
    




def main():
    input = torch.zeros((4, 256, 45)).cuda()
    model = Transformer(input_size = 256, input_channel = 45, num_label = 6).cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))

if __name__ == '__main__':
	main()
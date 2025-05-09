import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import math

class resConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, layer_num):
        super(resConv1dBlock, self).__init__()
        self.layer_num = layer_num
        self.conv1 = nn.ModuleList([
            nn.Conv1d(in_channels = in_channels, out_channels = 2 * in_channels, kernel_size = kernel_size, stride = stride, padding = int((kernel_size - 1) / 2) )
            for i in range(layer_num)])

        self.bn1 = nn.ModuleList([
            nn.BatchNorm1d(2 * in_channels)
            for i in range(layer_num)])

        self.conv2 = nn.ModuleList([ 
            nn.Conv1d(in_channels = 2 * in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = int((kernel_size - 1) / 2) )
            for i in range(layer_num)])

        self.bn2 = nn.ModuleList([
            nn.BatchNorm1d(out_channels)
            for i in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            tmp = F.relu(self.bn1[i](self.conv1[i](x)))
            x = F.relu(self.bn2[i](self.conv2[i](tmp)) + x)
        return x


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


class ResNet(nn.Module):
    def __init__(self, window_size, input_channel, weights = None):
        super(ResNet, self).__init__()
        base_hidden = 32
        self.conv1 = nn.Conv1d(input_channel, base_hidden, kernel_size = 1, stride = 1)
        self.res1 = resConv1dBlock(base_hidden, base_hidden, kernel_size = 3, stride = 1, layer_num = 2)
        self.pool1 = nn.AvgPool1d(kernel_size = 2)

        self.conv2 = nn.Conv1d(base_hidden, 2 * base_hidden, kernel_size = 1, stride = 1)
        self.res2 = resConv1dBlock(2 *base_hidden, 2 * base_hidden, kernel_size = 3, stride = 1, layer_num = 2)
        self.pool2 = nn.AvgPool1d(kernel_size = 2)

        self.conv3 = nn.Conv1d(2 * base_hidden, 4 * base_hidden, kernel_size = 1, stride = 1)
        self.res3 = resConv1dBlock(4 * base_hidden,4 * base_hidden,  kernel_size = 3, stride = 1, layer_num = 2)
        self.pool3 = nn.AvgPool1d(kernel_size = 2)

        self.conv4 = nn.Conv1d(4 * base_hidden, 8 * base_hidden, kernel_size = 1, stride = 1)
        self.res4 = resConv1dBlock(8 * base_hidden, 8 * base_hidden, kernel_size = 3, stride = 1, layer_num = 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        hidden_dim = 8 * base_hidden
        #self.dp = nn.Dropout(p = 0.3)

        self.classifier = MatchingNet(hidden_dim, weights)

    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool1(self.res1(x))
        x = F.relu(self.conv2(x))
        x = self.pool2(self.res2(x))
        x = F.relu(self.conv3(x))
        x = self.pool3(self.res3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(self.res4(x))

        x = x.view(x.size(0), -1)
        #x = self.dp(x)
        prob = self.classifier(x)
        return prob



def main():
    input = torch.zeros((4, 256, 45)).cuda()
    model = ResNet( input_size = 256, input_channel = 45, num_label = 6).cuda()
    # h = model.init_hidden(4)
    # model.apply(init_weights)
    # o = model(input, h)
    # print(o.size())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))

if __name__ == '__main__':
	main()
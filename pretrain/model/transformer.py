import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import warnings

inward_ori_index_no_hand = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19)]

inward_ori_index_no_hand_placeholder = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (1, 22), (2, 22), (3, 22), (4, 22), (5, 22), (6, 22),
                    (7, 22), (8, 22), (9, 22), (10, 22), (11, 22), (12, 22), (13, 22), (14, 22),
                    (15, 22), (16, 22), (17, 22), (18, 22), (19, 22), (20, 22), (21, 22)]

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class Graph:
    def __init__(self, labeling_mode='spatial', inward_ori = inward_ori_index_no_hand):
        self.num_node = len(inward_ori) + 1
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = [(i - 1, j - 1) for (i, j) in inward_ori]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = self.get_adjacency_matrix(labeling_mode)


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = self.get_spatial_graph()
        else:
            raise ValueError()
        return A

    def get_spatial_graph(self):
        I = self.edge2mat(self.self_link)
        In = self.normalize_digraph(self.edge2mat(self.inward))
        Out = self.normalize_digraph(self.edge2mat(self.outward))
        A = np.stack((I, In, Out))
        return A

    def edge2mat(self, link):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in link:
            A[j, i] = 1
        return A


    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        h, w = A.shape
        Dn = np.zeros((w, w))
        for i in range(w):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD

graph = Graph()    

class transformer_layer(nn.Module):
    def __init__(self, num_nodes, num_channels, num_layers = 1):
        super(transformer_layer, self).__init__()
        encoder = nn.TransformerEncoderLayer(d_model = num_nodes * num_channels, nhead = num_nodes, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder, num_layers)

    def forward(self, x):
        # x: B, T, V, H
        B, T, V, H = x.size()
        x = x.reshape(B, T, V * H)
        x = self.encoder(x)
        x = x.reshape(B, T, V, H)
        return x



class gcn_layer(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, num_subset=3):
        super(gcn_layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
                                      3, 1, num_point, num_point]), dtype=torch.float32, requires_grad=True).repeat(1, groups, 1, 1), requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * num_subset)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
                              3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'), requires_grad=False)
        self.DropS = DropBlock_Ske(num_point)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(
            in_channels, out_channels * num_subset, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * num_subset)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * num_subset, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant_(self.Linear_bias, 1e-6)

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(num_point))
        self.eyes = nn.Parameter(torch.stack(
            eye_array).clone().detach().cuda(), requires_grad=False)  # [c,25,25]

    def norm(self, A):
        b, c, h, w = A.size()
        A = A.view(c, self.num_point, self.num_point)
        D_list = torch.sum(A, 1).view(c, 1, self.num_point)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0, keep_prob = 1.0):
        # x0: B, T, V, H
        x0 = x0.permute(0, 3, 1, 2)
        # print(x0.size())
        # print(self.A.size())
        x0 = self.DropS(x0, keep_prob, self.A) #It is essentially just masking
        learn_A = self.DecoupleA.repeat(
            1, self.out_channels // self.groups, 1, 1)
        norm_learn_A = torch.cat([self.norm(learn_A[0:1, ...]), self.norm(
            learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])], 0)

        x = torch.einsum(
            'nctw,cd->ndtw', (x0, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias
        x = self.bn0(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum('nkctv,kcvw->nctw', (x, norm_learn_A))

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        x = x.permute(0, 2, 3, 1)
        return x


class node_inv_conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(node_inv_conv, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: B, T, V, H
        B, T, V, H = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * V, H, T)
        x = self.bn(self.relu(self.conv(x)))
        x = x.reshape(B, V, x.size(1), -1)
        x = x.permute(0, 3, 1, 2)
        return x


class GraphTransMix(nn.Module):
    def __init__(self, window_size, num_nodes, in_channel, A):
        super(GraphTransMix, self).__init__()
        cur_window = window_size
        self.dgcn1 = gcn_layer(in_channel, 64, A, groups = 8, num_point = num_nodes)
        self.conv1 = node_inv_conv(64, 64, 3, 2, 1)
        cur_window = (cur_window - 1) // 2 + 1 # 75
        self.trans1 = transformer_layer(num_nodes, 64)
        # self.conv12 = node_inv_conv(64, 64, 3, 1, 1)
        # self.trans12 = transformer_layer(num_nodes, 64)

        self.dgcn2 = gcn_layer(64, 128, A, groups = 8, num_point = num_nodes)
        self.conv2 = node_inv_conv(128, 128, 3, 2, 1)
        cur_window = (cur_window - 1) // 2 + 1 # 38
        self.trans2 = transformer_layer(num_nodes, 128)
        # self.conv22 = node_inv_conv(128, 128, 3, 1, 1)
        # self.trans22 = transformer_layer(num_nodes, 128)

        self.dgcn3 = gcn_layer(128, 256, A, groups = 16, num_point = num_nodes)
        self.conv3 = node_inv_conv(256, 256, 3, 2, 1)
        cur_window = (cur_window - 1) // 2 + 1 # 19
        self.trans3 = transformer_layer(num_nodes, 256)
        # self.conv32 = node_inv_conv(256, 256, 3, 1, 1)
        # self.trans32 = transformer_layer(num_nodes, 256)

        self.dgcn4 = gcn_layer(256, 256, A, groups = 16, num_point = num_nodes)
        self.conv4 = node_inv_conv(256, 256, 3, 2, 1)
        cur_window = (cur_window - 1) // 2 + 1 # 10
        self.trans4 = transformer_layer(num_nodes, 256)
        # self.conv42 = node_inv_conv(256, 256, 3, 1, 1)
        # self.trans42 = transformer_layer(num_nodes, 256)

    def forward(self, x, prob = 1.0):
        x = self.dgcn1(x, prob)
        x = self.conv1(x)
        x = self.trans1(x)
        # x = self.conv12(x)
        # x = self.trans12(x)

        x = self.dgcn2(x, 1.0)
        x = self.conv2(x)
        x = self.trans2(x)
        # x = self.conv22(x)
        # x = self.trans22(x)

        x = self.dgcn3(x, 1.0)
        x = self.conv3(x)
        x = self.trans3(x)
        # x = self.conv32(x)
        # x = self.trans32(x)

        x = self.dgcn4(x, 1.0)
        x = self.conv4(x)
        x = self.trans4(x)
        # x = self.conv42(x)
        # x = self.trans42(x)
        #x = F.normalize(x, dim=-1)
        return x

class GraphTrans(nn.Module):
    def __init__(self, window_size, num_nodes, in_channel, A):
        super(GraphTrans, self).__init__()
        self.in_conv = node_inv_conv(in_channel, 64, 3, 1, 1)

        self.scale_1 = gcn_layer(64, 64, A, groups=8, num_point=num_nodes)
        cur_window = window_size
        # 150

        self.scale_2 = nn.Sequential(
        node_inv_conv(64, 64, 3, 2, 1),
        gcn_layer(64, 64, A, groups=8, num_point=num_nodes)
        )
        # 75

        self.scale_3 = nn.Sequential(
        node_inv_conv(64, 64, 3, 2, 1),
        node_inv_conv(64, 64, 3, 2, 1),
        gcn_layer(64, 64, A, groups=8, num_point=num_nodes)
        )
        # 38

        self.scale_4 = nn.Sequential(
        node_inv_conv(64, 64, 3, 2, 1),
        node_inv_conv(64, 64, 3, 2, 1),
        node_inv_conv(64, 64, 3, 2, 1),
        gcn_layer(64, 64, A, groups=8, num_point=num_nodes)
        )
        # 19

        self.transformers = nn.Sequential(
            transformer_layer(num_nodes, 256),
            transformer_layer(num_nodes, 256),
            transformer_layer(num_nodes, 256),
            transformer_layer(num_nodes, 256),
            transformer_layer(num_nodes, 256),
            transformer_layer(num_nodes, 256),
        )

    def forward(self, x):
        # N, T, V, H
        x = self.in_conv(x)
        x1 = self.scale_1(x)
        x2 = self.scale_2(x)
        x3 = self.scale_3(x)
        x4 = self.scale_4(x)

        x4 = torch.cat([x4, x4.clone()], dim = 1)
        x3 = torch.cat([x3, x4], dim = -1)
        x3 = torch.cat([x3, x3.clone()[:, :-1] ], dim = 1)
        x2 = torch.cat([x2, x3], dim = -1)
        x2 = torch.cat([x2, x2.clone()], dim = 1)
        x = torch.cat([x1, x2], dim = -1)

        x = self.transformers(x)
        return x[:, 0].mean(1)

class DropBlock_Ske(nn.Module):
    def __init__(self, num_point):
        super(DropBlock_Ske, self).__init__()
        self.keep_prob = 0.0
        self.num_point = num_point

    def forward(self, input, keep_prob, A):  # n,c,t,v
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.size()

        input_abs = torch.mean(torch.mean(
            torch.abs(input), dim=2), dim=1).detach()
        input_abs = input_abs / torch.sum(input_abs) * input_abs.numel()
        if self.num_point == 25:  # Kinect V2
            gamma = (1. - self.keep_prob) / (1 + 1.92)
        elif self.num_point == 20:  # Kinect V1
            gamma = (1. - self.keep_prob) / (1 + 1.9)
        else:
            gamma = (1. - self.keep_prob) / (1 + 1.92)
            warnings.warn('undefined skeleton graph')
        input_abs = input_abs.unsqueeze(1).unsqueeze(1)
        input_abs = input_abs.expand((n, 1, t, v))
        M_seed = torch.bernoulli(torch.clamp(
            input_abs * gamma, max=1.0)).to(device=input.device, dtype=input.dtype)
        M = torch.einsum('nctv,vk->nctk', M_seed, A)
        #M = torch.matmul(M_seed, A)
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0
        mask = (1 - M).view(n, 1, t, self.num_point)
        return input * mask * mask.numel() / mask.sum()


class SimpCLS(nn.Module):
    def __init__(self, num_class = 120):
        super(SimpCLS, self).__init__()
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x):
        return self.fc(x)

class AngleDecoder(nn.Module):
    def __init__(self, num_chunks, window_size):
        super(AngleDecoder, self).__init__()
        self.conv0 = nn.ConvTranspose2d(256, 256, (3, 3), stride=(1, 1), padding = (1, 2))
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.ConvTranspose1d(256, 256, 3, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.ConvTranspose1d(256, 256, 3, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.ConvTranspose1d(256, 256, 3, stride = 2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.ConvTranspose1d(256, 256, 3, stride=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.trans5 = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)

        self.classifier1 = nn.Linear(256, num_chunks)
        self.classifier2 = nn.Linear(256, num_chunks)
        self.classifier3 = nn.Linear(256, num_chunks)
        self.window_size = window_size


    def forward(self, x):
        # N, T, V, H
        B = x.size(0)
        x = torch.permute(x, (0, 3, 1, 2))
        x = F.relu(self.bn0(self.conv0(x)))
        x = x.squeeze(-1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = x[:, :, :self.window_size]
        x = torch.permute(x, (0, 2, 1))
        x = torch.reshape(x, (-1, 256))
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x3 = self.classifier3(x)
        return x1, x2, x3

class TimeDecoder(nn.Module):
    def __init__(self, window_size):
        super(TimeDecoder, self).__init__()
        self.conv0 = nn.ConvTranspose1d(256, 256, 3, stride= 1, padding = 1)
        self.bn0 = nn.BatchNorm1d(256)
        self.conv1 = nn.ConvTranspose1d(256, 256, 3, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.ConvTranspose1d(256, 256, 3, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.ConvTranspose1d(256, 256, 3, stride = 2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.ConvTranspose1d(256, 256, 3, stride=2)
        self.bn4 = nn.BatchNorm1d(256)

        self.linear = nn.Linear(256, 3)
        self.window_size = window_size


    def forward(self, x):
        # B, T, V, H
        B, V = x.size(0), x.size(2)
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.reshape((-1, x.size(2), x.size(3)))

        x = F.relu(self.bn0(self.conv0(x)))
        x = x.squeeze(-1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = x[:, :, :self.window_size]
        x = torch.reshape(x, (B, V, -1, self.window_size))
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.linear(x)
        return x

def main():

    model = TimeDecoder(16, window_size= 150).cuda()
    x = torch.ones((2, 10, 21, 256)).cuda()
    o = model(x)
    print(o.size())

if __name__ == '__main__':
    main()
        


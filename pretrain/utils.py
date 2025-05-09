import numpy as np
import time
from datetime import datetime, date
import os
import torch
import random

inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]

inward_ori_index_no_hand = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19)]

graph_angle_index = [(23, 11, 10), (11, 10, 9), (10, 9, 8), (9, 8, 20),
               (21, 7, 6), (7, 6, 5), (6, 5, 4), (5, 4, 20),
               (19, 18, 17), (18, 17, 16), (15, 14, 13), (14, 13, 12)]

graph_angle_index_no_hand = [(11, 10, 9), (10, 9, 8), (9, 8, 20),
               (7, 6, 5), (6, 5, 4), (5, 4, 20),
               (19, 18, 17), (18, 17, 16), (15, 14, 13), (14, 13, 12)]


class Graph:
    def __init__(self, labeling_mode='spatial', inward_ori = inward_ori_index, num_node = 21):
        self.num_node = num_node
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


def logging(file):
    def write_log(s, printing = True):
        if printing:
            print(s)
        with open(file, 'a') as f:
            f.write(s+'\n')
    return write_log

def set_up_logging(args):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    log_path = os.path.join(args.log_path, f'{args.decoder}_{args.load_ep+1}({args.ep})_{date.today()}.txt')
    print(log_path)
    log = logging(log_path)
    log(f'Init experiment at time {datetime.now().strftime("%H:%M:%S")}')
    for k, v in vars(args).items():
        log("%s:\t%s" % (str(k), str(v)))
    return log

def save_model(model, step, args):
    task_path = f'/home/shuheng/data/checkpoints/{args.task}_{args.model}_{date.today()}'
    if not os.path.exists(task_path):
        os.mkdir(task_path)

    save_path = os.path.join(task_path, f'{model.__class__.__name__}_{step}.pt')
    torch.save(model.state_dict(), save_path)

def load_opt(model, args):
    if args.opt == 'SGD':
        return load_SGD(model, args)
    if args.opt == 'Adam':
        return load_Adam(model, args)

def load_SGD(model, args):
    params_dict = dict(model.named_parameters())
    params = []

    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0

        lr_mult = 1.0
        weight_decay = 1e-5

        params += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult,
                    'decay_mult': decay_mult, 'weight_decay': weight_decay}]

    optimizer = torch.optim.SGD(
        params,
        momentum=0.9)
    return optimizer


def load_Adam(model, args):
    params_dict = dict(model.named_parameters())
    params = []

    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0

        lr_mult = 1.0
        weight_decay = 1e-5

        params += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult,
                    'decay_mult': decay_mult, 'weight_decay': weight_decay}]

    optimizer = torch.optim.AdamW(
        params)
    return optimizer

def adjust_lr(optim, lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr
    return lr

def get_prob(ep):
    if ep < 200:
        prob = 1
    elif ep < 400:
        prob = 0.95
    elif ep < 600:
        prob = 0.9
    else:
        prob = 0.850
    return prob

def sample_node(feats, angles, args):
    i = random.randint(0, len(graph_angle_index_no_hand) - 1)
    l, a, r= graph_angle_index_no_hand[i]
    l_feat = feats[:, :, l, :]
    a_feat = feats[:, :, a, :]
    r_feat = feats[:, :, r, :]

    angles = angles[:, :, i, :] # N, T, H

    return torch.stack([l_feat, a_feat, r_feat], dim=2), angles


import numpy as np
import random
import torch
import argparse
import os
import copy
import time
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import *
from datetime import datetime
from torch.utils.data.dataset import Dataset


class PretrainDataset(Dataset):
    def __init__(self, x, args):
        self.x = x.astype(np.float32)
        means = np.mean(self.x, (0, 1, 2), keepdims=True)
        stds = np.std(self.x, (0, 1, 2), keepdims=True)
        self.tgt = (self.x - means) / stds
        
        ## Load Angle Data ##
        angle_path = os.path.join(args.data_path, 'cached/angles.npy')
        if os.path.exists(angle_path):
            self.angles = np.load(angle_path)
        else:
            self.angles = calculate_angles(x, graph_angle_index_no_hand).astype(np.int64)
            np.save(angle_path, self.angles)

        self.num_chunks = np.max(self.angles)
        self.angles = (self.angles + self.num_chunks) % self.num_chunks
        print(self.x.shape)

    def __getitem__(self, index):
        return self.x[index], self.angles[index], self.tgt[index]
    
    def __len__(self):
        return len(self.x)

def load_NTURGBD(args):
    # x: [N, T, V, H]
    # T: # timestamps
    # V: # nodes
    # H: # channel
    data_path = os.path.join(args.data_path, 'NTURGBD')
    xtrain = np.load(os.path.join(data_path, 'X_standard_train.npy'))
    ytrain = np.load(os.path.join(data_path, 'Y_standard_train.npy'))
    xtest = np.load(os.path.join(data_path, 'X_standard_test.npy'))
    ytest = np.load(os.path.join(data_path, 'Y_standard_test.npy'))
    return xtrain, ytrain, xtest, ytest


def load_Human3D():
    if os.path.exists("/data1/shuheng"):
        data_path = os.path.join('/data1/shuheng', 'HumanML3D')
    elif os.path.exists("/home/shuheng/data"):
        data_path = os.path.join('/home/shuheng/data', 'HumanML3D')
    x = np.load(os.path.join(data_path, 'HumanML3D.npy'))
    if np.count_nonzero(np.isnan(x)):
        new_x = []
        for i in range(len(x)):
            if not np.count_nonzero(np.isnan(x[i])):
                new_x.append(x[i].tolist())
            else:
                print(i)
        x = np.array(new_x)
        np.save(os.path.join(data_path, 'HumanML3D.npy'), x)
    return x


def load_pretrain_data(args):
    xtrain, _, xtest, _ = load_NTURGBD(args)
    xtrain = np.concatenate((xtrain, xtest), axis = 0)
    xtrain = xtrain[:, :, :21, :]
    x = load_Human3D()
    xtrain = np.concatenate((xtrain, x), axis=0)
    return xtrain



def calculate_angles(x, angle_index = graph_angle_index_no_hand):
    angles = np.zeros((x.shape[0], x.shape[1], len(angle_index), 3))
    for i in tqdm(range(len(angle_index))):
        l, a, r = angle_index[i]

        e0 = x[:, :, l, :] - x[:, :, a, :]
        e1 = x[:, :, r, :] - x[:, :, a, :]
        c = np.cross(e0, e1)

        theta = np.stack([np.arctan2(c[:, :, 0], c[:, :, 1]), np.arctan2(c[:, :, 1], c[:, :, 2]), np.arctan2(c[:, :, 2], c[:, :, 0])], axis = 2)
        
        angles[:, :, i, :] = np.rint(theta * 15 / np.pi)
        #angles[:, :, i, :] = theta  / np.pi

    return angles
import os
import csv
import yaml
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import random
from model import *
import torch
import torch.nn as nn


class BaseDataset(Dataset):
    def __init__(self, x, y):
        self.x = np.array(x).astype(np.float32)
        self.labels = y

    def __getitem__(self, index):
        return self.x[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
    
def read_weights(args):
    path = os.path.join('../weights/', args.labelweight + '.pt')
    args.log(path)
    weights = torch.load(path)
    return weights

def get_model(args):
    if args.model == 'ResNet' or 'resnet':
        model = ResNet(args.window_size, args.input_channel, args.weights).to(args.device)
    if args.model == 'transformer' or 'Transformer':
        model = IMUTransformerEncoder(args.window_size, args.input_channel, args.weights, args.dropout).to(args.device)
    return model


def read_data(args):
    path = os.path.join(args.data_path, f'./har_downstream/{args.dataset}')
    xtrain = np.load(os.path.join(path, 'x_train.npy')).astype('float32')#.tolist()
    ytrain = np.load(os.path.join(path, 'y_train.npy')).astype('int64')#.tolist()
    xtest = np.load(os.path.join(path, 'x_test.npy')).astype('float32')#.tolist()
    ytest = np.load(os.path.join(path, 'y_test.npy')).astype('int64')#.tolist()

    idx, cnt = np.unique(ytrain, return_counts=True)
    summary_train = dict(zip(idx, cnt))
    summary_train = dict(sorted(summary_train.items()))
    args.log(f"Train Label num cnt: {str(summary_train)}")
    idx, cnt = np.unique(ytest, return_counts=True)
    summary_test = dict(zip(idx, cnt))
    summary_test = dict(sorted(summary_test.items()))
    args.log(f"Test Label num cnt: {str(summary_test)}")

    args.log(f"The size of training data is {xtrain.shape}") # B * T * N
    args.input_channel = xtrain.shape[2]
    args.window_size = xtrain.shape[1]
    args.num_labels = max(ytrain) + 1
    return xtrain, ytrain, xtest, ytest

def split_valid(args, xtrain, ytrain):
    np.random.seed(args.seed)
    np.random.shuffle(xtrain)
    np.random.seed(args.seed)
    np.random.shuffle(ytrain)
    lens = len(xtrain)*9//10
    xvalid = xtrain[lens:]
    yvalid = ytrain[lens:]
    xtrain = xtrain[:lens]
    ytrain = ytrain[:lens]
    return xtrain, ytrain, xvalid, yvalid
     

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_config(path):
    return AttrDict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))

def logging(file):
    def write_log(s, printing = True):
        if printing:
            print(s)
        with open(file, 'a') as f:
            f.write(s+'\n')
    return write_log

def set_up_logging(args):
    log = logging(os.path.join(args.log_path, args.model+'.txt'))
    log(str(args))
    return log


import argparse
import os
import time
import torch
import json
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from utils import *

from sklearn.metrics import recall_score, f1_score, accuracy_score

dataset_mapping = {
    'opportunity_lc': 'OPP',
    'pamap2': 'PMP',
}

def parse_args():
    parser = argparse.ArgumentParser(description='task4--few_shot')
    parser.add_argument('--dataset', default = 'easy_imu_all', type = str,
                        choices=['hard_imu_phone', 'easy_imu_phone', 'hard_imu_all', 'easy_imu_all', 'fs_hard_phone', 'wifi', 'wifi_med', '5fs_hard_phone'])
    parser.add_argument('--model', default='transformer', type=str,
                        choices=['ResNet', 'transformer'])
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--n_gpu', default=0, type =int)
    parser.add_argument('--shots', default = 5, type = int)

    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)

    parser.add_argument('--opt', default='adam', type=str)
    
    parser.add_argument('--epochs', default = 200, type = int)
    parser.add_argument('--lr', default = 1e-4, type = float)
    parser.add_argument('--batch_size', default = 16, type = int)

    parser.add_argument('--labelweight', default = 'few_masd_weight', type = str)
    args = parser.parse_args()

    with open('../config.json', 'r') as f:
        config = json.load(f)
        args.data_path = config['data_path']
        args.log_path = config['log_path']


    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    torch.cuda.set_device(args.n_gpu)
    return args

args = parse_args()
log = set_up_logging(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.log = log


def test(model, TestLoader):
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for i, (x, y) in enumerate(TestLoader):
            x = x.to(device)
            y_true += y.tolist()
            prob = model(x)

            pred = torch.argmax(prob, dim = -1)
            y_pred += pred.cpu().tolist()

    acc = accuracy_score(y_true, y_pred)
    return acc


def main():
    log("Start time:" + time.asctime( time.localtime(time.time())) )

    xtrain, ytrain, xtest, ytest = read_data(args)

    weights = read_weights(args)
    weights = weights / torch.norm(weights, dim = 0, p = 2).unsqueeze(0)
    args.weights = weights
    model = get_model(args)

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_params = sum(p.numel() for p in model.parameters())
    log('Total parameters: ' + str(total_params))

    xtrain, ytrain, _, _ = split_valid(args, xtrain, ytrain) 
    xtrain, ytrain = few_shot_data(xtrain, ytrain, args)


    TrainDataset = BaseDataset(xtrain, ytrain)
    TrainLoader = DataLoader(TrainDataset, batch_size = args.batch_size, shuffle = True)
    TestDataset = BaseDataset(xtest, ytest)
    TestLoader = DataLoader(TestDataset, batch_size=args.batch_size, shuffle=False)
    loss_func = nn.CrossEntropyLoss()
    
    best_loss = 0xfff

    pbar = tqdm(range(args.epochs))
    for ep in pbar:
        model.train()
        epoch_loss = 0
        for i, (x, y) in enumerate(TrainLoader):
            x = x.to(device)
            y = y.to(device)

            prob = model(x)
            loss = loss_func(prob, y)
            epoch_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            return_acc = test(model, TestLoader)
            log(f"{ep} / {args.epochs} Training loss : {epoch_loss / i} Testing acc now is :{return_acc:.4f}", False)
            pbar.set_description(f"Training loss : {epoch_loss / i} Testing acc now is :{return_acc:.4f}")

    return return_acc


if __name__ == '__main__':
    accs = []
    for seed in range(5):
        args.seed = seed
        acc = main()
        accs.append(acc)
        log(f"Mean acc is {np.mean(accs):.4f}, std is {np.std(accs):.4f}")

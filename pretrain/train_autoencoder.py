import numpy as np
import random
import torch
import json
import argparse
import os
import copy
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score, accuracy_score
from tqdm import tqdm
from model.transformer import *
from model.loss import *
from utils import *
from data import *


def parse_args():
    parser = argparse.ArgumentParser(description='train skeleton representation')
    parser.add_argument('--decoder', default='angle', type=str, choices=['angle', "time"])
    parser.add_argument('--batch_size', default = 64, type = int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--opt', default='SGD', type=str, choices=['SGD', 'Adam'])
    parser.add_argument('--ep', default=1000, type=int)
    # misc
    parser.add_argument('--n_gpu', default = 0, type = int)
    parser.add_argument('--load_ep', default=-1, type=int)
    parser.add_argument('--ckpt_name', default = '205' , type = str)

    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.n_gpu)
    
    with open('./config.json', 'r') as f:
        config = json.load(f)
        args.data_path = config['data_path']
        args.log_path = config['log_path']


    args.ckpt_path = os.path.join(args.data_path, 'checkpoint')
    args.ckpt_name = f'{args.ckpt_name}_{args.decoder}'
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    
    return args

args = parse_args()
log = set_up_logging(args)
args.log = log
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    xtrain = load_pretrain_data(args)
    # x: [N, T, V, H]
    # T: # timestamps
    # V: # nodes
    # H: # channel
    train_dataset = PretrainDataset(xtrain, args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    skeleton_graph = Graph(inward_ori= inward_ori_index_no_hand, num_node=21)
    
    Encoder = GraphTransMix(xtrain.shape[1], skeleton_graph.A.shape[1], xtrain.shape[3], skeleton_graph.A).to(device)

    if args.decoder == 'angle':
        Decoder = AngleDecoder(int(train_dataset.num_chunks), xtrain.shape[1]).to(device)
    elif args.decoder == 'time':
        Decoder = TimeDecoder(xtrain.shape[1]).to(device)

    if args.load_ep != -1:
        log(f'Loading checkpoint from {args.ckpt_name} of epoch {args.load_ep}')
        save_path = os.path.join(args.ckpt_path, args.ckpt_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        enc_name = f'{args.load_ep}_enc.pt'
        dec_name = f'{args.load_ep}_dec.pt'
        enc_name = os.path.join(save_path, enc_name)
        dec_name = os.path.join(save_path, dec_name)

        Encoder.load_state_dict(torch.load(enc_name, map_location=device))
        Decoder.load_state_dict(torch.load(dec_name, map_location=device))


    start_ep = args.load_ep + 1

    if args.opt == 'SGD':
        enc_opt = load_SGD(Encoder, args)
        dec_opt = load_SGD(Decoder, args)
    elif args.opt == 'Adam':
        enc_opt = load_Adam(Encoder, args)
        dec_opt = load_Adam(Decoder, args)

    xe_lossfn = nn.CrossEntropyLoss()
    mse_lossfn = nn.MSELoss()

    step = 0
    pbar = tqdm(range(start_ep, args.ep, 1))
    for ep in pbar:
        prob = get_prob(ep)
        ep_loss = 0
        for i, (x, angle, tgt) in enumerate(train_loader):
            x, angle, tgt = x.to(device), angle.to(device), tgt.to(device)
            
            feat = Encoder(x, prob)

            if 'angle' in args.decoder:
                feat, ang = sample_node(feat, angle, args)
                x1, x2, x3 = Decoder(feat)
                ang = torch.reshape(ang, (-1, 3))
                y1, y2, y3 = ang[:, 0], ang[:, 1], ang[:, 2]
                loss = xe_lossfn(x1, y1) + xe_lossfn(x2, y2) + xe_lossfn(x3, y3)

            if 'time' in args.decoder:
                out = Decoder(feat)[:, :, :21, :]
                loss = torch.sqrt(mse_lossfn(out, tgt) + 1e-6)

            ep_loss += loss
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(Encoder.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(Decoder.parameters(), 1)

            enc_opt.step()
            dec_opt.step()
            step += 1


        pbar.set_description(f'Epoch {ep + 1}/{args.ep}, step: {step}, loss: {ep_loss / i :.4f}')
        log(f'Epoch {ep + 1}/{args.ep}, step: {step}, loss: {ep_loss / 100 :.4f}', False)

        if ep % 50 == 49:
            save_path = os.path.join(args.ckpt_path, args.ckpt_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            enc_name = f'{ep}_enc.pt'
            dec_name = f'{ep}_dec.pt'
            enc_name = os.path.join(save_path, enc_name)
            dec_name = os.path.join(save_path, dec_name)

            torch.save(Encoder.state_dict(), enc_name)
            torch.save(Decoder.state_dict(), dec_name)


if __name__ == '__main__':
    main()

import torch
import argparse
import os
import numpy as np
from model.transformer import *
from utils import *


DATA_PATH = '../few_shot_skeleton'
x_path = os.path.join(DATA_PATH, 'x.npy')
y_path = os.path.join(DATA_PATH, 'y.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def parse_args():
    parser = argparse.ArgumentParser(description='generate label representations')
    parser.add_argument('--n_gpu', default = 0, type = int)
    parser.add_argument('--num_shots', default = 5, type = int)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.n_gpu)
    return args

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    x = np.load(x_path)
    y = np.load(y_path)

    np.random.seed(4)
    np.random.shuffle(x)
    np.random.seed(4)
    np.random.shuffle(y)

    graph = Graph(inward_ori = inward_ori_index_no_hand)
    Encoder = GraphTransMix(150, 21, 3, graph.A).to(device) 
    Encoder_Path = '/home/shuheng/data/checkpoint/205_mix_angle_MAE/999_enc.pt'
    Encoder.load_state_dict(torch.load(Encoder_Path, map_location=device))

    num_shots = args.num_shots
    num_class = 27

    cnt_y = [0 for _ in range(num_class)]
    embeddings = [0 for _ in range(num_class)]
    ys = []
    Encoder.eval()
    with ((torch.no_grad())):
        for i in range(len(x)):
            if cnt_y[y[i]] < num_shots:
                cnt_y[y[i]] += 1
                X = x[i]
                X = torch.tensor(X).float().unsqueeze(0)
                X = X.view(X.size(0), X.size(1), -1, 3).to(device)
                out = Encoder(X, prob = 1.0).mean(1).squeeze()
                
                embeddings[y[i]] += out.cpu() / num_shots
                ys.append(y[i])
            if np.sum(cnt_y) == num_class * num_shots:
                print('Break')
                break

    embeddings = torch.stack(embeddings, dim = 0)
    print(embeddings.size())
    torch.save(embeddings, 'MASD_205PT999User9AVG5.pt')




if __name__ == '__main__':
    main()

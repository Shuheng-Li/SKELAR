import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y) + self.eps)

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.fn = nn.MarginRankingLoss(margin)

    def forward(self, feats):
        randind = torch.randint(0, 10, (feats.size(0),)).to(feats.device)
        randnode = torch.randint(0, 21, (feats.size(0) // 3,)).to(feats.device).repeat(3)
        feats = feats[torch.arange(feats.size(0)).to(feats.device),randind,randnode]

        N = feats.size(0) // 3
        anc_feat, pos_feat, neg_feat = feats[:N], feats[N:2 * N], feats[2 * N:]
        dist_pos = F.pairwise_distance(anc_feat, pos_feat, 2)
        dist_neg = F.pairwise_distance(anc_feat, neg_feat, 2)

        target = torch.ones(N).to(feats.device)
        target = Variable(target)
        return self.fn(dist_pos, dist_neg, target)

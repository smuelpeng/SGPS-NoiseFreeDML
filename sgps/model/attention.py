import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionSoftmax(nn.Module):
    def __init__(self, cfg):
        super(AttentionSoftmax, self).__init__()
        self.n_support = cfg.NF.N_SUPPORT
        self.hidden_dim = cfg.feature_dim
        self.T = cfg.NF.ATTEN_T

    def forward(self, feat_sub):
        # feat_sub bz x n_support x dim
        correlation = torch.bmm(feat_sub, feat_sub.permute(0, 2, 1) )  # [N,S,S]
        attn = (torch.sum(correlation, dim=1) - 1) / (self.n_support)
        weights = F.softmax(attn / self.T, dim=1).unsqueeze(dim=2)
        return weights

class AttentionMax(nn.Module):
    def __init__(self, cfg):
        super(AttentionMax, self).__init__()
        self.n_support = cfg.NF.N_SUPPORT
        self.hidden_dim = cfg.feature_dim
        self.T = cfg.NF.ATTEN_T

    def forward(self, feat_query, feat_sub):
        # feat_sub bz x n_support x dim
        bz = feat_sub.shape[0]
        # correlation = torch.bmm(feat_sub, feat_sub.permute(0, 2, 1) )  # [N,S,S]
        # values, indices = correlation.topk(1, )
        if len(feat_query.shape)==2:
            feat_query = feat_query.unsqueeze(1)

        correlation = torch.bmm(feat_query, feat_sub.permute(0,2,1))
        correlation = correlation.reshape(bz, self.n_support)
        indices = torch.argmax(correlation, dim=1)
        # attn = (torch.sum(correlation, dim=1) - 1) / (self.n_support)
        # weights = F.softmax(attn / self.T, dim=1).unsqueeze(dim=2)
        weights = torch.zeros((bz, self.n_support)).cuda()
        # import pdb;pdb.set_trace()
        for i in range(bz):
            weights[i, indices[i]] = 1
        weights = weights.unsqueeze(dim=2)
        return weights


class AttentionMean(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_support = cfg.NF.N_SUPPORT
        self.hidden_dim = cfg.feature_dim
        self.T = cfg.NF.ATTEN_T

    def forward(self, feat_sub):
        bz = feat_sub.shape[0]
        weights = torch.ones((bz, self.n_support)).cuda()
        weights = weights.unsqueeze(dim=2)
        weights = weights * 1.0 / self.n_support
        return weights

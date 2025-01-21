import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class SwitchLoss(nn.Module):
    def __init__(self, cfg):
        super(SwitchLoss, self).__init__()
        # self.scale = cfg.RSPP.SCALE
        self.scale = cfg.NF.SW_SCALE
        self.margin = cfg.NF.SW_MARGIN
        self.feature_dim = cfg.MODEL.HEAD.DIM
        self.cfg = cfg

    def forward(self,
                feat_q,
                feat_sub,
                attention_mask=None,
                mask_neg=None,
                keep_bool=None,
                extra_feat=None,
                extra_targets=None,
                xbm_feats=None
                ):
        """
        feat_q: bz x dim
        feat_sub: bz x IA_SUB_NUM x dim
        attention_mask
        """
        # if len(attention_mask.shape) == 3:
        if self.cfg.MODEL.ATTENTION_MODULE in ['trans_proto', ]:
            attention_mask = attention_mask.squeeze(1)
            pos_scores = F.cosine_similarity(feat_q, attention_mask)  # N
        else:
            feat_sub_proto = torch.sum(
                feat_sub * attention_mask, dim=1)  # [N,d]
            pos_scores = F.cosine_similarity(feat_q, feat_sub_proto)  # N

        pos_scores = pos_scores - self.margin
        if xbm_feats is not None:
            neg_scores = torch.mm(feat_q, xbm_feats.reshape(-1, self.feature_dim).T)
        else:
            neg_scores = torch.mm(feat_q, feat_sub.reshape(-1, self.feature_dim).T)
        exp_l_pos = torch.exp(pos_scores * self.scale)
        exp_l_neg = torch.exp(neg_scores * self.scale)
        if torch.isnan(exp_l_pos).any() or torch.isnan(exp_l_neg).any():
            return 0
        exp_l_neg = exp_l_neg * mask_neg
        logits = exp_l_pos / (exp_l_pos + exp_l_neg.sum(1))
        if keep_bool is not None :
            logits_main = logits[~keep_bool]
        else:
            logits_main = logits
        if len(logits_main) == 0:
            loss = 0
        else:
            loss = - torch.log(logits_main).mean()
        return loss

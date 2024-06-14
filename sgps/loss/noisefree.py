from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from dataclasses import dataclass, field
import math


import sgps
from ..utils.config import parse_structured
import torch
from torch import nn
from sgps.model.xbm import XBM
import numpy as np

eps = 1e-5


class NoiseFreeLoss(nn.Module):
    def __init__(self, cfg=None):
        super(NoiseFreeLoss, self).__init__()
        self.cfg = cfg
        self.clean_batch_loss = sgps.find(self.cfg.clean_batch_loss)(cfg)
        self.clean_bank_loss = sgps.find(self.cfg.clean_bank_loss)(cfg)
        self.noise_spp_loss = sgps.find(self.cfg.noise_sgps_loss)(cfg)
        self.attention_module = sgps.find(self.cfg.attention_module)(cfg)

        self.xbm = XBM(cfg)
        self.xbm_all = XBM(cfg)

        self.is_full = False

        self.selected_feats = None
        self.selected_tar = None
        self.selected_indices = None

        self.selected_feats_all = None
        self.selected_tar_all = None
        self.selected_indices_all = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

    def cal_neg_mask(self, targets_q, targets_k):
        '''
        filter pairs with same main_label tb_label bt_label;
        '''
        # if self.cfg.RSPP.TARGET_NUM == 3:
        assert (targets_k.shape[2] == 5) and (targets_q.shape[2] == 5)
        mask_main = targets_q[:, :,
                              0].reshape(-1, 1) != targets_k[:, :, 0].reshape(1, -1)
        mask_bt = targets_q[:, :,
                            1].reshape(-1, 1) != targets_k[:, :, 1].reshape(1, -1)
        mask_tb = targets_q[:, :,
                            3].reshape(-1, 1) != targets_k[:, :, 3].reshape(1, -1)
        mask = mask_main & mask_bt & mask_tb
        return mask

    def cal_neg_mask_bank(self, targets_q, targets_k, pos_indices=None, bank_indices=None):
        '''
        filter pairs with same main_label tb_label bt_label;
        '''
        assert (targets_q.shape[2] == 5)
        mask_main = targets_q[:, :,
                              0].reshape(-1, 1) != targets_k.reshape(1, -1)
        # mask_bt = targets_q[:, :,
        #                     2].reshape(-1, 1) != targets_k.reshape(1, -1)
        # mask_tb = targets_q[:, :,
        #                     4].reshape(-1, 1) != targets_k.reshape(1, -1)

        # mask = mask_main & mask_bt & mask_tb
        mask = mask_main
        if pos_indices is not None and bank_indices is not None:
            for i, pos_indice in enumerate(pos_indices):
                pos_indice = torch.Tensor(pos_indice[0]).cuda()
                mask_i = bank_indices.reshape(-1,
                                              1) == pos_indice.reshape(1, -1)
                mask_i = mask_i.sum(1) == 0
                mask[i] = mask[i] & mask_i
            # mask = mask
        return mask

    def cal_pos_mask(self, targets_q, targets_k):
        '''
        filter pairs with same main_label tb_label bt_label;
        '''
        # if self.cfg.RSPP.TARGET_NUM == 3:
        assert (targets_k.shape[2] == 3) and (targets_q.shape[2] == 3)
        mask_main = targets_q[:, :,
                              0].reshape(-1, 1) == targets_k[:, :, 0].reshape(1, -1)
        mask_bt = targets_q[:, :,
                            1].reshape(-1, 1) == targets_k[:, :, 1].reshape(1, -1)
        mask_tb = targets_q[:, :,
                            2].reshape(-1, 1) == targets_k[:, :, 2].reshape(1, -1)
        # mask = mask_main & mask_bt & mask_tb
        mask = mask_main | mask_bt | mask_tb
        return mask

    def forward(self, input: dict):
        cfg = self.cfg
        feat_q = input['feat_q']
        targets = input['targets']
        indices = input['indices']
        pos_indices = input['pos_indices']

        feat_q = F.normalize(feat_q)
        # Noise Free Step
        group_size = len(feat_q) // self.batch_size
        feat_q_reshape = feat_q.reshape(
            self.batch_size, group_size, self.feature_dim)

        targets = targets.reshape(
            self.batch_size,
            group_size,
            self.cfg.group_num)

        indices = indices.reshape(self.batch_size, group_size)
        feat_query = torch.index_select(feat_q_reshape, 1, torch.Tensor(
            [0]).type(torch.int32).cuda()).squeeze(1)
        feat_sub = torch.index_select(
            feat_q_reshape, 1, torch.arange(1, group_size).cuda())
        query_label = targets[:, 0:1]
        sub_label = targets[:, 1:]

        # PRISM Step
        main_targets = targets[:, :, 0]
        main_targets = main_targets.type(torch.int64)
        main_targets = main_targets.reshape(-1)

        if self.selected_feats is not None:
            # enable backward or xbm will distrupt z
            self.xbm.enqueue_dequeue(
                self.selected_feats, self.selected_tar, self.selected_indices)

        self.selected_feats_all = feat_q.detach()
        self.selected_tar_all = main_targets.detach()
        self.selected_indices_all = indices.reshape(-1).detach()
        self.xbm_all.enqueue_dequeue(
            self.selected_feats_all, self.selected_tar_all, self.selected_indices_all)

        xbm_feats, xbm_targets, xbm_indices = self.xbm.get()
        query_label_target = query_label[:, :, 0]
        query_label_target = query_label_target.type(torch.int64)
        query_label_target = query_label_target.reshape(-1)
        xbm_loss_query, p_in_query = self.clean_bank_loss(feat_query,
                                                          query_label_target,
                                                          xbm_feats,
                                                          xbm_targets,
                                                          update_center=True
                                                          )
        xbm_loss, p_in = self.clean_bank_loss(feat_q,
                                              main_targets,
                                              xbm_feats,
                                              xbm_targets,
                                              )
        selected_feats = feat_q[p_in]
        selected_tar = main_targets[p_in]
        batch_loss = self.clean_batch_loss(selected_feats,
                                           selected_tar,
                                           selected_feats,
                                           selected_tar)

        selected_feats_query = feat_query[p_in_query]
        selected_tar_query = query_label_target[p_in_query]
        selected_indices_query = indices[:, 0][p_in_query]

        batch_loss_query = self.clean_batch_loss(selected_feats_query,
                                                 selected_tar_query,
                                                 selected_feats_query,
                                                 selected_tar_query)

        loss_pr_query = batch_loss_query * self.cfg.lambda_clean_batch + \
            xbm_loss_query * self.cfg.lambda_clean_bank
        loss_pr_all = batch_loss * self.cfg.lambda_clean_batch + \
            xbm_loss * self.cfg.lambda_clean_bank

        self.selected_feats = selected_feats_query.detach()
        self.selected_tar = selected_tar_query.detach()
        self.selected_indices = selected_indices_query.detach()

        # switch loss
        keep_bool = p_in.reshape(self.batch_size, group_size)[:, 0]
        attention_mask = self.attention_module(feat_sub)
        maks_neg_switch = self.cal_neg_mask(query_label, sub_label)
        maks_neg_switch_bank = self.cal_neg_mask_bank(query_label,
                                                      xbm_targets,
                                                      pos_indices,
                                                      xbm_indices
                                                      )
        loss_spp = self.noise_spp_loss(feat_query,
                                       feat_sub,
                                       attention_mask,
                                       maks_neg_switch,
                                       keep_bool
                                       )
        loss_spp_bank = self.noise_spp_loss(feat_query,
                                            feat_sub,
                                            attention_mask,
                                            maks_neg_switch_bank,
                                            keep_bool,
                                            xbm_feats=xbm_feats,
                                            )

        if math.isnan(loss_spp) or math.isnan(loss_spp_bank):
            loss_spp = 0
            loss_spp_bank = 0

        loss_all = loss_pr_query * self.cfg.lambda_clean_query + loss_pr_all * self.cfg.lambda_clean_all + \
            loss_spp * self.cfg.lambda_noise_batch + \
            loss_spp_bank * self.cfg.lambda_noise_bank

        return loss_all

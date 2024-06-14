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

import torch
from torch import nn
from sgps.model.xbm import XBM
import numpy as np

eps = 1e-5
class NoiseFreeLoss(nn.Module):
    @dataclass
    class NoiseFreeLossConfig:
        clean_batch_loss: str = ""
        clean_bank_loss: str = ""
        noise_sgps_loss: str = ""        

    def __init__(self, cfg=None):

        super(NoiseFreeLoss, self).__init__()
        self.cfg = cfg
        # self.clean_batch_loss = LOSS_REGISTRY[cfg.NF.CLEAN_BATCH_LOSS](cfg)
        # self.clean_bank_loss = LOSS_REGISTRY[cfg.NF.CLEAN_BANK_LOSS](cfg)

        # self.noise_spp_loss = LOSS_REGISTRY[cfg.NF.NOISE_SPP_LOSS](cfg)
        # self.noise_reg_loss = LOSS_REGISTRY[cfg.NF.NOISE_REG_LOSS](cfg)

        # self.attention_module = ATTENTION_REGISTRY[self.cfg.MODEL.ATTENTION_MODULE](
        #     self.cfg)

        self.xbm = XBM(cfg)
        self.xbm_all = XBM(cfg)

        self.W = torch.nn.Parameter(
            torch.Tensor(cfg.MODEL.HEAD.DIM,
                         cfg.num_classes)
        )
        self.reset_parameters()
        self.is_full = False
        # cls hyper param
        self.la = 20
        self.margin = self.cfg.NF.CLS_MARGIN
        self.batch_size = self.cfg.DATA.TRAIN_BATCHSIZE
        self.feature_dim = self.cfg.MODEL.HEAD.DIM

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
        # if self.cfg.RSPP.TARGET_NUM == 3:
        # targets_k = targets_k.reshape(1,-1)
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
                mask_i = bank_indices.reshape(-1,1) == pos_indice.reshape(1,-1)
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
        # else:
        #     raise NotImplementedError

    def clean_trans_loss(self, proto_feats, proto_target, xbm_feats, xbm_targets, keep_bool):
        proto_feat_clean = proto_feats[keep_bool]
        proto_target_clean = proto_target[keep_bool]

        mask = proto_target_clean.reshape(-1,1) == xbm_targets.reshape(1,-1)
        mask_sum_num = torch.sum(mask, dim = 1)
        mask_sum_num_neg = torch.sum(~mask, dim = 1)

        ssgpss = 1 - torch.mm(proto_feat_clean, xbm_feats.T)
        pos_ssgpss = ssgpss * mask
        neg_ssgpss = ssgpss *(~mask)
        pos_ssgpss = pos_ssgpss.sum(1) / (mask_sum_num+eps)
        neg_ssgpss = neg_ssgpss.sum(1) / (mask_sum_num_neg+eps)
        
        loss = torch.mean(neg_ssgpss + 0.5 - pos_ssgpss)
        return loss
        
    def get_pos_cond_feat(self, pos_indices):
        xbm_feats, xbm_targets , xbm_indices = self.xbm_all.get()            
        positive_cond_feats = []
        if len(xbm_feats)==0:
            return positive_cond_feats

        for i, indices in enumerate(pos_indices):
            # indices = torch.Tensor(indices[0]).cuda()
            indices = indices[0]
            # import pdb
            # pdb.set_trace()
            indices_pos = np.where(indices.reshape(-1,1) == xbm_indices.cpu().numpy().reshape(1,-1))[1]
            indices_pos = np.random.choice(indices_pos, self.cfg.NF.POS_COND_NUM)
            pos_features = xbm_feats[indices_pos]
            positive_cond_feats.append(pos_features)
        positive_cond_feats = torch.stack(positive_cond_feats)
        return positive_cond_feats

    def forward(self, input: dict):
        cfg = self.cfg
        feat_q = input['feat_q']
        # feat_k = input['feat_k']
        targets = input['targets']
        iteration = input['iteration']
        is_noise = input['is_noise']
        indices = input['indices']
        pos_indices = input['pos_indices']
        

        feat_q = F.normalize(feat_q)
        # feat_k = F.normalize(feat_k)
        # Noise Free Step
        group_size = len(feat_q) // self.batch_size
        feat_q_reshape = feat_q.reshape(
            self.batch_size, group_size, self.feature_dim)
            
        targets = targets.reshape(
            self.batch_size,
            group_size,
            self.cfg.NF.GROUP_NUM)
        indices = indices.reshape(self.batch_size, group_size)
        feat_query = torch.index_select(feat_q_reshape, 1, torch.Tensor(
            [0]).type(torch.int32).cuda()).squeeze(1)
        feat_sub = torch.index_select(
            feat_q_reshape, 1, torch.arange(1, group_size).cuda())
        query_label = targets[:, 0:1]
        sub_label = targets[:, 1:]

        # warm up learning
        if iteration <= cfg.XBM.START_ITERATION:
            loss = self.clean_batch_loss(feat_q, targets, feat_q, targets)
            log_info["batch_loss"] = loss.item()
            return loss

        # PRISM Step
        main_targets = targets[:, :, 0]
        main_targets = main_targets.type(torch.int64)
        main_targets = main_targets.reshape(-1)

        if self.selected_feats is not None:
            # enable backward or xbm will distrupt z
            self.xbm.enqueue_dequeue(self.selected_feats, self.selected_tar, self.selected_indices)

        
        self.selected_feats_all = feat_q.detach()
        self.selected_tar_all = main_targets.detach()
        self.selected_indices_all = indices.reshape(-1).detach()
        self.xbm_all.enqueue_dequeue(self.selected_feats_all, self.selected_tar_all, self.selected_indices_all)

        xbm_feats, xbm_targets, xbm_indices = self.xbm.get()
        query_label_target = query_label[:, :, 0]
        query_label_target = query_label_target.type(torch.int64)
        query_label_target = query_label_target.reshape(-1)
        is_noise_query = is_noise.reshape(self.batch_size, group_size)[:,0]
        xbm_loss_query, p_in_query = self.clean_bank_loss(feat_query,
                                              query_label_target,
                                              xbm_feats,
                                              xbm_targets,
                                              is_noise=is_noise_query,
                                              update_center=True
                                              )

        xbm_loss, p_in = self.clean_bank_loss(feat_q,
                                              main_targets,
                                              xbm_feats,
                                              xbm_targets,
                                              is_noise=is_noise
                                              )
        selected_feats = feat_q[p_in]
        selected_tar = main_targets[p_in]
        batch_loss = self.clean_batch_loss(selected_feats,
                                           selected_tar,
                                           selected_feats,
                                           selected_tar)

        selected_feats_query = feat_query[p_in_query]
        selected_tar_query = query_label_target[p_in_query]
        selected_indices_query = indices[:,0][p_in_query]

        batch_loss_query = self.clean_batch_loss(selected_feats_query,
                                           selected_tar_query,
                                           selected_feats_query,
                                           selected_tar_query)

        log_info["batch_loss"] = batch_loss.item()
        loss_pr_all = cfg.XBM.BASE_WEIGHT * batch_loss + cfg.XBM.WEIGHT * xbm_loss
        
        loss_pr_query = cfg.XBM.BASE_WEIGHT * batch_loss_query + cfg.XBM.WEIGHT * xbm_loss_query
        loss_pr = cfg.NF.PR_QUERY_W * loss_pr_query + cfg.NF.PR_ALL_W * loss_pr_all

        self.selected_feats = selected_feats_query.detach()
        self.selected_tar = selected_tar_query.detach()
        self.selected_indices = selected_indices_query.detach()

        # self.selected_feats_all = feat_q.detach()
        # self.selected_tar_all = main_targets.detach()
        # self.selected_indices_all = indices.reshape(-1).detach()

        # Reg Loss
        # clean sample use class loss
        ew = self.W / torch.norm(self.W, 2, 1, keepdim=True)
        logits_clean = torch.mm(selected_feats_query, ew)
        loss_cls = F.cross_entropy(
            self.la * (logits_clean - self.margin),
            selected_tar_query,
        )

        if torch.isnan(loss_cls.detach()):
            loss_cls = 0

        # switch loss
        keep_bool = p_in.reshape(self.batch_size, group_size)[:, 0]

        if self.cfg.MODEL.ATTENTION_MODULE in ['trans_proto2','trans_proto3', 'trans_proto4','trans_proto5','trans_proto6','max' ]:
            attention_mask = self.attention_module(feat_query, feat_sub)
        elif self.cfg.MODEL.ATTENTION_MODULE in ['trans_proto7','trans_proto8']:
            feat_cond = self.get_pos_cond_feat(pos_indices)
            attention_mask = self.attention_module(feat_sub, feat_cond)
        else:
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
                                       xbm_feats = xbm_feats,
                                       )

        # if self.cfg.MODEL.ATTENTION_MODULE in ['trans_proto','trans_proto2', 'trans_proto3', 'trans_proto4','trans_proto5','trans_proto6', 'trans_proto7','trans_proto8'] and len(xbm_feats) > 0:
        if self.cfg.MODEL.ATTENTION_MODULE.startswith('trans_proto') and len(xbm_feats) > 0:
            attention_mask = attention_mask.squeeze(1)
            attention_mask = F.normalize(attention_mask)
            loss_trans = self.clean_trans_loss(attention_mask, query_label_target, xbm_feats, xbm_targets, keep_bool)
        else:
            loss_trans = 0
        # swith loss xbm 
        # loss_spp_xbm = self.noise_spp_loss(feat_query,
        #                                xbm_feats,
        #                                attention_mask,
        #                                maks_ne g_switch,
        #                                keep_bool
        #                                )
        # noise sample use KL Reg Loss
        if sum(~keep_bool) > 0:
            W_detach = ew.detach()
            feat_query_noise = feat_query[~keep_bool]
            feat_sub_noise = feat_sub[~keep_bool].squeeze(
                0).reshape(-1, self.feature_dim)

            logits_query = torch.mm(feat_query_noise, W_detach)
            logits_sub = torch.mm(feat_sub_noise, W_detach)
            loss_reg = self.noise_reg_loss(
                logits_query, logits_sub
            )
        else:
            loss_reg = torch.Tensor([0]).cuda()

        if math.isnan(loss_spp) or math.isnan(loss_spp_bank):
            loss_spp = 0
            loss_spp_bank = 0

        loss_all = loss_pr * self.cfg.NF.PR_W + loss_cls * self.cfg.NF.CLS_W  + \
                    loss_spp * self.cfg.NF.SPP_W + loss_trans * self.cfg.NF.TRANS_W \
                    + loss_spp_bank * self.cfg.NF.SPP_BANK_W
        #+ loss_reg * self.cfg.NF.REG_W
        if iteration % 20 == 0:
            logger.info(
                f'iter: {iteration} loss_all: {loss_all.item(): .4f}, batch_loss: {batch_loss}, xbm_loss: {xbm_loss}, loss_pr: {loss_pr: .4f}, \
                    loss_cls: {loss_cls: .4f}, loss_spp: {loss_spp: .4f}, loss_spp_bank: {loss_spp_bank: .4f}, \
                    loss_trans: {loss_trans: .4f}')
                    # loss_reg: {loss_reg.item():.4f}')

        if math.isnan(loss_all):
            print(
                f'iter: {iteration} loss_pr: {loss_pr}, loss_cls: {loss_cls}, loss_spp: {loss_spp} loss_reg: {loss_reg}')
            # logger.info(
            #     )
        return loss_all

    def Param_groups(self, lr=None):
        params = list(filter(lambda x: x.requires_grad, self.parameters()))

        if len(params):
            if lr is not None:
                return [{'params': params, 'lr': lr}]
            else:
                return [{'params': params}]
        else:
            return []

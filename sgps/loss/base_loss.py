from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

import torch
from torch import nn
from core.utils.general.log_helper import default_logger as logger
from core.utils.general.registry_factory import LOSS_REGISTRY, ATTENTION_REGISTRY


@LOSS_REGISTRY.register("base_loss")
class RSPPLoss(nn.Module):
    def __init__(self, cfg=None):
        super(RSPPLoss, self).__init__()
        self.cfg = cfg
        self.cls_margin = cfg.RSPP.CLS_MARGIN
        self.feature_dim = cfg.MODEL.HEAD.DIM
        self.queue_size = cfg.XBM.SIZE

        self.momentum = cfg.RSPP.MOMENTUM
        self.scale = cfg.RSPP.SCALE
        self.con_margin = cfg.RSPP.CON_MARGIN

        self.batch_size = cfg.DATA.TRAIN_BATCHSIZE
        self.num_classes = cfg.MODEL.HEAD.NUM_CLASSES

        self.IA_K = cfg.RSPP.IA_K
        self.IA_NUM_MAIN = cfg.RSPP.IA_NUM_MAIN
        self.IA_NUM_SUB = cfg.RSPP.IA_NUM_SUB

        self.IT_K = cfg.RSPP.IT_K
        self.IA_MODE_MAIN = cfg.RSPP.IA_MODE_MAIN
        self.IA_MODE_SUB = cfg.RSPP.IA_MODE_SUB

        self.IT_MODE_MAIN = cfg.RSPP.IT_MODE_MAIN
        self.IT_MODE_SUB = cfg.RSPP.IT_MODE_SUB

        self.noise_rate = cfg.NOISE.NOISE_RATE
        self.noise_window_size = cfg.NOISE.WINDOW_SIZE
        self.contrast_margin = cfg.RSPP.CON_MARGIN

        self.cls_weight = cfg.RSPP.CLS_WEIGHT
        self.con_weight = cfg.RSPP.CON_WEIGHT

        self.batch_weight = cfg.RSPP.BATCH_WEIGHT
        self.bank_weight = cfg.RSPP.BANK_WEIGHT

        self.noise_margins = []
        self.DCQ_K = cfg.RSPP.DCQ_K
        self.queue_size = self.queue_size * self.DCQ_K

        self.register_buffer(
            "weight_queue", torch.randn([self.feature_dim, self.queue_size]))
        self.weight_queue = F.normalize(self.weight_queue, dim=0)
        self.weight_queue.requires_grad = False

        # self.weight = Parameter(torch.Tensor(
        #     self.feature_dim, self.num_classes))
        # self.reset_parameters()
        self.register_buffer("label_queue", torch.randn(
            [self.cfg.RSPP.TARGET_NUM, self.queue_size]))
        self.register_buffer(
            "queue_ptr", torch.zeros([1, ], dtype=torch.int64))
        self.label_queue.requires_grad = False
        self.queue_ptr.requires_grad = False

        # self.clean_queue.requires_grad = False
        self.is_full = False
        # build attention module
        self.attention_module = ATTENTION_REGISTRY[self.cfg.MODEL.ATTENTION_MODULE](
            self.cfg)
        self.iteration = 0

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = keys.reshape(-1, self.feature_dim)
        labels = labels.reshape(-1, self.cfg.RSPP.TARGET_NUM)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # for simplicity
        assert self.queue_size % batch_size == 0, f'{self.queue_size} {batch_size}'
        # replace the keys at ptr (dequeue and enqueue)
        self.weight_queue[:, ptr:ptr + batch_size] = keys.transpose(1, 0)
        self.label_queue[:, ptr:ptr + batch_size] = labels.transpose(1, 0)
        if ptr + batch_size >= self.queue_size:
            self.is_full = True
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _get_weight_queue(self):
        if self.is_full:
            return self.weight_queue.t().detach().clone()
        else:
            ptr = int(self.queue_ptr)
            return self.weight_queue[:, :ptr].t().detach().clone()

    @torch.no_grad()
    def _get_label_queue(self):
        if self.is_full:
            # return self.label_queue[0, :], self.label_queue[1, :], self.clean_queue[0, :]
            return self.label_queue.t()
        else:
            ptr = int(self.queue_ptr)
            return self.label_queue[:, :ptr].t()
            # return self.label_queue[0, :ptr].t()
            # return self.label_queue[0, :ptr], self.label_queue[1, :ptr], self.clean_queue[0, :ptr]

    def get_PRISM_filter(self, feat_q, target_q, use_batch_queue=False):
        #  target_q, sub_label_q=None, use_batch_queue=False):
        """
        get prototype weight from bank;
        """
        feat_q = feat_q.detach()
        bz_im_q = len(feat_q)
        target_q = target_q.squeeze(1)
        if use_batch_queue or (self.queue_ptr.item() == 0 and self.is_full == False):
            weight_queue = feat_q
            label_queue = target_q[:, 0]
            sub_label_queue = target_q[:, 1]
        else:
            weight_queue = self._get_weight_queue()
            label_queue = weight_queue[:, 0]
            sub_label_queue = weight_queue[:, 1]
        # TODO use clean sample to get filter
        label_queue = label_queue.reshape(-1)
        sub_label_queue = sub_label_queue.reshape(-1)

        index_main = torch.where(sub_label_queue == 0)
        label_main = label_queue[index_main]
        weight_main = weight_queue[index_main]

        prototype_label = torch.unique(label_main)
        prototype_weight = torch.zeros(len(prototype_label), self.feature_dim)

        for i, label in enumerate(prototype_label):
            label = label.item()
            mask = (label_main == label)
            prototype_weight[i] = torch.mean(weight_main[mask], dim=0)

        prototype_weight = prototype_weight.cuda()
        prototype_weight = F.normalize(prototype_weight)
        logits = torch.matmul(feat_q, prototype_weight.t())
        exp_logits = torch.exp(logits)
        p_main = []
        for i in range(bz_im_q):
            if not target_q[:, 0][i] in prototype_label:
                p_main.append(1.0)
            else:
                # import pdb
                # pdb.set_trace()
                mask = target_q[:, 0][i] == prototype_label
                pos_logits = exp_logits[i][mask]
                p_main.append(pos_logits.item() / exp_logits.sum(1)[i].item())

        p_main = torch.Tensor(p_main)
        idx_sorted = torch.argsort(p_main)
        to_remove = idx_sorted[:int(self.noise_rate * len(p_main))]

        # update window
        if not torch.isnan(p_main[to_remove[-1]]) and p_main[to_remove[-1]] != 1.0:
            self.noise_margins.append(p_main[to_remove[-1]].item())
            self.noise_margins = self.noise_margins[-self.noise_window_size:]

        if len(self.noise_margins) > 0:
            keep_bool = (p_main > sum(self.noise_margins) /
                         len(self.noise_margins))
            if torch.any(keep_bool) == False:
                keep_bool = torch.ones_like(p_main, dtype=bool)
                keep_bool[to_remove] = False
                self.noise_margins = self.noise_margins[-1:]
        else:
            keep_bool = torch.ones_like(p_main, dtype=bool)
            keep_bool[to_remove] = False
        return keep_bool, prototype_weight, prototype_label

    def loss_main_class(self, feat_q, feat_main, keep_bool, mask_neg):
        """
        feat_q: bz x dim
        feat_mian: bz x 1 x dim

        a simple tuplet loss with keep_bool filter.;
        mainly used for main class learning;
        """ 
        l_pos = torch.bmm(feat_q.unsqueeze(1), feat_main.transpose(1, 2))
        l_pos = l_pos - self.cls_margin
        l_neg = torch.mm(feat_q, feat_main.squeeze(1).T)
        exp_l_pos = torch.exp(l_pos * self.scale)
        exp_l_neg = torch.exp(l_neg * self.scale) * mask_neg
        logits = exp_l_pos / (exp_l_pos + exp_l_neg.sum(1))
        logits_main = logits[keep_bool]
        loss = - torch.log(logits_main).mean()
        return loss

    def loss_main_contrastive(self, feat_q, feat_main, keep_bool, mask_neg):
        n = feat_q.size(0)
        l_pos = torch.bmm(feat_q.unsqueeze(1), feat_main.transpose(1, 2))
        l_pos = l_pos - self.cls_margin
        l_neg = torch.mm(feat_q, feat_main.squeeze(1).T)
        mask_neg = mask_neg & (l_neg > 0.5)
        l_neg = l_neg[mask_neg]
        pos_loss = torch.sum(-l_pos+1)

        neg_loss = torch.sum(l_neg) / l_neg.numel() if l_neg.numel() > 0 else 0
        if self.iteration % 20 == 0:
            logger.info(
                f'pos_loss: {pos_loss/n} neg_loss: {neg_loss} n:{n}, neg_n:{l_neg.numel()}')
        loss = (pos_loss / n) + neg_loss
        return loss

    def loss_sub_contrastive(self,  feat_q, feat_sub, attention_mask=None, mask_neg=None):
        n = feat_q.size(0)
        if len(attention_mask.shape) == 3:
            feat_sub_proto = torch.sum(
                feat_sub * attention_mask, dim=1)  # [N,d]
            l_pos = F.cosine_similarity(feat_q, feat_sub_proto)  # N
        elif len(attention_mask.shape) == 2:
            l_pos = F.cosine_similarity(feat_q, attention_mask)  # N

        l_neg = torch.mm(feat_q, feat_sub.reshape(-1, self.feature_dim).T)
        mask_neg = mask_neg & (l_neg > 0.5)
        l_neg = l_neg[mask_neg]
        pos_loss = torch.sum(-l_pos+1)
        neg_loss = torch.sum(l_neg) / l_neg.numel() if l_neg.numel() > 0 else 0
        if self.iteration % 20 == 0:
            logger.info(
                f'pos_loss: {pos_loss/n} neg_loss: {neg_loss} n:{n}, neg_n:{l_neg.numel()}')
        loss = (pos_loss / n) + neg_loss
        return loss

    def loss_bank_neg(self, feat_q, feat_bank, mask_neg=None):
        l_neg = torch.mm(feat_q, feat_bank.reshape(-1, self.feature_dim).T)
        mask_neg = mask_neg & (l_neg > 0.5)
        l_neg = l_neg[mask_neg]
        neg_loss = torch.sum(l_neg) / l_neg.numel() if l_neg.numel() > 0 else 0
        if self.iteration % 20 == 0:
            logger.info(
                f'neg_loss: {neg_loss}, neg_n:{l_neg.numel()}')
        loss = neg_loss
        return loss


    def loss_bank_pos(self, feat_q, feat_bank, mask_pos=None):
        l_pos = torch.mm(feat_q, feat_bank.reshape(-1, self.feature_dim).T)
        mask_pos = mask_pos  & (l_pos < 0.99)
        l_pos = l_pos[mask_pos]
        pos_loss = torch.sum(-l_pos+1) / l_pos.numel() if l_pos.numel() > 0 else 0
        if self.iteration % 20 == 0:
            logger.info(
                f'pos_loss: {pos_loss}, pos_n:{l_pos.numel()}')
        loss = pos_loss
        return loss            

    def loss_switch_attention(self, feat_q, feat_sub, attention_mask=None, mask_neg=None):
        """
        feat_q: bz x dim
        feat_sub: bz x IA_SUB_NUM x dim
        attention_mask
        """

        if len(attention_mask.shape) == 3:
            feat_sub_proto = torch.sum(
                feat_sub * attention_mask, dim=1)  # [N,d]
            pos_scores = F.cosine_similarity(feat_q, feat_sub_proto)  # N

        elif len(attention_mask.shape) == 2:
            pos_scores = F.cosine_similarity(feat_q, attention_mask)  # N

        neg_scores = torch.mm(feat_q, feat_sub.reshape(-1, self.feature_dim).T)
        exp_l_pos = torch.exp(pos_scores * self.scale)
        exp_l_neg = torch.exp(neg_scores * self.scale) * mask_neg
        logits = exp_l_pos / (exp_l_pos + exp_l_neg.sum(1))
        # logits_main = logits[~keep_bool]
        logits_main = logits
        loss = - torch.log(logits_main).mean()
        return loss

    def cal_neg_mask(self, targets_q, targets_k):
        '''
        filter pairs with same main_label tb_label bt_label;
        '''
        if self.cfg.RSPP.TARGET_NUM == 4:
            mask_main = targets_q[:, :,
                                  0].reshape(-1, 1) != targets_k[:, :, 0].reshape(1, -1)
            mask_bt = targets_q[:, :,
                                2].reshape(-1, 1) != targets_k[:, :, 2].reshape(1, -1)
            mask_tb = targets_q[:, :,
                                3].reshape(-1, 1) != targets_k[:, :, 3].reshape(1, -1)
            mask = mask_main & mask_bt & mask_tb
            return mask
        else:
            raise NotImplementedError


    def cal_pos_mask(self, targets_q, targets_k):
        '''
        filter pairs with same main_label tb_label bt_label;
        '''
        if self.cfg.RSPP.TARGET_NUM == 4:
            mask_main = targets_q[:, :,
                                  0].reshape(-1, 1) == targets_k[:, :, 0].reshape(1, -1)                                
            mask_bt = targets_q[:, :,
                                2].reshape(-1, 1) == targets_k[:, :, 2].reshape(1, -1)
            mask_tb = targets_q[:, :,
                                3].reshape(-1, 1) == targets_k[:, :, 3].reshape(1, -1)
            # mask = mask_main & mask_bt & mask_tb
            mask = mask_main | mask_bt | mask_tb
            return mask
        else:
            raise NotImplementedError


    def forward(self,
                input: dict,
                ):
        """
        calculate RSPP loss 
        required input:
        feat_q:  batch_size x IA_K x feat_dim # IA_K ~ [1, IA_NUM_MAIN, IA_NUM_SUB, IA_NUM_SUB]
        feat_k:  batch_size x IA_K x feat_dim
        targets: batch_size x IA_K x 4 # [main_label, sub_label, tb_label, bt_label]
        attention_mask: batch_size x (IA_NUM_SUB + IA_NUM_SUB)

        """
        feat_q = input['feat_q']
        feat_k = input['feat_k']
        targets = input['targets']

        feat_q = F.normalize(feat_q)
        feat_k = F.normalize(feat_k)

        group_size = len(feat_q) // self.batch_size
        assert group_size == self.DCQ_K, f'{group_size}, {self.DCQ_K}'

        feat_q = feat_q.reshape(self.batch_size, group_size, self.feature_dim)

        feat_k = feat_k.reshape(self.batch_size, group_size, self.feature_dim)
        targets = targets.reshape(
            self.batch_size, group_size, self.cfg.RSPP.TARGET_NUM)

        if self.queue_ptr > 0:
            feat_queue = self._get_weight_queue()
            label_queue = self._get_label_queue()
            # print(label_queue.shape)
            mask_neg_bank = self.cal_neg_mask(
                query_label, label_queue.unsqueeze(1))
            # with torch.autograd.set_detect_anomaly(True):            

            mask_pos_bank = self.cal_pos_mask(
                query_label, label_queue.unsqueeze(1))
               
            loss_bank_neg = self.loss_bank_neg(
                feat_query, feat_queue, mask_neg_bank)
            loss_bank_pos = self.loss_bank_pos(
                feat_query, feat_queue, mask_pos_bank)
        else:
            loss_bank_neg = 0.0
            loss_bank_pos = 0.0

        # loss_main = self.loss_main_class(
        #     feat_query, feat_main, keep_bool, maks_neg_main)
        # loss_switch = self.loss_switch_attention(
        #     feat_query, feat_sub, attention_mask, maks_neg_switch)

        loss_main = self.loss_main_contrastive(
            feat_query, feat_main, keep_bool, maks_neg_main)

        loss_switch = self.loss_sub_contrastive(
            feat_query, feat_sub, attention_mask, maks_neg_switch)

        batch_loss = self.cls_weight * loss_main + \
            self.con_weight * loss_switch + loss_bank_neg + loss_bank_pos

        # batch_loss = loss_main + loss_bank_neg
        if self.iteration % 20 == 0:
            logger.info(
                f'loss_main: {loss_main.item()} loss_switch: {loss_switch.item()} loss_bank_neg: {loss_bank_neg}')            
            # logger.info(f'attention mask: {attention_mask}')
        self.iteration += 1
        self._dequeue_and_enqueue(feat_q, targets)
        loss = batch_loss
        return loss

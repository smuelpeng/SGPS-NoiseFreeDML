from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import gc
import sgps
from sgps.systems.base import BaseSystem
from sgps.utils.typing import *
from sgps.utils.misc import time_recorder as tr
from sgps.utils.config import gen_log
from sgps.loss.noisefree import NoiseFreeLoss
from tqdm import tqdm
import time
import numpy as np
from collections import defaultdict, deque
from FeatureServer import NFClient2 as NFClient
from sgps.utils.evaluations.eval import AccuracyCalculator


@dataclass
class XBMconfig:
    SIZE: int = 81920
    FEATURE_DIM: int = 128
    FEATURE_FILE: str = ""
    TARGET_FILE: str = ""

@dataclass
class NosieConfig:
    NOISE_RATE: float = 0.5
    WINDOW_SIZE: int = 30
    WARM_UP: int = 0
    CHECK_NOISE: bool = False

@dataclass
class NoiseFreeLossConfig:
    clean_batch_loss: str = "sgps.loss.contrastive_loss.ContrastiveLoss"
    clean_bank_loss: str = "sgps.loss.memory_contrastive_loss_w_PRISM.MemoryContrastiveLossPRISM"
    noise_sgps_loss: str = "sgps.loss.sw_loss.SwitchLoss"        
    attention_module: str = "sgps.model.attention.AttentionSoftmax"

    lambda_clean_query: float = 1.0
    lambda_clean_all: float = 1.0

    lambda_clean_batch: float = 1.0
    lambda_clean_bank: float = 1.0
    lambda_noise_batch: float = 1.0
    lambda_noise_bank: float = 1.0

    XBM: XBMconfig = field(default_factory=XBMconfig)
    NOISE: NosieConfig = field(default_factory=NosieConfig)
    NF: dict = field(default_factory=dict)
    group_num: int = 5
    num_classes: int = 1000
    feature_dim: int = 128

class SGPSNF(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)
        head: dict = field(default_factory=dict)
        loss_cls: str = "NoiseFreeLoss"
        loss:NoiseFreeLossConfig = field(default_factory=NoiseFreeLossConfig)
        NF_port: int = 5870

    cfg: Config

    def configure(self):
        super().configure()
        self.criterion = NoiseFreeLoss(self.cfg.loss)
        self.backbone = sgps.find(self.cfg.backbone_cls)(self.cfg.backbone)
        self.NF_client = NFClient(0, self.cfg.NF_port)
        self.validation_step_outputs = []
        self.validation_step_labels = []

    def forward(self, imgs):     
        x_feat, logits = self.backbone(imgs)
        return x_feat, logits

    def training_step(self, batch, batch_idx):
        imgs, targets, indices, positive_indices = batch        
        x_feat, logits = self(imgs)
        input = {'feat_q': x_feat,
                 'targets': targets,
                 'indices': indices,
                 'pos_indices': positive_indices
                 }
        loss = self.criterion(input)
        self.NF_client.update_feature(
            mid='0',
            index=indices.reshape(-1),
            feature=x_feat.detach().cpu().numpy()
        )
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets, indices = batch
        # outputs = self(imgs)
        x_feat, logits = self.backbone(imgs)
        self.validation_step_outputs.append(x_feat)
        self.validation_step_labels.append(targets)
        return x_feat

    def on_validation_epoch_end(self):
        feats_query = torch.cat(self.validation_step_outputs, dim=0)
        feats_query = feats_query.cpu().numpy()
        feats_gallery = feats_query
        labels_query = torch.cat(self.validation_step_labels, dim=0)
        labels_query = labels_query.reshape(-1)
        labels_query = labels_query.cpu().numpy()
        labels_gallery = labels_query

        log_info = {}
        ret_metric = AccuracyCalculator(include=(
            "precision_at_1", "mean_average_precision_at_r", "r_precision", 'mean_average_precision_at_100'), exclude=())
        ret_metric = ret_metric.get_accuracy(
            feats_query, feats_gallery, labels_query, labels_gallery, True)
        mapr_curr = ret_metric['precision_at_1']
        for k, v in ret_metric.items():
            log_info[f"e_{k}"] = v
            print(f"e_{k} : {v}")
        return log_info

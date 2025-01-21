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

def create_model_res50(num_classes=1000):
    from sgps.model.resnet import resnet50

    import torchvision.models as models
    pretrain_model = models.resnet50(pretrained=True)
    pretrain_model.fc = nn.Linear(2048, num_classes)    
    model = resnet50(num_classes=num_classes)
    params  = pretrain_model.named_parameters()
    params1 = model.named_parameters() 
    dict_params1 = dict(params1)
    for name1, param in params:
        if name1 in dict_params1:
            dict_params1[name1].data.copy_(param.data)    
    return model

class DML(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)
        head: dict = field(default_factory=dict)
        loss_dml_cls: str = "NoiseFreeLoss"
        loss_dml:dict = field(default_factory=dict)
        NF_port: int = 5870

    cfg: Config

    def configure(self):
        super().configure()
        # self.criterion = NoiseFreeLoss(self.cfg.loss)
        self.backbone = sgps.find(self.cfg.backbone_cls)(self.cfg.backbone)
        self.criterion = sgps.find(self.cfg.loss_dml_cls)(self.cfg.loss_dml)

        self.NF_client = NFClient(0, self.cfg.NF_port)
        self.validation_step_outputs = []
        self.validation_step_labels = []

    def forward(self, imgs):
        # imgs, labels, indices, pos_indices = batch
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

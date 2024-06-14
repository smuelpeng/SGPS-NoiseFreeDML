from dataclasses import dataclass, field

import torch
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

class SGPSNF(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        # loss: TextMassLossConfig = TextMassLossConfig()
        loss_clean: dict = field(default_factory=dict)
        loss_noise: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        head_cls: str = ""
        head: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        super.configure()
        self.loss_clean = NoiseFreeLoss(**self.cfg.loss_clean)
        self.backbone = sgps.find(self.cfg.backbone_cls)(self.cfg.backbone)
        self.head = sgps.find(self.cfg.head_cls)(self.cfg.head)


    def forward(self, x):
        x = self.backbone(x)
        feat =  self.head(x)
        return {"feat": feat}
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.loss(outputs, batch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.loss(outputs, batch)
        return loss
    
    




import json
import math
import os
import random
from dataclasses import dataclass, field
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transform
from PIL import Image
import imageio
import cv2
import pandas as pd
from collections import defaultdict

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import sgps
from ..utils.config import parse_structured
from ..utils.typing import *
from ..utils.misc import load_json

from FeatureServer import NFClient2 as NFClient
import re
from sgps.data.common_function import read_image
import torchvision.transforms as T
import pytorch_lightning as pl
from .transforms import build_transforms
from .base_dataset import BaseDataSet as BaseValDMLDataset


@dataclass
class NoiseFreeDMLDataModuleConfig:
    dataset: str = ""
    root: str = ""
    train_file: str = ""
    test_file: str = ""
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 4
    num_instances: int = 4
    num_classes: int = 1000
    sampler_cls: str = ""
    NF_port: int = 5870
    INPUT: dict = field(default_factory=dict)
    max_iters: int = 8000
    


class NoiseFreeDMLDataset(BaseValDMLDataset):
    def __init__(self, cfg) -> None:
        super().__init__(cfg, mode="train")
        self.transforms = build_transforms(cfg, is_train=True)
        self.feat_client = None
    
    def init_client(self):
        if self.feat_client is None:
            self.mid = random.randint(1, 1000)
            self.feat_client = NFClient(self.mid, self.cfg.NF_port)
        else:
            return

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def load_one_sample(self, index):
        path = self.path_list[index]
        filename = os.path.join(self.root, path)
        label = self.label_list[index]
        img = read_image(filename, mode=self.mode)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __getitem__(self, index):
        self.init_client()
        sample_indices,positive_indices = self.feat_client.get_groups_with_positive_indices(
            mid=self.mid, index=[index])
        sample_index = sample_indices[:, 0]
        labels = sample_indices[:, 1:]
        img, label = self.load_one_sample(index)
        imgs = []
        for s_index in sample_index:
            img, label = self.load_one_sample(s_index)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        return imgs, labels, sample_index.reshape(-1), [positive_indices]
    
    def _collate_fn(self, batch):
        """
        """
        imgs, labels, sample_indexes, pos_indices = zip(*batch)
        imgs = torch.cat(imgs, dim=0)
        labels = np.array(labels).astype(np.int64)
        labels = torch.Tensor(labels)
        sample_indexes = np.array(sample_indexes).astype(np.int64)
        sample_indexes = torch.Tensor(sample_indexes)
        return imgs, labels, sample_indexes,pos_indices



class NoiseFreeDMLDataModule(pl.LightningDataModule):
    cfg: NoiseFreeDMLDataModuleConfig
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(NoiseFreeDMLDataModuleConfig, cfg)

    def setup(self, stage: str) -> None:
        # return super().setup(stage)
        if stage in [None, "fit"]:
            self.train_dataset = NoiseFreeDMLDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = BaseValDMLDataset(self.cfg, "val")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        if self.cfg.sampler_cls is not None:
            sampler = sgps.find(self.cfg.sampler_cls)(
                self.train_dataset,
                self.cfg.batch_size,
                num_instances = self.cfg.num_instances,
                max_iters = self.cfg.max_iters
            )
        else:
            sampler = None
        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=None if sampler is not None else True,
            collate_fn=self.train_dataset._collate_fn,
            pin_memory = False, 
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.val_dataset._collate_fn,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.test_dataset._collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

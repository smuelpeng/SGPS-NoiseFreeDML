# encoding: utf-8

import os
import re
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
from sgps.data.common_function import read_image
from .transforms import build_transforms

def str2int(s):
    if s.isdigit() or s.startswith("-"):
        return int(s)
    else:
        ascii_str = "".join([str(ord(k))[0] for k in s])
        return -int(ascii_str)

def find_clean_dataset(img_source):
    idx = img_source.find('noised_')+len('noised_')
    return img_source[:idx]+'cleaned_train.csv'


def get_is_noise(img_source, path_list):
    clean_img_source = find_clean_dataset(img_source)
    clean_path_list = list()
    with open(clean_img_source, "r") as f:
        for line in f:
            _path, _label = re.split(r",", line.strip())
            clean_path_list.append(_path)
    clean_path_list = np.asarray(clean_path_list)
    path_list = np.asarray(path_list)
    is_noise = np.isin(path_list, clean_path_list)
    return torch.BoolTensor(~is_noise)

class BaseDataSet(Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """

    def __init__(self,cfg, mode='train'):
        self.cfg = cfg        
        self.mode = "RGB"
        self.root = cfg.root
        is_train = mode == 'train'
        if is_train:
            self.img_source = cfg.train_file
        else:
            self.img_source = cfg.test_file

        self.transforms = build_transforms(cfg, is_train=False)
        self._load_data()

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__(            
        )

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        self.label_list = list()
        self.path_list = list()
        with open(self.img_source, "r") as f:
            for line in f:
                data_line = re.split(r",", line.strip())
                if len(data_line) == 2:
                    _path, _label = data_line
                else:
                    _path, _ , _label = data_line
                self.path_list.append(_path)
                self.label_list.append(_label)
        self.label_index_dict = self._build_label_index_dict()

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]
        img = read_image(img_path, mode=self.mode)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label, index    
    
    def _collate_fn(self, batch):
        """s
        """
        imgs, labels, indices = zip(*batch)
        labels = [str2int(k) for k in labels]
        labels = torch.tensor(labels, dtype=torch.int64)
        indices = torch.tensor(indices, dtype=torch.int64)
        return torch.stack(imgs, dim=0), labels, indices
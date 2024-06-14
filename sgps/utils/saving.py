import json
import os
import re
import shutil

import cv2
# import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
# import wandb
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
from pytorch_lightning.loggers import WandbLogger

import sgps
from ..utils.typing import *


class SaverMixin:
    _save_dir: Optional[str] = None
    _wandb_logger: Optional[WandbLogger] = None

    def set_save_dir(self, save_dir: str):
        self._save_dir = save_dir

    def get_save_dir(self):
        if self._save_dir is None:
            raise ValueError("Save dir is not set")
        return self._save_dir
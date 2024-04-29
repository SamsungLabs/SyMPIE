################################################################################
# Copyright (c) 2024 Samsung Electronics Co., Ltd.
#
# Author(s):
# Francesco Barbato (f.barbato@samsung.com; francesco.barbato@dei.unipd.it)
# Umberto Michieli (u.michieli@samsung.com)
# Mehmet Yucel (m.yucel@samsung.com)
# Pietro Zanuttigh (zanuttigh@dei.unipd.it)
# Mete Ozay (m.ozay@samsung.com)
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# For conditions of distribution and use, see the accompanying LICENSE.md file.
################################################################################

import json
from os import path

import cv2 as cv
import numpy as np

import torch
from torch.utils.data import Dataset

class VizWiz(Dataset):
    """
        vizwiz dataset
    """
    def __init__(self,
                 root_path=None,
                 split_path=None,
                 resize_to=232,
                 crop_to=224,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 interpolation=cv.INTER_AREA):

        self.root_path = root_path
        self.resize_to = resize_to # in opencv notation: (W, H)
        self.crop_to = crop_to
        self.mean = mean
        self.std = std
        self.interpolation = interpolation

        with open(path.join(split_path, "annotations.json"), encoding='utf-8') as fin:
            anno = json.load(fin)

        self.valid = [k["id"] for k in anno["categories"]]
        self.items = anno["images"]

    def __getitem__(self, item):
        impath = self.items[item] # VizWiz_val_00003893.jpg
        _, folder, _ = impath.split("_")

        im = cv.imread(path.join(self.root_path, folder, impath))

        # preprocessing same as torch
        gh, gw = im.shape[:2]
        if gh < gw:
            im = cv.resize(im, (round(gw*self.resize_to/gh),
                                self.resize_to), interpolation=self.interpolation)
        else:
            im = cv.resize(im, (self.resize_to,
                                round(gh*self.resize_to/gw)), interpolation=self.interpolation)

        gh, gw = im.shape[:-1]
        if gh - self.crop_to > 0:
            delta, pad = divmod(gh - self.crop_to, 2)
            im = im[delta:-delta-pad].copy()
        if gw - self.crop_to > 0:
            delta, pad = divmod(gw - self.crop_to, 2)
            im = im[:, delta:-delta-pad].copy()

        # set to rgb and normalize
        im = im[...,::-1].astype(np.float32)/255.
        # standardize using torch mean and std
        im = (im - self.mean)/self.std

        return torch.from_numpy(im.astype(np.float32)).permute(2,0,1), impath

    def to_rgb(self, ts):
        """
            tensor to rgb
        """
        im = ts.permute(1,2,0).numpy()
        im = im*self.std + self.mean
        im = np.clip(im, a_min=0, a_max=1)
        return im

    def __len__(self):
        return len(self.items)

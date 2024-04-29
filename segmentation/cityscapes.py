################################################################################
# This code is adapted from: https://github.com/lttm/lsr)
# Copyright (c) 2021.
#
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

import cv2 as cv
import numpy as np

import torch
from torch.utils.data import Dataset

class Cityscapes(Dataset):
    """
        cityscapes dataset
    """
    def __init__(self,
                 root_path=None,
                 split="train",
                 repeats=1):

        self.root_path = root_path
        self.split = split
        self.repeats = repeats

        with open(root_path+split+'.txt', encoding='utf-8') as fin:
            self.items = [l.strip().split() for l in fin]
        self.nitems = len(self.items)
        self.items *= repeats

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.flipids = torch.arange(1023,-1,-1)

        self.cnames = [
                            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'trafflight',
                            'traffsign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 
                            'truck', 'bus', 'train', 'motorcycle', 'bicycle' #, 'unlabeled'
                      ]
        self.cmap = np.array([
                [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
                [0, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                [0, 80, 100], [0, 0, 230], [119, 11, 32], [  0,   0,   0]
                ])
        self.maptotrain = {
                -1: -1, 0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: 0, 8: 1, 9: -1,
                10: -1, 11: 2, 12: 3, 13: 4, 14: -1, 15: -1, 16: -1, 17: 5, 18: -1, 19: 6,
                20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: -1,
                30: -1, 31: 16, 32: 17, 33: 18
                }


    def totrain(self, gt):
        """
            map label to training indices
        """
        gt_train = -np.ones(gt.shape, dtype=np.int64)
        for k,v in self.maptotrain.items():
            gt_train[gt==k] = v
        return gt_train

    def colorlabel(self, gt):
        """
            color the label
        """
        return self.cmap[gt]/255.

    def torgb(self, ts):
        """
            convert the tensor to rgb
        """
        im = ts.permute(1,2,0).numpy()
        return np.clip(im*self.std + self.mean, 0, 1)

    def flip(self, item, im, gt):
        """
            l/r flip
        """
        if self.repeats == 1:
            if np.random.rand() < .5:
                im = im[:,::-1].copy()
                gt = gt[:,::-1].copy()
        else:
            if (item//self.nitems) % 2 == 1:
                im = im[:,::-1].copy()
                gt = gt[:,::-1].copy()
        return im, gt

    def blur(self, im):
        """
            random gaussian blur
        """
        if np.random.rand() < .5:
            im = cv.GaussianBlur(im, (0,0), 2*np.random.rand())
        return im

    def awgn(self, im):
        """
            random awgn noise
        """
        if np.random.rand() < .5:
            im = im + 10*np.random.randn(*im.shape)
        return im

    def sensor(self, im):
        """
            random sensor noise
        """
        if np.random.rand() < .5:
            im = im + im*np.random.randn(*im.shape)/20
        return im

    def clip(self, im, gt):
        """
            random clip
        """
        w0 = np.random.randint(im.shape[1]-512+1)
        h0 = np.random.randint(im.shape[0]-512+1)
        im = im[h0:h0+512, w0:w0+512]
        gt = gt[h0:h0+512, w0:w0+512]
        im = np.clip(im, 0, 255)
        return im, gt

    def __getitem__(self, item):
        imp, gtp = self.items[item]

        im = cv.imread(self.root_path+imp)[...,::-1] # load in rgb
        gt = cv.imread(self.root_path+gtp, cv.IMREAD_UNCHANGED)

        gh, gw = im.shape[:2]
        if gh > gw:
            resize = (int(1024*gw/gh), 1024)
        else:
            resize = (1024, int(1024*gh/gw))

        im = cv.resize(im, resize, interpolation=cv.INTER_AREA)
        gt = cv.resize(gt, resize, interpolation=cv.INTER_NEAREST_EXACT)

        if self.split == "train":
            im, gt = self.flip(item, im, gt)
            im = self.blur(im)
            im = self.awgn(im)
            im = self.sensor(im)
            im, gt = self.clip(im, gt)

        im = (im/255. - self.mean)/self.std
        gt = self.totrain(gt)

        return torch.from_numpy(im.astype(np.float32)).permute(2,0,1), torch.from_numpy(gt)

    def __len__(self):
        return len(self.items)

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

from os import path
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 as cv
cv.setNumThreads(0)

class ImagenetC(Dataset):
    """
        imagenetc dataset
    """
    def __init__(self,
                 root_path=None,
                 split_path=None,
                 split='images',
                 resize_to=232,
                 crop_to=224,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 interpolation=cv.INTER_AREA,
                 subsample=False,
                 subsample_rate=50):
        super().__init__()

        self.root_path = root_path
        self.split = split
        self.resize_to = resize_to # in opencv notation: (W, H)
        self.crop_to = crop_to
        self.mean = mean
        self.std = std
        self.interpolation = interpolation

        corruptions = ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                       "contrast", "elastic_transform", "jpeg_compression", "pixelate",
                       "gaussian_noise", "impulse_noise", "shot_noise",
                       "brightness", "fog", "frost", "snow",
                       "colwarp", "clean"]
        self.corruptions = {corr: i for i, corr in enumerate(corruptions)}
        self.folders = [(self.get_category(corr), corr) for corr in corruptions]
        self.intensities = list(range(1,6))

        with open(path.join(split_path, split+'.txt'), encoding='utf-8') as fin:
            images = [l.strip().split() for l in fin]
        self.num_samples = len(images)

        self.items = [
                        [
                            category+"/"+ \
                                self.get_path(category, corruption, intensity) +\
                                    (im if category not in ["clean", "color"]
                                        else im.replace(".JPEG", ".jpg")),
                            class_id,
                            intensity,
                            corr_id
                        ] for im, class_id in images for intensity in
                            self.intensities for corr_id, (category, corruption) in
                            enumerate(self.folders)
                     ]

        if subsample:
            np.random.seed(42)
            per = np.random.permutation(len(self.items))[:len(self.items)//subsample_rate]
            self.items = [self.items[i] for i in per]
        self.classes = 1000

    def to_rgb(self, ts):
        """
            convert tensor to rgb
        """
        im = ts.permute(1,2,0).numpy()
        im = im*self.std + self.mean
        im = np.clip(im, a_min=0, a_max=1)
        return im

    def __len__(self):
        return len(self.items)

    @staticmethod
    def get_category(corr):
        """
            get category name from corruption
        """
        if corr in ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"]:
            return "blur"
        if corr in ["contrast", "elastic_transform", "jpeg_compression", "pixelate"]:
            return "digital"
        if corr in ["gaussian_noise", "impulse_noise", "shot_noise"]:
            return "noise"
        if corr in ["brightness", "fog", "frost", "snow"]:
            return "weather"
        if corr == "clean":
            return "clean"
        if corr == "colwarp":
            return "color"
        raise ValueError(f"Unrecognized Corruption [{corr}]")

    @staticmethod
    def get_path(category, corruption, intensity):
        """
            change directory based on the category
        """
        if category == "clean":
            return ""
        if category == "color":
            return str(intensity)+"/"
        return corruption+"/"+str(intensity)+"/"

    def __getitem__(self, item):
        impath, class_id, intensity, corr_id = self.items[item]
        im = cv.imread(path.join(self.root_path, impath))

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

        return torch.from_numpy(im.astype(np.float32)).permute(2,0,1),\
                int(class_id), int(intensity)-1, corr_id

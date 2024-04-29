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
import cv2 as cv

from imagenetc import ImagenetC 

cv.setNumThreads(0)

class ImagenetCBar(ImagenetC):
    """
        imagenetc-bar dataset
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

        corruptions = ["blue_noise_sample", "brownish_noise", "caustic_refraction",
                       "checkerboard_cutout", "cocentric_sine_waves", "inverse_sparkles", 
                       "perlin_noise", "plasma_noise", "single_frequency_greyscale",
                       "sparkles", "clean"]
        self.corruptions = {corr: i for i, corr in enumerate(corruptions)}
        self.intensities = list(range(1,6))

        with open(path.join(split_path, split+'.txt'), encoding='utf-8') as fin:
            images = [l.strip().split() for l in fin]
        self.num_samples = len(images)

        self.items = [
            [
                self.get_path(None, corruption, intensity) +\
                    (im if corruption not in ["clean", "color"] else im.replace(".JPEG", ".jpg")),
                class_id,
                intensity,
                corr_id
            ] for im, class_id in images for intensity in self.intensities
                    for corr_id, corruption in enumerate(self.corruptions)
            ]

        if subsample:
            np.random.seed(42)
            per = np.random.permutation(len(self.items))[:len(self.items)//subsample_rate]
            self.items = [self.items[i] for i in per]
        self.classes = 1000

    @staticmethod
    def get_path(category, corruption, intensity): 
        """
            change directory based on the category
        """
        if corruption == "clean":
            return "clean/"
        return corruption+"/"+str(intensity)+"/"

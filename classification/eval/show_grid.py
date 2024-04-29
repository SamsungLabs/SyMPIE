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

import argparse
import json
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default='json/none_rn50_2.json', type=str,
                        help='path to json file to display')

    args = parser.parse_args()

    with open(args.json, encoding='utf-8') as fdata:
        data = np.array(json.load(fdata)).T

    if data.shape[1] == 11:
        corruptions = ["blue_noise_sample", "brownish_noise", "caustic_refraction",
                        "checkerboard_cutout", "cocentric_sine_waves", "inverse_sparkles", 
                        "perlin_noise", "plasma_noise", "single_frequency_greyscale",
                        "sparkles", "clean"]
    else:
        corruptions = ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                    "contrast", "elastic_transform", "jpeg_compression", "pixelate",
                    "gaussian_noise", "impulse_noise", "shot_noise",
                    "brightness", "fog", "frost", "snow",
                    "colwarp", "clean"]

    data = np.concatenate([data, data.mean(axis=0, keepdims=True)], axis=0)
    data = np.concatenate([data, data[:,:-1-int(data.shape[1] != 11)].mean(
                                                axis=1, keepdims=True)], axis=1)

    intensities = [str(i) for i in range(1,6)]

    corruptions += ["avg."]
    intensities += ["avg."]

    plt.figure(figsize=(9,4))
    plt.imshow(data)
    plt.colorbar()

    ax = plt.gca()
    ax.xaxis.set_label_position('top')
    plt.xticks(np.arange(data.shape[1]), labels=corruptions, ha="right", rotation=45)
    plt.yticks(np.arange(data.shape[0]), labels=intensities)

    plt.xlabel("Corruption Type")
    plt.ylabel("Intensity")

    cutoff = (data.max()+data.min())/2 + (data.max()-data.min())/4
    for (j,i), label in np.ndenumerate(data):
        ax.text(i, j, "%.2f"%label, ha='center', va='center', color="w" if label < cutoff else "k")

    plt.tight_layout()
    plt.show()

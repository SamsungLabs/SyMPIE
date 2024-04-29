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

import sys
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import yaml

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, CenterCrop, Compose

from cityscapes import Cityscapes
from metrics import Metrics
from training_LSR.graphs.models.models import DeeplabNW
from renorm import ReNorm

sys.path.append("../classification/training")
from model import ColModule, ColWarp, Clamp

def torgb(ts):
    """
        convert tensor to rgb
    """
    im = ts.permute(1,2,0).numpy()
    return np.clip((im + [104.00698793, 116.66876762, 122.67891434])/255., 0, 1)[...,::-1]

if __name__ == "__main__":
    with open('../config_paths.yaml', encoding='utf-8') as yamlf:
        config = yaml.load(yamlf, yaml.BaseLoader)
    city = Cityscapes(root_path=config['acdc'],
                      split='val')


    CKPT_PATH="../classification/training/checkpoints/cityscapesfinal.pth"
    WARP = True
    PATH = "qualitatives/%s/%s/"

    model = DeeplabNW(19, 'ResNet50', pretrained=False)
    model.load_state_dict({k.replace("module.", ""):v for k,v in
                           torch.load(CKPT_PATH)["state_dict"].items()})
    model.to('cuda')
    model.eval()

    colm = ColModule()
    colm.load_state_dict(torch.load('../classification/training/checkpoints/sympie.pth'))
    colm.to('cuda')
    colm.eval()

    renorm = ReNorm()
    renorm.to('cuda')

    resize = Compose([
                        Resize(232, antialias=True),
                        CenterCrop(224)
                     ])

    colw = ColWarp()
    colw.eval()
    colw.to('cuda')

    clamp = Clamp()
    clamp.eval()
    clamp.to('cuda')

    for d in ["rgb", "gt", "pred"]:
        os.makedirs(PATH%("warp" if WARP else "orig", d), exist_ok=True)

    with torch.inference_mode():
        dloader = DataLoader(city,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=8)

        for ii, (x, y) in enumerate(tqdm(dloader)):
            metrics = Metrics(city.cnames)
            x, y = x.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)

            if WARP:
                ecol = colm(resize(x))
                x = colw(x, ecol)
                x = clamp(x)

            x = renorm(x)

            o, _ = model(x)
            f, _ = model(x[...,city.flipids])

            p = (o+f[...,city.flipids]).argmax(dim=1)

            metrics(p, y)

            plt.imsave(PATH%("warp" if WARP else "orig", "rgb")+"%04d.jpg"%ii, torgb(x[0].cpu()))
            plt.imsave(PATH%("warp" if WARP else "orig", "gt")+"%04d.png"%ii,
                        city.colorlabel(y[0].cpu()))
            plt.imsave(PATH%("warp" if WARP else "orig", "pred")+"%04d_%.2f.png"%(ii,
                        metrics.percent_mIoU()), city.colorlabel(p[0].cpu()))

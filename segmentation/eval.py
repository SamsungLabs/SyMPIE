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
from tqdm import tqdm
import yaml

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, CenterCrop, Compose

from cityscapes import Cityscapes
from metrics import Metrics
from renorm import ReNorm
from training_LSR.graphs.models.models import DeeplabNW

sys.path.append("../classification/training")
from model import ColModule, ColWarp, Clamp

def evaluate_city(colm, warp=True,
                  ckpt_path="training_LSR/log/train/cityscapesfinal.pth",
                  print_perclass=False):
    """
        evaluate on the cityscapes dataset
    """
    with open('../config_paths.yaml', encoding='utf-8') as yamlf:
        config = yaml.load(yamlf, yaml.BaseLoader)

    city = Cityscapes(root_path=config['cityscapes'],
                      split='val')

    model = DeeplabNW(19, 'ResNet50', pretrained=False)
    model.load_state_dict({k.replace("module.", ""):v for k,v in
                           torch.load(ckpt_path)["state_dict"].items()})
    model = DataParallel(model)
    model.to('cuda')
    model.eval()

    renorm = ReNorm(scale=255.,
                    new_mean=[104.00698793, 116.66876762, 122.67891434],
                    new_std=[1,1,1],
                    flip_chs=True)
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

    with torch.inference_mode():
        dloader = DataLoader(city,
                             batch_size=2,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=8)
        metrics = Metrics(city.cnames)

        pbar = tqdm(dloader, desc=f"Evaluating on CityScapes, current mIoU: {0}")
        for x, y in pbar:
            x, y = x.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)

            if warp:
                ecol = colm(resize(x))
                x = colw(x, ecol)
                x = clamp(x)

            x = renorm(x)

            o, _ = model(x)
            f, _ = model(x[...,city.flipids])

            p = (o+f[...,city.flipids]).argmax(dim=1)

            metrics(p, y)
            pbar.set_description(f"Evaluating on CityScapes, \
                                 current mIoU: {metrics.percent_mIoU()}")
    if print_perclass:
        print(metrics)
    return metrics.percent_mIoU()

def evaluate_acdc(colm, warp=True,
                  ckpt_path="training_LSR/log/train/cityscapesfinal.pth",
                  print_perclass=False):
    """
        evaluate on the acdc dataset
    """
    with open('../config_paths.yaml', encoding='utf-8') as yamlf:
        config = yaml.load(yamlf, yaml.BaseLoader)
    acdc = Cityscapes(root_path=config['acdc'],
                      split='val')

    model = DeeplabNW(19, 'ResNet50', pretrained=False)
    model.load_state_dict({k.replace("module.", ""):v for k,v
                           in torch.load(ckpt_path)["state_dict"].items()})
    model = DataParallel(model)
    model.to('cuda')
    model.eval()

    renorm = ReNorm(scale=255.,
                    new_mean=[104.00698793, 116.66876762, 122.67891434],
                    new_std=[1,1,1],
                    flip_chs=True)
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

    with torch.inference_mode():
        dloader = DataLoader(acdc,
                             batch_size=2,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=8)
        metrics = Metrics(acdc.cnames)

        pbar = tqdm(dloader, desc=f"Evaluating on ACDC, current mIoU: {0}")
        for x, y in pbar:
            x, y = x.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)

            if warp:
                ecol = colm(resize(x))
                x = colw(x, ecol)
                x = clamp(x)

            x = renorm(x)

            o, _ = model(x)
            f, _ = model(x[...,acdc.flipids])

            p = (o+f[...,acdc.flipids]).argmax(dim=1)

            metrics(p, y)
            pbar.set_description(f"Evaluating on ACDC, current mIoU: {metrics.percent_mIoU()}")
    if print_perclass:
        print(metrics)
    return metrics.percent_mIoU()

def evaluate_zurich(colm, warp=True,
                    ckpt_path="training_LSR/log/train/cityscapesfinal.pth",
                    print_perclass=False):
    """
        evaluate on darkzurich
    """
    with open('../config_paths.yaml', encoding='utf-8') as yamlf:
        config = yaml.load(yamlf, yaml.BaseLoader)
    acdc = Cityscapes(root_path=config['darkzurich'],
                      split='val')

    model = DeeplabNW(19, 'ResNet50', pretrained=False)
    model.load_state_dict({k.replace("module.", ""):v for k,v in
                           torch.load(ckpt_path)["state_dict"].items()})
    model = DataParallel(model)
    model.to('cuda')
    model.eval()

    renorm = ReNorm(scale=255.,
                    new_mean=[104.00698793, 116.66876762, 122.67891434],
                    new_std=[1,1,1],
                    flip_chs=True)
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

    with torch.inference_mode():
        dloader = DataLoader(acdc,
                             batch_size=2,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=8)
        metrics = Metrics(acdc.cnames)

        pbar = tqdm(dloader, desc=f"Evaluating on Dark Zurich, current mIoU: {0}")
        for x, y in pbar:
            x, y = x.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)

            if warp:
                ecol = colm(resize(x))
                x = colw(x, ecol)
                x = clamp(x)

            x = renorm(x)

            o, _ = model(x)
            f, _ = model(x[...,acdc.flipids])

            p = (o+f[...,acdc.flipids]).argmax(dim=1)

            metrics(p, y)
            pbar.set_description(f"Evaluating on Dark Zurich, \
                                 current mIoU: {metrics.percent_mIoU()}")
    if print_perclass:
        print(metrics)
    return metrics.percent_mIoU()

if __name__ == "__main__":

    module = ColModule()

    module.load_state_dict(torch.load('../imagenet/checkpoints/our_acdc.pth'))
    module.to('cuda')
    module.eval()

    BASE = "training_LSR/log/train/cityscapesfinal.pth"
    #BASE = "training_LSR/log/acdc/cityscapesfinal.pth"

    WARP = True

    print(evaluate_city(module, warp=WARP, ckpt_path=BASE, print_perclass=True))
    print(evaluate_acdc(module, warp=WARP, ckpt_path=BASE, print_perclass=True))
    print(evaluate_zurich(module, warp=WARP, ckpt_path=BASE, print_perclass=True))

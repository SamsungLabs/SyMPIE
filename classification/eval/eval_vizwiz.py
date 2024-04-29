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
import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet50, ResNet50_Weights

from vizwiz import VizWiz

sys.path.append("../training")
from model import ColModule, ColWarp, Clamp

if __name__ == "__main__":

    colm = ColModule()
    colm.load_state_dict(torch.load('../training/checkpoints/sympie.pth'))
    colm.to('cuda')
    colm.eval()

    colw = ColWarp()
    colw.to('cuda')
    colw.eval()

    clamp = Clamp()
    clamp.to('cuda')
    clamp.eval()

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    #model = resnet50()
    #model.load_state_dict({k.replace("module.", ""): v for k,v in
    #    torch.load("../imagenet/checkpoints/3_ours_da_augmix_stronger_blur.pth")\
    #    ["state_dict"].items()})
    model.to('cuda')
    model.eval()

    dset = VizWiz()
    dloader = DataLoader(dset,
                         batch_size=1,
                         shuffle=False,
                         drop_last=True,
                         num_workers=8,
                         pin_memory=True)

    dict_warp = {}
    dict_orig = {}

    with torch.inference_mode():
        for im, name in tqdm(dloader):
            im = im.to('cuda')

            ecol = colm(im)
            im2 = colw(im, ecol)
            im2 = clamp(im2)

            out_orig = model(im)[:,dset.valid].argmax(dim=1)
            out_warp = model(im2)[:,dset.valid].argmax(dim=1)

            dict_orig[name[0]] = dset.valid[out_orig.item()]
            dict_warp[name[0]] = dset.valid[out_warp.item()]

    with open("pred_warp_8gpu_yucel.json", "w", encoding='utf-8') as fwarp:
        json.dump(dict_warp, fwarp)

    with open("pred_orig_8gpu_yucel.json", "w", encoding='utf-8') as forig:
        json.dump(dict_orig, forig)

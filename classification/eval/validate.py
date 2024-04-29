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
import sys
import os
from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from torchvision.models.resnet import resnet50, ResNet50_Weights,\
        resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from torchvision.models.vgg import vgg13, VGG13_Weights, vgg13_bn,\
        VGG13_BN_Weights, vgg16, VGG16_Weights, vgg16_bn, VGG16_BN_Weights
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights

import clip

from imagenetc import ImagenetC
from imagenetcbar import ImagenetCBar

os.environ['OMP_NUM_THREADS'] = '1'

sys.path.append("../training")
from model import ColWarp, ColModule, Clamp

sys.path.append("../clip")
from parallel_clip import ImageCLIP, image_logits

def get_clas(mname):
    """
        parse classifier from name
    """
    if mname == "rn50_1":
        l_clas = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif mname == "rn50_2":
        l_clas = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif mname == "yucel":
        l_clas = resnet50()
        l_clas.load_state_dict({k.replace("module.", ""): v for k,v in
            torch.load("../training/checkpoints/3_ours_da_augmix_stronger_blur.pth")["state_dict"].items()})
    elif mname == "prime":
        l_clas = resnet50()
        l_clas.load_state_dict({k.replace("module.", ""): v for k,v in
            torch.load("../training/checkpoints/ResNet50_ImageNet_PRIME_noJSD.ckpt").items()})
    elif mname == "pixmix":
        l_clas = resnet50()
        l_clas.load_state_dict({k.replace("module.", ""): v for k,v in
            torch.load("../training/checkpoints/PIXMIX.pth")["state_dict"].items()})
    elif mname == "rn18":
        l_clas = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif mname == "rn34":
        l_clas = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    elif mname == "vgg13bn":
        l_clas = vgg13_bn(weights=VGG13_BN_Weights.IMAGENET1K_V1)
    elif mname == "vgg13":
        l_clas = vgg13(weights=VGG13_Weights.IMAGENET1K_V1)
    elif mname == "vgg16bn":
        l_clas = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
    elif mname == "vgg16":
        l_clas = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    elif mname == "mnet_1":
        l_clas = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    elif mname == "mnet_2":
        l_clas = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    elif mname == "vitb16":
        l_clas = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    elif mname == "swint":
        l_clas = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    elif mname == "clip":
        l_clas = ImageCLIP(clip.load("RN50")[0])
    else:
        raise ValueError(f"Unknown Model {mname}")
    l_clas.eval()
    return l_clas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clas", type=str, default="rn50_2",
                        choices=["rn50_1", "rn50_2", "yucel", "rn18",
                                 "rn34", "vgg13bn", "vgg13", "vgg16bn",
                                 "vgg16", "mnet_1", "mnet_2", "vitb16",
                                 "swint", "prime", "pixmix", "clip"])
    parser.add_argument("--colm", type=str, default="none")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--bar", action="store_true")
    args = parser.parse_args()

    print(f"Running with: {vars(args)}")

    with open('../../config_paths.yaml', encoding='utf-8') as yamlf:
        config = yaml.load(yamlf, yaml.BaseLoader)

    if args.bar:
        dset = ImagenetCBar(root_path=config['imagenetc-bar'],
                            split_path=config['imagenetc-bar'],
                            subsample=args.quick)
    else:
        dset = ImagenetC(root_path=config['imagenetc'],
                         split_path=config['imagenetc'],
                         subsample=args.quick)
    dloader = DataLoader(dataset=dset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.workers if not (args.debug or args.quick) else 8,
                         pin_memory=True,
                         drop_last=False,
                         persistent_workers=True and args.workers > 0)

    colm = ColModule()
    if args.colm != "none":
        colm.load_state_dict(torch.load(f"../training/checkpoints/{args.colm}.pth"))
    colm = DataParallel(colm)
    colm.eval()
    colm.to('cuda')

    warp = ColWarp()
    warp = DataParallel(warp)
    warp.eval()
    warp.to('cuda')

    clas = get_clas(args.clas)
    clas = DataParallel(clas)
    clas.eval()
    clas.to('cuda')

    clamp = Clamp()
    clamp = DataParallel(clamp)
    clamp.eval()
    clamp.to('cuda')

    if args.clas == "clip":
        with open("../clip/clip_rn50.json", encoding='utf-8') as fin:
            clas_embeds = torch.tensor(json.load(fin), dtype=torch.float16).to('cuda')

    out_grid = torch.zeros(len(dset.corruptions), len(dset.intensities), device='cuda')
    cts_grid = torch.zeros(len(dset.corruptions), len(dset.intensities), device='cuda')

    racc = 0. 
    it = 0 
    with torch.no_grad():
        pbar = tqdm(dloader, smoothing=0, desc="Evaluating, batch accuracy: %.2f"%0)
        for im, cl, ii, ci in pbar:
            im = im.to('cuda', dtype=torch.float32, non_blocking=True)
            cl = cl.to('cuda', dtype=torch.long, non_blocking=True)

            if args.colm != 'none':
                im = warp(im, colm(im))
                im = clamp(im)

            out = clas(im)
            if args.clas == "clip":
                out = image_logits(out, clas_embeds)
            pred = out.argmax(dim=1)

            for b in range(args.batch_size):
                out_grid[ci[b], ii[b]] += pred[b] == cl[b]
                cts_grid[ci[b], ii[b]] += 1

            racc += (pred==cl).float().mean().item()
            it += 1
            pbar.set_description("Evaluating, average accuracy: %.2f"%(100 * racc/it))

            if args.debug:
                break

    out_grid *= 100/cts_grid
    os.makedirs("json", exist_ok=True)
    with open(f"json/{('bar_' if args.bar else '')}{('quick_' if args.quick else '')}{args.colm}_{args.clas}.json", "w", encoding='utf-8') as fout:
        json.dump(out_grid.cpu().numpy().tolist(), fout)

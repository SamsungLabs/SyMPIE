################################################################################
# This code is adapted from: https://github.com/pytorch/vision)
# Copyright (c) 2016.
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

import time
import sys
import random

from numpy import random as npr

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.models.resnet import resnet50, ResNet50_Weights, \
    resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from torchvision.models.vgg import vgg13, VGG13_Weights, vgg13_bn, \
    VGG13_BN_Weights, vgg16, VGG16_Weights, vgg16_bn, VGG16_BN_Weights
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights

from imagenetc import DistortImageFolder
from model import CorrectColMulti

sys.path.append("../../segmentation")
from eval import evaluate_city, evaluate_acdc

def set_seed(seed):
    """
        set random sed
    """
    torch.manual_seed(seed)
    npr.seed(seed)
    random.seed(seed)

def clamp_grad_norm(grad: torch.Tensor):
    """
        clamp gradient norm
    """
    norm = grad.norm()
    if norm < 1:
        return grad
    return grad/norm

def get_clas(mname):
    """
        get classifier from name
    """
    if mname == "rn50_1":
        clas = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif mname == "rn50_2":
        clas = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif mname == "yucel":
        clas = resnet50()
        clas.load_state_dict({k.replace("module.", ""): v for k,v in \
            torch.load("checkpoints/3_ours_da_augmix_stronger_blur.pth")["state_dict"].items()})
    elif mname == "rn18":
        clas = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif mname == "rn34":
        clas = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    elif mname == "vgg13bn":
        clas = vgg13_bn(weights=VGG13_BN_Weights.IMAGENET1K_V1)
    elif mname == "vgg13":
        clas = vgg13(weights=VGG13_Weights.IMAGENET1K_V1)
    elif mname == "vgg16bn":
        clas = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
    elif mname == "vgg16":
        clas = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    elif mname == "mnet_1":
        clas = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    elif mname == "mnet_2":
        clas = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unknown Model {mname}")
    clas.eval()
    return clas

def init_model(cnames):
    """
        initialize models with multiple classifiers
    """
    classifiers = [get_clas(mname) for mname in cnames]
    model = CorrectColMulti(classifiers)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.train()
    for clas in model.classifiers:
        clas.eval()
        for p in clas.parameters():
            p.requires_grad = False

    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(clamp_grad_norm)

    return model

def to_h_m_s(secs):
    """
        convert seconds to hours mins secs
    """
    h = secs//3600
    secs -= h*3600
    m = secs//60
    secs -= m*60
    return h, m, secs

def main(rank, world_size, seed):
    """
        main body, as required by DDP
    """
    dist_url = "env://"

    # select the correct cuda device
    torch.cuda.set_device(rank)

    # initialize the process group
    print(f"| distributed init (rank {rank}): {dist_url}", flush=True)
    dist.init_process_group("nccl",
                            rank=rank,
                            init_method=dist_url,
                            world_size=world_size)
    dist.barrier()

    iterations = 50000
    iters_per_epoch = 500
    epochs = int(round(iterations/iters_per_epoch))

    mnames = ["rn50_2"]
    ema_rate = .9

    set_seed(seed)

    if rank == 0:
        # import stuff here, we need them only on rank 0 process
        # imports at the top level slow everything down
        from matplotlib import pyplot as plt
        from torch.utils.tensorboard.writer import SummaryWriter
        from shutil import rmtree 

        rmtree(args.logdir, ignore_errors=True)
        os.makedirs(args.logdir+"/ckpts", exist_ok=True)
        os.makedirs(args.logdir+"/ckpts_ema", exist_ok=True)
        writer = SummaryWriter(args.logdir, flush_secs=.5)
        plt.switch_backend('agg')

        # download weights on rank 0 process if needed:
        for mname in mnames:
            get_clas(mname)

    dist.barrier()
    tset = DistortImageFolder(args.imagenet,
                              use_colwarp=True,
                              max_corruptions=1)
    vset = DistortImageFolder(args.imagenet,
                              use_colwarp=True,
                              max_corruptions=1,
                              is_val=True)

    tsampler = DistributedSampler(tset, num_replicas=world_size, rank=rank, shuffle=True)
    tloader = DataLoader(tset,
                         48,
                         pin_memory=True,
                         drop_last=True,
                         num_workers=4,
                         sampler=tsampler)

    vsampler = DistributedSampler(vset, num_replicas=world_size, rank=rank, shuffle=True)
    vloader = DataLoader(vset,
                         48,
                         pin_memory=True,
                         drop_last=True,
                         num_workers=1,
                         sampler=vsampler)

    model = init_model(mnames)
    model.to("cuda")
    model = DDP(model, device_ids=[rank])

    model_ema = init_model(mnames)
    model_ema.eval()
    model_ema.to('cuda')
    model_ema = DDP(model_ema, device_ids=[rank])

    celoss = CrossEntropyLoss()
    celoss.to("cuda")

    optim = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optim, epochs*min(len(tloader), iters_per_epoch))

    buw = float("-inf")
    buo = float("-inf")
    bcity = float("-inf")
    bacdc = float("-inf")

    it = 0
    for e in range(epochs):
        tloader.sampler.set_epoch(e)
        model.module.colm.train()
        model.module.colw.train()
        for ii, (x, y, z) in enumerate(tloader):
            optim.zero_grad()

            x, y, z = x.to("cuda", dtype=torch.float32, non_blocking=True), \
                            y.to("cuda", dtype=torch.long, non_blocking=True), \
                                z.to("cuda", dtype=torch.float32, non_blocking=True)

            clasid = it%len(mnames)

            # use mean teacher to get prediction
            with torch.no_grad():
                _, _, eu1 = model_ema(x, clasid)

            p1, ecol, u1 = model(x, clasid)
            ce1 = celoss(p1, y)
            ce1.backward()

            p2, _, u2 = model(eu1, clasid)
            ce2 = celoss(p2, y)
            (.5*ce2).backward()

            l = ce1 + ce2

            with torch.inference_mode():
                pred = p1.argmax(dim=-1)
                uacc = 100*(y==pred).float().mean()

                p1 = model.module.classifiers[clasid](z)
                pred = p1.argmax(dim=-1)
                oacc = 100*(y==pred).float().mean()

                p1 = model.module.classifiers[clasid](x)
                pred = p1.argmax(dim=-1)
                wacc = 100*(y==pred).float().mean()

            optim.step()
            scheduler.step()

            # reset the classifier and update ema model
            model.module.classifiers[clasid].load_state_dict(get_clas(mnames[clasid]).state_dict())
            #if it > 0 and it%32 == 0:
            esdict = model_ema.module.state_dict()
            sdict = model.module.state_dict()
            for k in esdict:
                esdict[k] = ema_rate*esdict[k] + (1-ema_rate)*sdict[k]
            model_ema.module.load_state_dict(esdict)

            if rank == 0:
                dlen = min(len(tloader), iters_per_epoch)
                if ii % 10 == 0:
                    if ii == 0:
                        start = time.time()
                        print("Starting Training Epoch [%03d/%03d]"%(e+1, epochs))
                    else:
                        dtime = time.time()-start
                        print("Training Epoch [%03d/%03d] - Step [%05d/%05d] - Model [%s] - N. Corrupt. [%d] - Epoch Time [%02dh %02dm %02.3fs] - Epoch ETA [%02dh %02dm %02.3fs]- Loss %02.3f"% (
                                e+1, epochs, ii, dlen, mnames[clasid], tloader.dataset.max_corruptions, *to_h_m_s(dtime), *to_h_m_s((dlen-ii)*dtime/max(ii,1)), l.item()))

                writer.add_scalar("train/lr", optim.param_groups[0]['lr'], it)
                writer.add_scalar("train/loss", l.item(), it)
                writer.add_scalars("train/ce", {"round1": ce1.item(), "round2": ce2.item()}, it)

                writer.add_scalar("train/dacc_uw", uacc.item()-wacc.item(), it)
                writer.add_scalar("train/dacc_uo", uacc.item()-oacc.item(), it)
                writer.add_scalars("train/acc", {"original": oacc.item(),
                                                "warped": wacc.item(), "unwarped": uacc.item()}, it)

                if it % min(500, len(tloader)-1) == 0:
                    fig, axs = plt.subplots(2,2)
                    axs[0,0].imshow(tset.to_rgb(x[0].detach().cpu()))
                    axs[0,0].axis('off')
                    axs[0,0].set_title("input1")

                    axs[0,1].imshow(tset.to_rgb(eu1[0].detach().cpu()))
                    axs[0,1].axis('off')
                    axs[0,1].set_title("input2")

                    axs[1,0].imshow(tset.to_rgb(u1[0].detach().cpu()))
                    axs[1,0].axis('off')
                    axs[1,0].set_title("pred1")

                    axs[1,1].imshow(tset.to_rgb(u2[0].detach().cpu()))
                    axs[1,1].axis('off')
                    axs[1,1].set_title("pred2")

                    fig.tight_layout()
                    writer.add_figure("image", fig, it, close=True)

                    fig, ax = plt.subplots(1,1)
                    im = ax.imshow(ecol[0,:12].reshape(4,3).detach().cpu().numpy())
                    fig.colorbar(im, ax=ax)
                    writer.add_figure("warp/color", fig, it, close=True)

                    fig, ax = plt.subplots(1,1)
                    im = ax.imshow(ecol[0,12:].reshape(5,5).detach().cpu().numpy())
                    fig.colorbar(im, ax=ax)
                    writer.add_figure("warp/spatial", fig, it, close=True)
            it += 1
            if ii >= iters_per_epoch:
                break

        vloader.sampler.set_epoch(e)
        model.module.colm.eval()
        model.module.colw.eval()
        with torch.inference_mode():
            a_uacc = 0.
            a_oacc = 0.
            a_wacc = 0.
            for ii, (x, y, z) in enumerate(vloader):
                optim.zero_grad()

                x, y, z = x.to("cuda", dtype=torch.float32, non_blocking=True), \
                                y.to("cuda", dtype=torch.long, non_blocking=True), \
                                    z.to("cuda", dtype=torch.float32, non_blocking=True)

                clasid = 0 #it%len(mnames)

                # use mean teacher to get prediction
                with torch.no_grad():
                    _, _, eu1 = model_ema(x, clasid)

                p1, ecol, u1 = model(x, clasid)
                p2, _, u2 = model(eu1, clasid)

                pred = p1.argmax(dim=-1)
                a_uacc += 100*(y==pred).float().mean()

                p1 = model.module.classifiers[clasid](z)
                pred = p1.argmax(dim=-1)
                a_oacc += 100*(y==pred).float().mean()

                p1 = model.module.classifiers[clasid](x)
                pred = p1.argmax(dim=-1)
                a_wacc += 100*(y==pred).float().mean()

                if rank == 0:
                    dlen = len(vloader)
                    if ii % 10 == 0:
                        if ii == 0:
                            start = time.time()
                            print("Starting Validation Epoch [%03d/%03d]"%(e+1, epochs))
                        else:
                            dtime = time.time()-start
                            print("Validation Epoch [%03d/%03d] - Step [%05d/%05d] - Model [%s] - N. Corrupt. [%d] - Epoch Time [%02dh %02dm %02.3fs] - Epoch ETA [%02dh %02dm %02.3fs]- Loss %02.3f"%(
                                    e+1, epochs, ii, dlen, mnames[clasid], vloader.dataset.max_corruptions, *to_h_m_s(dtime), *to_h_m_s((dlen-ii)*dtime/max(ii,1)), l.item()))

            if rank == 0:
                d_uw = (a_uacc.item()-a_wacc.item())/len(vloader)
                d_uo = (a_uacc.item()-a_oacc.item())/len(vloader)

                if buw < d_uw:
                    buw = d_uw
                    torch.save(model.module.colm.state_dict(), args.logdir+"/best_uw.pth")
                if buo < d_uo:
                    buo = d_uo
                    torch.save(model.module.colm.state_dict(), args.logdir+"/best_uo.pth")

                city = evaluate_city(model.module.colm,
                        ckpt_path="checkpoints/cityscapesfinal.pth")

                if bcity < city:
                    bcity = city
                    torch.save(model.module.colm.state_dict(), args.logdir+"/best_city.pth")

                acdc = evaluate_acdc(model.module.colm,
                        ckpt_path="checkpoints/cityscapesfinal.pth")

                if bacdc < acdc:
                    bacdc = acdc
                    torch.save(model.module.colm.state_dict(), args.logdir+"/best_acdc.pth")

                writer.add_scalar("val/dacc_uw", d_uw, it)
                writer.add_scalar("val/dacc_uo", d_uo, it)
                writer.add_scalar("val/city", city, it)
                writer.add_scalar("val/acdc", acdc, it)

        dist.barrier()

    if rank == 0:
        torch.save(model.module.colm.state_dict(), args.logdir+"/final.pth")
        torch.save(model_ema.module.colm.state_dict(), args.logdir+"/final_ema.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    import yaml
    import os

    with open('../../config_paths.yaml', encoding='utf-8') as yamlf:
        config = yaml.load(yamlf, yaml.BaseLoader)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", default='logs/sympie_rerun')
    for k, v in config.items():
        parser.add_argument('--'+k, default=v)
    args = parser.parse_args()

    main(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), args.seed)

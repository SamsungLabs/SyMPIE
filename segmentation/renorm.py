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

import torch
from torch.nn import Module

class ReNorm(Module):
    """
        re-normalizes the input image to change 
        between segmentation and unwarping modules
    """
    def __init__(self,
                 old_mean=(0.485, 0.456, 0.406),
                 old_std=(0.229, 0.224, 0.225),
                 scale=255.,
                 new_mean=(104.00698793, 116.66876762, 122.67891434),
                 new_std=(1,1,1),
                 flip_chs=True):
        super().__init__()

        self.scale = scale
        self.chs_ids = torch.arange(2,-1,-1) if flip_chs else torch.arange(3)
        self.register_buffer('old_mean', torch.tensor(old_mean).reshape(1,3,1,1), persistent=False)
        self.register_buffer('old_std', torch.tensor(old_std).reshape(1,3,1,1), persistent=False)
        self.register_buffer('new_mean', torch.tensor(new_mean).reshape(1,3,1,1), persistent=False)
        self.register_buffer('new_std', torch.tensor(new_std).reshape(1,3,1,1), persistent=False)

    def forward(self, ts):
        """
            torch module forward
        """
        ts = ts*self.old_std + self.old_mean
        ts = self.scale*ts[:,self.chs_ids].clone()
        ts = (ts-self.new_mean)/self.new_std
        return ts

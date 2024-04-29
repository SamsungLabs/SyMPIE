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
from torch import nn

class ReNorm(nn.Module):
    """
        re-normalizes the tensors to align between
        unwarper and clip
    """
    def __init__(self,
                 old_mean=(0.485, 0.456, 0.406),
                 old_std=(0.229, 0.224, 0.225),
                 scale=1,
                 new_mean=(0.48145466, 0.4578275, 0.40821073),
                 new_std=(0.26862954, 0.26130258, 0.27577711),
                 flip_chs=False):
        super().__init__()

        self.scale = scale
        self.chs_ids = torch.arange(2,-1,-1) if flip_chs else torch.arange(3)
        self.register_buffer('old_mean', torch.tensor(old_mean).reshape(1,3,1,1), persistent=False)
        self.register_buffer('old_std', torch.tensor(old_std).reshape(1,3,1,1), persistent=False)
        self.register_buffer('new_mean', torch.tensor(new_mean).reshape(1,3,1,1), persistent=False)
        self.register_buffer('new_std', torch.tensor(new_std).reshape(1,3,1,1), persistent=False)

    def forward(self, x):
        """
            torch module forward
        """
        x = x*self.old_std + self.old_mean
        x = self.scale*x[:,self.chs_ids].clone()
        x = (x-self.new_mean)/self.new_std
        return x

class TextCLIP(nn.Module):
    """
        wrapper for text prediction
    """
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        """
            torch module forward
        """
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    """
        wrapper for image prediction
    """
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        self.renorm = ReNorm()

    def forward(self,image):
        """
            torch module forward
        """
        image = self.renorm(image)
        return self.model.encode_image(image)

def image_logits(x1,x2):
    """
        compute logits for image classification
    """
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  x1 @ x2.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1

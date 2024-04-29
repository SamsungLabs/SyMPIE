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

import os, shutil
from tqdm import tqdm

def get_class(path):
    with open(path, 'r') as f:
        for _ in range(13):
            f.readline()
        c = f.readline().split('>')[1].split('<')[0]
    return c

raw_path = "val_raw"
clean_path = "val"
clas_path = "../../Annotations/CLS-LOC/val"

fnames = [f.split('.')[0] for f in os.listdir(raw_path) if not f.startswith('.') and os.path.isfile(os.path.join(raw_path, f))]
classes = [get_class(os.path.join(clas_path, f+'.xml')) for f in fnames]

for c, f in tqdm(zip(classes, fnames)):
    fpath = os.path.join(clean_path, c)
    os.makedirs(fpath, exist_ok=True)
    shutil.copy(os.path.join(raw_path, f+'.JPEG'), os.path.join(fpath, f+'.jpg'))

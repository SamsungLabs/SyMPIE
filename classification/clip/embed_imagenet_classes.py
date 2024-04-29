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
import torch
import clip

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MNAME = "RN50" #"ViT-B/32"

    with open('classes.csv', encoding='utf-8') as fin:
        cnames = ["A photo of a "+l.strip().split(",")[1].replace('"', '')+"."
                    for l in fin if len(l.strip()) > 0]

    model, preprocess = clip.load(MNAME, device=DEVICE)
    model.eval()

    with torch.inference_mode():
        tokens = clip.tokenize(cnames).to(DEVICE)
        text_features = model.encode_text(tokens)
        print(text_features.shape)

    with open("clip_"+MNAME.lower().replace("/", "")+".json", "w", encoding='utf-8') as fout:
        json.dump(text_features.cpu().numpy().tolist(), fout)

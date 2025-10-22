# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.utils.type_conversion import convert_data_type
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class LabelToBinaryClassd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, offset=2) -> None:
        super().__init__(keys, allow_missing_keys)
        self.offset = offset

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = int(torch.max(d[key]))
            d[key] = label - self.offset if label else 0
        return d

class SaveImagePred(MapTransform):
    def __init__(
        self,
        keys: KeysCollection
    ):
        super().__init__(keys)

    def __call__(self, data):
        print("____start SaveImagePre____")
        d = dict(data)
        for key in self.keys:
            img = np.squeeze(d[key] * 255).astype(np.uint8)
            print(img)
            print(img.shape)
            out = Image.fromarray(img)
#             out = out.rotate(-90, expand=True)
#             out = out.transpose(Image.FLIP_LEFT_RIGHT)
            out.save('/claraDevDay/testPre_inf.png')
            
        print("____end SaveImagePre____")
        return d


class PVTNetSumOutd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection
    ):
        super().__init__(keys)

    # def __call__(self, data):
    #     d = dict(data)
    #     for key in self.keys:
    #         out = d[key]
    #         out_tensor = convert_data_type(out, torch.Tensor)
    #         print(out)
    #         print(out_tensor)
    #         #tup_s = (out_tensor[0])
    #         #s = torch.stack(tup_s)
    #         s = torch.stack([out_tensor[0]])
          
    #         p = torch.sum(s, dim = 0, keepdim=False)
    #         d[key] = p
    #     return d
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            out = d[key]
            out_tensor = convert_data_type(out, torch.Tensor)
            if isinstance(out_tensor, (list, tuple)):
                for i, t in enumerate(out_tensor):
                    if torch.is_tensor(t):
                        min_val = torch.min(t).item()
                        max_val = torch.max(t).item()
                        print(f"  Level {i}: shape={t.shape}, min={min_val:.4f}, max={max_val:.4f}, type={type(t)}")
                    else:
                        print(f"  Level {i}: not a tensor, type={type(t)}")
            else:
                if torch.is_tensor(out_tensor):
                    min_val = torch.min(out_tensor).item()
                    max_val = torch.max(out_tensor).item()
                    print(f"  Single tensor: shape={out_tensor.shape}, min={min_val:.4f}, max={max_val:.4f}, type={type(out_tensor)}")
                else:
                    print(f"  Single object of type {type(out_tensor)}")
            

            s = torch.stack([out_tensor[0]])
            p = torch.sum(s, dim=0, keepdim=False)

            print(f"[PVTNetSumOutd] after sum shape: {p.shape}")
            print(f"[PVTNetSumOutd] after sum min={p.min().item():.4f}, max={p.max().item():.4f}")

            d[key] = p
        return d
    

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
from typing import Callable, Sequence, Dict

from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    AddGuidanceSignald,
    Fetch2DSliced,
    ResizeGuidanced,
    RestoreLabeld,
    SpatialCropGuidanced,
)
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    # AddChanneld,
    # AsChannelFirstd,
    AsChannelLastd,
    AsDiscreted,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    Spacingd,
    ToNumpyd,
    SaveImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ResizeWithPadOrCropd,
    KeepLargestConnectedComponentd,
    Invertd,
)

from monailabel.transform.post import (Restored, DumpImagePrediction2Dd,)
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from lib.transforms import (SaveImagePred, PVTNetSumOutd,)
from monailabel.transform.writer import Writer

import logging
logger = logging.getLogger(__name__)

class Usg(BasicInferTask):
    def __init__(self, path: str, network, conf: Dict[str, str], **kwargs):
        # Override Labels
        # self.labels = {"Tumor": 255}
        # self.label_colors = {"Tumor": (255, 255, 0)}
        super().__init__(path=path, network=network, type=InferType.SEGMENTATION, **kwargs, dimension=2, description="USG segment")
        self.labels = {"Tumor": 255}
        print(conf)
        logger.info("init infer")



    def pre_transforms(self, data=None) -> Sequence[Callable]:
        logger.info("Applying LoadImaged")
        t = [
            LoadImaged(keys="image"),
            #EnsureTyped(keys="image"),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityd(keys="image"),
            #RandAxisFlipd(keys="image",prob="0.2"),
            #Resized(keys="image",spatial_size=(512,512),mode="nearest"	),
            ResizeWithPadOrCropd(keys="image", spatial_size=[512, 512]),
            #Spacingd(keys="image", pixdim=[1.0] * self.dimension, mode="bilinear"),
        ]
        '''
        self.add_cache_transform(t, data)
        t.append(AddGuidanceFromPointsd(ref_image="image", guidance="guidance", spatial_dims=self.dimension))

        if self.dimension == 2:
            t.append(Fetch2DSliced(keys="image", guidance="guidance"))
        t.extend(
            [
                AddChanneld(keys="image"),
                SpatialCropGuidanced(keys="image", guidance="guidance", spatial_size=self.spatial_size),
                Resized(keys="image", spatial_size=self.model_size, mode="area"),
                ResizeGuidanced(guidance="guidance", ref_image="image"),
                NormalizeIntensityd(keys="image", subtrahend=208, divisor=388),  # type: ignore
                AddGuidanceSignald(image="image", guidance="guidance"),
                EnsureTyped(keys="image", device=data.get("device") if data else None),
            ]
        )
        '''
        return t
    
    
    def inferer(self, data=None) -> Inferer:
        logger.info("inference")
        return SimpleInferer()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            PVTNetSumOutd(keys="pred"),
            Activationsd(keys="pred", sigmoid=True),
 #           Invertd(
 #           keys="pred",
 #            transform=test_org_transforms,
 #           orig_keys="image",
 #           meta_keys="pred_meta_dict",
 #           orig_meta_keys="image_meta_dict",
 #           meta_key_postfix="meta_dict",
 #           nearest_interp=False,
 #           to_tensor=True,
#             ),
            #AsDiscreted(keys="pred",labels=self.labels, threshold=0.5),
            AsDiscreted(keys="pred", threshold=0.5),
#            PVTNetSumOutd(keys="pred"),
            #ToNumpyd(keys="pred"),
            KeepLargestConnectedComponentd(keys="pred"),
            #RestoreLabeld(keys="pred", ref_image="image", mode="nearest"),
            #AsChannelLastd(keys="pred"),
            
            #SaveImagePred(keys="pred"),
            #Restored(keys="pred", ref_image="image"),
            
            #SaveImagePred(keys="pred"),
            #SaveImaged(keys="pred", output_dir=".", output_ext=".nii.gz", output_postfix="seg"),
#            DumpImagePrediction2Dd("/claraDevDay/image.png","/claraDevDay/pred.png"),
            #Restored(keys="pred", ref_image="image"),
            AsChannelLastd(keys="pred"),


        ]
    
    def writer(self, data, extension=None, dtype=None):
        writer = Writer(label=self.output_label_key, key_extension="nii.gz",nibabel=True)
        return writer(data)



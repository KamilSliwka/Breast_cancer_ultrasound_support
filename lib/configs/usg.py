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
import os
from typing import Any, Dict, Optional, Union
import torch
import lib.infers
import lib.trainers
from monai.bundle import download

from monailabel.config import settings
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.utils.others.generic import strtobool
from monailabel.utils.others.generic import download_file, strtobool
from monai.networks.nets import UNet
from lib.models import PolypPVT

logger = logging.getLogger(__name__)


#

#from lib.infers.test import TestInfer

class Usg(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        self.labels = [
            "tumor",
        ]

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{self.name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{self.name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            url = f"{url}/radiology_deepgrow_2d_bunet.pt"
            download_file(url, self.path[0])

        device = "cuda" if torch.cuda.is_available() else "cpu" 
        # Network
        self.network = PolypPVT() 
        '''        self.network = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2,2,2,2),
            num_res_units=3,
            dropout=0.2,
            norm='BATCH',
        )
        '''
        #logger.info(f"Network w configs/usg.py {self.network}")
        self.config = conf
        self.planner = planner

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        #task: InferTask = lib.infers.ToolTracking(self.bundle_path, self.conf)
        logger.info("infer")
        task: InferTask = lib.infers.Usg(
            path=self.path,
            network=self.network,
            labels=self.labels,
            #preload=strtobool(self.conf.get("preload", "false")),
            conf={"load_strict": True} #self.config, #{"cache_transforms": True, "cache_transforms_in_memory": True, "cache_transforms_ttl": 300},
        )
    
        
        
        
        return task
    
    
    

    def trainer(self) -> Optional[TrainTask]:
        #task: TrainTask = lib.trainers.ToolTracking(self.bundle_path, self.conf)
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.Usg(
            model_dir=output_dir,
            network=self.network,
            load_path=load_path,
            publish_path=self.path[1],
            description="Train 2D Deepgrow model",
            dimension=2,
            labels=self.labels,
            roi_size=(256, 256),
            model_size=(256, 256),
            max_train_interactions=10,
            max_val_interactions=5,
            val_interval=5,
            config={
                "max_epochs": 10,
                "train_batch_size": 16,
                "val_batch_size": 16,
            },
        )
        return task
'''
    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        strategies: Dict[str, Strategy] = {}
        if self.epistemic_enabled:
            strategies[f"{self.name}_epistemic"] = Epistemic()
        return strategies

    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        methods: Dict[str, ScoringMethod] = {}

        if self.epistemic_enabled:
            methods[f"{self.name}_epistemic"] = CVATEpistemicScoring(
                top_k=int(self.conf.get("epistemic_top_k", "10")),
                infer_task=lib.infers.ToolTracking(
                    self.bundle_path,
                    self.conf,
                    dropout=0.2,
                    train_mode=True,
                    skip_writer=True,
                ),
                function="monailabel.endoscopy.tooltracking",
                max_samples=self.epistemic_max_samples,
                simulation_size=self.epistemic_simulation_size,
                use_variance=True,
            )
        return methods
'''



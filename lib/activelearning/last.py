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
import random

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.scoring import ScoringMethod

logger = logging.getLogger(__name__)


class Last(Strategy):
    """
    Consider implementing a first strategy for active learning
    """

    def __init__(self):
        super().__init__("Get Last Sample")

    def __call__(self, request, datastore: Datastore):
        images = datastore.get_unlabeled_images()
        if not len(images):
            return None

        images.sort()
        image = images[-1]

        logger.info(f"First: Selected Image: {image}")
        return {"id": image}
# class SelectImageWithMyScore(Strategy):
    
#     """
#     Place for my own active learning strategy
#     """
    
#     def __init__(self):
#         super().__init__("Get Sample with the best score")

#     def __call__(self, request, datastore: Datastore):
#         images = datastore.get_unlabeled_images()
#         if not len(images):
#             return None
        
#         image = random.choice(images)
#         logger.info(f"Randomly selected Image '{image}'")
     
#         return {"id": image}
    
class SelectImageWithMyScore(Strategy):
    def __init__(self,scoring_method: ScoringMethod):
        super().__init__("Get Sample with the best score")
        self.scoring_method = scoring_method
        self.current_strategy = 0
    def __call__(self, request, datastore: Datastore):
        self.scoring_method(request, datastore) 
        images = datastore.get_unlabeled_images()
        if not len(images):
            return None

        my_scores = {
            image: {
                "target": datastore.get_image_info(image).get("my_score_target", 0),
                "boundary_driven": datastore.get_image_info(image).get("my_score_boundary_driven", 0)
            }
            for image in images
        }
        score_type = ["target","boundary_driven"]
        # default to picking at random if `my_score` is not available
        score = [my_scores[image][score_type[self.current_strategy]] for image in images]
        if sum(score) == 0:
            image = random.choice(images)
            logger.info(f"Randomly selected Image from Selective Uncertainty '{image}'")
        else:
            my_max_score = max(score)
            ind = score.index(my_max_score)
            image = images[ind]
            if self.current_strategy == 0:
                logger.info(f"Selected image '{image}' using `my_score_target` ({my_max_score})")
            else:
                logger.info(f"Selected image '{image}' using `my_score_boundary_driven` ({my_max_score})")

        self.current_strategy = (self.current_strategy + 1) % 2
        
        return {"id": image}
  

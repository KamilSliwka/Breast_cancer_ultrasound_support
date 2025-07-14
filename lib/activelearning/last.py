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
    def __call__(self, request, datastore: Datastore):
        self.scoring_method(request, datastore)  # <-- scoring uruchamiany tutaj
        images = datastore.get_unlabeled_images()
        if not len(images):
            return None

        my_scores = {image: datastore.get_image_info(image).get("my_score", 0) for image in images}

        # default to picking at random if `my_score` is not available
        if sum(my_scores.values()) == 0:
            image = random.choice(images)
            logger.info(f"Randomly selected Image from My Score '{image}'")
        else:
            my_max_score, image = max(zip(my_scores.values(), my_scores.keys()))
            logger.info(f"Selected image '{image}' using `my_score` ({my_max_score})")
        return {"id": image}
  

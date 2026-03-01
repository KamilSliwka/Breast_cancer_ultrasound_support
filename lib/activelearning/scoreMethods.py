import logging
import random
import os
import torch
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.scoring import ScoringMethod

logger = logging.getLogger(__name__)

class MyScoreGeneratorMethod(ScoringMethod):
    # scoring_model w init
    def __init__(self,  device: str = "cuda", num_samples: int = 100):
        super().__init__("My scoring method")
        #self.scoring_model_path = scoring_model_path
        self.device = device
        self.num_samples = num_samples
    


    def __call__(self, request, datastore: Datastore):
        result = {}
        model_path = "/claraDevDay/new_endoscopy/lib/models/scoring_model.pth"
        scoring_model_timestamp = int(os.stat(model_path).st_mtime)
        scoring_model = torch.load(model_path, weights_only=False)

        if not scoring_model:
            return None

        scoring_model = scoring_model.to(self.device).eval()

        skipped = 0
        unlabeled_images = datastore.get_unlabeled_images()
        num_samples = request.get("num_samples", self.num_samples)

        for image_id in unlabeled_images:
            image_info = datastore.get_image_info(image_id)
            prev_timestamp = image_info.get("my_score_timestamp", 0)

            #if the timestamps match we dont' need to recompute score
            if prev_timestamp == scoring_model_timestamp:
                skipped += 1
                continue

            with torch.no_grad():
            
                image_path = datastore.get_image_uri(image_id)
                logger.info(f"[SCORING] image_path (raw): {image_path}")
                image_path = os.path.abspath(image_path)
                logger.info(f"[SCORING] exists: {os.path.exists(image_path)} | isfile: {os.path.isfile(image_path)}")

                if not os.path.isfile(image_path):
                    logger.error(f"[SCORING] Błąd — brak pliku: {image_path}")
                    raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")

                try:
                    import itk
                    _ = itk.imread(image_path)
                except Exception as e:
                    logger.error(f"[DEBUG] ITK nie wczytał {image_path}: {e}")
                    raise
                
                if not os.path.exists(image_path):
                    logger.error(f"[SCORING] Nie znaleziono pliku obrazu: {image_path}")
                    raise FileNotFoundError(f"Brakuje pliku: {image_path}")

                data = {"image": image_path}
                logger.info(f"[SCORING] Przekazuję do modelu: {data}")
                my_score = scoring_model(data)

            if self.device == "cuda":
                torch.cuda.empty_cache()

            # add `my_score` in datastore to use later in `SelectImageWithMyScore`
            info = {
                "my_score_target": my_score["target"],
                "my_score_boundary_driven": my_score["boundary_driven"],
                "my_score_timestamp": scoring_model_timestamp
            }
          
            datastore.update_image_info(image_id, info)
            result[image_id] = info

        return result
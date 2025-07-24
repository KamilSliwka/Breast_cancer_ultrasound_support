import torch
from torch import nn
from typing import Dict,Any
import logging
from monai.transforms import Compose
logger = logging.getLogger(__name__)

class MyScoringModel(nn.Module):
    def __init__(self, usg_task):
        super().__init__()
        self.usg_task = usg_task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        logger.info(f"[SCORING MODEL] Przekazano dane: {list(data.keys())}")
        logger.info(f"[SCORING MODEL] Typ image: {type(data['image'])}")
       
    
        with torch.no_grad():
            # 1. Preprocessing (np. LoadImaged, Resized, Normalized)
            transforms = Compose(self.usg_task.pre_transforms())
            d = transforms(data)
            
            #d = self.usg_task.pre_transforms(data)
            if isinstance(d, list):
                d = d[0]  # take first element from list
           
            # 2. Preprocessed input image
            image = d["image"].unsqueeze(0).to(self.device)
            
            # 3. Inference
            with torch.no_grad():
                pred = self.usg_task.network(image)

            # 4. entropy calculation

            probs = torch.sigmoid(pred)  # Binary probability map
            probs = probs.squeeze()
            print(probs.shape)
            
            param1 = 0.4
            param2 = 0.1           
            
            print(probs)
            
            # path = "/claraDevDay/new_endoscopy/tensor.txt"
            # torch.set_printoptions(threshold=float('inf'))

            # with open(path, "w") as f:
            #     f.write(str(probs))

            # torch.set_printoptions(profile="default")
            
            # Wyciągamy indeksy pikseli "pewnych klasy pozytywnej"
            # take pexels which are "certain" to be part of tumor
            
            target_1_pixels = torch.where(probs <= param1)

            # take pexels which are "uncertain"(close 0.5)
            uncertain_1_pixels = torch.where(torch.abs(probs - 0.5) <= param2)

            # binary entropy
            entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
            entropy = entropy.view(-1)  # [H*W]

            # konverting indexes from 2D to 1D
            H, W = probs.shape
            target_1_indices_1d = target_1_pixels[0] * W + target_1_pixels[1]
            uncertain_1_indices_1d = uncertain_1_pixels[0] * W + uncertain_1_pixels[1]

            # average uncertain for "certain" pexels
            if len(target_1_indices_1d) > 0:
                target_uncertainty_avg = entropy[target_1_indices_1d].mean()
            else:
                target_uncertainty_avg = torch.tensor(0.0)

            # average uncertain for "uncertain" pexels
            if len(uncertain_1_indices_1d) > 0:
                uncertain_uncertainty_avg = entropy[uncertain_1_indices_1d].mean()
            else:
                uncertain_uncertainty_avg = torch.tensor(0.0)
                        
            # Mean entropy as uncertainty score:

            print("Probs stats:", probs.min().item(), probs.max().item(), probs.mean().item())
            print("Entropy stats:", entropy.min().item(), entropy.max().item(), entropy.mean().item())
            logger.info(f"[SCORING] Score target : {target_uncertainty_avg}")
            logger.info(f"[SCORING] Score uncertain : {uncertain_uncertainty_avg}")
            logger.info(f"[SCORING] Entropy: {entropy}")
            logger.info(f"[SCORING] Pred probs: {probs}")

        return {
            "target": float(target_uncertainty_avg),
            "uncertain": float(uncertain_uncertainty_avg)
        }
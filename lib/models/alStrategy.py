import torch
from torch import nn
from typing import Dict,Any
import logging
from monai.transforms import Compose
from monai.transforms import (
    Activationsd,
    EnsureTyped,
)
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from lib.transforms import (SaveImagePred, PVTNetSumOutd,)

logger = logging.getLogger(__name__)

class MyScoringModel(nn.Module):
    def __init__(self, usg_task):
        super().__init__()
        self.usg_task = usg_task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.counter = 0
   
    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        self.usg_task.network.eval()
        logger.info(f"[SCORING MODEL] Przekazano dane: {list(data.keys())}")
        logger.info(f"[SCORING MODEL] Typ image: {type(data['image'])}")
       
    
        with torch.no_grad():
            # 1. Preprocessing (np. LoadImaged, Resized, Normalized)
            transforms = Compose(self.usg_task.pre_transforms())
            d = transforms(data)

            if isinstance(d, list):
                d = d[0]  # take first element from list
           
            # 2. Preprocessed input image
            image = d["image"].unsqueeze(0).to(self.device)
            
            # 3. Inference
           
            print("=== PRE-TRANSFORM STATS ===")
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")
            print(f"Image min: {image.min().item():.4f}, max: {image.max().item():.4f}, mean: {image.mean().item():.4f}") 
                      
            with torch.no_grad():
                pred = self.usg_task.network(image)
                print("=== MODEL RAW OUTPUT (logits) ===")
                print(f"Type: {type(pred)}")
                if isinstance(pred, torch.Tensor):
                    print(f"Shape: {pred.shape}")
                    print(f"Min: {pred.min().item():.4f}, Max: {pred.max().item():.4f}, Mean: {pred.mean().item():.4f}")
                custom_post_transforms = Compose([
                        EnsureTyped(keys="pred", device=data.get("device") if data else None),
                        PVTNetSumOutd(keys="pred"),
                        Activationsd(keys="pred", sigmoid=True),
                    ])
                    
                output = custom_post_transforms({"pred": pred})
                probs = output["pred"]
            # 4. entropy calculation
    
           # probs = torch.sigmoid(pred)  # Binary probability map
            probs = probs.squeeze()
            print(probs.shape)
        
            
            param1 = 0.9
            param2 = 0.1           
            print("max: ", torch.max(probs))
            print(probs)
            
            target_1_pixels = torch.where(probs >= param1)

            # take pexels which are "uncertain"(close 0.5) Boundary-Driven
            boundary_driven_1_pixels = torch.where(torch.abs(probs - 0.5) <= param2)

            # binary entropy
            entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
            entropy = entropy.view(-1)  # [H*W]

            # konverting indexes from 2D to 1D
            H, W = probs.shape
            target_1_indices_1d = target_1_pixels[0] * W + target_1_pixels[1]
            boundary_driven_1_indices_1d = boundary_driven_1_pixels[0] * W + boundary_driven_1_pixels[1]

            # average uncertain for "certain" pexels
            if len(target_1_indices_1d) > 0:
                target_uncertainty_avg = entropy[target_1_indices_1d].mean()
            else:
                target_uncertainty_avg = torch.tensor(0.0)

            # average uncertain for "uncertain" pexels
            if len(boundary_driven_1_indices_1d) > 0:
                boundary_driven_uncertainty_avg = entropy[boundary_driven_1_indices_1d].mean()
            else:
                boundary_driven_uncertainty_avg = torch.tensor(0.0)
                        
            # Mean entropy as uncertainty score:

            print("Probs stats:", "min: ", probs.min().item(),"max: ", probs.max().item(),"mean: ", probs.mean().item())
            print("Entropy stats:","min: ",  entropy.min().item(),"max: ",  entropy.max().item(),"mean: ",  entropy.mean().item())
            logger.info(f"[SCORING] Score target : {target_uncertainty_avg}")
            logger.info(f"[SCORING] Score boundary_driven : {boundary_driven_uncertainty_avg}")
            logger.info(f"[SCORING] Entropy: {entropy}")
            logger.info(f"[SCORING] Pred probs: {probs}")

        return {
            "target": float(target_uncertainty_avg),
            "boundary_driven": float(boundary_driven_uncertainty_avg)
        }
        
        
     
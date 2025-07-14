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
        # Oczekujemy: data = {"image": <ścieżka do pliku>}
        logger.info(f"[SCORING MODEL] Przekazano dane: {list(data.keys())}")
        logger.info(f"[SCORING MODEL] Typ image: {type(data['image'])}")
        # with torch.no_grad():
        #     result = self.usg_task(data)  # usg_task to np. BasicInferTask
        #     pred = result["pred"]
        #     probs = torch.softmax(pred, dim=1)
        #     entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        #     score = entropy.mean()
        # return score
    
        with torch.no_grad():
            # 1. Preprocessing (np. LoadImaged, Resized, Normalized)
            transforms = Compose(self.usg_task.pre_transforms())
            d = transforms(data)
            
            #d = self.usg_task.pre_transforms(data)
            if isinstance(d, list):
                d = d[0]  # pobierz pierwszy element z listy
            # 2. Przetworzony obraz wejściowy
            #image = d["image"].unsqueeze(0).to(self.usg_task.device)
            image = d["image"].unsqueeze(0).to(self.device)
            # 3. Inference
            with torch.no_grad():
                pred = self.usg_task.network(image)

            # 4. Przykład obliczenia entropii lub innych metryk
            # print("Test pred shape:", pred.shape)
            # print("Test pred stats:", pred.min().item(), pred.max().item(), pred.mean().item())
            

            # probs = torch.softmax(pred, dim=1)
            # entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            # score = entropy.mean()
            
            # pred.shape: [B, 1, H, W]
            probs = torch.sigmoid(pred)  # Binary probability map

            # Binary entropy per pixel:
            entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)

            # Mean entropy as uncertainty score:
            score = entropy.mean()

            print("Probs stats:", probs.min().item(), probs.max().item(), probs.mean().item())
            print("Entropy stats:", entropy.min().item(), entropy.max().item(), entropy.mean().item())
            logger.info(f"[SCORING] Score: {score}")
            logger.info(f"[SCORING] Entropy: {entropy}")
            logger.info(f"[SCORING] Pred probs: {probs}")

        return float(score)
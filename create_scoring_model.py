# create_scoring_model.py
import torch
from lib.models import PolypPVT, MyScoringModel
from lib.infers import Usg

# Ścieżka do wag pretrenowanego segmentora
pretrained_weights = "/claraDevDay/new_endoscopy/lib/models/pvt_v2_b2.pth"

# Inicjalizacja i załadowanie wag
segmentor = PolypPVT()
model_dict = segmentor.state_dict()
save_model = torch.load(pretrained_weights)
save_model = {k.replace('module.', ''): v for k, v in save_model.items()}
save_model = {k: v for k, v in save_model.items() if k in model_dict}
model_dict.update(save_model)
segmentor.load_state_dict(model_dict)
segmentor.eval()

# Zadanie inferencyjne
usg_task = Usg(path=pretrained_weights, network=segmentor, conf={"labels": None})

# Tworzenie scoring modelu
scoring_model = MyScoringModel(usg_task)
scoring_model.eval()

# Zapis do pliku
torch.save(scoring_model, "/claraDevDay/new_endoscopy/lib/models/scoring_model.pth")
print("Scoring model saved.")

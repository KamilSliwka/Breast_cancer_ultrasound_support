# create_scoring_model.py
import torch
from lib.models import PolypPVT, MyScoringModel
from lib.infers import Usg
from monai.utils import set_determinism
import numpy as np, random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
set_determinism(seed=seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
ckpt_path = "/claraDevDay/new_endoscopy/model/usg_pvt.pt"
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model"]

# Create model instance and load state dict
model = PolypPVT().to(device)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
model.eval()

# print("Model loaded successfully!")
# print("Missing keys:", missing)
# print("Unexpected keys:", unexpected)

# with torch.no_grad():
    
#     dummy = torch.randn(1, 3, 512, 512).to(device)
#     out = model(dummy)
# print("Output stats:", out.min().item(), out.max().item(), out.mean().item())

usg_task = Usg(path="/claraDevDay/new_endoscopy/model/usg_pvt.pt", network=model, conf={"labels": None})
usg_task.network.eval()

# Create scoring model
scoring_model = MyScoringModel(usg_task)
scoring_model.eval()

# Save to file
torch.save(scoring_model, "/claraDevDay/new_endoscopy/lib/models/scoring_model.pth")
print("Scoring model saved.")


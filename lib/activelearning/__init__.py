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

###

import torch
from .last import Last
from .last import SelectImageWithMyScore
from .scoreMethods import MyScoreGeneratorMethod
# from lib.models import PolypPVT
# from lib.infers import Usg
# from monai.transforms import Compose
# from lib.models import MyScoringModel 

# # Ścieżka do wag
# path = '/claraDevDay/new_endoscopy/lib/models/pvt_v2_b2.pth'

# # Inicjalizacja modelu
# segmentor = PolypPVT()
# model_dict = segmentor.state_dict()

# # Załaduj stan zapisany w .pth z obsługą niezgodnych kluczy
# save_model = torch.load(path)
# # Usuń ewentualny prefix "module."
# save_model = {k.replace('module.', ''): v for k, v in save_model.items()}
# # Filtruj pasujące klucze
# save_model = {k: v for k, v in save_model.items() if k in model_dict}
# model_dict.update(save_model)
# segmentor.load_state_dict(model_dict)
# segmentor.eval()

# # Ustawienie zadania inferencyjnego

# usg_task = Usg(path='/usg/new_endoscopy/lib/models/pvt_v2_b2.pth', network=segmentor, conf={"labels": None})

# scoring_model = MyScoringModel(usg_task)

# # Zapisz model Pythona jako obiekt (pickle)
# torch.save(scoring_model, "/claraDevDay/new_endoscopy/lib/models/scoring_model.pth")

# import torch

model_path = "/claraDevDay/new_endoscopy/lib/models/scoring_model.pth"
#scoring_model = torch.load(model_path)
scoring_model = torch.load(model_path, weights_only=False)
scoring_model.eval()


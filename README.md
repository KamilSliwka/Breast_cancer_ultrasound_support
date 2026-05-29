# Breast Cancer Ultrasound Support System

A **MONAI Label**-based application for supporting breast cancer diagnosis in ultrasound examinations, developed as an engineering thesis project. Built on top of an existing MONAI Label template and extended with a custom active learning pipeline designed to minimize the number of annotated samples needed to train a high-quality deep learning model.

---

## Motivation

A literature review conducted as part of the thesis identified the primary obstacle to building AI models capable of genuinely supporting radiologists: **the scarcity of high-quality labeled medical datasets**. Acquiring expert annotations in clinical imaging is slow and expensive.

This project tackles that problem directly by implementing **Active Learning** — a paradigm in which the model identifies the most informative unlabeled samples and requests expert labels only for those, achieving strong generalization with significantly fewer annotations.

---

## Key Contributions

- **Migrated the codebase** to a newer, supported version of MONAI Label, resolving breaking API changes and deprecated interfaces to restore a working baseline.
- **Designed and implemented three custom Active Learning strategies** on top of MONAI Label's `Strategy` / `ScoringMethod` interfaces:
  - **Target-Aware Uncertainty Sampling (TAUS)** — focuses uncertainty estimation on regions that are clinically relevant to the segmentation task, avoiding wasted annotations on uninformative background areas.
  - **Boundary-Driven Uncertainty Sampling (BDUS)** — targets samples where the model is most uncertain near lesion boundaries; providing labels for these samples yields the greatest gain in the model's ability to generalize and delineate precise contours.
  - **Selective Uncertainty Sampling (SUS)** — combines TAUS and BDUS into a single balanced strategy, designed to achieve the best overall labeling efficiency.
- **Built a scoring model pipeline** (`create_scoring_model.py`) that loads a pretrained `PolypPVT` checkpoint, wraps it in a `MyScoringModel`, and serializes it for use during active learning scoring passes.

- **Implemented a result caching mechanism** — scoring runs only when the model weights have changed (detected via file modification timestamp), avoiding redundant inference across the unlabeled pool on every query.

- **Containerized the application** using Docker for reproducible deployment of the full MONAI Label server and inference stack.
- **Evaluated end-user utility** of the system from the perspective of a clinical user.

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Framework | Python, PyTorch |
| Medical AI Platform | MONAI Label |
| Model Architecture | PolypPVT (transformer-based segmentation) |
| Active Learning | Custom strategies: TAUS, BDUS, SUS |
| Containerization | Docker |
| Notebook Experimentation | Jupyter Notebook |



---

 
## Architecture
 

 
```
lib/
├── activelearning/
│   ├── scoreMethods.py       — MyScoreGeneratorMethod (scoring + caching logic)
│   └── strategy.py           — SelectImageWithMyScore (TAUS/BDUS alternation + fallback)
├── models/
│   ├── alStrategyModel.py    — MyScoringModel (PVT without binarization post-transform)
│   ├── scoring_model.pth     — serialized scoring model
│   └── pvt.pth               — base segmentation weights
└── transforms/               — custom data transformation scripts
 
images_rgb/
├── labels/
│   ├── final/                — radiologist-approved masks (moves image to Training Set)
│   └── original/             — pre-edit versions
├── cg*.nii.gz                — USG volumes
└── datastore.json            — MONAI Label state file (scores, timestamps, annotation status)
```
 
---


## Active Learning Loop

```
┌─────────────────────────────────────────────────────────┐
│  1. SCORE   — run inference on unlabeled pool           │
│              compute TAUS + BDUS scores per image       │
│              cache results in datastore.json            │
│                                                         │
│  2. SELECT  — alternate between max(TAUS) / max(BDUS)   │
│              return most informative image to 3D Slicer │
│                                                         │
│  3. SEGMENT — model auto-segments the image             │
│              radiologist reviews and corrects mask      │
│                                                         │
│  4. LABEL   — Submit Label: mask saved to labels/final/ │
│              image moves from Unlabeled → Training Set  │
│                                                         │
│  5. TRAIN   — retrain/fine-tune model on updated set    │
│              model timestamp updated → cache invalidated│
│                                                         │
│  6. REPEAT  ──────────────────────────────────────────  │
└─────────────────────────────────────────────────────────┘
```
---



## License

Based on the [MONAI Label](https://github.com/Project-MONAI/MONAILabel) framework, licensed under the Apache License 2.0.

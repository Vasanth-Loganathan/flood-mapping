# Rapid Flood Extent Mapping from Sentinel-1 SAR

**A controlled study of multi-temporal U-Nets on Kuro Siwo, with cross-dataset evaluation on Sen1Floods11**


## Overview

Synthetic aperture radar (SAR) maps floodwater through cloud and at night. This project asks how much pre-event imagery, multi-temporal architectures, terrain data, speckle filtering and pretrained encoders help when the flood events in the test set were never seen in training.

* **Data:** Kuro Siwo (Sentinel-1 VV/VH, two pre-event and one post-event image, 224×224 patches at 10 m, three-class mask) and Sen1Floods11 (hand-labelled chips, one image each).
* **Splits:** strictly event-disjoint. 22 training events, 3 validation events, and a fresh test of 10 unseen events (10,839 patches) that was frozen and evaluated once.
* **Proposed model:** `mt_g2`, a shared-encoder U-Net with a gated, gap-aware temporal fusion (the days between acquisitions are an input) and temporal dropout.
* **Compared with:** change-detection rules, post-event-only and early-fusion U-Nets, difference-fusion and temporal-attention variants, DEM channels, Lee and median speckle filters, a U-Net with a pretrained ResNet-50 encoder and SegFormer-B2.
* **Analysis:** per-event, per-gap, flood-free-patch and permanent-water error analysis; Grad-CAM, exact Shapley values over input channels and gate maps.

## Main results

Flood-class IoU, 500-pixel minimum-area rule, mean ± sd over 3 seeds (ensemble of the 3 seeds in the last column).

| Model | Validation | Frozen test (10 unseen events) | Test ensemble |
|---|---|---|---|
| U-Net, post-event only | 0.532 ± 0.003 | | |
| U-Net, pre-event 1 + post-event | 0.684 ± 0.008 | 0.586 ± 0.019 | 0.599 |
| `mt_g2` (proposed) | 0.669 ± 0.004 | 0.587 ± 0.016 | 0.597 |
| U-Net + ResNet-50 encoder* | 0.693 ± 0.007 | 0.587 ± 0.021 | 0.606 |
| SegFormer-B2* | 0.696 ± 0.003 | 0.583 ± 0.005 | 0.589 |

\*Evaluated on the test post hoc (exploratory), because they were created after the test was frozen.

* One pre-event image gives most of the gain (+0.15 IoU over post-event only). A second pre-event image, attention, gating, DEM channels, speckle filters and pretrained encoders do not give a measurable gain on unseen events.
* About 40% of false-positive pixels on the test come from flood-free patches.
* Cross-dataset (post-event-only U-Net, permanent water ignored): trained on Kuro Siwo and tested on Sen1Floods11, IoU is 0.233; trained and tested on Sen1Floods11, 0.524. Trained on Sen1Floods11 and tested on Kuro Siwo test shard 1, 0.370 against 0.599 in domain.

## Repository contents

| File | Purpose |
|---|---|
| `flood_mapping.ipynb` | The full pipeline: data download and pre-processing, models, experiments, frozen test, analysis, Sen1Floods11, explainability, results summary |
| `flood_paper_IEEE_v3.pdf` | The paper (IEEE format) |
| `requirements.txt` | Python dependencies |
| `results/`  | Result tables and the frozen configuration |

## Data

* Kuro Siwo WebDataset: `orion-ai-lab/Kuro-Siwo-Webdataset` on Hugging Face (shards are downloaded one at a time, about 10 GB each).
* Sen1Floods11: public bucket `gs://sen1floods11` (the notebook reads it over HTTPS), or attach it as a Kaggle dataset and set `SEN_LOCAL_DIR`.

The datasets are not included in this repository.

## How to run

The notebook was developed on a Kaggle notebook with a single T4 GPU and internet access. All files are written under `/kaggle/working/flood_mapping_project`.

1. Upload `flood_mapping_clean.ipynb` to Kaggle and install the requirements (`pip install -r requirements.txt`; the notebook installs missing packages itself).
2. Set the flags in section 0:

| Flag | Meaning |
|---|---|
| `RUN_DOWNLOADS` | allow downloading data from Hugging Face / Google Cloud Storage |
| `RUN_TRAINING` |  allow training; finished runs are always skipped |
| `RUN_FINAL_TEST` |  the frozen test evaluation (see below) |
| `RUN_POSTHOC` | post-hoc test evaluation of models created after the freeze |

3. From scratch, set `RUN_DOWNLOADS = True` and `RUN_TRAINING = True` and run the cells in order. This takes roughly 25 GPU hours and needs about 12 GB of free disk while a shard is downloaded.
4. Every training and evaluation step is resumable: results are stored in `results/<run>.json` and checkpoints in `checkpoints/<run>.pt`.

**Frozen test.** The final test configuration was written to `results/FROZEN_CHOICES.json` before the test data was evaluated, and the evaluation was run once. The corresponding notebook cell refuses to run again if `results/fresh_test_final.csv` exists, so keep `RUN_FINAL_TEST = False` and read the saved tables.

## Reproducibility notes

* Seeds 0, 1, 2 for every model; ensembles average the sigmoid outputs of the three seeds.
* Hyperparameters were set in advance and not tuned on test data. Decision rules were fixed before the results were seen: a variant replaces the baseline only if it is at least 0.015 IoU better on validation (0.005 for an ensemble).
* Event locations cannot be derived from the Kuro Siwo metadata (`geom` is in projected metres), so the claims are about unseen events, not unseen regions.
* Pretrained weights for the baselines (`timm/resnet50.a1_in1k`, `nvidia/mit-b2`) are downloaded from the Hugging Face Hub.


## Acknowledgements

Kuro Siwo (Bountos et al., NeurIPS 2024) and Sen1Floods11 (Bonafilia et al., CVPRW 2020) are the work of their authors; please follow their terms of use.

# Project Documentation

## Rapid Flood Extent Mapping from Sentinel-1 SAR Imagery using a Multi-Temporal Attention U-Net

**Status:** All planned models are trained and compared on validation (3 seeds each). Main findings: (1) pre-event imagery is essential (+0.15 IoU over post-event only); (2) neither a second pre-event image, nor the multi-temporal encoder, nor temporal attention improved validation IoU over a plain U-Net with one pre-event + post-event image; (3) the test set differs strongly from training (many flood-free patches, much longer pre-to-post gaps). (4) The improvement round (section 5.7: change-channel inputs, better training, ensemble + TTA) did **not** beat the plain 1 pre + post U-Net as a single model; only a 3-seed ensemble helped (+0.008 IoU). (5) The multi-temporal round (section 5.8: gated, gap-aware, temporally regularised) did not beat it either: best `mt_g2` 0.669 +/- 0.004 single and 0.678 as ensemble, against 0.684 and 0.692 for the plain U-Net, also in the long-gap bin; the learned gate mostly ignores the older pre-event image. (6) The DEM round (section 5.9) did not help either (`dem_unet` 0.675 vs 0.684; `mt_g2_dem` 0.659 vs 0.669 on one seed); no DEM model, no mixed ensemble was adopted. (7) The choices were frozen and the fresh test (shard 1 + eligible events of shard 2, 10,839 patches, 10 events) was run **once** (section 5.10): plain U-Net ensemble IoU 0.599, multi-temporal `mt_g2` ensemble 0.597; the difference is -0.003 with a bootstrap interval of [-0.008, +0.003], i.e. the two are indistinguishable on the test. About 40% of the false-positive pixels come from flood-free patches. **The test has now been used; no further model decisions may be based on it.** (8) The extra checks (section 5.11) found no benefit from speckle filtering (Lee 0.683, median 0.680 vs 0.684), permanent water causes 16-32% of false-positive pixels (upper-bound gain +0.02 to +0.03 IoU), event locations cannot be derived from `geom` (projected coordinates), and tiling works. **Modelling is stopped; next: figures and report.**
**Dataset:** Kuro Siwo (WebDataset). Sen1Floods11 considered only as a possible later external benchmark.
**Environment:** Kaggle Notebook (Tesla T4, 15.6 GB GPU, 4 CPU cores, ~33 GB RAM, ~20 GB disk).

---

## 1. Project overview

Optical imagery is often blocked by cloud; SAR is largely unaffected by cloud and daylight, so it is useful during and after severe weather. The project maps floodwater from Sentinel-1 SAR using two pre-flood and one post-flood observation of the same patch.

**Research question:** Does a Multi-Temporal Attention U-Net improve flood segmentation over simpler approaches, and how well does it generalize to flood events not seen in training?

**Comparisons**
1. Simple change-detection baseline. (done)
2. Conventional U-Net with 2, 4 and 6 input channels. (done)
3. Multi-temporal U-Net without attention (`mt_diff`), with attention (`mt_attn`), and the gated, gap-aware multi-temporal U-Net (`mt_g2`, the proposed multi-temporal model). (done on validation; fresh-test evaluation pending)
4. Terrain (DEM) channels for the plain and the multi-temporal U-Net (`dem_unet`, `mt_g2_dem`). (section 5.9, planned)

**Metrics:** flood-class IoU, F1, precision, recall, pooled over valid pixels, plus per-event tables (per-patch IoU is undefined for patches with no flood).

---

## 2. Dataset

### 2.1 Kuro Siwo (primary)
Repository: https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset

Each sample has two pre-event and one post-event SAR observation (VV and VH each), a 3-class mask, a valid-pixel mask, DEM, and metadata with event IDs and acquisition dates. Stored as WebDataset TAR shards of `.npy` arrays plus `info.json`.

| Split | Shards | Size |
|---|---|---|
| `train_GRD` | 5 | 46.76 GB |
| `test_GRD` | 12 | 122.77 GB |

Kaggle disk (~20 GB) holds only one large shard at a time. Shards are downloaded, processed, then deleted.

### 2.2 Sen1Floods11 (optional, later)
Different sample structure (no 2-pre + 1-post arrangement, no 3-class mask). Would need compatible inputs/labels designed before use as an external test.

### 2.3 Sample structure
Each sample is one 224x224 patch; arrays are `(1, 224, 224)` float32.

| Field | Meaning | Role |
|---|---|---|
| `sec1_vv`, `sec1_vh` | First pre-flood VV / VH (SL1) | Input |
| `sec2_vv`, `sec2_vh` | Second pre-flood VV / VH (SL2) | Input |
| `flood_vv`, `flood_vh` | Post-flood VV / VH | Input |
| `mask` | 0 background, 1 permanent water, 2 floodwater | Target |
| `valid_mask` | Valid-pixel indicator | Loss/metric masking |
| `dem` | Elevation | Optional later input |
| `info` | JSON metadata | Splitting, date analysis |

> **Caveat on location IDs:** `aoiid` is probably an index *within* an activation, not a global location. `grid_id` is **unique per patch** (25,707 distinct values for 25,707 patches), so it identifies a patch, not a place. Neither can show whether different events cover the same ground. The report therefore claims generalization to **unseen flood events only**, not to unseen locations.

### 2.4 Modelling choices
- **Input:** 6 channels, fixed order `sec1_vv, sec1_vh, sec2_vv, sec2_vh, flood_vv, flood_vh` -> `(B, 6, 224, 224)`.
- **Target:** `label == 2` (flood = 1; background and permanent water = 0).
- **Invalid pixels:** label 255, excluded from loss and metrics (derived from pixel-level `valid_mask`).
- **DEM:** not used so far.
- **SAR scaling:** linear backscatter -> dB, clipped to [-30, +5] dB, stored as uint8, standardized per channel with statistics from the original training split.
- **Pre-event timing:** median days from pre-event image to post-event image is 36 for SL1 (`sec1`) and 60 for SL2 (`sec2`), so `sec1` is the closer pre-event image.

---

## 3. Data findings

### 3.1 Training set (all 5 shards, metadata only)
- 25,707 patches, 25 flood events (2017-2022); no missing metadata; no duplicate sample IDs.
- Per-patch flood coverage: mean 15.8%, median 4.4%, max 100% (highly skewed).
- Top three events (1111009: 7,556; 1111013: 6,008; 1111004: 4,086) are about 69% of the data.

### 3.2 Train, validation and test differ (measured)

| Split | Patches | Events | Flood-free patches | Mean flood % | Gap median (days) | Gap p10 / p90 | Gap > 60 d |
|---|---|---|---|---|---|---|---|
| Train | 7,925 | 22 | 14.2% | 15.2 | 24 | 12 / 108 | 32.3% |
| Validation | 3,684 | 3 | 27.3% | 10.8 | 48 | 12 / 102 | 18.6% |
| Test (shard 0) | 6,000 | 9 | **69.0%** | 6.3 | **168** | **144** / 228 | **100%** |

(Gap = days from the first pre-event image to the post-event image.) Test events are flood-free in 35-91% of their patches (event 147, the largest, 88%). The test set therefore sits far outside the training range in both gap and flood-free share.

### 3.3 Test set (`test_GRD` shard 0)
9 events (118, 130, 147, 174, 205, 267, 273, 275, 277), **none** shared with training/validation, so the official test split is event-disjoint.

---

## 4. Final split and preprocessing (implemented)

| Split | Source | Events | Patches |
|---|---|---|---|
| Train | `train_GRD`, all events except validation, max 1,500 random patches per event | 22 | 7,925 |
| Train, extra | 1,160 additional flood-free patches (permanent-water patches first, at most 1,000 per event) from training events | - | 1,160 |
| Validation | `train_GRD`: events 1111007, 1111011, 555 (capped at 1,500 per event) | 3 | 3,684 |
| Test shard 0 | `test_GRD`, all patches (already looked at) | 9 | 6,000 |
| Fresh test, shard 1 | `test_GRD` shard 1, all patches (converted, not yet evaluated) | to be seen | 6,000 |
| Fresh test, shard 2 | `test_GRD` shard 2, **eligible events only**: 427, 437, 445 (events 470 and 492 also occur in training/validation and are excluded: 1,161 patches) | 3 | 4,839 |

Preprocessing: one streaming pass per shard; SAR as uint8 dB in `X_<split>.npy`; labels in `Y_<split>.npy` (255 = invalid); manifests and `norm_stats.json` in `data/processed/`. Training augmentation: random flips and 90-degree rotations.

**Test-set discipline:** test shard 0 has been used for analysis and must not drive decisions. All decisions (recipe, post-processing rule, architecture) were made on validation. The minimum-area rule (500 pixels) is fixed in advance. The fresh test shards are evaluated once; no tuning or re-running based on those numbers.

---

## 5. Results

### 5.1 Baselines and first U-Net (single seed)

| Model | Split | IoU | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Change detection (VV change, threshold -3.0 dB) | Val | 0.576 | 0.731 | 0.681 | 0.788 |
| Change detection (VV change, threshold -3.0 dB) | Test 0 | 0.524 | 0.688 | 0.583 | 0.840 |
| Post-event VV only (threshold -14.0 dB) | Val | 0.480 | 0.649 | 0.579 | 0.738 |
| Post-event VV only (threshold -14.0 dB) | Test 0 | 0.311 | 0.474 | 0.328 | 0.856 |
| U-Net, 6 channels, first run | Val | 0.674 | 0.805 | 0.836 | 0.777 |
| U-Net, 6 channels, first run | Test 0 | 0.661 | 0.796 | 0.730 | 0.875 |

### 5.2 Training recipe
Recipe A (lr 1e-3, 10 epochs) and B (lr 3e-4, 12 epochs) gave the same validation IoU (0.672 vs 0.674); B was used for all later runs. Validation IoU is noisy from epoch to epoch (swings of 0.03-0.05), so single-run differences below about 0.02 are not trustworthy. A probability threshold of 0.41 instead of 0.5 gives no real gain (0.6750 vs 0.6739); 0.5 is kept.

### 5.3 Model comparison on validation (3 seeds, recipe B)

| Model | Parameters | IoU | F1 | Precision | Recall | Time per epoch |
|---|---|---|---|---|---|---|
| Post-event only U-Net (2 ch) | 7.76 M | 0.532 +/- 0.003 | 0.695 | 0.639 | 0.762 | 80 s |
| **U-Net, 1 pre + post (4 ch)** | 7.76 M | **0.684 +/- 0.008** | 0.812 | 0.816 | 0.809 | 81 s |
| U-Net, 2 pre + post (6 ch) | 7.76 M | 0.668 +/- 0.005 | 0.801 | 0.793 | 0.810 | 81 s |
| `mt_diff` (shared encoder, difference fusion) | 7.75 M | 0.666 +/- 0.010 | 0.799 | 0.793 | 0.806 | 168 s |
| `mt_attn` (temporal attention + attention gates) | 8.13 M | 0.665 +/- 0.001 | 0.798 | 0.784 | 0.814 | 218 s |
| U-Net, 1 pre + post, with extra flood-free patches | 7.76 M | 0.686 +/- 0.003 | - | - | - | 90 s |

Validation IoU per event (mean over seeds):

| Model | 555 | 1111007 | 1111011 |
|---|---|---|---|
| Post-event only | 0.372 | 0.481 | 0.606 |
| U-Net 1 pre + post | 0.532 | 0.632 | 0.750 |
| U-Net 2 pre + post | 0.516 | 0.625 | 0.723 |
| `mt_diff` | 0.503 | 0.615 | 0.726 |
| `mt_attn` | 0.506 | 0.624 | 0.717 |

Best validation epochs: `mt_diff` 2, 1, 2; `mt_attn` 1, 1, 6 (U-Net 1 pre + post: 8, 3, 5). The multi-temporal models reach their best validation IoU almost immediately and then stay flat or decline while training loss keeps falling (about 0.70 -> 0.25).

### 5.4 Extra flood-free training patches and the minimum-area rule (validation, 3 seeds)
1,160 extra flood-free patches raised the training flood-free share from 14% to about 25% (the available pool was smaller than hoped). Result:

| Validation scoring | 1 pre + post U-Net | + extra flood-free patches |
|---|---|---|
| As is | 0.684 +/- 0.008 | 0.686 +/- 0.003 |
| Flood-free patches re-weighted to 50% | 0.677 +/- 0.009 | 0.680 +/- 0.003 |
| Flood-free patches re-weighted to 69% | 0.664 +/- 0.011 | 0.667 +/- 0.002 |

Differences are within noise; the seed spread is smaller with extra patches (tentative, 3 seeds only). Minimum-area rule (patches with fewer predicted flood pixels become "no flood"): at 500 px, re-weighted (69%) IoU rises by about 0.005 (0.664 -> 0.669; 0.667 -> 0.671) while unweighted IoU is unchanged (0.6837 vs 0.6841); at 1,000 px the unweighted IoU drops slightly; at 2,500-5,000 px it drops clearly (5,000 px: 0.649). **500 px was chosen in advance for the fresh test evaluation.**

### 5.5 Diagnostics of the first U-Net
- By patch flood fraction, validation / test IoU: 0-5% flood: 0.348 / 0.297; 5-20%: 0.492 / 0.570; above 20%: 0.750 / 0.818. Small floods are hard.
- Test IoU per event ranges from 0.22 to 0.87 (first U-Net: 118: 0.224; 130: 0.873; 147: 0.407; 174: 0.612; 205: 0.434; 267: 0.409; 273: 0.398; 275: 0.755; 277: 0.753).
- **Flood-free patches are the main test-set problem.** On validation, flood-free patches cause only 3% of false-positive pixels and removing them would change IoU from 0.674 to 0.677. On test shard 0, they cause 47% of false-positive pixels (691 false-positive pixels per flood-free patch vs 97 on validation) and removing them would raise IoU from 0.661 to 0.747.
- **The "long gaps hurt" hypothesis is not supported.** Within test shard 0, IoU rises with gap: 0.425 (60-90 d, 168 patches), 0.641 (90-180 d), 0.715 (180-365 d). The weakest bin is small and probably dominated by one or two events.
- Validation cannot reproduce the test problem: re-weighting validation to 69% flood-free patches costs only about 0.02 IoU, versus about 0.09 on test. Validation therefore cannot tell us whether the extra flood-free patches fix the test behaviour.

### 5.6 Conclusions so far
1. **Pre-event imagery is the key ingredient:** +0.15 IoU on validation (0.532 -> 0.684), far larger than the seed spread. A classical change rule shows the same pattern.
2. **A second pre-event image does not help,** in a plain U-Net (0.668 vs 0.684) or in the multi-temporal models. The second image is older (median 60 vs 36 days).
3. **The proposed Multi-Temporal Attention U-Net did not outperform the plain U-Net on validation** (0.665 vs 0.684; equal to `mt_diff` at 0.666 and the 6-channel U-Net at 0.668), while costing about 2.7 times the training time. All differences among these three are within about 2 standard deviations; the attention mechanism shows no measurable benefit on these three validation events.
4. **Validation is a weak judge:** only 3 events, all with gaps far shorter than the test set, flood-free share 27% vs 69%, and noisy epoch-to-epoch scores. The fresh test shards are the real test of whether temporal attention helps when pre-event images are old (gaps of 144-228 days), which is exactly where choosing between pre-event images could matter.
5. Training behaviour suggests that performance is limited by event-to-event shift, not model capacity: validation IoU peaks within 1-3 epochs even though training loss keeps falling.

### 5.7 Improvement round (validation only; planned, results to be filled in)
Motivation: the evidence in 5.3-5.6 says performance is limited by event-to-event shift and by false alarms on flood-free patches, not by model capacity (peak validation IoU within 1-3 epochs while training loss keeps falling; 47% of test false-positive pixels come from flood-free patches). The round therefore changes inference, inputs and training, not the architecture.

| # | Method | Configuration name | What changes |
|---|---|---|---|
| 1 | Seed ensemble + test-time augmentation | (applies to any config) | Average the sigmoid outputs of the 3 seeds and of 4 or 8 flip/90-degree-rotation views; no retraining |
| 2 | Explicit change channels | `v2_inputs` | 4 input channels + 3 dB-difference channels: post-pre VV, post-pre VH, post VV minus post VH (7 channels; same 7.76 M-parameter U-Net) |
| 3 | Better training | `v2_train` | Tversky loss (alpha_fp 0.6, so false alarms cost more than misses; 0.5 = old Dice) + hard-negative sampling of flood-free patches (weights from the old model's false positives, includes the 1,160 extra flood-free patches) + shift augmentation (random dB offsets for the whole stack and per acquisition, pixel noise) + EMA of weights |
| 2+3 | Combined | `v2_both` | Both of the above |

Baseline for comparison: `base` = the existing `abl_pre1_post4` checkpoints (3 seeds). All numbers use the same validation events, the same recipe (lr 3e-4, 12 epochs, patience 5) and 3 seeds.

**Metrics reported per configuration:** IoU as is; IoU with the fixed 500 px minimum-area rule; IoU with flood-free patches re-weighted to 69% plus the 500 px rule (the best available proxy for the test problem); F1, precision, recall; single models (mean +/- sd) vs ensemble vs ensemble + TTA.

**Adoption rule (fixed before looking at results):**
- A training/input variant replaces `base` as the headline model only if its mean single-model validation IoU is at least 0.015 higher (about 2 seed sd) **and** its 69%-reweighted IoU is not lower.
- Ensembling / TTA is adopted if it adds at least 0.005 IoU over the mean single model of the same configuration. It is reported separately as an inference technique (cost: 3 models x up to 8 views per patch).
- Hyperparameters (alpha_fp 0.6, noise levels, EMA decay 0.998, hard-negative boost 3, reference 2,000 px) were set a priori and are **not** tuned on the validation results of this round; if a variant fails, report it as such.

**Results (validation, 3 seeds each; IoU pooled over valid pixels):**

| Configuration | Single IoU (mean +/- sd) | Best epochs | Precision / recall | Ensemble x3 | Ensemble x3 + TTA8 | Single, 69% flood-free re-weighted, min area 500 | Ensemble x3, same scoring | Adopted? |
|---|---|---|---|---|---|---|---|---|
| `base` (`abl_pre1_post4`) | 0.684 +/- 0.008 | 8, 3, 5 | 0.816 / 0.809 | **0.692** | 0.693 | 0.669 +/- 0.011 | 0.678 | yes (as ensemble) |
| `v2_inputs` (+ change channels) | 0.683 +/- 0.006 | 1, 6, 6 | 0.811 / 0.814 | 0.687 | 0.687 | 0.665 +/- 0.014 | 0.672 | no |
| `v2_train` (Tversky + hard negatives + shift aug + EMA + extra patches) | 0.674 +/- 0.003 | 5, 5, 5 | 0.825 / 0.787 | 0.676 | 0.675 | 0.662 +/- 0.005 | 0.665 | no |
| `v2_both` | 0.668 +/- 0.001 | 1, 1, 3 | 0.797 / 0.805 | 0.670 | 0.670 | 0.650 +/- 0.002 | 0.653 | no |

Adoption rule applied: change from `base` in single-model IoU / in 69%-reweighted IoU: `v2_inputs` -0.001 / -0.004; `v2_train` -0.010 / -0.007; `v2_both` -0.016 / -0.019. None reaches the +0.015 required, so **no variant replaces `base`**. Ensembling `base` gains +0.008 over its single-model mean (rule: at least 0.005) and is adopted; TTA adds only +0.001 on top of the ensemble (and +0.002 for a single model), below the rule, so **TTA is not used** in the headline (8x the inference cost for no measurable gain).

**What the round shows**
- The change channels add nothing: the U-Net already learns the post-minus-pre difference from the raw channels (-0.001, inside the seed spread).
- The Tversky loss moved the model in the intended direction (precision 0.816 -> 0.825, recall 0.809 -> 0.787), but the net IoU is lower. The combined `v2_both` is the worst and peaks in epochs 1-3 with very low seed spread, so its extra changes appear to make it fit the training events faster without generalizing better to the validation events.
- Ensembling helps (about +0.008, most visible on the 69%-reweighted score, +0.009) because it averages out seed-specific errors, which is consistent with variance between runs, not model capacity, being a part of the problem.
- Validation cannot judge the main target of `v2_train` (false alarms on flood-free test patches; see 5.5), so `v2_train` and the others are also run once on the fresh test shards as **exploratory** configurations. They are not used for selection and are reported as exploratory, whatever the outcome.
- Hyperparameters were fixed in advance and not tuned; a variant of the loss weight or the learning rate could behave differently, but tuning more on three validation events would mostly fit noise.

### 5.8 Multi-temporal round (validation; done)
Requirement: the final proposed model must be a multi-temporal U-Net using all three observations (pre1, pre2, post). The earlier `mt_diff` / `mt_attn` (0.666 / 0.665) peaked in epochs 1-2 and trailed the plain U-Net (0.684), with a recipe tuned for the plain U-Net. Round design (validation only; fresh test untouched):

| Config | Change relative to the previous one |
|---|---|
| `mt_g0` | **Gated temporal fusion** (ref = pre1 + g (pre2 - pre1), gate starts near 0 so the model begins as pre1-vs-post and uses pre2 only if it helps) + **stabilised recipe** (lr 1e-4, weight decay 1e-2, EMA weights, bottleneck dropout 0.1, 14 epochs, patience 6) |
| `mt_g1` | + **gap-aware gate**: days from pre1 / pre2 to the post image (log, standardized) fed into the gate |
| `mt_g2` | + **temporal dropout**: with probability 0.3 one pre-event image is removed during training |
| `mt_g3` | + **GroupNorm** instead of BatchNorm (no shared batch statistics between pre and post images) |

Procedure: screen all four with one seed (about 40-45 min each), pick 1-2 finalists (highest IoU; the simpler configuration when within 0.01), train them with 3 seeds, compare with `base` as single models and ensembles, IoU by pre-to-post gap, and the learned gate values by gap. Hyperparameters were set a priori and are not tuned further. Decision rule: the best multi-temporal configuration by 3-seed mean validation IoU is the proposed model and is reported next to the plain U-Net whatever the outcome (an honest negative result is an acceptable finding). Parameters: 8.05 M (plain U-Net 7.76 M).

| Configuration | Screening IoU (seed 0) | 3-seed IoU | Ensemble x3 | Long-gap IoU (>60 d) | Mean gate g | Notes |
|---|---|---|---|---|---|---|
| `mt_g0` | 0.669 | - | - | - | - | best epoch 2, curve noisy (0.62-0.67) |
| `mt_g1` | 0.670 | - | - | - | - | best epoch 8 |
| `mt_g2` | 0.673 | **0.669 +/- 0.004** (F1 0.802, precision 0.788, recall 0.816, best epochs 8, 9, 5) | 0.678 | 0.529 (ensemble) | 0.08 (finest) to 0.26 (coarsest scale) | finalist; proposed multi-temporal model |
| `mt_g3` | 0.667 | - | - | - | - | GroupNorm, no gain |
| reference: `mt_attn` (old recipe) | 0.665 | 0.665 +/- 0.001 | - | - | - | |
| reference: `base` | - | 0.684 +/- 0.008 | **0.692** | 0.538 (ensemble) | - | plain U-Net, 1 pre + post |

Screening (1 seed): all four variants lie between 0.667 and 0.673, i.e. within noise of each other; `mt_g2` was taken forward because its validation curve is the most stable (0.65-0.67 from epoch 1) and run with 3 seeds.

**Comparison with the plain U-Net (validation, 3 seeds; IoU as is, the 500 px rule changes it by less than 0.001; "69%" = flood-free patches re-weighted to 69% plus the 500 px rule):**

| Model | Single IoU (s0 / s1 / s2) | Single mean | Ensemble x3 | 69% re-weighted, single mean | 69% re-weighted, ensemble | Precision / recall (ensemble) |
|---|---|---|---|---|---|---|
| `base` | 0.689 / 0.689 / 0.675 | 0.684 | **0.692** | 0.669 | **0.678** | 0.828 / 0.808 |
| `mt_g2` | 0.673 / 0.669 / 0.665 | 0.669 | 0.678 | 0.656 | 0.666 | 0.798 / 0.818 |

IoU by days from the closer pre-event image to the post image (validation, 3-seed ensembles): (0-30 d, n = 1,500) base 0.757 vs `mt_g2` 0.727; (30-60 d, n = 1,500) 0.640 vs 0.638; (>60 d, n = 684) 0.538 vs 0.529.

Mean gate per pre-event gap bin (`mt_g2` seed 0; g = 0 means only pre1, g = 1 only pre2): finest scale 0.085 / 0.076 / 0.079 for gaps (0, 30], (30, 60], (100, inf); coarsest scale 0.290 / 0.264 / 0.169.

**What the round shows**
- `mt_g2` is about 0.015 IoU below the plain U-Net for single models and 0.014 for ensembles. This is larger than the seed spread (0.004-0.008) and the ordering is the same on every metric, so it is a real, if small, deficit and not noise. Precision is lower (0.788 vs 0.816) and recall slightly higher: more false alarms.
- It does not win in the long-gap bin either (0.529 vs 0.538), which is the case a temporal model should help most.
- Ensembling helps both models by the same amount (+0.009 vs +0.008), so ensembling does not close the gap.
- The gate stays small (0.08-0.26): the model uses the older image only a little, mostly at coarse scales. The gate is **lower** for old pre1 images (0.29 to 0.17 at the coarsest scale), the opposite of the hypothesis that it would lean on the second image when the first is old. Likely reason: the second pre-event image is older than the first in the median case (60 vs 36 days), so it is rarely a better reference. Checked on train + validation metadata: **pre2 is older than pre1 in 100.0% of patches** (never closer, never equal), so the second image can never be the better reference, which explains why the gate has nothing to exploit.
- Per-seed values in the table are validation numbers from the notebook; the 3-seed means are in `results/model_comparison_val.csv` / `results/mt_round_val_detail.csv`.
- Decision rule applied (fixed in advance): the best multi-temporal configuration by 3-seed mean validation IoU is the proposed model, so **`mt_g2` is the proposed multi-temporal model** and is reported next to the plain U-Net. The result is an honest negative finding.

**Not included (deliberately):** DEM / slope channels. DEM is in the dataset and is probably the most informative missing input for flood mapping, but it needs a new preprocessing pass over all 5 training shards and the test shards (about 1-2 hours of downloads, one shard on disk at a time) and a change of the stored arrays. Kept as the next candidate extra if the round above leaves time. Pretrained encoders and 3-class training are also left out of this round.

### 5.9 DEM round (validation only; done, no DEM model adopted)
Motivation: terrain is the most informative input that is still missing (water pools in low, flat areas and does not stand on slopes), and the main failure modes (false alarms, event-to-event shift) are the kind terrain information can reduce. Everything else stays as it was.

| Config | Input / change | Compared with |
|---|---|---|
| `dem_unet` | plain U-Net, 4 SAR channels (pre1 + post) + 2 DEM channels (6 in total), recipe B | `base` |
| `mt_g2_dem` | `mt_g2` + a small DEM encoder whose features are added at every scale through 1x1 convolutions initialised to zero (starts exactly as `mt_g2`) | `mt_g2` |

DEM channels (computed from the `dem` field of each patch, stored as float16 in `data/processed/D_<split>.npy`): **relief** = DEM minus the patch median (metres, clipped at +/-3 sd of the training relief) and **slope** in degrees (10 m pixels, log1p, standardized). Patch-relative relief is used on purpose: absolute elevation differs between events and would not transfer to unseen events. Statistics from 400 training patches (`dem_stats.json`). Training arrays for DEM need one more pass over the five training shards (about 1-1.5 h); test arrays are made only if a DEM model is part of the frozen list.

**Adoption rules (fixed before any DEM result is seen):**
- R1: `dem_unet` replaces `base` as the plain-U-Net headline if its 3-seed mean single IoU is at least 0.015 higher **and** its 69%-re-weighted IoU (500 px rule) is not lower.
- R2: `mt_g2_dem` replaces `mt_g2` as the proposed model under the same rule (3 seeds required). It is trained with seed 0 first and gets seeds 1, 2 only if seed 0 is at least 0.010 above the `mt_g2` mean.
- R3: a mixed ensemble (3 plain + 3 multi-temporal models, probabilities averaged) is adopted if its IoU is at least 0.005 above the 3-model ensemble of the plain headline and its 69%-re-weighted IoU is not lower. Caveat for the report: 6 models against 3.
- Nothing is tuned; the decision is written to `results/dem_decision.json` before the freeze.

**Results (validation; 500 px rule; "69%" = flood-free patches re-weighted to 69%):**

| Configuration | Seeds | Single IoU (mean +/- sd) | Precision / recall | Ensemble | 69% re-weighted, single | Rule result |
|---|---|---|---|---|---|---|
| `base` | 3 | 0.684 +/- 0.008 | 0.819 / 0.806 | 0.691 | 0.669 | reference |
| `dem_unet` | 3 | 0.675 +/- 0.008 (s0 0.683, s1 0.675, s2 0.667) | 0.806 / 0.807 | 0.684 | 0.661 | R1: dIoU -0.008, d69 -0.008 -> not adopted |
| `mt_g2` | 3 | 0.669 +/- 0.005 | 0.791 / 0.812 | 0.677 | 0.656 | reference |
| `mt_g2_dem` | 1 | 0.658 (seed 0 only) | 0.814 / 0.775 | - | 0.649 | seed 0 was 0.010 below `mt_g2` (needed +0.010 above) -> seeds 1-2 not trained; R2: not adopted |
| mixed ensemble `base` + `mt_g2` (6 models) | - | - | 0.827 / 0.811 | **0.693** | ens 0.682 | R3: +0.0016 over `base` ensemble (needed +0.005) -> not adopted |

IoU by gap (validation, ensembles, 500 px rule): (0, 30] d `base` 0.757 vs `mt_g2` 0.727; (30, 60] 0.640 vs 0.637; >60 d 0.533 vs 0.524.

**What the round shows**
- Terrain did not help. Adding relief and slope lowers validation IoU by about 0.008 for the plain U-Net (about one seed sd, but in the wrong direction and on both metrics) and by about 0.010 for the multi-temporal model. The DEM statistics look sensible (relief p1/p50/p99 = -42 / 0 / 59 m, slope p50/p90/p99 = 1.0 / 3.8 / 14.5 degrees, no empty DEM patches), so this is not a data bug.
- Likely reasons (not tested): the SAR pair already carries most of the information; patch-relative relief at 10 m mostly describes the same flat floodplains in every event; and with three validation events the networks can use terrain to memorise the training events instead of generalising.
- The mixed ensemble gains only +0.0016 IoU, which is below the pre-registered +0.005; it is reported but not adopted.
- Hyperparameters were not tuned; the decision is in `results/dem_decision.json`.

### 5.10 Fresh test evaluation (shards 1 and 2; run once)

**Procedure.** Choices frozen on validation (`results/FROZEN_CHOICES.json`): headline plain model `base`, proposed multi-temporal model `mt_g2`, 3 seeds each, single models and ensemble x3, no TTA, 500 px minimum-area rule, no DEM, no mixed ensemble, **no exploratory models** (the list was left empty, so `v2_*`, `abl_*`, `extra_*`, `mt_diff` and `mt_attn` were not evaluated on the test). The notebook execution counters show the freeze cell ran before the final cell, and the final cell ran once. Data: `test_01` (6,000 patches) plus the eligible events of shard 2 (427, 437, 445; 4,839 patches) = 10,839 patches in 10 events (277, 279, 321, 324, 411, 417, 421, 427, 437, 445), none in training or validation. Event 277 also occurs in test shard 0 and event 427 in both shards 1 and 2, so events are not independent across shards.

| Model | Single IoU (mean +/- sd, 3 seeds) | Ensemble x3 IoU | Ensemble F1 | Precision / recall (ensemble) | IoU without the 500 px rule (ensemble) | IoU if flood-free patches had no false positives (ensemble) | FP pixels from flood-free patches (ensemble) |
|---|---|---|---|---|---|---|---|
| `base` (plain U-Net, 1 pre + post) | 0.586 +/- 0.019 | **0.599** | 0.750 | 0.709 / 0.795 | 0.593 | 0.666 | 40.4% |
| `mt_g2` (multi-temporal) | 0.587 +/- 0.016 | 0.597 | 0.747 | 0.700 / 0.801 | 0.590 | 0.672 | 43.9% |

Patch-level bootstrap of the ensemble difference `mt_g2` minus `base`: -0.0026 (95% interval -0.0079 to +0.0030). Patches within an event are correlated, so the interval is optimistic; the conclusion is that the two models are indistinguishable on the test.

Per-event IoU (ensembles, 500 px rule):

| Event | 277 | 279 | 321 | 324 | 411 | 417 | 421 | 427 | 437 | 445 |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` | 0.785 | 0.255 | 0.520 | 0.603 | 0.715 | 0.906 | 0.620 | 0.245 | 0.583 | 0.767 |
| `mt_g2` | 0.780 | 0.261 | 0.538 | 0.630 | 0.718 | 0.909 | 0.637 | 0.243 | 0.543 | 0.758 |

IoU by days from the closer pre-event image to the post image (ensembles; no patches in 120-180 d):

| Gap | 0-60 d (n=1,227) | 60-120 d (n=506) | 180-240 d (n=8,674) | >240 d (n=432) |
|---|---|---|---|---|
| `base` | 0.657 | 0.906 | 0.540 | 0.520 |
| `mt_g2` | 0.669 | 0.909 | 0.532 | 0.538 |

**What the test shows**
1. **The validation-to-test drop is large:** ensemble IoU 0.69 (validation) -> 0.60 (test). The test has 80% of its patches at 180-240 days gap, far outside the training range (median 24 d), and many flood-free patches.
2. **The multi-temporal model matches but does not beat the plain U-Net.** On validation it was 0.014 below; on the test the gap vanishes (-0.003, interval spans zero). It is slightly better in three of four gap bins and on 6 of 10 events, slightly worse in the 180-240 d bin that holds most patches. No statement of superiority can be made; its only edge is a small recall gain with a lower precision, and it costs about 2.7 times the training time.
3. **The minimum-area rule fixed in advance helps on the test as well** (+0.006 ensemble, +0.007 single), so it was a good choice.
4. **False alarms on flood-free patches remain the main fixable error:** removing them would raise IoU from 0.599 to 0.666 (+0.067); they supply about 40% of all false-positive pixels.
5. **Two events fail badly:** 279 (0.26) and 427 (0.24) for both models; the other eight range from 0.52 to 0.91. The cause of the two failures has not been examined (prediction maps and metadata for these events are the next diagnostic).
6. The ensemble is better than the average single model by +0.013 (`base`) and +0.010 (`mt_g2`), consistent with validation (+0.008, +0.009).

### 5.11 Additional methodological checks (after the test; descriptive; done)
These close four items of the project brief that the main experiments did not cover (`cells_extra_checks.py`, notebook cells E1-E4). None of them changes the frozen headline or the proposed model.

**E1 Speckle filtering (validation, 3 seeds each; same recipe B and normalisation as `base`).** The data are used as delivered (dB scaling, no filter) in all main experiments. Ablation with a Lee-type adaptive filter (5x5, linear intensity, ENL 4.4; `base_lee5`) and a 3x3 median filter in dB (`base_med3`), applied on the fly to the four input channels:

| Model | Single IoU (mean +/- sd) | 69% re-weighted | Precision / recall | Ensemble x3 | Rule (+0.015 and 69% not lower) |
|---|---|---|---|---|---|
| `base` (no filter) | 0.684 +/- 0.008 | 0.669 +/- 0.011 | 0.819 / 0.806 | 0.691 | reference |
| `base_lee5` | 0.683 +/- 0.004 | 0.669 +/- 0.005 | 0.824 / 0.800 | 0.689 | dIoU -0.001 -> no measurable benefit |
| `base_med3` | 0.680 +/- 0.013 | 0.669 +/- 0.014 | 0.838 / 0.784 | 0.685 | dIoU -0.004 -> no measurable benefit |

Conclusion: speckle filtering gives no gain; the U-Net learns its own smoothing. The median filter trades recall for precision (precision +0.019, recall -0.022) with a larger seed spread. Not run on the test (the optional post-hoc flag was left off).

**E2 Permanent water (`base` ensemble, pixel level, no minimum-area rule; fresh-test splits restricted to events not seen in training).**

| Split | Patches | Patches with permanent water | IoU | IoU if permanent water ignored | FP pixels on permanent water | Permanent-water pixels called flood |
|---|---|---|---|---|---|---|
| validation | 3,684 | 64.6% | 0.692 | 0.726 | 32.3% | 11.9% |
| test shard 1 | 6,000 | 17.6% | 0.640 | 0.667 | 16.0% | 6.1% |
| test shard 2 (eligible) | 4,839 | 36.3% | 0.372 | 0.392 | 25.7% | 0.4% |

Conclusion: permanent water causes 16-32% of the false-positive pixels, and ignoring it would raise IoU by only +0.020 to +0.034, an upper bound for any fix. That is smaller than the flood-free-patch problem (+0.067 on the pooled test, section 5.10), so a 3-class model is a modest, not a priority, improvement. (Per-split pixel-level IoUs differ from the pooled 500 px-rule numbers of section 5.10.)

**E3 Cross-region generalisation.** The planned distance-to-training analysis could not be run: the `geom` field of Kuro Siwo is in a local projected coordinate system (example: `POLYGON ((32415 1087705, ...))`, metres without a zone), not longitude/latitude, so patches cannot be placed on a map or compared across events from this field. What can be claimed is therefore generalisation to **unseen flood events** only, as stated since section 2.3, not to unseen regions or continents. Per-event table of the fresh test (ensembles, 500 px rule):

| Event | Patches | Mean flood fraction (%) | IoU `base` | IoU `mt_g2` |
|---|---|---|---|---|
| 277 | 873 | 8.47 | 0.785 | 0.780 |
| 279 | 2,381 | 0.73 | 0.255 | 0.261 |
| 321 | 432 | 4.29 | 0.520 | 0.538 |
| 324 | 548 | 2.30 | 0.603 | 0.630 |
| 411 | 149 | 1.38 | 0.715 | 0.718 |
| 417 | 506 | 6.24 | 0.907 | 0.909 |
| 421 | 137 | 2.15 | 0.620 | 0.637 |
| 427 | 5,283 | 0.47 | 0.245 | 0.243 |
| 437 | 169 | 0.75 | 0.583 | 0.543 |
| 445 | 361 | 1.90 | 0.767 | 0.758 |

Event-mean IoU: `base` 0.600 +/- 0.217, `mt_g2` 0.602 +/- 0.216 (range 0.24-0.91): the spread between events is much larger than any difference between models. The two failing events (279, 427) hold 7,664 of the 10,839 test patches (71%) and have a very small mean flood fraction (0.5-0.7%); with so little flood, a few false alarms are enough to pull IoU down. This is consistent with the flood-free / small-flood failure mode (section 5.5) but has not been verified event by event (prediction maps, label quality). Strict cross-region testing (leave-one-region-out training) would need event locations and is future work.

**E4 Tiling and stitching.** The dataset is delivered as 224x224 patches, so the experiments needed no tiling. `scene_to_input` and `predict_scene` (reflect padding, overlapping 224 tiles, Hann-weighted blending, ensemble average) handle full scenes. Self-check on a 2x2 mosaic of validation patches: with no overlap the stitched map differs from patch-wise prediction by at most 0.0008 in probability (half-precision inference), with 32 px overlap the flood mask agrees on 99.97% of pixels; non-multiple sizes (300 x 390) work.

---

## 6. Model definitions

**Baseline 1 - change detection:** threshold on `post_dB - mean(pre_dB)` (and variants), threshold fitted on training data.
**Baseline 2 - conventional U-Net:** early fusion, 7.76 M parameters; evaluated with 2, 4 and 6 input channels, with and without extra flood-free patches.
**`mt_diff`:** one encoder shared across the three timesteps (pre1, pre2, post; 2 channels each); fusion at every scale = concatenate [post, post minus mean(pre), mean(pre)] then 1x1 convolution; U-Net decoder.
**Improvement-round variants (`v2_*`):** plain U-Net (base 32) on `closer pre-event image + post-event image` (4 channels), optionally plus 3 dB-difference channels (`diff`: standardized with statistics of 400 training patches, `data/processed/diff_stats.json`). Training variants add Tversky loss, hard-negative sampling (`data/processed/hard_neg_weights.npy`, mined once with `abl_pre1_post4_s0`, which never saw `train_extra`), shift augmentation and EMA weights; the saved checkpoint is the EMA model with the best validation IoU. See notebook section 16.
**`mt_g2` (proposed multi-temporal model):** shared encoder over pre1, pre2 and post (2 channels each, base width 30, 8.05 M parameters). At every scale the reference pre-event feature is `ref = pre1 + g (pre2 - pre1)` with a learned per-pixel gate g that starts near 0 (bias -3) and is also fed the standardized log days from pre1 and from pre2 to the post image; fusion = 1x1 conv on [post, post - ref, ref]; U-Net decoder. Training: temporal dropout (one pre-event image removed with probability 0.3), lr 1e-4, weight decay 1e-2, EMA weights, bottleneck dropout 0.1, 14 epochs, patience 6. See notebook MT cell A.
**`dem_unet` / `mt_g2_dem`:** see section 5.9.
**`mt_attn` (first multi-temporal version):** as `mt_diff`, but the reference pre-event feature is a per-pixel softmax-weighted mix of the two pre-event features (weights from similarity to the post-event features), plus additive attention gates on the decoder skip connections.

---

## 7. Evaluation protocol

- Fresh test shards 1 and 2, evaluated once: shard 1 (6,000 patches) and the eligible events of shard 2 (427, 437, 445; 4,839 patches). Events present in training/validation (470, 492) are excluded and reported.
- Headline: plain-U-Net head chosen by rule R1 (default `base`), 3 seeds, single models (mean +/- sd) and ensemble x3, no TTA, 500-pixel rule. Proposed multi-temporal model: `mt_g2` (or `mt_g2_dem` under R2), same reporting. Mixed ensemble only if R3 adopts it.
- Exploratory (reported as such, not used for selection): `v2_inputs`, `v2_train`, `v2_both`, `abl_post2`, `abl_all6`, `extra_pre1_post4`, `mt_diff`, `mt_attn`, and the non-adopted counterpart of each DEM pair if its test arrays exist. Each with and without the 500-pixel rule.
- **Frozen before any fresh-test number is seen:** the choices above are written to `results/FROZEN_CHOICES.json` (with SHA-256 of every checkpoint) by the freeze cell; the final cell refuses to run twice (`results/fresh_test_final.csv`).
- Also reported: IoU by pre-to-post gap, a patch-level bootstrap of the multi-temporal-minus-plain difference (optimistic, patches within an event are correlated), and the flood-free-patch false-positive share.
- Report pooled IoU, F1, precision, recall (mean +/- sd over seeds), per-event IoU, and an "IoU if flood-free patches had no false positives" column to expose that failure mode.
- Differences between models are interpreted against the seed spread; claims are limited to unseen flood events (not unseen locations).
- Test shard 0 is used only for descriptive analysis in the report.
- **As actually run:** the freeze cell recorded only the two headline models (exploratory list empty, no checkpoint hashes in the record); the final cell therefore evaluated only `base` and `mt_g2`. Any later evaluation of other configurations on shards 1-2 must be labelled post-hoc and must not change the headline or the proposed model.

---

## 8. Project stages

| Stage | Work | Status |
|---|---|---|
| 1-4 | Data selection, metadata audit, split, preprocessing, Dataset, metrics | Done |
| 5 | Change-detection baseline | Done |
| 6 | Conventional U-Net, recipe selection, channel ablation (3 seeds) | Done |
| 7 | Diagnostics; train/val/test distribution check | Done |
| 8 | `mt_diff` and `mt_attn` (3 seeds each) | Done (validation) |
| 9 | Extra flood-free training patches; minimum-area rule | Done (validation); no clear gain |
| 10 | Improvement round on validation (section 5.7; notebook section 16) | Done: no variant beat `base`; 3-seed ensemble adopted (+0.008); TTA not adopted |
| 10b | Multi-temporal round (section 5.8): gated, gap-aware, temporally regularised multi-temporal U-Net | Done: `mt_g2` proposed; 0.669 single / 0.678 ensemble, below the plain U-Net |
| 10c | DEM round (section 5.9; `cells_dem_freeze.py`): DEM pass, `dem_unet`, `mt_g2_dem`, mixed ensemble, rules R1-R3 | Done: nothing adopted |
| 11 | Freeze (`FROZEN_CHOICES.json`), then fresh test evaluation (shard 1 and eligible events of shard 2), run once | **Done** (section 5.10); the test is used up |
| 12 | Optional further modelling (pretrained encoder, 3-class training, flood / no-flood head, gap augmentation): only as a new round with a **new** untouched test (shards 3-11, see 9) | Optional, not started |
| 12b | Post-hoc test ablation of the not-yet-evaluated original configurations (`abl_post2`, `abl_all6`, `extra_pre1_post4`, `mt_diff`, `mt_attn`, `v2_*`), labelled exploratory | Optional; allowed because it informs no choice |
| 12c | Extra checks E1-E4 (section 5.11): speckle filter ablation, permanent water, event-level generalisation, tiling | **Done** |
| 13 | Final analysis, figures, report | **Next (modelling stopped)** |

---

## 9. Next actions

1. **(Done)** Improvement round (5.7), multi-temporal round (5.8), DEM round (5.9), freeze and fresh test (5.10). Back up `checkpoints/`, `results/`, `data/processed/` (including `D_*.npy`, `dem_stats.json`, `gap_stats.json`, `FROZEN_CHOICES.json`, `fresh_test_final*.csv`) as a Kaggle dataset or download; 27+ checkpoints are needed to reproduce the numbers.
2. **Decision: stop modelling and write up (recommended), or open a new round.** The fresh test is spent. Further model changes that look at the test numbers would be tuning on the test. If more performance work is wanted, the clean way is:
   - develop on validation **plus** the already-seen test shards 0-2 (treat them as extra validation: about 20 events instead of 3, which also fixes the weak validation set);
   - evaluate once on **new** shards: `test_GRD` shards 3-11 exist and have not been touched (about 11 GB each; two shards, about 12,000 patches, are enough);
   - candidate ideas, in order: (a) **gap augmentation** (train on pre2 + post as well as pre1 + post, so training sees the long gaps of the test; mt_g2's temporal dropout partly did this and matched the plain model at long gaps), (b) a **patch-level flood / no-flood head** to cut false alarms (worth up to +0.067 IoU if perfect, realistically a fraction), (c) 3-class training with permanent water as its own class, (d) a pretrained encoder, (e) a larger ensemble.
2b. **Run the extra checks E1-E4** (section 5.11): about 2 hours (six short U-Net trainings in E1, the rest is minutes) and fill in the tables.
3. **Post-hoc ablation on the test (optional, small):** evaluate `abl_post2` and `abl_all6` (and `mt_diff`, `mt_attn`, `v2_*`) on shards 1-2 into a separate CSV, labelled exploratory. This gives the key "pre-event image is essential" result on the test set as well as validation.
4. **Diagnose events 279 and 427** (IoU about 0.25): prediction maps, flood fraction, gap, share of flood-free patches, label quality; also failure events from shard 0 (118, 147, 267, 273).
5. **Figures and tables for the report:** channel ablation, model comparison (validation), the test table and per-event / per-gap tables from 5.10, prediction examples, false positives on flood-free patches, gate values by gap (5.8), the DEM negative result.
6. **Report:** methods, results, limitations (unseen-event generalization only; 3 validation events; training dominated by a few large events; label noise not examined; strong test shift in gap (80% of test patches at 180-240 d) and flood-free share; events 277 and 427 appear in two test shards; single recipe for all models; freeze record without checkpoint hashes; exploratory configurations not evaluated on the test). Framing: a controlled study showing that one pre-event image is what matters; neither a second pre-event image, attention, a gated multi-temporal encoder nor terrain channels improved on a plain U-Net, and the multi-temporal model matched but did not beat it on the unseen test events.

---

## 10. Environment and housekeeping

- Kaggle project directory: `/kaggle/working/flood_mapping_project/{data, checkpoints, logs, results, configs, figures}`; config in `configs/project_config.json`.
- Processed data in `data/processed/` (X/Y arrays for train, val, train_extra, test splits; manifests; `norm_stats.json`; `done.json`).
- Per-run results: `results/<run>.json`; summaries: `results/ablation_channels_val.csv`, `results/model_comparison_val.csv`, `results/extra_empty_pre1_post4_val.csv`, `results/fresh_test_all_models.csv` (after step 10); training logs in `logs/`; checkpoints in `checkpoints/`.
- **New files from the improvement round:** `data/processed/diff_stats.json`, `data/processed/hard_neg_weights.npy`, `results/v2_*_s*.json`, `results/improvement_round_val.csv`, `results/improvement_round_val_detail.csv`, `results/fresh_test_improvement_round.csv` (after the frozen cell), checkpoints `v2_*_s*.pt` (9 more, 27 in total; all 27 are evaluated on the fresh test).
- **New files from the multi-temporal and DEM rounds:** `data/processed/gap_stats.json`, `D_train.npy`, `D_val.npy` (and `D_test_01/02.npy` if needed), `dem_stats.json`, `dem_done.json`, `test_02_ready.json`, `manifest_test_02_full.csv` (full shard-2 manifest; `manifest_test_02.csv` is rewritten to the eligible events with `split` and `row`), `results/mt_g*_s*.json`, `mt_g2_dem_s*.json`, `dem_unet_s*.json`, `dem_round_val_detail.csv`, `dem_decision.json`, `FROZEN_CHOICES.json`, `fresh_test_final*.csv`, `fresh_test_patchstats_*_ens.csv`.
- **Notebook changes during the DEM/test round:** DEM downloads go to `/tmp/kuro_siwo_dem` (disk space on `/kaggle/working`); a REPAIR cell rebuilds the dataset classes after the dataset definition was changed; the freeze cell was simplified (no hashes).
- **Back up** `data/processed/`, `results/`, `checkpoints/` and the metadata CSVs outside `/kaggle/working` (for example as a Kaggle dataset). 27 checkpoints are needed for the final evaluation.
- Set an `HF_TOKEN` to avoid Hub rate limits; keep only one large shard on disk at a time.

## 5.12 Final experiments (Sen1Floods11, recent baselines, explainability) - done
* Recent baselines (3 seeds, same recipe, lr fixed a priori): validation IoU U-Net/ResNet-50 0.693 +/- 0.007 (ens. 0.703), SegFormer-B2 0.696 +/- 0.003 (ens. 0.700) vs plain 0.684. Below the +0.015 rule. Fresh test (post hoc, exploratory): single 0.587 / 0.583 vs 0.586 (plain), ensembles 0.606 / 0.589 vs 0.599. No measurable difference.
* Sen1Floods11 (446 chips; 252/89/90 + 15 Bolivia): post-only U-Net trained on it: test 0.524 +/- 0.012 (water 0.661), Bolivia 0.608 +/- 0.065. Kuro-trained zero-shot: 0.233 +/- 0.124 (Bolivia 0.230 +/- 0.371, ensemble 0.073). Sen->Kuro: val 0.577, shard 1 0.370, shard 2 0.138.
* Explainability: Shapley (val) post VV 0.469 of 0.685 total; mt_g2 dates: pre1 0.056, pre2 0.046, post 0.572. Grad-CAM covers flood in events 417/277, almost nothing in 427. Gate shows no consistent link to flooded pixels.
* Paper updated (v2, 9 pages); only the repository link and affiliation remain as placeholders.
* Cleaned notebook delivered (`flood_mapping_clean.ipynb`), tested on fake data on CPU, not on Kaggle.

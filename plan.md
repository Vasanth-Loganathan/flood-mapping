# Project Documentation

## Rapid Flood Extent Mapping from Sentinel-1 SAR Imagery using Multi-Temporal Attention U-Net

**Status:** Initial dataset exploration completed; preprocessing is
next.\
**Primary dataset:** Kuro Siwo.\
**Secondary dataset considered:** Sen1Floods11.\
**Current environment:** Kaggle Notebook; Google Colab is also being
considered.

## 1. Project overview

The project aims to map flood extent from Sentinel-1 Synthetic Aperture
Radar (SAR) imagery. Optical imagery can be obscured by clouds, whereas
SAR is useful during and after severe weather because it is much less
affected by cloud cover and does not depend on daylight.

The proposed model is a **Multi-Temporal Attention U-Net**. It will use
observations of the same geographical patch from two pre-flood dates and
one post-flood date to predict floodwater pixels.

**Research question:** Does a Multi-Temporal Attention U-Net improve
flood segmentation compared with simpler approaches, and how well does
it generalize to flood events not used during training?

### Planned comparisons

1.  Simple change-detection baseline.
2.  Conventional U-Net.
3.  Proposed Multi-Temporal Attention U-Net.

### Planned metrics

-   Intersection over Union (IoU)
-   F1 score
-   Precision
-   Recall

## 2. Dataset selection

### 2.1 Kuro Siwo --- primary dataset

Repository:
https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset

Kuro Siwo was selected because its sample structure provides multiple
Sentinel-1 SAR observations of the same geographical patch at different
times. This matches the proposed multi-temporal model.

The data is distributed as WebDataset TAR shards. A sample consists of
numerical arrays and a JSON metadata field, rather than one JPG or PNG
file.

### 2.2 Sen1Floods11 --- alternative / possible external benchmark

Repository: https://github.com/cloudtostreet/Sen1Floods11

Sen1Floods11 contains Sentinel-1 SAR and Sentinel-2 optical data. Its
standard sample structure does not provide the same two-pre-event plus
one-post-event six-channel SAR arrangement as Kuro Siwo. Its commonly
used water labels also do not provide the same straightforward
three-class mask structure. It may be considered later for external
evaluation, but compatible inputs and labels would need to be designed.

### 2.3 Why Kuro Siwo was selected

-   Two pre-flood and one post-flood SAR observations are available for
    a sample.
-   Each observation contains VV and VH polarization.
-   The mask distinguishes background, permanent water, and floodwater.
-   Metadata includes flood-event identifiers and acquisition dates.
-   Event identifiers can support event-wise data splitting to reduce
    leakage.

## 3. Kuro Siwo sample structure

Each sample represents one geographical patch. Inspected arrays have
shape `(1, 224, 224)`. Removing the leading singleton dimension gives
`(224, 224)`.

  Field          Meaning                            Planned role
  -------------- ---------------------------------- -----------------------------------
  `sec1_vv`      First pre-flood VV observation     Input
  `sec1_vh`      First pre-flood VH observation     Input
  `sec2_vv`      Second pre-flood VV observation    Input
  `sec2_vh`      Second pre-flood VH observation    Input
  `flood_vv`     Post-flood VV observation          Input
  `flood_vh`     Post-flood VH observation          Input
  `mask`         Ground-truth segmentation labels   Target / evaluation
  `valid_mask`   Valid-pixel indicator              Loss and metric masking
  `dem`          Digital Elevation Model            Possible later input
  `info`         JSON metadata                      Event/date analysis and splitting

### 3.1 SAR polarization

-   VV: vertical transmit, vertical receive.
-   VH: vertical transmit, horizontal receive.

VV and VH are different radar polarization measurements, not RGB color
channels.

### 3.2 Temporal arrangement

  Time               VV           VH
  ------------------ ------------ ------------
  First pre-flood    `sec1_vv`    `sec1_vh`
  Second pre-flood   `sec2_vv`    `sec2_vh`
  Post-flood         `flood_vv`   `flood_vh`

The three observations refer to the same geographical patch at different
acquisition times.

### 3.3 Mask labels

    Value Meaning
  ------- ------------------------
        0 Background / non-water
        1 Permanent water
        2 Floodwater

The initial planned task is binary flood segmentation: class 2 becomes
target 1, and classes 0 and 1 become target 0. Retain the original
three-class mask for later analysis and separate evaluation. This is a
proposed modelling choice, not a change to the original dataset labels.

### 3.4 Valid pixels

`valid_mask` identifies pixels that should contribute to training and
evaluation. Invalid pixels must be excluded from loss and metrics rather
than treated as background.

### 3.5 DEM

`dem` contains elevation information. It may help describe terrain and
flood susceptibility. It will be excluded from the first six-channel
experiment and may be added as a seventh channel in a later controlled
experiment.

### 3.6 Metadata

The `info` JSON includes fields such as: - `actid`: flood-event
identifier. - `aoiid`: Area of Interest identifier. - `flood_date`:
flood event date/time. - `sources.SL1`: first pre-event source
information. - `sources.SL2`: second pre-event source information. -
`sources.MS1`: post-event/master source information. - Other fields
including `pflood`, `pwater`, `pcovered`, `gvalid`, and `grid_id`.

In the inspected sample, acquisition dates were accessed through
`sources[*].source_date`. Metadata will support temporal analysis and
event-aware splitting.

## 4. Download and storage constraints

The project was explored in Kaggle Notebooks with Internet enabled.

Repository shard structure inspected:

  Split           Shards   Approximate total size
  ------------- -------- ------------------------
  `train_GRD`          5                  46.8 GB
  `test_GRD`          12                 122.8 GB

Training shard sizes were approximately 10.91 GB for the first four
shards and 3.11 GB for the fifth. Test shards were approximately 10.9 GB
each.

Kaggle working storage was about 20.96 GB total, with about 6.9 GB free
at the time of successful exploration. Downloads failed when
insufficient space remained. The successful exploration configuration
used:

``` python
TRAIN_SHARD_IDS = [0]
TEST_SHARD_IDS = []
```

The first training shard already existed locally at the last inspection.
A previous attempt also left a 3.11 GB shard 00004. Check actual file
sizes and free space before downloading anything. Do not extract the
entire TAR archive; use streaming reads.

A Hugging Face DNS/network failure was resolved by enabling Internet in
Kaggle.

## 5. Exploration completed

The current exploration notebook inspected every fourth sample in
`train_GRD/shard-00000.tar`, scanning 1,500 samples. These counts and
statistics describe the sampled subset only, not the complete shard or
dataset.

### 5.1 Event IDs in the sampled subset

    `actid`   Sampled patches
  --------- -----------------
        470                42
        492                53
        496                 5
        497               100
        498                25
        502               112
        518                76
        520                20
        554                18
        555               171
        559                 6
        561                10
        562                16
        567                24
    1111002                 8
    1111003                 3
    1111004               811

There were 17 event IDs in the sampled subset. Event IDs appeared
contiguous in shard order in this scan. This observation applies only to
the sampled data in this shard.

### 5.2 Shapes and types

Inspected arrays had shape `(1, 224, 224)` and numerical arrays were
observed as `float32`. The mask values ranged from 0 to 2. The
exploration code uses `np.squeeze` to remove the leading singleton
dimension.

### 5.3 SAR scale

The inspected SAR values were non-negative and consistent with
linear-scale values, not dB. For post-event VV:

  Statistic           Linear value
  ----------------- --------------
  1st percentile            0.0080
  Median                    0.1192
  99th percentile           0.8669

After applying `10 * log10(clip(x, 1e-4, None))`:

  Statistic              dB
  ----------------- -------
  1st percentile      -21.0
  Median               -9.2
  99th percentile      -0.6

The conversion used in exploration was:

``` python
10 * np.log10(np.clip(array, 1e-4, None))
```

Normalization statistics and the final normalization method are not yet
decided.

### 5.4 Class distribution

Mean per-patch pixel fractions across the scanned samples: - Permanent
water: approximately 3.85%. - Floodwater: approximately 7.88%. -
Background/non-water: approximately 88.27%.

Among scanned patches: - 14.7% had zero flood pixels. - 39.3% had more
than 5% flood pixels.

This indicates class imbalance and a meaningful number of patches
without positive flood pixels. Sampling, loss design, and metric
reporting should account for this.

### 5.5 Valid pixels

The inspected `gvalid` metadata value was true for all scanned samples.
The minimum valid-pixel fraction among scanned patches was approximately
0.438. About 3% of scanned patches had a valid fraction below 0.9.

Use the pixel-level `valid_mask` during training and evaluation;
metadata `gvalid` is not a substitute.

### 5.6 Metadata consistency

The correlation between `pflood` from metadata and flood fraction
calculated from mask class 2 was approximately 0.9983 in the scanned
subset.

### 5.7 Acquisition date gaps

For the scanned samples: - Median gap from first pre-event observation
to post-event observation: approximately 24 days. - Minimum gap: 6
days. - Maximum gap: 528 days. - Median gap from flood date to
post-event observation: approximately 1 day. - Maximum gap from flood
date to post-event observation: 9 days.

The long pre-to-post gaps matter: temporal differences may include
seasonal or other changes in addition to flooding. Retain acquisition
dates and consider them during analysis or filtering.

### 5.8 Visual checks

The notebook visualized first pre-event VV in dB, second pre-event VV in
dB, post-event VV in dB, post-event minus first pre-event VV in dB, DEM,
and the ground-truth mask. Mask display colors were black for
background, blue for permanent water, and red for floodwater.

## 6. Initial model input and target proposal

### 6.1 Six-channel input

The first experiment is planned to use:

1.  `sec1_vv`
2.  `sec1_vh`
3.  `sec2_vv`
4.  `sec2_vh`
5.  `flood_vv`
6.  `flood_vh`

After stacking, one sample should have channel-first shape
`(6, 224, 224)`. A batch should have shape `(batch_size, 6, 224, 224)`.
Keep this channel order fixed and documented.

### 6.2 Binary target

Proposed conversion:

``` python
target = (mask == 2).astype("float32")
```

Floodwater becomes 1; background and permanent water become 0. Preserve
the original mask and use `valid_mask` to exclude invalid pixels from
loss and metrics.

### 6.3 DEM experiment

Do not include DEM in the first six-channel baseline. Once the SAR-only
pipeline works, test DEM as an additional input channel and compare
results.

## 7. Data splitting and leakage prevention

Do not randomly split individual patches without considering event and
geography. Patches from the same event or nearby locations may be
strongly related, causing leakage and overly optimistic results.

Use `actid` as a starting point for event-aware splitting: - Training
events. - Validation events. - Test events.

Finalize the split only after examining enough of the dataset to ensure
each split has adequate samples. Keep test events untouched during model
selection. Current event counts come from a sample of one training shard
and are not sufficient to finalize the split.

## 8. Project stages

  -----------------------------------------------------------------------
  Stage                   Work                    Status
  ----------------------- ----------------------- -----------------------
  1                       Dataset selection,      Completed for sampled
                          structure, statistics   shard 00000
                          and visual sanity       
                          checks                  

  2                       Preprocessing design    Next
                          and implementation      

  3                       Event-aware split and   Pending
                          streaming data loader   

  4                       Simple change-detection Pending
                          baseline                

  5                       Conventional U-Net      Pending

  6                       Multi-Temporal          Pending
                          Attention U-Net         

  7                       Evaluation, ablation    Pending
                          and cross-event         
                          generalization          

  8                       Final analysis,         Pending
                          documentation and       
                          report                  
  -----------------------------------------------------------------------
Computing Environment
We selected Kaggle Notebook with GPU acceleration for model development.
Component	Configuration
GPU	NVIDIA Tesla T4
GPU memory	15.64 GB
CPU	4 cores
RAM	33.66 GB
Python	3.13.15
PyTorch	2.11.0+cu128
CUDA	12.8
Available disk initially	20.94 GB

Created the project directory:
/kaggle/working/flood_mapping_project/
│
├── data/
├── checkpoints/
├── logs/
├── results/
├── configs/
└── figures/


A project configuration file was also created at:
configs/project_config.json


2. Dataset Selection and Download
Selected dataset: Kuro Siwo WebDataset.

Repository: https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset

The repository contains:
Dataset split	Number of shards	Approximate size
train_GRD	5	46.76 GB
test_GRD	12	122.77 GB
Total	17	169.53 GB

Due to Kaggle's limited disk space, we decided not to download the entire dataset.

Downloaded the first training shard:
train_GRD/shard-00000.tar


- File size: approximately 10.91 GB.
- Number of samples: 6,000.
- Data was kept inside the TAR archive without extraction.
3. Dataset Structure
Each sample represents a geographical patch of 224 × 224 pixels.

All samples contain the following ten fields:
Field	Description
sec1_vv	First pre-flood VV polarization
sec1_vh	First pre-flood VH polarization
sec2_vv	Second pre-flood VV polarization
sec2_vh	Second pre-flood VH polarization
flood_vv	Post-flood VV polarization
flood_vh	Post-flood VH polarization
mask	Ground-truth segmentation mask
valid_mask	Valid-pixel indicator
dem	Digital Elevation Model
info	JSON metadata

All numerical arrays have:
- Shape: (1, 224, 224)
- Data type: float32
Proposed model input
We selected six SAR channels in the following order:
1. sec1_vv
2. sec1_vh
3. sec2_vv
4. sec2_vh
5. flood_vv
6. flood_vh
Expected model input shape:
(batch_size, 6, 224, 224)


DEM will initially be excluded to isolate the contribution of multi-temporal SAR observations.
Ground-truth labels
The mask contains three possible classes:
Value	Meaning
0	Background / non-water
1	Permanent water
2	Floodwater

Our initial task is binary flood segmentation:
target = (mask == 2).astype("float32")



Therefore:
- Floodwater = 1
- Background and permanent water = 0
Invalid pixels will be excluded from loss calculations and evaluation metrics using valid_mask.
4. Dataset-Level Analysis
A complete metadata scan of the first training shard identified:
Property	Result
Total samples	6,000
Unique flood events	17
Unique AOIs	9
Patches containing floodwater	5,131
Patches with zero floodwater	869
Samples with gvalid=True	6,000

Flood coverage statistics
Statistic	Coverage
Mean	7.656%
Standard deviation	12.280%
Minimum	0%
25th percentile	0.381%
Median	2.591%
75th percentile	9.461%
Maximum	99.657%

The distribution is highly skewed. Most patches contain relatively little floodwater, while some patches contain extensive flooding.

This indicates class imbalance that must be considered during model training and evaluation.
Event distribution
The shard contains 17 flood events.

The most significant observation is that event 1111004 contributes 3,246 patches, representing approximately 54.1% of the entire shard.

This creates a potential event-imbalance problem.

Random patch-level splitting could cause leakage between training and validation because patches from the same flood event could appear in both sets.
5. SAR Data Analysis
The six SAR channels were inspected using 100 samples.

All channels contain positive float32 values in linear backscatter scale.
Numerical statistics
Channel	Mean	Standard deviation	Median
sec1_vv	0.04953	0.05162	0.04234
sec1_vh	0.00938	0.00920	0.00720
sec2_vv	0.04438	0.04927	0.03720
sec2_vh	0.00813	0.00928	0.00570
flood_vv	0.10614	0.08574	0.10374
flood_vh	0.02376	0.01814	0.02516

Additional observations:
- No zero-valued pixels were found in the inspected valid SAR pixels.
- No non-finite values were found.
- The inspected samples had a valid-pixel percentage of 99.970%.
SAR visualization
A representative sample was visualized using:
- Pre-flood 1 VV and VH.
- Pre-flood 2 VV and VH.
- Post-flood VV and VH.
- DEM.
- Ground-truth mask.
- Valid-pixel mask.
The visualization confirmed that the six channels have distinct backscatter characteristics and that SAR imagery exhibits speckle noise.
Linear-to-decibel conversion
The SAR data is stored in linear scale. We investigated conversion into decibels:
\[
X_{dB}=10\log_{10}(\max(X,10^{-4}))
\]

The resulting histograms showed that the dB representation provides a more interpretable distribution than the original linear scale.

For initial preprocessing, we propose:
1. Convert linear SAR backscatter to dB.
2. Perform channel-wise standardization.
3. Calculate normalization statistics using training data only.
\[
X_{norm}=\frac{X_{dB}-\mu_c}{\sigma_c}
\]

The final normalization parameters have not yet been calculated.
6. Flood Event and Geographical Analysis
The metadata contains two important identifiers:
- actid: Flood event identifier.
- aoiid: Area of interest identifier.
The event–AOI relationship was analyzed to understand geographical overlap.
Important event relationships
Event	Number of AOIs	AOIs
502	6	1, 2, 3, 4, 5, 7
518	2	1, 2
567	2	8, 15

The remaining events in this shard each belong to one AOI.
AOI distribution
AOI 1 contains 5,300 samples, representing approximately 88.33% of the shard.

Other AOIs contain substantially fewer samples.
Connected event–AOI groups
A graph-based analysis identified three disconnected groups:
Group	Flood events	AOIs	Samples
1	567	8, 15	96
2	15 events	1, 2, 3, 4, 5, 7	5,837
3	562	13	67

This reveals that geographical diversity is highly imbalanced.

A geographical generalization experiment may be difficult to establish conclusively using this shard alone.
7. Current Research Decisions
The following decisions have been established or proposed:
Component	Current decision
Task	Binary flood segmentation
Input	Six temporal SAR channels
Patch size	224 × 224
Label	mask == 2
Invalid pixels	Excluded using valid_mask
Initial model	Conventional U-Net baseline
Proposed model	Multi-Temporal Attention U-Net
SAR transformation	Linear to dB
Normalization	Channel-wise standardization
Primary evaluation	Unseen flood events
Secondary evaluation	Geographical generalization, if sufficient data is available

The planned model comparisons are:
1. Simple change-detection baseline.
2. Conventional U-Net.
3. Proposed Multi-Temporal Attention U-Net.
Evaluation metrics:
- Intersection over Union (IoU).
- F1-score.
- Precision.
- Recall.
8. Next Steps
Current stage: Dataset investigation
The following tasks remain:
1. Inspect remaining training shards: Examine shard-00001.tar through shard-00004.tar without downloading all of them permanently.
2. Complete metadata collection: Identify every flood event, AOI and sample count across all training shards.
3. Finalize data splitting: Create event-wise training, validation and testing partitions. Ensure that no flood event appears in multiple partitions.
4. Investigate event imbalance: Determine how to handle events with very few samples and the dominance of event 1111004.
5. Build an efficient TAR dataset loader: Read samples directly from TAR archives without extracting the entire dataset.
6. Calculate normalization statistics: Use only the training partition.
7. Build preprocessing pipeline: Implement SAR conversion, normalization, binary masks and valid-pixel handling.
8. Visualize processed samples: Verify the final tensors and labels.
9. Implement baseline models: Start with change detection and conventional U-Net.
10. Implement the proposed Multi-Temporal Attention U-Net.
11. Train, evaluate and compare all models.
12. Perform event-wise generalization analysis and document results.
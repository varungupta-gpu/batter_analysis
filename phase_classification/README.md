# Batting phase classification — BiGRU

This package classifies each video frame into **preparation**, **downswing**,
**followthrough**, or **no_phase**. It contains the original trained BiGRU and its
available batting phase data, plus scripts to train, extract features, predict, and validate.

## Folder guide

```text
phase_classification/
├── code/                         Python code
│   ├── deep_bigru_architecture.py Model architecture
│   ├── train_deep_bigru.py        Dataset loading and training implementation
│   ├── train.py                  Training command
│   ├── predict.py                Prediction command
│   ├── extract_features.py       Base feature calculations
│   ├── stance_normalizer.py      Right-handed pose normalization
│   ├── prepare_features.py       Feature extraction command
│   └── validate_alignment.py     Dataset/model consistency audit
├── models/deep_bigru/             Original trained checkpoint and metadata
├── data/
│   ├── videos/                   179 unique original source videos
│   ├── annotations/              Original batting.xlsx annotation workbook
│   ├── labels/                   71 manual per-frame phase label CSVs
│   ├── keypoints/
│   │   ├── normalized/           126 normalized keypoint CSVs
│   │   └── raw/                  127 raw batsman keypoint CSVs
│   ├── features/                 126 original per-frame feature CSVs
│   └── manual_centers/           38 saved pose target selections
├── visualizations/
│   └── training_overlays/        60 existing labeling visualization videos
├── reports/
│   ├── dataset_manifest.csv      Matching files and original model split
│   ├── additional_videos.csv     53 source-only videos without feature CSVs
│   ├── alignment_report.json     Passed dataset/model alignment checks
│   └── checksums.sha256           File integrity hashes, relative to this folder
├── requirements.txt              Python dependencies
└── README.md                     This guide
```

Generated predictions, new checkpoints, and regenerated features go in `runs/`,
which is excluded from Git. The four phase classes are labels within the CSVs;
a video can contain several phases, so each source video is stored once.

## Start here

1. Download videos and weights with Git LFS.
2. Install dependencies.
3. Use `reports/dataset_manifest.csv` to find the video, labels, keypoints, features,
   and saved train/test split for the same `video_id`.
4. Run prediction with the saved checkpoint, or train a new model.

All commands below run from the repository root:

```sh
git lfs pull
python -m pip install -r requirements.txt
```

## Predict phases

```sh
python phase_classification/code/predict.py phase_classification/data/features/camera2_clip_20260318_190437_156428_features.csv --output phase_classification/runs/predictions.csv
```

The output contains `video_id`, `frame_no`, and predicted `phase`. The default checkpoint
is `models/deep_bigru/deep_bigru_phase_best.pth`. Prediction covers every supplied frame
using overlapping 80-frame windows, mean logits, and edge padding for short clips.
This package adds that inference policy; training uses action-centered crops instead.

## Train a new model

```sh
python phase_classification/code/train.py --device cpu --epochs 100
```

Use `--device cuda:0` for an available CUDA GPU. New weights go to `runs/deep_bigru`;
the original checkpoint stays unchanged. Retraining uses seed 42 and creates a new split.
The original training used an unspecified random seed; its exact split and frame windows
are preserved in `models/deep_bigru/deep_bigru_phase_metadata.json`.

## Generate features from keypoints

```sh
python phase_classification/code/prepare_features.py phase_classification/data/keypoints/normalized/camera2_clip_20260318_190437_156428_keypoints.csv --output-dir phase_classification/runs/features
```

This normalizes a copy of the keypoints to the right-handed convention and extracts
base features, with speed features normalized to 30 FPS. Raw-video pose extraction
requires the original pose detector pipeline; detector weights are outside this BiGRU package.

## Verify alignment

```sh
python phase_classification/code/validate_alignment.py
```

The audit checks matching video IDs, ordered and unique frame numbers, manual label
bounds, raw/normalized keypoint alignment, video decoding, checkpoint feature/class order,
disjoint train/test videos, exact saved 80-frame windows, and finite model output.
It regenerates the reports and checksums. Full video decoding can take several minutes.

## Model and data details

- Architecture: separate forward/backward GRU stacks with 64, 128, and 256 hidden units;
  61 input features; 938,692 parameters; 80-frame sequences.
- Class index order: `preparation`, `downswing`, `followthrough`, `no_phase`.
- Original saved split: 100 training videos and 18 test videos. Eight feature sequences
  were shorter than 80 frames and excluded by the original trainer.
- Frame IDs are zero-based original-video indices. Join files using `video_id` and
  `frame_no`. Never interpret a CSV row number as a frame ID.
- All original feature values, label CSVs, keypoint CSVs, and checkpoint bytes are preserved.
  Identical duplicate source video copies were deduplicated.
- Six feature files lack nine legacy distance features. The original trainer and prediction
  script fill missing columns with zero and use checkpoint feature order. Missing/nonfinite
  values follow the original forward-fill/backward-fill/zero treatment. Affected files are
  identified by `zero_filled_features` in the dataset manifest. The current extractor omits
  those nine legacy features, so regenerated features use the same documented zero-fill.
- Of the 118 included model samples, 52 have no manual label CSV and were treated as
  `no_phase` by the original trainer. These are assumed negatives, not verified annotations.
  Unlisted frames in manually labeled videos also default to `no_phase`.
- Additional videos and one extra raw keypoint CSV are retained as available source data;
  their presence does not imply they were used to train the saved model.
- Alignment checks establish structural consistency, not the semantic correctness of human
  labels. Historical dataset hashes were not saved with the checkpoint, so historical
  feature values cannot be independently proven.

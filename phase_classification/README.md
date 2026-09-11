# Batting phase classification with Deep BiGRU

This self-contained folder packages the existing trained phase classifier, source videos,
manual phase CSVs, normalized and raw batsman keypoints, base features, and training overlays.
The checkpoint is preserved unchanged. Other classifiers and their predictions are excluded.
The repository's pre-existing trigger detection files remain outside this folder.

Install Git LFS before cloning, then run `git lfs pull` to download videos and weights.
From the repository root:

```sh
python -m pip install -r phase_classification/requirements.txt
python phase_classification/validate_alignment.py
python phase_classification/train.py --device cpu --epochs 100
python phase_classification/predict.py phase_classification/output/features/camera2_clip_20260318_190437_156428_features.csv --output phase_classification/runs/predictions.csv
```

Training writes to `runs/deep_bigru`, preserving the bundled checkpoint. Use `--device cuda:0`
for an available CUDA GPU. Retraining uses seed 42 and creates a new split; the original
training used an unspecified random seed. The exact original split and action-centered
80-frame windows are preserved in the checkpoint metadata and checked by the validator.

| Path | Contents |
| --- | --- |
| `input/` | Original phase source videos and original annotation workbook |
| `input/test/` | Additional unique source videos; folder name is not the model split |
| `output/features/` | Original per-frame base feature CSVs |
| `output/keypoints/` | Original normalized and raw batsman keypoint CSVs |
| `output/training/*_phase_labels.csv` | Manual per-frame phase labels |
| `output/training/*_training_overlay.mp4` | Existing labeling visualization videos |
| `output/manual_centers/` | Saved pose target selection coordinates |
| `output/models/deep_bigru/` | Trained weights and original metadata |
| `dataset_manifest.csv` | Matching video, features, keypoints, labels, and original split |
| `additional_videos.csv` | Source-only videos without extracted feature CSVs |
| `alignment_report.json` | Results of the reproducible alignment audit |
| `checksums.sha256` | File integrity manifest |

The architecture has separate forward/backward GRU stacks of 64, 128, and 256 hidden units,
61 input features, and 938,692 parameters. Class indices are ordered exactly as
`preparation`, `downswing`, `followthrough`, `no_phase`. Training uses 80-frame sequences.

CSV frame numbers are zero-based original-video frame indices. Match on `video_id` and
`frame_no`, never CSV row position across different videos. The manifest resolves source
videos in both input folders. Byte-identical source duplicates were removed from this package.
All original CSV feature values are retained. Six files lack nine checkpoint features;
the original trainer and the inference script explicitly fill these columns with zero,
then order columns using the checkpoint's feature list. Missing/nonfinite values receive
the original forward-fill/backward-fill/zero treatment. The manifest identifies these files.

The saved model used 100 training and 18 test videos. Eight feature sequences are shorter
than 80 frames and were excluded by the original trainer. There are 71 manual label CSVs;
52 of the included model samples have no manual label file and were treated as `no_phase`
by the original trainer. These are assumptions, not verified negative annotations.
Unlisted frames in a manual label CSV also default to `no_phase`. No new labels are invented.
Alignment checks establish file/frame/schema consistency; they do not establish that a
human annotation is semantically correct. No historical dataset hashes were saved with
the checkpoint, so exact historical feature values cannot be independently proven.

`predict.py` loads the checkpoint strictly and predicts all supplied frames using overlapping
80-frame windows, mean logits, and edge padding for short clips. This inference policy is
newly provided for this package; it is distinct from training's action-centered cropping.

To regenerate base features from existing keypoints:

```sh
python phase_classification/prepare_features.py phase_classification/output/keypoints/camera2_clip_20260318_190437_156428_keypoints.csv --output-dir phase_classification/runs/features
```

This normalizes a copy to the right-handed convention and uses the existing feature extractor
with speed features normalized to 30 FPS. The current extractor omits nine legacy distance
features, so checkpoint inference zero-fills them as described above. Raw-video pose extraction
requires the original pose detection pipeline; its detector weights are outside this BiGRU package.

# Batting phase classification

This branch contains only the batting phase classification BiGRU package.

**[Open the full folder guide and commands](phase_classification/README.md).**

| Folder | What it contains |
| --- | --- |
| [code](phase_classification/code/) | Model architecture, training, prediction, feature extraction, validation |
| [models](phase_classification/models/deep_bigru/) | Original trained BiGRU weights and metadata |
| [videos](phase_classification/data/videos/) | 179 unique source videos |
| [labels](phase_classification/data/labels/) | 71 manual phase label CSVs |
| [keypoints](phase_classification/data/keypoints/) | Raw and normalized batsman keypoints |
| [features](phase_classification/data/features/) | 126 feature CSVs |
| [annotations](phase_classification/data/annotations/) | Original annotation workbook |
| [manual centers](phase_classification/data/manual_centers/) | Saved pose target selections |
| [visualizations](phase_classification/visualizations/training_overlays/) | Training overlay videos |
| [reports](phase_classification/reports/) | File mapping, saved split, alignment results, checksums |

```sh
git lfs pull
python -m pip install -r requirements.txt
```

Use the [dataset manifest](phase_classification/reports/dataset_manifest.csv) to find matching
files by video ID. The checkpoint uses 100 training and 18 test videos.

Existing data limitations are documented in the guide: six feature files need missing-feature
zero-filling, and 52 unlabeled model samples were treated as `no_phase` by the original trainer.

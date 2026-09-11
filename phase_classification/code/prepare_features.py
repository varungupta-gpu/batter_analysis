"""Normalize a copy of keypoints and extract the phase model's base features."""
import argparse
from pathlib import Path
import shutil
from extract_features import extract_features
from stance_normalizer import normalize_keypoint_csv_to_right_handed

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('keypoints', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    target = a.output_dir/a.keypoints.name
    if target.resolve() == a.keypoints.resolve():
        raise ValueError('Choose an output directory separate from the source keypoints')
    shutil.copy2(a.keypoints, target)
    normalize_keypoint_csv_to_right_handed(target)
    extract_features(target, a.output_dir/target.name.replace('_keypoints.csv', '_features.csv'))

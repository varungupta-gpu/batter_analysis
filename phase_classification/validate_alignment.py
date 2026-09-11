"""Validate source video/CSV/model alignment and write reproducible manifests."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
import torch
from predict import load_model
from train_deep_bigru import build_deep_bigru_samples, _load_video_labels

ROOT = Path(__file__).resolve().parent

def main():
    model_dir = ROOT/'output/models/deep_bigru'
    metadata = json.loads((model_dir/'deep_bigru_phase_metadata.json').read_text())
    model, checkpoint = load_model(model_dir/'deep_bigru_phase_best.pth')
    for key in ['feature_names', 'classes', 'sequence_length', 'hidden_dims', 'num_classes']:
        assert checkpoint[key] == metadata[key], key
    train, test = set(metadata['train_videos']), set(metadata['test_videos'])
    assert not train & test, 'Train/test video overlap'
    videos = {}
    for path in sorted((ROOT/'input').rglob('*.mp4')):
        assert path.stem not in videos, f'Duplicate source video: {path.stem}'
        videos[path.stem] = path
    records = []
    for path in sorted((ROOT/'output/features').glob('*_features.csv')):
        video_id = path.name.removesuffix('_features.csv')
        df = pd.read_csv(path)
        assert set(df.video_id.astype(str)) == {video_id}, video_id
        assert df.frame_no.is_unique and df.frame_no.is_monotonic_increasing, video_id
        assert np.equal(df.frame_no, df.frame_no.astype(int)).all(), video_id
        keypoints = ROOT/'output/keypoints'/f'{video_id}_keypoints.csv'
        kd = pd.read_csv(keypoints)
        assert kd.frame_no.is_unique and np.array_equal(kd.frame_no, df.frame_no), video_id
        raw = ROOT/'output/keypoints'/f'{video_id}_raw_batsman_keypoints.csv'
        if raw.exists():
            assert np.array_equal(pd.read_csv(raw).frame_no, df.frame_no), video_id
        video = videos[video_id]
        cap = cv2.VideoCapture(str(video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert cap.isOpened() and fps > 0 and count > df.frame_no.max() and df.frame_no.min() >= 0, video_id
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
        assert decoded > df.frame_no.max(), f'Cannot decode {video_id}: {decoded} frames'
        cap.release()
        label = ROOT/'output/training'/f'{video_id}_phase_labels.csv'
        if label.exists():
            ld = _load_video_labels(label)
            assert ld.frame_no.is_unique and set(ld.frame_no) <= set(df.frame_no), video_id
            assert set(ld.video_id.astype(str)) == {video_id}, video_id
        records.append(dict(video_id=video_id, video=str(video.relative_to(ROOT)).replace('\\','/'),
                            feature_csv=path.relative_to(ROOT).as_posix(), keypoints_csv=keypoints.relative_to(ROOT).as_posix(),
                            label_csv=label.relative_to(ROOT).as_posix() if label.exists() else '',
                            label_status='manual' if label.exists() else 'unlabeled_assumed_no_phase_by_original_trainer',
                            split='train' if video_id in train else 'test' if video_id in test else 'excluded_too_short',
                            video_frames=count, fps=fps, feature_rows=len(df),
                            zero_filled_features=';'.join(c for c in checkpoint['feature_names'] if c not in df)))
    samples, features, skipped = build_deep_bigru_samples(ROOT/'output/features', ROOT/'output/training')
    assert features == checkpoint['feature_names']
    assert {s['video_id'] for s in samples} == train | test
    windows = {w['video_id']: w for w in metadata['sample_windows']}
    for sample in samples:
        for key in ['start_frame','end_frame','active_start_frame','active_end_frame']:
            assert sample[key] == windows[sample['video_id']][key], (sample['video_id'], key)
    torch.set_num_threads(2)
    with torch.inference_mode():
        logits = model(torch.tensor(np.stack([s['features'] for s in samples])))
    assert logits.shape == (len(samples), 80, 4) and torch.isfinite(logits).all()
    pd.DataFrame(records).to_csv(ROOT/'dataset_manifest.csv', index=False)
    matched = {r['video_id'] for r in records}
    extra = [dict(video_id=v, video=p.relative_to(ROOT).as_posix(), status='source_only_no_feature_csv') for v,p in videos.items() if v not in matched]
    pd.DataFrame(extra).to_csv(ROOT/'additional_videos.csv', index=False)
    report = dict(status='passed', feature_video_sets=len(records), manually_labeled=sum(bool(r['label_csv']) for r in records),
                  source_videos=len(videos), additional_source_only_videos=len(extra), train_videos=len(train), test_videos=len(test),
                  model_features=len(features), schema_zero_fill_files=sum(bool(r['zero_filled_features']) for r in records),
                  loader_notes=skipped, checkpoint_forward_shape=list(logits.shape),
                  checks=['checkpoint metadata and weights', 'disjoint saved split', 'exact saved sample windows',
                          'video bounds and sequential decoding', 'unique ordered identical keypoint/feature frames',
                          'manual label frame and video IDs', 'finite inference for every saved model sample'])
    (ROOT/'alignment_report.json').write_text(json.dumps(report, indent=2)+'\n')
    hashes = []
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and 'runs' not in path.parts and path.name != 'checksums.sha256':
            hashes.append(hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()+'  '+path.relative_to(ROOT).as_posix())
    (ROOT/'checksums.sha256').write_text('\n'.join(hashes)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

"""Predict every CSV frame using overlapping 80-frame windows and mean logits."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from deep_bigru_architecture import create_deep_bigru_model

ROOT = Path(__file__).resolve().parents[1]

def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = create_deep_bigru_model(len(checkpoint['feature_names']), checkpoint['num_classes'],
                                    checkpoint['hidden_dims'], checkpoint['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model, checkpoint

def predict(features, output, checkpoint_path):
    model, checkpoint = load_model(checkpoint_path)
    df = pd.read_csv(features).sort_values('frame_no').reset_index(drop=True)
    if df.empty or df.frame_no.duplicated().any():
        raise ValueError('Features must contain unique, nonempty frame numbers')
    names = checkpoint['feature_names']
    missing = [c for c in names if c not in df]
    if missing:
        print(f'Zero-filling {len(missing)} missing features, matching the training loader: {missing}')
    x = df.reindex(columns=names, fill_value=0).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0).to_numpy(dtype=np.float32)
    length = checkpoint['sequence_length']
    total = np.zeros((len(x), checkpoint['num_classes']), dtype=np.float64)
    count = np.zeros((len(x), 1))
    starts = sorted(set(list(range(0, max(1, len(x)-length+1), length//2)) + [max(0, len(x)-length)]))
    with torch.inference_mode():
        for start in starts:
            window = x[start:start+length]
            valid = len(window)
            if valid < length:
                window = np.pad(window, ((0, length-valid), (0, 0)), mode='edge')
            logits = model(torch.from_numpy(window[None])).numpy()[0, :valid]
            total[start:start+valid] += logits
            count[start:start+valid] += 1
    result = df[['video_id', 'frame_no']].copy()
    result['phase'] = np.array(checkpoint['classes'])[np.argmax(total/count, axis=1)]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('features', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'models/deep_bigru/deep_bigru_phase_best.pth')
    a = p.parse_args()
    predict(a.features, a.output, a.checkpoint)

"""Train the phase BiGRU from the bundled feature and manual-label CSVs."""
import argparse
from pathlib import Path
from train_deep_bigru import train_deep_bigru_classifier

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=ROOT / 'runs/deep_bigru')
    a = p.parse_args()
    train_deep_bigru_classifier(ROOT / 'data/features', ROOT / 'data/labels',
                                a.output, num_epochs=a.epochs, device=a.device, random_state=a.seed)

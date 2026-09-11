"""
Train a deep bidirectional GRU for per-frame phase classification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from deep_bigru_architecture import DeepBidirectionalGRU, create_deep_bigru_model


PHASE_CLASSES = ["preparation", "downswing", "followthrough", "no_phase"]
PHASE_ALIASES = {
    "prep": "preparation",
    "preparation": "preparation",
    "downswing": "downswing",
    "followthrough": "followthrough",
    "follow_through": "followthrough",
    "no_phase": "no_phase",
}
ID_COLUMNS = {"video_id", "frame", "frame_no", "phase", "Phase"}


def _resolve_device(device: str) -> torch.device:
    if device.startswith("gpu"):
        device = device.replace("gpu", "cuda", 1)
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _normalize_phase(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    key = " ".join(key.split())
    normalized = PHASE_ALIASES.get(key) or PHASE_ALIASES.get(key.replace(" ", "_"))
    if normalized not in PHASE_CLASSES:
        raise ValueError(f"Unsupported phase label: {value}")
    return normalized


class DeepBiGRUPhaseDataset(Dataset):
    def __init__(self, samples: list[dict], label_encoder: LabelEncoder):
        self.video_ids = [sample["video_id"] for sample in samples]
        self.features = np.array([sample["features"] for sample in samples], dtype=np.float32)
        labels = [sample["labels"] for sample in samples]
        self.labels = np.array([label_encoder.transform(label_seq) for label_seq in labels], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.labels[idx])


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in ID_COLUMNS]


def _load_video_labels(label_csv: Path) -> pd.DataFrame:
    label_df = pd.read_csv(label_csv)
    if "frame" in label_df.columns and "frame_no" not in label_df.columns:
        label_df = label_df.rename(columns={"frame": "frame_no"})
    if "phase" not in label_df.columns:
        raise ValueError(f"Missing phase column in {label_csv}")
    label_df["phase"] = label_df["phase"].map(_normalize_phase)
    return label_df


def _overlay_phase_labels(feature_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    video_df = feature_df.copy()
    video_df["phase"] = "no_phase"

    if {"start_frame", "end_frame", "phase"}.issubset(label_df.columns):
        for _, row in label_df.iterrows():
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            mask = video_df["frame_no"].between(start_frame, end_frame)
            video_df.loc[mask, "phase"] = row["phase"]
    else:
        frame_labels = label_df[["frame_no", "phase"]].drop_duplicates("frame_no", keep="last")
        phase_map = dict(zip(frame_labels["frame_no"].astype(int), frame_labels["phase"]))
        mapped = video_df["frame_no"].astype(int).map(phase_map)
        video_df.loc[mapped.notna(), "phase"] = mapped[mapped.notna()].values

    return video_df


def _crop_to_sequence(
    video_df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int,
    video_id: str,
) -> Optional[dict]:
    if len(video_df) < sequence_length:
        return None

    active_positions = np.flatnonzero(video_df["phase"].values != "no_phase")
    if len(active_positions) == 0:
        # Unlabelled videos are valid negative examples. Use their middle 80
        # frames so no_phase is represented without needing a label CSV.
        start = max(0, (len(video_df) - sequence_length) // 2)
        crop = video_df.iloc[start : start + sequence_length].copy()
        return {
            "video_id": video_id,
            "features": crop[feature_cols].values.astype(np.float32),
            "labels": crop["phase"].astype(str).values,
            "start_frame": int(crop["frame_no"].iloc[0]),
            "end_frame": int(crop["frame_no"].iloc[-1]),
            "active_start_frame": None,
            "active_end_frame": None,
        }

    active_start = int(active_positions[0])
    active_end = int(active_positions[-1])
    active_length = active_end - active_start + 1
    if active_length > sequence_length:
        # Preserve the video instead of dropping it. A centered crop retains
        # the middle of the labelled action while maintaining a fixed input.
        start = active_start + (active_length - sequence_length) // 2
        crop = video_df.iloc[start : start + sequence_length].copy()
        return {
            "video_id": video_id,
            "features": crop[feature_cols].values.astype(np.float32),
            "labels": crop["phase"].astype(str).values,
            "start_frame": int(crop["frame_no"].iloc[0]),
            "end_frame": int(crop["frame_no"].iloc[-1]),
            "active_start_frame": int(video_df.iloc[active_start]["frame_no"]),
            "active_end_frame": int(video_df.iloc[active_end]["frame_no"]),
        }

    remaining = sequence_length - active_length
    before = min(active_start, remaining // 2)
    after = min(len(video_df) - active_end - 1, remaining - before)

    missing = remaining - before - after
    if missing > 0:
        extra_before = min(active_start - before, missing)
        before += extra_before
        missing -= extra_before
    if missing > 0:
        extra_after = min(len(video_df) - active_end - 1 - after, missing)
        after += extra_after
        missing -= extra_after
    if missing > 0:
        return None

    start = active_start - before
    end = start + sequence_length
    if end > len(video_df):
        end = len(video_df)
        start = end - sequence_length
    if start < 0 or active_start < start or active_end >= end:
        return None

    crop = video_df.iloc[start:end].copy()
    return {
        "video_id": video_id,
        "features": crop[feature_cols].values.astype(np.float32),
        "labels": crop["phase"].astype(str).values,
        "start_frame": int(crop["frame_no"].iloc[0]),
        "end_frame": int(crop["frame_no"].iloc[-1]),
        "active_start_frame": int(video_df.iloc[active_start]["frame_no"]),
        "active_end_frame": int(video_df.iloc[active_end]["frame_no"]),
    }


def build_deep_bigru_samples(
    features_dir: Path,
    labels_dir: Path,
    sequence_length: int = 80,
) -> tuple[list[dict], list[str], dict]:
    features_dir = Path(features_dir)
    labels_dir = Path(labels_dir)
    feature_files = sorted(features_dir.glob("*_features.csv"))
    if not feature_files:
        raise FileNotFoundError(f"No feature CSV files found in {features_dir}")

    samples: list[dict] = []
    feature_names = sorted(
        {
            column
            for feature_csv in feature_files
            if feature_csv.name not in {"combined_features.csv", "phase_training_dataset.csv"}
            for column in _feature_columns(pd.read_csv(feature_csv, nrows=0))
        }
    )
    if not feature_names:
        raise ValueError("No feature columns found in the base feature CSVs.")
    skipped = {
        "included_without_labels": 0,
        "too_short": 0,
        "no_active_phase": 0,
        "active_longer_than_sequence": 0,
    }

    for feature_csv in feature_files:
        if feature_csv.name in {"combined_features.csv", "phase_training_dataset.csv"}:
            continue

        feature_df = pd.read_csv(feature_csv)
        if "frame" in feature_df.columns and "frame_no" not in feature_df.columns:
            feature_df = feature_df.rename(columns={"frame": "frame_no"})
        if "video_id" not in feature_df.columns or "frame_no" not in feature_df.columns:
            continue

        video_id = str(feature_df["video_id"].iloc[0])
        label_csv = labels_dir / f"{video_id}_phase_labels.csv"
        source_feature_cols = _feature_columns(feature_df)
        feature_df = feature_df.reindex(columns=[*ID_COLUMNS.intersection(feature_df.columns), *feature_names])
        for column in feature_names:
            if column not in source_feature_cols:
                feature_df[column] = 0.0

        feature_df = feature_df.sort_values("frame_no").reset_index(drop=True)
        feature_df[feature_names] = feature_df[feature_names].replace([np.inf, -np.inf], np.nan)
        feature_df[feature_names] = feature_df[feature_names].ffill().bfill().fillna(0.0)

        if len(feature_df) < sequence_length:
            skipped["too_short"] += 1
            continue

        if label_csv.exists():
            label_df = _load_video_labels(label_csv)
            video_df = _overlay_phase_labels(feature_df, label_df)
        else:
            skipped["included_without_labels"] += 1
            video_df = feature_df.copy()
            video_df["phase"] = "no_phase"

        sample = _crop_to_sequence(video_df, feature_names, sequence_length, video_id)
        if sample is None:
            active_count = int((video_df["phase"] != "no_phase").sum())
            if active_count == 0:
                skipped["no_active_phase"] += 1
            elif active_count > sequence_length:
                skipped["active_longer_than_sequence"] += 1
            else:
                skipped["too_short"] += 1
            continue
        samples.append(sample)

    if not samples:
        raise ValueError(f"No 80-frame BiGRU samples were built. Skipped: {skipped}")

    return samples, feature_names, skipped


def _run_epoch(
    model: DeepBidirectionalGRU,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, float, list[int], list[int]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.set_grad_enabled(is_train):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds = logits.argmax(dim=-1)
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.numel()
            all_preds.extend(preds.detach().cpu().reshape(-1).tolist())
            all_labels.extend(labels.detach().cpu().reshape(-1).tolist())

    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1), all_preds, all_labels


def train_deep_bigru_classifier(
    features_dir: Path,
    labels_dir: Path,
    output_model_dir: Path,
    sequence_length: int = 80,
    hidden_dims: tuple[int, int, int] = (64, 128, 256),
    dropout: float = 0.3,
    batch_size: int = 8,
    num_epochs: int = 100,
    learning_rate: float = 0.0005,
    test_size: float = 0.15,
    random_state: Optional[int] = None,
    device: str = "cuda:0",
) -> tuple[DeepBidirectionalGRU, LabelEncoder, list[str], list[str]]:
    output_model_dir = Path(output_model_dir)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    torch_device = _resolve_device(device)
    print(f"Using device: {torch_device}")

    samples, feature_names, skipped = build_deep_bigru_samples(features_dir, labels_dir, sequence_length)
    video_ids = np.array([sample["video_id"] for sample in samples])

    if len(samples) < 2:
        raise ValueError("Need at least two valid videos for train/test split.")

    train_videos, test_videos = train_test_split(
        video_ids,
        test_size=test_size,
        random_state=random_state,
    )
    train_set = set(train_videos)
    test_set = set(test_videos)
    train_samples = [sample for sample in samples if sample["video_id"] in train_set]
    test_samples = [sample for sample in samples if sample["video_id"] in test_set]

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(PHASE_CLASSES, dtype=object)
    train_dataset = DeepBiGRUPhaseDataset(train_samples, label_encoder)
    test_dataset = DeepBiGRUPhaseDataset(test_samples, label_encoder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = create_deep_bigru_model(
        input_dim=len(feature_names),
        output_dim=len(label_encoder.classes_),
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(torch_device)

    print(f"Built samples: {len(samples)}")
    print(f"Dataset notes: {skipped}")
    print(f"Train videos: {len(train_samples)}")
    print(f"Test videos: {len(test_samples)}")
    print(f"Input dim: {len(feature_names)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Total parameters: {model.get_total_params():,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_path = output_model_dir / "deep_bigru_phase_best.pth"

    print(f"Starting Deep BiGRU training ({num_epochs} epochs)...")
    for epoch in range(num_epochs):
        train_loss, train_acc, _, _ = _run_epoch(model, train_loader, criterion, torch_device, optimizer)
        val_loss, val_acc, _, _ = _run_epoch(model, test_loader, criterion, torch_device)
        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_encoder": label_encoder,
                    "feature_names": feature_names,
                    "sequence_length": sequence_length,
                    "hidden_dims": list(hidden_dims),
                    "dropout": dropout,
                    "num_classes": len(label_encoder.classes_),
                    "classes": list(label_encoder.classes_),
                    "model_type": "DeepBidirectionalGRU",
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_loss, train_acc, train_preds, train_labels = _run_epoch(model, train_loader, criterion, torch_device)
    test_loss, test_acc, test_preds, test_labels = _run_epoch(model, test_loader, criterion, torch_device)

    print("\nTRAINING SET CLASSIFICATION REPORT")
    print(classification_report(train_labels, train_preds, labels=list(range(4)), target_names=PHASE_CLASSES, zero_division=0))
    print(f"Training Accuracy: {train_acc:.2f}%")

    print("\nTEST SET CLASSIFICATION REPORT")
    print(classification_report(test_labels, test_preds, labels=list(range(4)), target_names=PHASE_CLASSES, zero_division=0))
    print(f"Test Accuracy: {test_acc:.2f}%")

    metadata = {
        "model_type": "DeepBidirectionalGRU",
        "num_classes": len(label_encoder.classes_),
        "classes": list(label_encoder.classes_),
        "feature_names": feature_names,
        "sequence_length": sequence_length,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "test_size": test_size,
        "best_val_loss": best_val_loss,
        "train_videos": list(train_videos),
        "test_videos": list(test_videos),
        "skipped": skipped,
        "sample_windows": [
            {
                "video_id": sample["video_id"],
                "start_frame": sample["start_frame"],
                "end_frame": sample["end_frame"],
                "active_start_frame": sample["active_start_frame"],
                "active_end_frame": sample["active_end_frame"],
            }
            for sample in samples
        ],
        "total_params": model.get_total_params(),
    }
    with (output_model_dir / "deep_bigru_phase_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to: {output_model_dir}")
    return model, label_encoder, feature_names, list(test_videos)

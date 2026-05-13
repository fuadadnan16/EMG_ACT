"""
predict.py — Inference with trained multi-label EMG hand activation model
=========================================================================
Load a trained model and predict hand zone activations from new EMG data.

Usage:
  python predict.py --model checkpoints_multilabel/best_model.pt \
                    --emg_file new_recording.csv
"""

import pathlib, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import model classes from train script
# (In production, these would be in a shared module)

NUM_LABELS = 10
NUM_EMG_CHANNELS = 8
TARGET_EMG_CHANNELS = 16

LABEL_COLUMNS = [
    "label_palm", "label_thumb",
    "label_index_tip", "label_index_seg",
    "label_middle_tip", "label_middle_seg",
    "label_ring_tip", "label_ring_seg",
    "label_pinky_tip", "label_pinky_seg",
]

ZONE_NAMES = [col.replace("label_", "") for col in LABEL_COLUMNS]


class EMGMultiLabelModelStandalone(nn.Module):
    def __init__(self, num_labels=NUM_LABELS, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(NUM_EMG_CHANNELS, 32, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def load_model(model_path: str, device="cpu"):
    """Load trained model from checkpoint."""
    ckpt = torch.load(model_path, map_location=device)

    if ckpt.get("use_pretrained", False):
        raise NotImplementedError(
            "For pretrained-backbone models, load the emg2pose checkpoint too. "
            "Use the standalone model for simpler inference."
        )

    model = EMGMultiLabelModelStandalone()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    return model, ckpt


def predict_emg_file(model, emg_file: str, fs=500, window_sec=2.0,
                     stride_sec=0.5, threshold=0.5, device="cpu"):
    """
    Run sliding window predictions on an EMG file.
    Returns DataFrame with timestamp, predictions, and zone activations.
    """
    from scipy.signal import butter, filtfilt, iirnotch

    emg_cols = [f"EMG{i}_Raw" for i in range(1, 9)]
    df = pd.read_csv(emg_file)
    raw = df[emg_cols].values.astype(np.float32)
    timestamps = df["unix_time_s"].values

    # Preprocess
    raw -= raw.mean(axis=0)
    nyq = 0.5 * fs
    b_notch, a_notch = iirnotch(60.0 / nyq, Q=30)
    filtered = filtfilt(b_notch, a_notch, raw, axis=0).astype(np.float32)
    b, a = butter(4, [20.0 / nyq, 200.0 / nyq], btype="band")
    filtered = filtfilt(b, a, filtered, axis=0).astype(np.float32)
    std = filtered.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    filtered = (filtered - filtered.mean(axis=0, keepdims=True)) / std

    window_size = int(window_sec * fs)
    stride = int(stride_sec * fs)

    results = []
    with torch.no_grad():
        for start in range(0, len(filtered) - window_size + 1, stride):
            end = start + window_size
            window = torch.tensor(
                filtered[start:end][np.newaxis], dtype=torch.float32
            ).to(device)

            logits = model(window)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            active = (probs >= threshold).astype(int)

            center_idx = start + window_size // 2
            center_time = timestamps[min(center_idx, len(timestamps) - 1)]

            result = {
                "unix_time": center_time,
                "window_start_idx": start,
                "window_end_idx": end,
            }
            for i, zone in enumerate(ZONE_NAMES):
                result[f"prob_{zone}"] = round(float(probs[i]), 3)
                result[f"active_{zone}"] = int(active[i])

            # Human-readable active zones
            active_zones = [ZONE_NAMES[i] for i in range(NUM_LABELS) if active[i]]
            result["active_zones"] = ", ".join(active_zones) if active_zones else "rest"

            results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to trained model .pt")
    p.add_argument("--emg_file", required=True, help="Path to EMG CSV file")
    p.add_argument("--fs", type=int, default=500)
    p.add_argument("--window_sec", type=float, default=2.0)
    p.add_argument("--stride_sec", type=float, default=0.25)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", default=None, help="Output CSV path")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.model, device)
    print(f"Loaded model (epoch {ckpt['epoch']}, val_f1={ckpt['val_f1']:.4f})")

    results = predict_emg_file(
        model, args.emg_file, args.fs,
        args.window_sec, args.stride_sec, args.threshold, device
    )

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    else:
        print(results[["unix_time", "active_zones"]].to_string())

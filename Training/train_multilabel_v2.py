"""
train_multilabel_v2.py — EMG Multi-Label Hand Activation Model (v2)
====================================================================
Improvements over v1:
  - FIX: torch.load weights_only=False for PyTorch 2.6 compatibility
  - FIX: torch.save without numpy arrays (avoid unpickling error)
  - BETTER SPLIT: utterance-based split (not time-based) so val sees
    complete command patterns, with ~25% val for more val windows
  - ANTI-OVERFITTING: heavier dropout (0.5), weight decay (1e-3),
    label smoothing, smaller model, spectral augmentation
  - DATA AUGMENTATION: time-shift, channel dropout, Gaussian noise,
    amplitude scaling — applied online during training
  - BETTER ARCHITECTURE: added temporal attention pooling, residual
    connection in classifier

Usage:
  python train_multilabel_v2.py \
    --gt_csv ground_truth_labeled.csv \
    --emg_dir ./set_1
"""

# ── Patch emg2pose for Python 3.9 BEFORE any imports ─────────────────────────
import pathlib, sys, os

def _patch_emg2pose(repo_root: str = "emg2pose"):
    pkg = pathlib.Path(repo_root) / "emg2pose"
    if not pkg.exists():
        return
    patched = []
    for py_file in pkg.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        if "from __future__ import annotations" not in text:
            py_file.write_text(
                "from __future__ import annotations\n" + text, encoding="utf-8"
            )
            patched.append(py_file.name)
    if patched:
        print(f"[setup] Patched {len(patched)} file(s) for Python 3.9 compatibility.")

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse, csv, random, glob, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, iirnotch
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COLUMNS = [
    "label_palm", "label_thumb",
    "label_index_tip", "label_index_seg",
    "label_middle_tip", "label_middle_seg",
    "label_ring_tip", "label_ring_seg",
    "label_pinky_tip", "label_pinky_seg",
]

NUM_LABELS = len(LABEL_COLUMNS)  # 10
NUM_EMG_CHANNELS = 8
TARGET_EMG_CHANNELS = 16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_csv",        default="ground_truth_labeled.csv")
    p.add_argument("--emg_dir",       default=".")
    p.add_argument("--ckpt_path",
                   default="emg2pose_model_checkpoints/tracking_vemg2pose.ckpt")
    p.add_argument("--emg2pose_repo", default="emg2pose/")
    p.add_argument("--out_dir",       default="checkpoints_v2/")
    p.add_argument("--window_sec",    type=float, default=1.0,
                   help="Window size in seconds (shorter=more windows, less overfit)")
    p.add_argument("--stride_sec",    type=float, default=0.25,
                   help="Stride in seconds")
    p.add_argument("--epochs",        type=int,   default=80)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--fs",            type=int,   default=500)
    p.add_argument("--val_frac",      type=float, default=0.25,
                   help="Fraction of utterances for validation")
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--weight_decay",  type=float, default=1e-3)
    p.add_argument("--label_smooth",  type=float, default=0.05,
                   help="Label smoothing: 0=hard labels, 0.05=slight smooth")
    p.add_argument("--filter_type",   default="bpf",
                   choices=["bpf", "hpf", "none"])
    p.add_argument("--augment",       action="store_true", default=True,
                   help="Enable data augmentation during training")
    p.add_argument("--no_augment",    action="store_true", default=False)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(signal: np.ndarray, fs: int, filter_type: str) -> np.ndarray:
    nyq = 0.5 * fs
    if filter_type == "none":
        return signal
    b_notch, a_notch = iirnotch(60.0 / nyq, Q=30)
    signal = filtfilt(b_notch, a_notch, signal, axis=0).astype(np.float32)
    if filter_type == "bpf":
        b, a = butter(4, [20.0 / nyq, 200.0 / nyq], btype="band")
    elif filter_type == "hpf":
        b, a = butter(4, 20.0 / nyq, btype="high")
    else:
        return signal
    return filtfilt(b, a, signal, axis=0).astype(np.float32)


def load_emg_csv(csv_path: str, fs: int = 500,
                 filter_type: str = "bpf") -> tuple:
    emg_cols = [f"EMG{i}_Raw" for i in range(1, 9)]
    df = pd.read_csv(csv_path)
    raw = df[emg_cols].values.astype(np.float32)
    orig_ts = df["unix_time_s"].values.astype(np.float64)

    if len(raw) < 10:
        return raw, orig_ts

    # Resample to uniform grid
    t_start, t_end = orig_ts[0], orig_ts[-1]
    n_uniform = int((t_end - t_start) * fs) + 1
    if n_uniform < 10:
        return raw, orig_ts

    uniform_ts = np.linspace(t_start, t_end, n_uniform)
    interp_func = interp1d(orig_ts, raw, axis=0, kind='linear',
                           fill_value='extrapolate')
    resampled = interp_func(uniform_ts).astype(np.float32)

    resampled -= resampled.mean(axis=0)
    if len(resampled) > 20:
        filtered = apply_filters(resampled, fs, filter_type)
    else:
        filtered = resampled

    std = filtered.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    filtered = (filtered - filtered.mean(axis=0, keepdims=True)) / std

    return filtered, uniform_ts


# ─────────────────────────────────────────────────────────────────────────────
# DATA ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def build_label_intervals(gt_df: pd.DataFrame) -> list:
    intervals = []
    n = len(gt_df)
    for i in range(n):
        row = gt_df.iloc[i]
        start_t = row["end_unix"]
        end_t = gt_df.iloc[i + 1]["end_unix"] if i + 1 < n else start_t + 60.0
        labels = row[LABEL_COLUMNS].values.astype(np.float32)
        intervals.append({
            "start_unix": start_t,
            "end_unix": end_t,
            "labels": labels,
            "text": row["text"],
            "segment": int(row["segment"]),
        })
    return intervals


def assign_labels_to_emg(emg_timestamps, intervals):
    N = len(emg_timestamps)
    labels = np.zeros((N, NUM_LABELS), dtype=np.float32)
    for iv in intervals:
        mask = ((emg_timestamps >= iv["start_unix"]) &
                (emg_timestamps < iv["end_unix"]))
        labels[mask] = iv["labels"]
    return labels


def load_all_emg_files(emg_dir, fs, filter_type):
    """Load all EMG files, return list of (emg_data, timestamps, filename)."""
    emg_pattern = os.path.join(emg_dir, "emg_*.csv")
    emg_files = sorted(glob.glob(emg_pattern),
                       key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    if not emg_files:
        raise FileNotFoundError(f"No emg_*.csv files found in {emg_dir}")

    print(f"\n  Found {len(emg_files)} EMG files")
    all_files = []
    for emg_file in emg_files:
        fname = os.path.basename(emg_file)
        emg_data, timestamps = load_emg_csv(emg_file, fs, filter_type)
        if len(emg_data) >= 10:
            all_files.append((emg_data, timestamps, fname))
    return all_files


def build_utterance_segments(gt_df, all_files, fs):
    """
    Build per-utterance segments: each segment has EMG data + labels.
    Returns list of {emg: (N,8), labels: (10,), text: str, segment: int}
    """
    intervals = build_label_intervals(gt_df)
    segments = []

    for file_emg, file_ts, fname in all_files:
        file_labels = assign_labels_to_emg(file_ts, intervals)

        # Find which intervals this file belongs to
        for iv in intervals:
            mask = ((file_ts >= iv["start_unix"]) &
                    (file_ts < iv["end_unix"]))
            n_matching = mask.sum()
            if n_matching >= fs * 0.5:  # at least 0.5s of data
                segments.append({
                    "emg": file_emg[mask],
                    "labels": iv["labels"],
                    "text": iv["text"],
                    "segment": iv["segment"],
                    "n_samples": int(n_matching),
                })

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class EMGAugmenter:
    """Online data augmentation for EMG signals."""

    def __init__(self, p_noise=0.5, p_scale=0.5, p_shift=0.3,
                 p_channel_drop=0.2):
        self.p_noise = p_noise
        self.p_scale = p_scale
        self.p_shift = p_shift
        self.p_channel_drop = p_channel_drop

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, 8) single window."""
        # Gaussian noise
        if random.random() < self.p_noise:
            noise_std = random.uniform(0.01, 0.15)
            x = x + torch.randn_like(x) * noise_std

        # Amplitude scaling per channel
        if random.random() < self.p_scale:
            scale = torch.empty(1, x.shape[1]).uniform_(0.8, 1.2)
            x = x * scale

        # Time shift (circular)
        if random.random() < self.p_shift:
            shift = random.randint(-x.shape[0] // 10, x.shape[0] // 10)
            x = torch.roll(x, shifts=shift, dims=0)

        # Channel dropout (zero out 1 channel)
        if random.random() < self.p_channel_drop:
            ch = random.randint(0, x.shape[1] - 1)
            x[:, ch] = 0.0

        return x


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class EMGMultiLabelDataset(Dataset):
    def __init__(self, segments: list, window_size: int, stride: int,
                 augmenter=None, label_smooth=0.0):
        self.windows = []
        self.targets = []
        self.augmenter = augmenter
        self.label_smooth = label_smooth

        for seg in segments:
            emg = seg["emg"]
            labels = seg["labels"]
            for start in range(0, len(emg) - window_size + 1, stride):
                end = start + window_size
                self.windows.append(emg[start:end])
                self.targets.append(labels.copy())

        self.windows = np.array(self.windows, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.augmenter is not None:
            x = self.augmenter(x)

        if self.label_smooth > 0:
            y = y * (1 - self.label_smooth) + 0.5 * self.label_smooth

        return x, y


def prepare_loaders_v2(gt_df, all_files, fs, window_size, stride,
                       val_frac, batch_size, label_smooth, use_augment):
    """
    Utterance-based split: hold out val_frac of utterance indices.
    This ensures val sees complete command patterns the model hasn't trained on.
    """
    segments = build_utterance_segments(gt_df, all_files, fs)
    print(f"  Total segments: {len(segments)}")

    # Get unique segment (utterance) IDs
    seg_ids = sorted(set(s["segment"] for s in segments))
    n_val = max(1, int(len(seg_ids) * val_frac))

    # Hold out every Nth utterance (spread across the recording)
    # This is better than holding out the last block
    val_ids = set(seg_ids[i] for i in range(0, len(seg_ids), len(seg_ids) // n_val)[:n_val])
    train_ids = set(seg_ids) - val_ids

    train_segs = [s for s in segments if s["segment"] in train_ids]
    val_segs = [s for s in segments if s["segment"] in val_ids]

    print(f"  Train utterances: {len(train_ids)} → {len(train_segs)} segments")
    print(f"  Val utterances:   {len(val_ids)} → {len(val_segs)} segments")

    # Print val utterance examples
    val_texts = set(s["text"] for s in val_segs)
    print(f"  Val commands: {', '.join(list(val_texts)[:8])}...")

    augmenter = EMGAugmenter() if use_augment else None

    train_ds = EMGMultiLabelDataset(train_segs, window_size, stride,
                                    augmenter=augmenter,
                                    label_smooth=label_smooth)
    val_ds = EMGMultiLabelDataset(val_segs, window_size, stride,
                                  augmenter=None, label_smooth=0.0)

    print(f"  Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # Return label stats for pos_weight
    all_labels = np.array([s["labels"] for s in train_segs])
    return train_loader, val_loader, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (improved)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttentionPool(nn.Module):
    """Learned attention pooling over temporal dimension."""
    def __init__(self, feat_dim):
        super().__init__()
        self.attn = nn.Linear(feat_dim, 1)

    def forward(self, x):
        # x: (B, feat_dim, T)
        x_t = x.permute(0, 2, 1)           # (B, T, feat_dim)
        w = torch.softmax(self.attn(x_t), dim=1)  # (B, T, 1)
        return (x_t * w).sum(dim=1)         # (B, feat_dim)


class EMGMultiLabelModelV2(nn.Module):
    """
    Improved standalone model with:
      - Deeper feature extractor with residual-style connections
      - Temporal attention pooling (not just avg pool)
      - Heavier regularization
    """

    def __init__(self, num_labels=NUM_LABELS, hidden_dim=96, dropout=0.5):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(NUM_EMG_CHANNELS, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.attn_pool = TemporalAttentionPool(128)

        # Combine avg + attn pooling → 256 features
        feat_dim = 128 * 2

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"  Model V2: Conv1d(8→32→64→128→128) + AttentionPool")
        print(f"  Classifier: {feat_dim} → {hidden_dim} → {hidden_dim//2} → {num_labels}")
        print(f"  Total params: {total:,}")

    def forward(self, x):
        x = x.permute(0, 2, 1)             # (B, 8, T)
        feat = self.feature_extractor(x)    # (B, 128, T')

        # Dual pooling
        avg = self.avg_pool(feat).squeeze(-1)   # (B, 128)
        attn = self.attn_pool(feat)              # (B, 128)
        pooled = torch.cat([avg, attn], dim=1)   # (B, 256)

        return self.classifier(pooled)           # (B, 10)


class EMGMultiLabelModelPretrained(nn.Module):
    """With emg2pose backbone (if available)."""

    def __init__(self, pretrained_model, num_labels=NUM_LABELS,
                 hidden_dim=96, dropout=0.5):
        super().__init__()

        self.expand = nn.Linear(NUM_EMG_CHANNELS, TARGET_EMG_CHANNELS)

        full_model = pretrained_model.model
        conv_layers = []
        for m in full_model.modules():
            if isinstance(m, nn.Conv1d):
                conv_layers.append(m)
            if len(conv_layers) == 2:
                break

        if len(conv_layers) < 2:
            raise ValueError("Could not find 2 Conv1d layers in emg2pose model.")

        self.feature_extractor = nn.Sequential(
            conv_layers[0], nn.ReLU(),
            conv_layers[1], nn.ReLU(),
        )
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        feat_dim = conv_layers[1].out_channels
        print(f"  Backbone: 2 pretrained Conv1d layers (frozen, feat_dim={feat_dim})")

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.attn_pool = TemporalAttentionPool(feat_dim)
        combined_dim = feat_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
        )

    def forward(self, x):
        x = self.expand(x)
        x = x.permute(0, 2, 1)
        feat = self.feature_extractor(x)
        avg = self.avg_pool(feat).squeeze(-1)
        attn = self.attn_pool(feat)
        pooled = torch.cat([avg, attn], dim=1)
        return self.classifier(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_emg2pose_model(ckpt_path, repo_root):
    repo_pkg = str(pathlib.Path(repo_root) / "emg2pose")
    if repo_pkg not in sys.path:
        sys.path.insert(0, repo_pkg)

    from omegaconf import DictConfig, ListConfig
    try:
        from omegaconf.base import ContainerMetadata
        torch.serialization.add_safe_globals(
            [DictConfig, ListConfig, ContainerMetadata])
    except ImportError:
        torch.serialization.add_safe_globals([DictConfig, ListConfig])

    from lightning import Emg2PoseModule
    module = Emg2PoseModule.load_from_checkpoint(
        ckpt_path, map_location="cpu", weights_only=False
    )
    module.eval()
    print("  Pretrained emg2pose checkpoint loaded.")
    return module


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds, targets, threshold=0.5):
    pred_binary = (preds >= threshold).astype(np.float32)
    per_label_acc = (pred_binary == targets).mean(axis=0)
    exact_match = (pred_binary == targets).all(axis=1).mean()
    try:
        macro_f1 = f1_score(targets, pred_binary, average="macro", zero_division=0)
        per_label_f1 = f1_score(targets, pred_binary, average=None, zero_division=0)
    except:
        macro_f1 = 0.0
        per_label_f1 = np.zeros(NUM_LABELS)
    return {
        "exact_match": exact_match,
        "macro_f1": macro_f1,
        "per_label_acc": per_label_acc,
        "per_label_f1": per_label_f1,
        "hamming_acc": per_label_acc.mean(),
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(x)
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    # Undo label smoothing for metric computation
    all_targets = (all_targets > 0.5).astype(np.float32)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(all_preds)
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(x)
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(all_preds)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.no_augment:
        args.augment = False

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  EMG Multi-Label Hand Activation Model v2")
    print("=" * 70)
    print(f"  Device        : {device}")
    print(f"  Window        : {args.window_sec}s ({int(args.window_sec * args.fs)} samples)")
    print(f"  Stride        : {args.stride_sec}s")
    print(f"  Filter        : {args.filter_type}")
    print(f"  Dropout       : {args.dropout}")
    print(f"  Weight decay  : {args.weight_decay}")
    print(f"  Label smooth  : {args.label_smooth}")
    print(f"  Augmentation  : {args.augment}")
    print(f"  Val fraction  : {args.val_frac}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  LOADING DATA")
    print("─" * 70)

    gt_df = pd.read_csv(args.gt_csv)
    print(f"  Ground truth: {len(gt_df)} utterances")

    all_files = load_all_emg_files(args.emg_dir, args.fs, args.filter_type)

    # ── Create data loaders ───────────────────────────────────────────────────
    window_size = int(args.window_sec * args.fs)
    stride = int(args.stride_sec * args.fs)

    print("\n" + "─" * 70)
    print("  CREATING DATA LOADERS (utterance-based split)")
    print("─" * 70)

    train_loader, val_loader, train_labels = prepare_loaders_v2(
        gt_df, all_files, args.fs, window_size, stride,
        args.val_frac, args.batch_size, args.label_smooth, args.augment,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  BUILDING MODEL")
    print("─" * 70)

    use_pretrained = True
    ckpt_path = pathlib.Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"  No pretrained backbone → using standalone model v2")
        use_pretrained = False

    if use_pretrained:
        _patch_emg2pose(args.emg2pose_repo)
        pretrained = load_emg2pose_model(str(ckpt_path), args.emg2pose_repo)
        model = EMGMultiLabelModelPretrained(
            pretrained, dropout=args.dropout).to(device)
    else:
        model = EMGMultiLabelModelV2(dropout=args.dropout).to(device)

    # ── Pos weights ───────────────────────────────────────────────────────────
    label_means = train_labels.mean(axis=0)
    pos_weight = torch.tensor(
        [(1 - m) / max(m, 0.01) for m in label_means],
        dtype=torch.float32
    ).clamp(max=5.0).to(device)
    print(f"  Pos weights: {pos_weight.cpu().numpy().round(2)}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine annealing (better than ReduceLROnPlateau for small datasets)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  TRAINING")
    print("─" * 70)

    best_val_f1 = 0.0
    best_path = out_dir / "best_model.pt"
    history = []
    patience_counter = 0
    early_stop_patience = 25

    for epoch in range(1, args.epochs + 1):
        tr_metrics = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device)
        va_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        star = ""
        if va_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = va_metrics["macro_f1"]
            patience_counter = 0
            star = " ★"
            # FIX: Save only pure Python/torch types, no numpy
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": int(epoch),
                "val_f1": float(best_val_f1),
                "val_exact_match": float(va_metrics["exact_match"]),
                "val_hamming_acc": float(va_metrics["hamming_acc"]),
                "window_sec": float(args.window_sec),
                "fs": int(args.fs),
                "label_columns": list(LABEL_COLUMNS),
                "use_pretrained": bool(use_pretrained),
                "dropout": float(args.dropout),
            }, best_path)
        else:
            patience_counter += 1

        history.append({
            "epoch": epoch,
            "tr_loss": round(tr_metrics["loss"], 4),
            "tr_f1": round(tr_metrics["macro_f1"], 4),
            "va_loss": round(va_metrics["loss"], 4),
            "va_f1": round(va_metrics["macro_f1"], 4),
            "va_exact": round(va_metrics["exact_match"], 4),
            "va_hamming": round(va_metrics["hamming_acc"], 4),
            "lr": round(current_lr, 8),
        })

        # Show gap between train/val for overfitting monitoring
        gap = tr_metrics["macro_f1"] - va_metrics["macro_f1"]
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Tr F1 {tr_metrics['macro_f1']:.3f} | "
              f"Val F1 {va_metrics['macro_f1']:.3f} "
              f"Exact {va_metrics['exact_match']:.3f} "
              f"(gap={gap:.3f}){star}")

        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping — no improvement for {early_stop_patience} epochs.")
            break

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Best validation macro F1: {best_val_f1:.4f}")

    # FIX: weights_only=False for PyTorch 2.6
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    final_metrics = validate(model, val_loader, criterion, device)

    print(f"\n  Per-label results (validation):")
    print(f"  {'Zone':20s}  {'Acc':>6s}  {'F1':>6s}")
    print(f"  {'─' * 36}")
    for i, col in enumerate(LABEL_COLUMNS):
        zone_name = col.replace("label_", "")
        print(f"  {zone_name:20s}  "
              f"{final_metrics['per_label_acc'][i]:.3f}  "
              f"{final_metrics['per_label_f1'][i]:.3f}")
    print(f"  {'─' * 36}")
    print(f"  {'MACRO':20s}  "
          f"{final_metrics['hamming_acc']:.3f}  "
          f"{final_metrics['macro_f1']:.3f}")
    print(f"  {'EXACT MATCH':20s}  {final_metrics['exact_match']:.3f}")

    # Save history
    hist_path = out_dir / "training_history.csv"
    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n  Model saved → {best_path}")
    print(f"  History     → {hist_path}")
    print(f"  Config      → {config_path}")


if __name__ == "__main__":
    main()

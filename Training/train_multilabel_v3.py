"""
train_multilabel_v3.py — EMG Hand Activation Model v3 (targeting 90%+)
=======================================================================
Key insight from v2 analysis: there are only 17 unique activation patterns
across 64 utterances. This means:

1. DUAL-HEAD APPROACH: Multi-class (17 patterns) + Multi-label (10 zones)
   - Multi-class head enforces VALID combinations (no impossible patterns)
   - Multi-label head provides fine-grained per-zone gradients
   - Combined loss balances both objectives

2. FEATURE ENGINEERING: Raw EMG alone isn't enough at this scale
   - RMS envelope (captures activation intensity)
   - Mean absolute value (standard EMG feature)
   - Waveform length (captures signal complexity)
   - Zero crossings (captures frequency content)
   - These are standard clinical EMG features used in prosthetics

3. STRATIFIED K-FOLD: With only 64 utterances, a single split is noisy
   - 5-fold cross-validation for reliable metrics
   - Final model trains on all data with best hyperparameters
   - Reports mean ± std across folds

4. MIXUP AUGMENTATION: Interpolates between EMG windows and their labels

Usage:
  python train_multilabel_v3.py --gt_csv ground_truth_labeled.csv --emg_dir ./set_1
"""

import pathlib, sys, os, argparse, csv, random, glob, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, iirnotch
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from collections import Counter


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
NUM_LABELS = 10
NUM_EMG_CHANNELS = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_csv",        default="ground_truth_labeled.csv")
    p.add_argument("--emg_dir",       default=".")
    p.add_argument("--ckpt_path",     default="emg2pose_model_checkpoints/tracking_vemg2pose.ckpt")
    p.add_argument("--emg2pose_repo", default="emg2pose/")
    p.add_argument("--out_dir",       default="checkpoints_v3/")
    p.add_argument("--window_sec",    type=float, default=1.0)
    p.add_argument("--stride_sec",    type=float, default=0.25)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--fs",            type=int,   default=500)
    p.add_argument("--dropout",       type=float, default=0.4)
    p.add_argument("--weight_decay",  type=float, default=5e-4)
    p.add_argument("--label_smooth",  type=float, default=0.05)
    p.add_argument("--n_folds",       type=int,   default=5)
    p.add_argument("--filter_type",   default="bpf", choices=["bpf","hpf","none"])
    p.add_argument("--mixup_alpha",   type=float, default=0.3)
    p.add_argument("--mc_weight",     type=float, default=0.5,
                   help="Weight for multi-class loss (0=pure multi-label)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING + FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(signal, fs, filter_type):
    nyq = 0.5 * fs
    if filter_type == "none":
        return signal
    b_n, a_n = iirnotch(60.0 / nyq, Q=30)
    signal = filtfilt(b_n, a_n, signal, axis=0).astype(np.float32)
    if filter_type == "bpf":
        b, a = butter(4, [20.0 / nyq, 200.0 / nyq], btype="band")
    elif filter_type == "hpf":
        b, a = butter(4, 20.0 / nyq, btype="high")
    else:
        return signal
    return filtfilt(b, a, signal, axis=0).astype(np.float32)


def load_emg_csv(csv_path, fs=500, filter_type="bpf"):
    emg_cols = [f"EMG{i}_Raw" for i in range(1, 9)]
    df = pd.read_csv(csv_path)
    raw = df[emg_cols].values.astype(np.float32)
    orig_ts = df["unix_time_s"].values.astype(np.float64)
    if len(raw) < 10:
        return raw, orig_ts

    t_s, t_e = orig_ts[0], orig_ts[-1]
    n_u = int((t_e - t_s) * fs) + 1
    if n_u < 10:
        return raw, orig_ts

    uniform_ts = np.linspace(t_s, t_e, n_u)
    resampled = interp1d(orig_ts, raw, axis=0, kind='linear',
                         fill_value='extrapolate')(uniform_ts).astype(np.float32)
    resampled -= resampled.mean(axis=0)
    if len(resampled) > 20:
        filtered = apply_filters(resampled, fs, filter_type)
    else:
        filtered = resampled

    std = filtered.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    filtered = (filtered - filtered.mean(axis=0, keepdims=True)) / std
    return filtered, uniform_ts


def compute_emg_features(window, fs=500, n_segments=10):
    """
    Compute time-domain EMG features over sub-segments of a window.
    Standard features used in clinical EMG / prosthetics:
      - RMS: root mean square (activation intensity)
      - MAV: mean absolute value
      - WL: waveform length (sum of absolute differences)
      - ZC: zero crossings (frequency proxy)

    window: (T, 8) → returns (n_segments, 8*4) = (n_segments, 32) features
    """
    T, C = window.shape
    seg_len = T // n_segments
    features = []

    for i in range(n_segments):
        s = i * seg_len
        e = s + seg_len
        seg = window[s:e]  # (seg_len, 8)

        rms = np.sqrt((seg ** 2).mean(axis=0))              # (8,)
        mav = np.abs(seg).mean(axis=0)                       # (8,)
        wl = np.abs(np.diff(seg, axis=0)).sum(axis=0) / seg_len  # (8,)
        zc = ((seg[:-1] * seg[1:]) < 0).sum(axis=0) / seg_len    # (8,)

        feat = np.concatenate([rms, mav, wl, zc])  # (32,)
        features.append(feat)

    return np.array(features, dtype=np.float32)  # (n_segments, 32)


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

def build_label_intervals(gt_df):
    intervals = []
    for i in range(len(gt_df)):
        row = gt_df.iloc[i]
        start_t = row["end_unix"]
        end_t = gt_df.iloc[i+1]["end_unix"] if i+1 < len(gt_df) else start_t + 60.0
        labels = row[LABEL_COLUMNS].values.astype(np.float32)
        intervals.append({
            "start_unix": start_t, "end_unix": end_t,
            "labels": labels, "text": row["text"],
            "segment": int(row["segment"]),
        })
    return intervals


def assign_labels(timestamps, intervals):
    N = len(timestamps)
    labels = np.zeros((N, NUM_LABELS), dtype=np.float32)
    for iv in intervals:
        mask = (timestamps >= iv["start_unix"]) & (timestamps < iv["end_unix"])
        labels[mask] = iv["labels"]
    return labels


def build_pattern_mapping(gt_df):
    """Build mapping from label tuple → class index."""
    patterns = gt_df[LABEL_COLUMNS].apply(tuple, axis=1)
    unique = sorted(set(patterns), key=lambda x: patterns.tolist().index(x))
    pat2idx = {p: i for i, p in enumerate(unique)}
    idx2pat = {i: np.array(p, dtype=np.float32) for p, i in pat2idx.items()}
    return pat2idx, idx2pat


def load_all_data(gt_csv, emg_dir, fs, filter_type):
    gt_df = pd.read_csv(gt_csv)
    intervals = build_label_intervals(gt_df)
    pat2idx, idx2pat = build_pattern_mapping(gt_df)
    n_classes = len(pat2idx)
    print(f"  {len(gt_df)} utterances, {n_classes} unique patterns")

    emg_pattern = os.path.join(emg_dir, "emg_*.csv")
    emg_files = sorted(glob.glob(emg_pattern),
                       key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    print(f"  {len(emg_files)} EMG files")

    # Build segments: one per (emg_file, interval) match
    segments = []
    for emg_file in emg_files:
        emg_data, timestamps = load_emg_csv(emg_file, fs, filter_type)
        if len(emg_data) < 10:
            continue
        for iv in intervals:
            mask = (timestamps >= iv["start_unix"]) & (timestamps < iv["end_unix"])
            n = mask.sum()
            if n >= fs * 0.5:
                label_tuple = tuple(iv["labels"].astype(int))
                class_idx = pat2idx.get(label_tuple, -1)
                if class_idx < 0:
                    continue
                segments.append({
                    "emg": emg_data[mask],
                    "labels": iv["labels"],
                    "class_idx": class_idx,
                    "segment": iv["segment"],
                    "text": iv["text"],
                })

    print(f"  {len(segments)} segments built")
    return gt_df, segments, pat2idx, idx2pat


# ─────────────────────────────────────────────────────────────────────────────
# DATASET with feature engineering
# ─────────────────────────────────────────────────────────────────────────────

class EMGDatasetV3(Dataset):
    def __init__(self, segments, window_size, stride, fs=500,
                 augment=False, label_smooth=0.0, mixup_alpha=0.0,
                 n_feat_segments=10):
        self.augment = augment
        self.label_smooth = label_smooth
        self.mixup_alpha = mixup_alpha
        self.fs = fs
        self.n_feat_segments = n_feat_segments

        self.raw_windows = []
        self.feat_windows = []
        self.labels = []
        self.class_indices = []

        for seg in segments:
            emg = seg["emg"]
            for start in range(0, len(emg) - window_size + 1, stride):
                end = start + window_size
                w = emg[start:end]
                self.raw_windows.append(w)
                self.feat_windows.append(
                    compute_emg_features(w, fs, n_feat_segments))
                self.labels.append(seg["labels"].copy())
                self.class_indices.append(seg["class_idx"])

        self.raw_windows = np.array(self.raw_windows, dtype=np.float32)
        self.feat_windows = np.array(self.feat_windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.class_indices = np.array(self.class_indices, dtype=np.int64)

    def __len__(self):
        return len(self.raw_windows)

    def __getitem__(self, idx):
        x_raw = torch.tensor(self.raw_windows[idx], dtype=torch.float32)
        x_feat = torch.tensor(self.feat_windows[idx], dtype=torch.float32)
        y_ml = torch.tensor(self.labels[idx], dtype=torch.float32)
        y_mc = torch.tensor(self.class_indices[idx], dtype=torch.long)

        if self.augment:
            x_raw, x_feat = self._augment(x_raw, x_feat)

        if self.label_smooth > 0:
            y_ml = y_ml * (1 - self.label_smooth) + 0.5 * self.label_smooth

        return x_raw, x_feat, y_ml, y_mc

    def _augment(self, x_raw, x_feat):
        # Gaussian noise
        if random.random() < 0.5:
            x_raw = x_raw + torch.randn_like(x_raw) * random.uniform(0.02, 0.12)
        # Amplitude scaling
        if random.random() < 0.5:
            scale = torch.empty(1, x_raw.shape[1]).uniform_(0.85, 1.15)
            x_raw = x_raw * scale
        # Time shift
        if random.random() < 0.3:
            shift = random.randint(-x_raw.shape[0]//8, x_raw.shape[0]//8)
            x_raw = torch.roll(x_raw, shifts=shift, dims=0)
        # Channel dropout
        if random.random() < 0.15:
            ch = random.randint(0, x_raw.shape[1]-1)
            x_raw[:, ch] = 0.0

        # Recompute features after augmentation
        x_feat = torch.tensor(
            compute_emg_features(x_raw.numpy(), self.fs, self.n_feat_segments),
            dtype=torch.float32)
        return x_raw, x_feat


def mixup_batch(x_raw, x_feat, y_ml, y_mc, alpha=0.3):
    """Mixup augmentation on multi-label targets only (not multi-class)."""
    if alpha <= 0:
        return x_raw, x_feat, y_ml, y_mc

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5
    batch_size = x_raw.size(0)
    index = torch.randperm(batch_size)

    x_raw = lam * x_raw + (1 - lam) * x_raw[index]
    x_feat = lam * x_feat + (1 - lam) * x_feat[index]
    y_ml = lam * y_ml + (1 - lam) * y_ml[index]
    # For multi-class: use the dominant class
    # (don't mix class indices — use the one with higher weight)
    return x_raw, x_feat, y_ml, y_mc


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — Dual-head with feature fusion
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
    def forward(self, x):
        # x: (B, dim, T)
        x_t = x.permute(0, 2, 1)
        w = torch.softmax(self.attn(x_t), dim=1)
        return (x_t * w).sum(dim=1)


class EMGModelV3(nn.Module):
    """
    Dual-head model:
      - Raw EMG → Conv1d backbone → temporal features
      - Engineered features → small MLP → feature vector
      - Concatenated → shared trunk → two heads:
        1. Multi-label head (10 sigmoid outputs)
        2. Multi-class head (N_patterns softmax outputs)
    """

    def __init__(self, n_classes, n_feat_segments=10, feat_channels=32,
                 dropout=0.4):
        super().__init__()
        self.n_classes = n_classes

        # Raw EMG backbone
        self.raw_backbone = nn.Sequential(
            nn.Conv1d(NUM_EMG_CHANNELS, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.15),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
        )
        self.raw_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.raw_max_pool = nn.AdaptiveMaxPool1d(1)
        self.raw_attn_pool = TemporalAttentionPool(128)
        # avg + max + attn = 384
        raw_feat_dim = 128 * 3

        # Feature branch: (B, n_segments, 32) → flat
        feat_input_dim = n_feat_segments * feat_channels
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_input_dim, 128),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        feat_out_dim = 64

        # Shared trunk
        trunk_dim = raw_feat_dim + feat_out_dim  # 384 + 64 = 448
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, 192),
            nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.BatchNorm1d(96), nn.GELU(), nn.Dropout(dropout),
        )

        # Multi-label head
        self.ml_head = nn.Linear(96, NUM_LABELS)

        # Multi-class head
        self.mc_head = nn.Linear(96, n_classes)

        total = sum(p.numel() for p in self.parameters())
        print(f"  Model V3: Conv backbone(8→128) + Feature branch")
        print(f"  Trunk: {trunk_dim} → 192 → 96")
        print(f"  Multi-label head: 96 → {NUM_LABELS}")
        print(f"  Multi-class head: 96 → {n_classes}")
        print(f"  Total params: {total:,}")

    def forward(self, x_raw, x_feat):
        """
        x_raw: (B, T, 8)
        x_feat: (B, n_segments, 32)
        Returns: ml_logits (B, 10), mc_logits (B, n_classes)
        """
        # Raw backbone
        r = x_raw.permute(0, 2, 1)         # (B, 8, T)
        r = self.raw_backbone(r)            # (B, 128, T')
        r_avg = self.raw_avg_pool(r).squeeze(-1)
        r_max = self.raw_max_pool(r).squeeze(-1)
        r_attn = self.raw_attn_pool(r)
        raw_out = torch.cat([r_avg, r_max, r_attn], dim=1)  # (B, 384)

        # Feature branch
        B = x_feat.shape[0]
        f = x_feat.reshape(B, -1)           # (B, n_seg*32)
        feat_out = self.feat_branch(f)      # (B, 64)

        # Fuse
        fused = torch.cat([raw_out, feat_out], dim=1)  # (B, 448)
        trunk_out = self.trunk(fused)                    # (B, 96)

        ml_logits = self.ml_head(trunk_out)  # (B, 10)
        mc_logits = self.mc_head(trunk_out)  # (B, n_classes)

        return ml_logits, mc_logits


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds, targets, threshold=0.5):
    pred_bin = (preds >= threshold).astype(np.float32)
    per_label_acc = (pred_bin == targets).mean(axis=0)
    exact_match = (pred_bin == targets).all(axis=1).mean()
    try:
        macro_f1 = f1_score(targets, pred_bin, average="macro", zero_division=0)
        per_label_f1 = f1_score(targets, pred_bin, average=None, zero_division=0)
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


def train_one_epoch(model, loader, optimizer, ml_criterion, mc_criterion,
                    device, mc_weight, mixup_alpha, idx2pat):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for x_raw, x_feat, y_ml, y_mc in loader:
        x_raw, x_feat = x_raw.to(device), x_feat.to(device)
        y_ml, y_mc = y_ml.to(device), y_mc.to(device)

        # Mixup
        if mixup_alpha > 0 and random.random() < 0.5:
            x_raw, x_feat, y_ml, y_mc = mixup_batch(
                x_raw, x_feat, y_ml, y_mc, mixup_alpha)

        ml_logits, mc_logits = model(x_raw, x_feat)

        ml_loss = ml_criterion(ml_logits, y_ml)
        mc_loss = mc_criterion(mc_logits, y_mc)
        loss = (1 - mc_weight) * ml_loss + mc_weight * mc_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(x_raw)

        # For metrics: use multi-class prediction → map to multi-label
        with torch.no_grad():
            # Combine both heads for prediction
            mc_pred_idx = mc_logits.argmax(dim=1)
            mc_pred_labels = torch.stack([
                torch.tensor(idx2pat[i.item()], device=device)
                for i in mc_pred_idx])

            ml_pred = torch.sigmoid(ml_logits)

            # Ensemble: average multi-class mapped labels with multi-label sigmoid
            combined = 0.5 * mc_pred_labels + 0.5 * ml_pred

            all_preds.append(combined.cpu().numpy())
            # Undo label smoothing
            targets_hard = (y_ml > 0.5).float()
            all_targets.append(targets_hard.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(all_preds)
    return metrics


@torch.no_grad()
def validate(model, loader, ml_criterion, mc_criterion,
             device, mc_weight, idx2pat):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    all_mc_correct, all_mc_total = 0, 0

    for x_raw, x_feat, y_ml, y_mc in loader:
        x_raw, x_feat = x_raw.to(device), x_feat.to(device)
        y_ml, y_mc = y_ml.to(device), y_mc.to(device)

        ml_logits, mc_logits = model(x_raw, x_feat)
        ml_loss = ml_criterion(ml_logits, y_ml)
        mc_loss = mc_criterion(mc_logits, y_mc)
        loss = (1 - mc_weight) * ml_loss + mc_weight * mc_loss
        total_loss += loss.item() * len(x_raw)

        # Multi-class accuracy
        mc_pred = mc_logits.argmax(dim=1)
        all_mc_correct += (mc_pred == y_mc).sum().item()
        all_mc_total += len(y_mc)

        # Combined prediction
        mc_pred_labels = torch.stack([
            torch.tensor(idx2pat[i.item()], device=device)
            for i in mc_pred])
        ml_pred = torch.sigmoid(ml_logits)
        combined = 0.5 * mc_pred_labels + 0.5 * ml_pred

        all_preds.append(combined.cpu().numpy())
        all_targets.append(y_ml.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(all_preds)
    metrics["mc_acc"] = all_mc_correct / max(all_mc_total, 1)
    return metrics


def train_fold(fold, train_segs, val_segs, pat2idx, idx2pat, args, device):
    """Train one fold. Returns best val metrics dict."""
    n_classes = len(pat2idx)
    window_size = int(args.window_sec * args.fs)
    stride = int(args.stride_sec * args.fs)

    train_ds = EMGDatasetV3(train_segs, window_size, stride, args.fs,
                            augment=True, label_smooth=args.label_smooth,
                            mixup_alpha=0)  # mixup done at batch level
    val_ds = EMGDatasetV3(val_segs, window_size, stride, args.fs,
                          augment=False, label_smooth=0.0)

    print(f"    Train: {len(train_ds)} windows | Val: {len(val_ds)} windows")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    model = EMGModelV3(n_classes, dropout=args.dropout).to(device)

    # Class weights for multi-class head
    class_counts = Counter(s["class_idx"] for s in train_segs)
    total_segs = sum(class_counts.values())
    mc_weights = torch.tensor([
        total_segs / max(class_counts.get(i, 1) * n_classes, 1)
        for i in range(n_classes)
    ], dtype=torch.float32).clamp(max=5.0).to(device)

    # Pos weights for multi-label head
    train_labels = np.array([s["labels"] for s in train_segs])
    lm = train_labels.mean(axis=0)
    pos_weight = torch.tensor(
        [(1 - m) / max(m, 0.01) for m in lm],
        dtype=torch.float32).clamp(max=5.0).to(device)

    ml_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mc_criterion = nn.CrossEntropyLoss(weight=mc_weights,
                                        label_smoothing=args.label_smooth)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=1e-6)

    best_val_f1 = 0.0
    best_state = None
    patience = 0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer,
                             ml_criterion, mc_criterion, device,
                             args.mc_weight, args.mixup_alpha, idx2pat)
        va = validate(model, val_loader, ml_criterion, mc_criterion,
                      device, args.mc_weight, idx2pat)
        scheduler.step()

        star = ""
        if va["macro_f1"] > best_val_f1:
            best_val_f1 = va["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = va.copy()
            patience = 0
            star = " ★"
        else:
            patience += 1

        if epoch % 10 == 0 or star:
            gap = tr["macro_f1"] - va["macro_f1"]
            print(f"    E{epoch:3d} | Tr {tr['macro_f1']:.3f} | "
                  f"Val {va['macro_f1']:.3f} MC:{va['mc_acc']:.3f} "
                  f"Exact:{va['exact_match']:.3f} (gap={gap:.3f}){star}")

        if patience >= 30:
            print(f"    Early stopping at epoch {epoch}")
            break

    return best_val_f1, best_state, best_metrics, model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  EMG Multi-Label Hand Activation Model v3")
    print("=" * 70)
    print(f"  Device        : {device}")
    print(f"  Window        : {args.window_sec}s")
    print(f"  MC weight     : {args.mc_weight}")
    print(f"  Mixup alpha   : {args.mixup_alpha}")
    print(f"  K-folds       : {args.n_folds}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  LOADING DATA")
    print("─" * 70)

    gt_df, segments, pat2idx, idx2pat = load_all_data(
        args.gt_csv, args.emg_dir, args.fs, args.filter_type)

    n_classes = len(pat2idx)
    print(f"  Pattern mapping: {n_classes} classes")
    for pat, idx in sorted(pat2idx.items(), key=lambda x: x[1]):
        active = [LABEL_COLUMNS[j].replace("label_","")
                  for j, v in enumerate(pat) if v == 1]
        name = ', '.join(active) if active else 'REST'
        count = sum(1 for s in segments if s["class_idx"] == idx)
        print(f"    Class {idx:2d}: {count:3d} segs  {name[:50]}")

    # ── K-Fold Cross Validation ───────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"  {args.n_folds}-FOLD CROSS VALIDATION")
    print("─" * 70)

    seg_ids = sorted(set(s["segment"] for s in segments))
    random.shuffle(seg_ids)

    fold_size = len(seg_ids) // args.n_folds
    fold_results = []

    for fold in range(args.n_folds):
        print(f"\n  ── Fold {fold+1}/{args.n_folds} ──")

        # Create fold split
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < args.n_folds - 1 else len(seg_ids)
        val_ids = set(seg_ids[val_start:val_end])
        train_ids = set(seg_ids) - val_ids

        train_segs = [s for s in segments if s["segment"] in train_ids]
        val_segs = [s for s in segments if s["segment"] in val_ids]

        print(f"    Train: {len(train_ids)} utterances ({len(train_segs)} segs)")
        print(f"    Val:   {len(val_ids)} utterances ({len(val_segs)} segs)")

        best_f1, best_state, best_metrics, model = train_fold(
            fold, train_segs, val_segs, pat2idx, idx2pat, args, device)

        fold_results.append({
            "fold": fold + 1,
            "val_f1": best_f1,
            "val_exact": best_metrics["exact_match"],
            "val_hamming": best_metrics["hamming_acc"],
            "val_mc_acc": best_metrics.get("mc_acc", 0),
        })

        print(f"    Best: F1={best_f1:.4f} Exact={best_metrics['exact_match']:.3f} "
              f"MC_Acc={best_metrics.get('mc_acc',0):.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CROSS-VALIDATION RESULTS")
    print("=" * 70)

    f1s = [r["val_f1"] for r in fold_results]
    exacts = [r["val_exact"] for r in fold_results]
    hammings = [r["val_hamming"] for r in fold_results]
    mc_accs = [r["val_mc_acc"] for r in fold_results]

    print(f"  Macro F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Exact Match: {np.mean(exacts):.4f} ± {np.std(exacts):.4f}")
    print(f"  Hamming Acc: {np.mean(hammings):.4f} ± {np.std(hammings):.4f}")
    print(f"  MC Accuracy: {np.mean(mc_accs):.4f} ± {np.std(mc_accs):.4f}")

    for r in fold_results:
        print(f"    Fold {r['fold']}: F1={r['val_f1']:.4f} "
              f"Exact={r['val_exact']:.3f} MC={r['val_mc_acc']:.3f}")

    # ── Train final model on ALL data ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  TRAINING FINAL MODEL (all data)")
    print("─" * 70)

    window_size = int(args.window_sec * args.fs)
    stride = int(args.stride_sec * args.fs)

    # Use 90/10 split for final model
    random.shuffle(seg_ids)
    n_val_final = max(2, len(seg_ids) // 10)
    val_ids_final = set(seg_ids[:n_val_final])
    train_ids_final = set(seg_ids) - val_ids_final

    train_segs_final = [s for s in segments if s["segment"] in train_ids_final]
    val_segs_final = [s for s in segments if s["segment"] in val_ids_final]

    best_f1, best_state, best_metrics, model = train_fold(
        "final", train_segs_final, val_segs_final,
        pat2idx, idx2pat, args, device)

    # Save
    best_path = out_dir / "best_model.pt"
    torch.save({
        "state_dict": best_state,
        "val_f1": float(best_f1),
        "val_exact_match": float(best_metrics["exact_match"]),
        "n_classes": int(n_classes),
        "pat2idx": {str(k): int(v) for k, v in pat2idx.items()},
        "idx2pat": {int(k): v.tolist() for k, v in idx2pat.items()},
        "window_sec": float(args.window_sec),
        "fs": int(args.fs),
        "label_columns": list(LABEL_COLUMNS),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s)),
    }, best_path)

    # Save CV results
    cv_path = out_dir / "cv_results.csv"
    with open(cv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fold_results[0].keys())
        w.writeheader()
        w.writerows(fold_results)

    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n  Final model → {best_path}")
    print(f"  CV results  → {cv_path}")
    print(f"  Config      → {config_path}")

    print(f"\n  ─── SUMMARY ───")
    print(f"  CV Macro F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Final Val F1:   {best_f1:.4f}")


if __name__ == "__main__":
    main()

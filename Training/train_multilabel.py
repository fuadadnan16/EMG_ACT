"""
train_multilabel.py — EMG Multi-Label Hand Activation Model
=============================================================
A single model that predicts which of 10 hand zones are active
from 8-channel EMG signals at 500 Hz.

Hand zones (matching the hand drawing):
  0: palm          1: thumb
  2: index_tip     3: index_seg
  4: middle_tip    5: middle_seg
  6: ring_tip      7: ring_seg
  8: pinky_tip     9: pinky_seg

Pipeline:
  1. Load ground_truth_labeled.csv → get per-utterance labels + unix timestamps
  2. Load emg_*.csv files → match each EMG file to utterance(s) via unix timestamps
  3. For each utterance window, the label persists from end_unix of that utterance
     to end_unix of the next utterance (the command holds until the next one)
  4. Build windowed dataset (multi-label: 10 binary targets)
  5. Use emg2pose pretrained Conv1d backbone (first 2 layers, frozen)
     with Linear(8→16) channel expansion
  6. Train multi-label classifier with BCEWithLogitsLoss

Usage:
  python train_multilabel.py \
    --gt_csv ground_truth_labeled.csv \
    --emg_dir . \
    --ckpt_path emg2pose_model_checkpoints/tracking_vemg2pose.ckpt \
    --emg2pose_repo emg2pose/
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
from scipy.signal import butter, filtfilt, iirnotch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score


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
TARGET_EMG_CHANNELS = 16  # emg2pose expects 16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_csv",        default="ground_truth_labeled.csv",
                   help="Path to ground truth labeled CSV")
    p.add_argument("--emg_dir",       default=".",
                   help="Directory containing emg_*.csv files")
    p.add_argument("--ckpt_path",
                   default="emg2pose_model_checkpoints/tracking_vemg2pose.ckpt")
    p.add_argument("--emg2pose_repo", default="emg2pose/")
    p.add_argument("--out_dir",       default="checkpoints_multilabel/")
    p.add_argument("--window_sec",    type=float, default=2.0)
    p.add_argument("--stride_sec",    type=float, default=0.5)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--fs",            type=int,   default=500)
    p.add_argument("--train_frac",    type=float, default=0.80)
    # Filter pipeline
    p.add_argument("--filter_type",   default="bpf",
                   choices=["bpf", "hpf", "none"],
                   help="Filter type: bpf (20-200Hz), hpf (20Hz), or none")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(signal: np.ndarray, fs: int, filter_type: str) -> np.ndarray:
    """Apply filtering to EMG signals. signal: (N, 8)"""
    nyq = 0.5 * fs
    if filter_type == "none":
        return signal

    # 60 Hz notch filter (power line)
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
    """
    Load an EMG CSV file and resample to uniform `fs` Hz.

    The raw EMG data has irregular timestamps (burst sampling),
    so we interpolate to a uniform grid at the target sample rate.

    Returns:
        emg: (N, 8) filtered EMG signals at uniform fs
        timestamps: (N,) uniform unix timestamps
    """
    from scipy.interpolate import interp1d

    emg_cols = [f"EMG{i}_Raw" for i in range(1, 9)]
    df = pd.read_csv(csv_path)
    raw = df[emg_cols].values.astype(np.float32)
    orig_ts = df["unix_time_s"].values.astype(np.float64)

    if len(raw) < 10:
        return raw, orig_ts

    # Resample to uniform grid at target fs
    t_start, t_end = orig_ts[0], orig_ts[-1]
    n_uniform = int((t_end - t_start) * fs) + 1
    if n_uniform < 10:
        return raw, orig_ts

    uniform_ts = np.linspace(t_start, t_end, n_uniform)
    interp_func = interp1d(orig_ts, raw, axis=0, kind='linear',
                           fill_value='extrapolate')
    resampled = interp_func(uniform_ts).astype(np.float32)

    # Remove DC offset per channel
    resampled -= resampled.mean(axis=0)

    # Apply filtering
    if len(resampled) > 20:
        filtered = apply_filters(resampled, fs, filter_type)
    else:
        filtered = resampled

    # Z-score normalize per channel
    std = filtered.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    filtered = (filtered - filtered.mean(axis=0, keepdims=True)) / std

    return filtered, uniform_ts


# ─────────────────────────────────────────────────────────────────────────────
# DATA ALIGNMENT: Match EMG files to ground truth utterances
# ─────────────────────────────────────────────────────────────────────────────

def build_label_intervals(gt_df: pd.DataFrame) -> list:
    """
    Build label intervals from ground truth.

    Each utterance's label persists from its end_unix until the next
    utterance's end_unix. The command is active during the EMG recording
    that follows the utterance.

    Returns list of dicts:
        {start_unix, end_unix, labels: np.array(10,)}
    """
    intervals = []
    n = len(gt_df)

    for i in range(n):
        row = gt_df.iloc[i]
        # Label starts at the end of the utterance (command given)
        start_t = row["end_unix"]

        # Label ends at the end of the next utterance (new command)
        if i + 1 < n:
            end_t = gt_df.iloc[i + 1]["end_unix"]
        else:
            # Last utterance: extend by a generous margin (60s)
            end_t = start_t + 60.0

        labels = row[LABEL_COLUMNS].values.astype(np.float32)
        intervals.append({
            "start_unix": start_t,
            "end_unix": end_t,
            "labels": labels,
            "text": row["text"],
            "segment": row["segment"],
        })

    return intervals


def assign_labels_to_emg(emg_timestamps: np.ndarray,
                         intervals: list) -> np.ndarray:
    """
    For each EMG sample, find which label interval it belongs to.
    Returns: (N, 10) array of labels for each sample.
    If no interval matches, label is all zeros (rest state).
    """
    N = len(emg_timestamps)
    labels = np.zeros((N, NUM_LABELS), dtype=np.float32)

    for interval in intervals:
        mask = ((emg_timestamps >= interval["start_unix"]) &
                (emg_timestamps < interval["end_unix"]))
        labels[mask] = interval["labels"]

    return labels


def load_and_align_all_data(gt_csv: str, emg_dir: str,
                            fs: int, filter_type: str) -> tuple:
    """
    Load ground truth, load all EMG files, align them.

    Returns:
        all_emg: (total_samples, 8)
        all_labels: (total_samples, 10)
        file_boundaries: list of (start_idx, end_idx, filename) for splitting
    """
    # Load ground truth
    gt_df = pd.read_csv(gt_csv)
    print(f"\n  Ground truth: {len(gt_df)} utterances")
    print(f"  Label columns: {LABEL_COLUMNS}")
    print(f"  Time range: {gt_df['end_unix'].min():.1f} to {gt_df['end_unix'].max():.1f}")

    # Build label intervals
    intervals = build_label_intervals(gt_df)
    print(f"  Built {len(intervals)} label intervals")

    # Find and sort all EMG files
    emg_pattern = os.path.join(emg_dir, "emg_*.csv")
    emg_files = sorted(glob.glob(emg_pattern),
                       key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    if not emg_files:
        raise FileNotFoundError(f"No emg_*.csv files found in {emg_dir}")

    print(f"\n  Found {len(emg_files)} EMG files")

    all_emg_list = []
    all_labels_list = []
    file_boundaries = []
    total_samples = 0
    matched_files = 0
    unmatched_files = 0

    for emg_file in emg_files:
        fname = os.path.basename(emg_file)
        emg_data, timestamps = load_emg_csv(emg_file, fs, filter_type)

        if len(emg_data) < 10:
            print(f"    SKIP {fname}: too few samples ({len(emg_data)})")
            continue

        # Assign labels based on timestamp matching
        labels = assign_labels_to_emg(timestamps, intervals)

        # Check if any labels were assigned
        label_sum = labels.sum()
        if label_sum == 0:
            # This EMG file doesn't overlap with any utterance interval
            unmatched_files += 1
            print(f"    WARN {fname}: no matching labels "
                  f"(time: {timestamps[0]:.1f}-{timestamps[-1]:.1f})")
            # Still include as rest/background data (all zeros label)

        start_idx = total_samples
        all_emg_list.append(emg_data)
        all_labels_list.append(labels)
        total_samples += len(emg_data)
        file_boundaries.append((start_idx, total_samples, fname))
        matched_files += 1

    all_emg = np.concatenate(all_emg_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)

    print(f"\n  Total samples: {total_samples} ({total_samples/fs:.1f} sec)")
    print(f"  Matched files: {matched_files}, Unmatched: {unmatched_files}")

    # Print label distribution
    print("\n  Label distribution (fraction of active samples):")
    for i, col in enumerate(LABEL_COLUMNS):
        frac = all_labels[:, i].mean()
        print(f"    {col:20s}: {frac:.3f} ({int(all_labels[:, i].sum())} samples)")

    return all_emg, all_labels, file_boundaries


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class EMGMultiLabelDataset(Dataset):
    """
    Windowed EMG dataset with multi-label targets.
    Each window has a single label vector = majority vote within that window.
    """
    def __init__(self, emg: np.ndarray, labels: np.ndarray,
                 window_size: int, stride: int):
        self.windows = []
        self.targets = []

        for start in range(0, len(emg) - window_size + 1, stride):
            end = start + window_size
            window_labels = labels[start:end]

            # Use majority vote per label column for the window
            target = (window_labels.mean(axis=0) >= 0.5).astype(np.float32)

            self.windows.append(emg[start:end])
            self.targets.append(target)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (torch.tensor(self.windows[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32))


def prepare_loaders(all_emg, all_labels, window_size, stride,
                    train_frac, batch_size):
    """Time-based split."""
    total = len(all_emg)
    train_n = int(total * train_frac)

    emg_tr = all_emg[:train_n]
    lab_tr = all_labels[:train_n]
    emg_va = all_emg[train_n:]
    lab_va = all_labels[train_n:]

    train_ds = EMGMultiLabelDataset(emg_tr, lab_tr, window_size, stride)
    val_ds   = EMGMultiLabelDataset(emg_va, lab_va, window_size, window_size)

    print(f"  Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_emg2pose_model(ckpt_path: str, repo_root: str):
    """Load pretrained emg2pose model."""
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
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class EMGMultiLabelModel(nn.Module):
    """
    Multi-label hand activation model using emg2pose pretrained backbone.

    Architecture:
      Input: (B, T, 8)           — 8-channel EMG
      → Linear(8, 16)            — learned channel expansion to match emg2pose
      → permute → (B, 16, T)
      → Conv1d layer 0 (pretrained, frozen, stride=5)
      → ReLU
      → Conv1d layer 1 (pretrained, frozen, stride=2)
      → ReLU
      Total stride = 10x → at 500Hz: T/10 temporal features
      → AdaptiveAvgPool1d(1) → (B, feat_dim)
      → MLP → (B, 10)           — 10 sigmoid outputs (one per hand zone)
    """

    def __init__(self, pretrained_model, num_labels=NUM_LABELS,
                 hidden_dim=64, dropout=0.3):
        super().__init__()

        # Channel expansion: 8 → 16
        self.expand = nn.Linear(NUM_EMG_CHANNELS, TARGET_EMG_CHANNELS)

        # Extract first 2 Conv1d layers from pretrained model
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
            conv_layers[0],
            nn.ReLU(),
            conv_layers[1],
            nn.ReLU(),
        )

        # Freeze pretrained layers
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        feat_dim = conv_layers[1].out_channels
        print(f"  Backbone: 2 pretrained Conv1d layers "
              f"(frozen, stride=10x, feat_dim={feat_dim})")

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

        print(f"  Classifier: {feat_dim} → {hidden_dim} → {hidden_dim} → {num_labels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 8) raw EMG windows
        Returns: (B, 10) logits for each hand zone
        """
        x = self.expand(x)              # (B, T, 16)
        x = x.permute(0, 2, 1)          # (B, 16, T)
        x = self.feature_extractor(x)   # (B, feat_dim, T/10)
        x = self.pool(x).squeeze(-1)    # (B, feat_dim)
        return self.classifier(x)        # (B, 10) logits


class EMGMultiLabelModelStandalone(nn.Module):
    """
    Standalone model (no pretrained backbone) for cases where
    emg2pose checkpoint is not available.
    """

    def __init__(self, num_labels=NUM_LABELS, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # (B, 8, T)
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

        print(f"  Standalone model: Conv1d(8→32→64→128) → {hidden_dim} → {num_labels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)          # (B, 8, T)
        x = self.feature_extractor(x)   # (B, 128, T')
        x = self.pool(x).squeeze(-1)    # (B, 128)
        return self.classifier(x)        # (B, 10)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray, threshold=0.5):
    """Compute per-label and macro metrics."""
    pred_binary = (preds >= threshold).astype(np.float32)

    # Per-label accuracy
    per_label_acc = (pred_binary == targets).mean(axis=0)

    # Overall accuracy (exact match)
    exact_match = (pred_binary == targets).all(axis=1).mean()

    # Macro F1
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
        "hamming_acc": per_label_acc.mean(),  # average accuracy across all labels
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
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  EMG Multi-Label Hand Activation Model")
    print("=" * 70)
    print(f"  Device      : {device}")
    print(f"  Window      : {args.window_sec}s ({int(args.window_sec * args.fs)} samples)")
    print(f"  Stride      : {args.stride_sec}s")
    print(f"  Filter      : {args.filter_type}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Labels      : {NUM_LABELS} hand zones")

    # ── Load & align data ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  LOADING & ALIGNING DATA")
    print("─" * 70)

    all_emg, all_labels, file_boundaries = load_and_align_all_data(
        gt_csv=args.gt_csv,
        emg_dir=args.emg_dir,
        fs=args.fs,
        filter_type=args.filter_type,
    )

    # ── Create data loaders ───────────────────────────────────────────────────
    window_size = int(args.window_sec * args.fs)
    stride = int(args.stride_sec * args.fs)

    print("\n" + "─" * 70)
    print("  CREATING DATA LOADERS")
    print("─" * 70)

    train_loader, val_loader = prepare_loaders(
        all_emg, all_labels, window_size, stride,
        args.train_frac, args.batch_size,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  BUILDING MODEL")
    print("─" * 70)

    use_pretrained = True
    ckpt_path = pathlib.Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"  WARNING: Checkpoint not found: {ckpt_path}")
        print(f"  Falling back to standalone model (no pretrained backbone)")
        use_pretrained = False

    if use_pretrained:
        _patch_emg2pose(args.emg2pose_repo)
        pretrained = load_emg2pose_model(str(ckpt_path), args.emg2pose_repo)
        model = EMGMultiLabelModel(pretrained).to(device)
    else:
        model = EMGMultiLabelModelStandalone().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}  |  Trainable: {trainable_params:,}")

    # ── Compute class weights for imbalanced labels ───────────────────────────
    # Use pos_weight for BCEWithLogitsLoss to handle imbalance
    label_sums = all_labels.sum(axis=0)
    total_samples = len(all_labels)
    pos_weight = torch.tensor(
        [(total_samples - s) / max(s, 1) for s in label_sums],
        dtype=torch.float32
    ).to(device)
    # Clip extreme weights
    pos_weight = pos_weight.clamp(max=10.0)
    print(f"  Pos weights: {pos_weight.cpu().numpy().round(2)}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=7, factor=0.5, min_lr=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  TRAINING")
    print("─" * 70)

    best_val_f1 = 0.0
    best_path = out_dir / "best_model.pt"
    history = []
    patience_counter = 0
    early_stop_patience = 15

    for epoch in range(1, args.epochs + 1):
        tr_metrics = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device)
        va_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(va_metrics["macro_f1"])

        current_lr = optimizer.param_groups[0]["lr"]
        star = ""
        if va_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = va_metrics["macro_f1"]
            patience_counter = 0
            star = " ★"
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_f1": best_val_f1,
                "val_exact_match": va_metrics["exact_match"],
                "val_hamming_acc": va_metrics["hamming_acc"],
                "window_sec": args.window_sec,
                "fs": args.fs,
                "label_columns": LABEL_COLUMNS,
                "use_pretrained": use_pretrained,
            }, best_path)
        else:
            patience_counter += 1

        history.append({
            "epoch": epoch,
            "tr_loss": round(tr_metrics["loss"], 4),
            "tr_f1": round(tr_metrics["macro_f1"], 4),
            "tr_hamming": round(tr_metrics["hamming_acc"], 4),
            "va_loss": round(va_metrics["loss"], 4),
            "va_f1": round(va_metrics["macro_f1"], 4),
            "va_hamming": round(va_metrics["hamming_acc"], 4),
            "va_exact": round(va_metrics["exact_match"], 4),
            "lr": round(current_lr, 8),
        })

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Loss {tr_metrics['loss']:.4f} F1 {tr_metrics['macro_f1']:.3f} | "
              f"Val Loss {va_metrics['loss']:.4f} F1 {va_metrics['macro_f1']:.3f} "
              f"Exact {va_metrics['exact_match']:.3f}{star}")

        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping — no improvement for {early_stop_patience} epochs.")
            break

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Best validation macro F1: {best_val_f1:.4f}")

    # Load best model and evaluate
    ckpt = torch.load(best_path, map_location=device)
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

    # Save config
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n  Model saved → {best_path}")
    print(f"  History     → {hist_path}")
    print(f"  Config      → {config_path}")


if __name__ == "__main__":
    main()

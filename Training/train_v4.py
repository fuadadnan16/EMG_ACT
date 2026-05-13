"""
train_v4.py — EMG Hand Activation Model v4 (Maximum Accuracy)
==============================================================
Single model, no cross-validation overhead. Every trick that helps
small-data multi-label classification:

  1. FOCAL LOSS — focuses on hard-to-classify samples
  2. SWA (Stochastic Weight Averaging) — smoother generalization
  3. PER-ZONE THRESHOLD OPTIMIZATION — finds optimal threshold per zone
  4. PATTERN SNAPPING — constrains output to 17 valid patterns
  5. TEST-TIME AUGMENTATION (TTA) — averages over augmented inputs
  6. HYPERPARAMETER GRID SEARCH — tries multiple configs, picks best
  7. MIXUP on multi-label targets
  8. Utterance-based split with overlapping val windows

Usage:
  python3 train_v4.py --gt_csv ground_truth_labeled.csv --emg_dir ./set_1

  # Quick run (fewer configs):
  python3 train_v4.py --gt_csv ground_truth_labeled.csv --emg_dir ./set_1 --quick
"""

import pathlib, sys, os, argparse, csv, random, glob, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, iirnotch
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from copy import deepcopy
from itertools import product


# ─────────────────────────────────────────────────────────────────────────────
LABEL_COLUMNS = [
    "label_palm", "label_thumb", "label_index_tip", "label_index_seg",
    "label_middle_tip", "label_middle_seg", "label_ring_tip", "label_ring_seg",
    "label_pinky_tip", "label_pinky_seg",
]
NUM_LABELS = 10
NUM_CH = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_csv", default="ground_truth_labeled.csv")
    p.add_argument("--emg_dir", default=".")
    p.add_argument("--out_dir", default="checkpoints_v4/")
    p.add_argument("--fs", type=int, default=500)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--filter_type", default="bpf")
    p.add_argument("--quick", action="store_true",
                   help="Run fewer hyperparameter configs (faster)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_emg_csv(csv_path, fs=500, filter_type="bpf"):
    emg_cols = [f"EMG{i}_Raw" for i in range(1, 9)]
    df = pd.read_csv(csv_path)
    raw = df[emg_cols].values.astype(np.float32)
    ts = df["unix_time_s"].values.astype(np.float64)
    if len(raw) < 10:
        return raw, ts
    t0, t1 = ts[0], ts[-1]
    n = int((t1 - t0) * fs) + 1
    if n < 10:
        return raw, ts
    ut = np.linspace(t0, t1, n)
    rs = interp1d(ts, raw, axis=0, kind='linear',
                  fill_value='extrapolate')(ut).astype(np.float32)
    rs -= rs.mean(axis=0)
    nyq = 0.5 * fs
    if filter_type != "none" and len(rs) > 20:
        b_n, a_n = iirnotch(60.0 / nyq, Q=30)
        rs = filtfilt(b_n, a_n, rs, axis=0).astype(np.float32)
        if filter_type == "bpf":
            b, a = butter(4, [20.0/nyq, 200.0/nyq], btype="band")
        else:
            b, a = butter(4, 20.0/nyq, btype="high")
        rs = filtfilt(b, a, rs, axis=0).astype(np.float32)
    std = rs.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    rs = (rs - rs.mean(axis=0, keepdims=True)) / std
    return rs, ut


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data(gt_csv, emg_dir, fs, filter_type):
    gt = pd.read_csv(gt_csv)
    intervals = []
    for i in range(len(gt)):
        row = gt.iloc[i]
        s = row["end_unix"]
        e = gt.iloc[i+1]["end_unix"] if i+1 < len(gt) else s + 60.0
        intervals.append({
            "start": s, "end": e,
            "labels": row[LABEL_COLUMNS].values.astype(np.float32),
            "seg": int(row["segment"]), "text": row["text"],
        })

    # Build valid pattern set
    pats = gt[LABEL_COLUMNS].apply(tuple, axis=1)
    valid_patterns = np.array(sorted(set(pats)), dtype=np.float32)

    emg_files = sorted(glob.glob(os.path.join(emg_dir, "emg_*.csv")),
                       key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    segments = []
    for ef in emg_files:
        emg, ts = load_emg_csv(ef, fs, filter_type)
        if len(emg) < 10:
            continue
        for iv in intervals:
            mask = (ts >= iv["start"]) & (ts < iv["end"])
            n = mask.sum()
            if n >= fs * 0.3:
                segments.append({
                    "emg": emg[mask], "labels": iv["labels"],
                    "seg": iv["seg"], "text": iv["text"],
                })

    return gt, segments, valid_patterns


# ─────────────────────────────────────────────────────────────────────────────
# DATASET WITH AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class EMGDataset(Dataset):
    def __init__(self, segments, win, stride, augment=False,
                 label_smooth=0.0, mixup=0.0):
        self.augment = augment
        self.label_smooth = label_smooth
        self.mixup = mixup
        self.windows, self.labels = [], []
        for seg in segments:
            emg = seg["emg"]
            for s in range(0, len(emg) - win + 1, stride):
                self.windows.append(emg[s:s+win])
                self.labels.append(seg["labels"].copy())
        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx])
        y = torch.tensor(self.labels[idx])
        if self.augment:
            x = self._aug(x)
        if self.label_smooth > 0:
            y = y * (1 - self.label_smooth) + 0.5 * self.label_smooth
        return x, y

    def _aug(self, x):
        if random.random() < 0.5:
            x = x + torch.randn_like(x) * random.uniform(0.02, 0.15)
        if random.random() < 0.5:
            x = x * torch.empty(1, x.shape[1]).uniform_(0.8, 1.2)
        if random.random() < 0.3:
            x = torch.roll(x, random.randint(-x.shape[0]//8, x.shape[0]//8), 0)
        if random.random() < 0.2:
            x[:, random.randint(0, x.shape[1]-1)] = 0.0
        return x


def mixup_batch(x, y, alpha=0.3):
    if alpha <= 0:
        return x, y
    lam = max(np.random.beta(alpha, alpha), 0.55)
    idx = torch.randperm(x.size(0))
    return lam * x + (1-lam) * x[idx], lam * y + (1-lam) * y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)
    def forward(self, x):  # (B, d, T)
        a = torch.softmax(self.w(x.permute(0,2,1)), dim=1)
        return (x.permute(0,2,1) * a).sum(1)


class EMGModel(nn.Module):
    def __init__(self, hidden=96, dropout=0.5, n_conv_layers=4):
        super().__init__()
        layers = []
        ch_in = NUM_CH
        channels = [32, 64, 128, 128][:n_conv_layers]
        kernels = [15, 11, 7, 5][:n_conv_layers]
        for i, (ch_out, k) in enumerate(zip(channels, kernels)):
            layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size=k, stride=2, padding=k//2),
                nn.BatchNorm1d(ch_out),
                nn.GELU(),
                nn.Dropout(0.1 if i < 2 else 0.15),
            ])
            ch_in = ch_out

        self.backbone = nn.Sequential(*layers)
        feat_dim = ch_out

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.attn_pool = AttentionPool(feat_dim)
        pool_dim = feat_dim * 3  # avg + max + attn

        self.head = nn.Sequential(
            nn.Linear(pool_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, NUM_LABELS),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        f = self.backbone(x)
        p = torch.cat([
            self.avg_pool(f).squeeze(-1),
            self.max_pool(f).squeeze(-1),
            self.attn_pool(f),
        ], dim=1)
        return self.head(p)


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def optimize_thresholds(preds, targets):
    """Find optimal per-zone threshold that maximizes F1."""
    best_thresholds = np.full(NUM_LABELS, 0.5)
    for i in range(NUM_LABELS):
        best_f1 = 0
        for t in np.arange(0.2, 0.8, 0.02):
            pred_bin = (preds[:, i] >= t).astype(float)
            f1 = f1_score(targets[:, i], pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[i] = t
    return best_thresholds


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN SNAPPING
# ─────────────────────────────────────────────────────────────────────────────

def snap_to_pattern(preds_binary, valid_patterns):
    """Snap each prediction to nearest valid pattern (Hamming distance)."""
    snapped = np.zeros_like(preds_binary)
    for i in range(len(preds_binary)):
        dists = np.abs(valid_patterns - preds_binary[i]).sum(axis=1)
        snapped[i] = valid_patterns[np.argmin(dists)]
    return snapped


# ─────────────────────────────────────────────────────────────────────────────
# TEST-TIME AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_tta(model, x, n_aug=5):
    """Average predictions over original + augmented versions."""
    model.eval()
    preds = torch.sigmoid(model(x))
    for _ in range(n_aug):
        x_aug = x + torch.randn_like(x) * 0.05
        preds += torch.sigmoid(model(x_aug))
    return preds / (n_aug + 1)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds, targets, thresholds=None, valid_patterns=None):
    if thresholds is None:
        thresholds = np.full(NUM_LABELS, 0.5)
    pred_bin = np.zeros_like(preds)
    for i in range(NUM_LABELS):
        pred_bin[:, i] = (preds[:, i] >= thresholds[i]).astype(float)

    if valid_patterns is not None:
        pred_bin = snap_to_pattern(pred_bin, valid_patterns)

    per_f1 = f1_score(targets, pred_bin, average=None, zero_division=0)
    macro_f1 = per_f1.mean()
    exact = (pred_bin == targets).all(axis=1).mean()
    hamming = (pred_bin == targets).mean(axis=0).mean()
    return {"macro_f1": macro_f1, "exact": exact, "hamming": hamming,
            "per_f1": per_f1}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_one_config(config, train_segs, val_segs, valid_patterns,
                     device, epochs, verbose=True):
    """Train a single configuration. Returns best val metrics + state."""
    win = config["window"]
    stride_tr = config["stride"]
    stride_va = win // 2

    train_ds = EMGDataset(train_segs, win, stride_tr, augment=True,
                          label_smooth=config["label_smooth"],
                          mixup=config["mixup"])
    val_ds = EMGDataset(val_segs, win, stride_va, augment=False)

    if len(train_ds) < 10 or len(val_ds) < 10:
        return None

    train_dl = DataLoader(train_ds, batch_size=config["batch_size"],
                          shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=config["batch_size"],
                        shuffle=False, num_workers=0)

    model = EMGModel(hidden=config["hidden"], dropout=config["dropout"],
                     n_conv_layers=config["n_conv"]).to(device)

    # Pos weights
    lm = train_ds.labels.mean(axis=0)
    pw = torch.tensor([(1-m)/max(m,0.01) for m in lm]).clamp(max=5.0).to(device)

    if config["loss"] == "focal":
        criterion = FocalBCELoss(gamma=config["focal_gamma"], pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                   weight_decay=config["wd"])

    # SWA setup
    swa_model = None
    swa_start = max(epochs // 2, 20)
    if config["use_swa"]:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer, swa_lr=config["lr"] * 0.5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=1e-6)

    best_f1 = 0
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            if config["mixup"] > 0 and random.random() < 0.4:
                x, y = mixup_batch(x, y, config["mixup"])
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if config["use_swa"] and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Validate
        eval_model = swa_model if (config["use_swa"] and epoch >= swa_start) else model
        eval_model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                if config["use_tta"] and epoch > epochs - 5:
                    p = predict_with_tta(eval_model, x, n_aug=3)
                else:
                    p = torch.sigmoid(eval_model(x))
                all_p.append(p.cpu().numpy())
                all_t.append(y.numpy())

        preds = np.concatenate(all_p)
        tgts = np.concatenate(all_t)
        tgts = (tgts > 0.5).astype(float)

        # Try optimized thresholds
        opt_thresh = optimize_thresholds(preds, tgts)

        # Try with and without pattern snapping
        m_basic = compute_metrics(preds, tgts)
        m_thresh = compute_metrics(preds, tgts, opt_thresh)
        m_snap = compute_metrics(preds, tgts, opt_thresh, valid_patterns)

        # Pick the best
        best_m = max([m_basic, m_thresh, m_snap], key=lambda x: x["macro_f1"])
        val_f1 = best_m["macro_f1"]

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = deepcopy(eval_model.state_dict())
            best_metrics = best_m.copy()
            best_metrics["thresholds"] = opt_thresh.copy()
            best_metrics["preds"] = preds.copy()
            best_metrics["tgts"] = tgts.copy()
            patience = 0
        else:
            patience += 1

        if verbose and (epoch % 20 == 0 or epoch <= 3 or val_f1 >= best_f1):
            gap = 0  # skip train eval for speed
            print(f"    E{epoch:3d} | Val F1={val_f1:.3f} "
                  f"(basic={m_basic['macro_f1']:.3f} "
                  f"thresh={m_thresh['macro_f1']:.3f} "
                  f"snap={m_snap['macro_f1']:.3f}) "
                  f"Exact={best_m['exact']:.3f}"
                  f"{' ★' if val_f1 >= best_f1 else ''}")

        if patience >= 30:
            break

    return {
        "best_f1": best_f1,
        "best_state": best_state,
        "best_metrics": best_metrics,
        "config": config,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }


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
    print("  EMG Hand Activation Model v4 — Maximum Accuracy")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Mode:   {'Quick' if args.quick else 'Full'} grid search")

    # Load data
    print("\n  Loading data...")
    gt, segments, valid_patterns = load_all_data(
        args.gt_csv, args.emg_dir, args.fs, args.filter_type)
    print(f"  {len(segments)} segments, {len(valid_patterns)} valid patterns")

    # Utterance-based split (spread across recording)
    seg_ids = sorted(set(s["seg"] for s in segments))
    random.shuffle(seg_ids)
    n_val = max(3, len(seg_ids) // 5)
    val_ids = set(seg_ids[i] for i in range(0, len(seg_ids),
                  max(1, len(seg_ids) // n_val))[:n_val])
    train_ids = set(seg_ids) - val_ids

    train_segs = [s for s in segments if s["seg"] in train_ids]
    val_segs = [s for s in segments if s["seg"] in val_ids]
    print(f"  Train: {len(train_segs)} segs | Val: {len(val_segs)} segs")

    # ── Define hyperparameter grid ──
    if args.quick:
        configs = [
            {"window": 500, "stride": 125, "batch_size": 64, "lr": 3e-4,
             "wd": 5e-4, "dropout": 0.5, "hidden": 96, "n_conv": 4,
             "loss": "focal", "focal_gamma": 2.0, "label_smooth": 0.05,
             "mixup": 0.2, "use_swa": True, "use_tta": True},
            {"window": 500, "stride": 125, "batch_size": 64, "lr": 5e-4,
             "wd": 1e-3, "dropout": 0.4, "hidden": 128, "n_conv": 4,
             "loss": "bce", "focal_gamma": 0, "label_smooth": 0.03,
             "mixup": 0.3, "use_swa": True, "use_tta": True},
            {"window": 250, "stride": 62, "batch_size": 128, "lr": 5e-4,
             "wd": 5e-4, "dropout": 0.5, "hidden": 96, "n_conv": 3,
             "loss": "focal", "focal_gamma": 1.5, "label_smooth": 0.05,
             "mixup": 0.2, "use_swa": True, "use_tta": True},
        ]
    else:
        # Full grid: systematically explore key dimensions
        base = {"batch_size": 64, "n_conv": 4, "use_swa": True, "use_tta": True}
        configs = []
        for win in [250, 500, 750]:
            for lr in [3e-4, 5e-4, 1e-3]:
                for dropout in [0.4, 0.5, 0.6]:
                    for loss_type in ["bce", "focal"]:
                        for mixup in [0.0, 0.2]:
                            c = {**base,
                                 "window": win,
                                 "stride": win // 4,
                                 "lr": lr,
                                 "wd": 5e-4,
                                 "dropout": dropout,
                                 "hidden": 96,
                                 "loss": loss_type,
                                 "focal_gamma": 2.0 if loss_type == "focal" else 0,
                                 "label_smooth": 0.05,
                                 "mixup": mixup}
                            configs.append(c)

    print(f"\n  Running {len(configs)} configurations...")

    # ── Grid search ──
    results = []
    best_overall_f1 = 0
    best_overall = None

    for i, config in enumerate(configs):
        print(f"\n  ── Config {i+1}/{len(configs)} ──")
        print(f"    win={config['window']} lr={config['lr']} "
              f"drop={config['dropout']} loss={config['loss']} "
              f"mixup={config['mixup']}")

        t0 = time.time()
        result = train_one_config(
            config, train_segs, val_segs, valid_patterns,
            device, args.epochs, verbose=True)
        elapsed = time.time() - t0

        if result is None:
            print(f"    SKIP (not enough data for this window size)")
            continue

        f1 = result["best_f1"]
        exact = result["best_metrics"]["exact"]
        print(f"    Result: F1={f1:.4f} Exact={exact:.3f} ({elapsed:.0f}s)")

        results.append(result)

        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall = result
            print(f"    ★ NEW BEST OVERALL")

    # ── Final results ──
    print("\n" + "=" * 70)
    print("  GRID SEARCH RESULTS")
    print("=" * 70)

    results.sort(key=lambda x: x["best_f1"], reverse=True)
    print(f"\n  {'Rank':>4s}  {'F1':>6s}  {'Exact':>6s}  {'Win':>4s}  "
          f"{'LR':>7s}  {'Drop':>5s}  {'Loss':>6s}  {'Mix':>4s}")
    print(f"  {'─'*52}")
    for i, r in enumerate(results[:10]):
        c = r["config"]
        print(f"  {i+1:4d}  {r['best_f1']:.4f}  {r['best_metrics']['exact']:.3f}  "
              f"{c['window']:4d}  {c['lr']:7.5f}  {c['dropout']:.2f}  "
              f"{c['loss']:>6s}  {c['mixup']:.1f}")

    # ── Detailed best model results ──
    bm = best_overall["best_metrics"]
    print(f"\n  BEST MODEL: F1={best_overall_f1:.4f} Exact={bm['exact']:.3f}")
    print(f"  Config: {best_overall['config']}")

    print(f"\n  Per-zone results:")
    print(f"  {'Zone':20s}  {'F1':>6s}  {'Threshold':>9s}")
    print(f"  {'─'*38}")
    zone_names = [c.replace("label_","") for c in LABEL_COLUMNS]
    for i, zn in enumerate(zone_names):
        print(f"  {zn:20s}  {bm['per_f1'][i]:.3f}  {bm['thresholds'][i]:.2f}")
    print(f"  {'─'*38}")
    print(f"  {'MACRO':20s}  {bm['macro_f1']:.3f}")

    # ── Save best model ──
    save_path = out_dir / "best_model_v4.pt"
    torch.save({
        "state_dict": best_overall["best_state"],
        "config": best_overall["config"],
        "thresholds": bm["thresholds"].tolist(),
        "valid_patterns": valid_patterns.tolist(),
        "val_f1": float(best_overall_f1),
        "val_exact": float(bm["exact"]),
        "val_hamming": float(bm["hamming"]),
        "per_f1": bm["per_f1"].tolist(),
        "label_columns": list(LABEL_COLUMNS),
    }, save_path)

    # Save grid search results
    grid_path = out_dir / "grid_search_results.csv"
    with open(grid_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "f1", "exact", "window", "lr", "dropout",
                     "loss", "mixup", "swa", "n_train", "n_val"])
        for i, r in enumerate(results):
            c = r["config"]
            w.writerow([i+1, round(r["best_f1"],4), round(r["best_metrics"]["exact"],3),
                        c["window"], c["lr"], c["dropout"], c["loss"],
                        c["mixup"], c["use_swa"], r["n_train"], r["n_val"]])

    print(f"\n  Model saved → {save_path}")
    print(f"  Grid results → {grid_path}")


if __name__ == "__main__":
    main()

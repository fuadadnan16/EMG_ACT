# EMG Multi-Label Hand Activation Model

## Overview

A single model that predicts **which of 10 hand zones are active** from 8-channel wrist-mounted EMG signals at 500 Hz. Inspired by the **emg2pose** (NeurIPS 2024) and **FEEL** (ECCV) papers, this system replaces force sensors with EMG sensors for hand activation detection.

## Hand Zone Mapping

```
Zone 0: Palm           Zone 1: Thumb
Zone 2: Index Tip      Zone 3: Index Segment
Zone 4: Middle Tip     Zone 5: Middle Segment
Zone 6: Ring Tip       Zone 7: Ring Segment
Zone 8: Pinky Tip      Zone 9: Pinky Segment
```

## Architecture

```
Input: (B, T, 8) — 8-channel EMG at 500Hz
  │
  ├─ Linear(8 → 16) — learned channel expansion (emg2pose expects 16 ch)
  │
  ├─ Conv1d Layer 0 (pretrained from emg2pose, frozen, stride=5)
  ├─ ReLU
  ├─ Conv1d Layer 1 (pretrained from emg2pose, frozen, stride=2)
  ├─ ReLU
  │   Total temporal stride = 10x
  │
  ├─ AdaptiveAvgPool1d(1) — global temporal pooling
  │
  ├─ Linear(feat_dim → 64) → ReLU → Dropout(0.3)
  ├─ Linear(64 → 64) → ReLU → Dropout(0.3)
  └─ Linear(64 → 10) — multi-label logits
```

**Fallback**: If the emg2pose checkpoint is not available, a standalone Conv1d backbone trains from scratch.

## Data Pipeline

1. **Ground Truth** (`ground_truth_labeled.csv`): Contains 64 voice commands with timestamps and 10 binary labels per command
2. **EMG Files** (`emg_0.csv` through `emg_60.csv`): 61 recordings with 8-channel raw EMG + timestamps
3. **Alignment**: Each command's label persists from its `end_unix` until the next command's `end_unix`
4. **Resampling**: Raw EMG (irregular timestamps) is interpolated to uniform 500 Hz grid
5. **Windowing**: 2-second windows with 0.5-second stride for training

## Usage

### Training

```bash
# With emg2pose pretrained backbone
python train_multilabel.py \
  --gt_csv ground_truth_labeled.csv \
  --emg_dir /path/to/emg_csvs/ \
  --ckpt_path emg2pose_model_checkpoints/tracking_vemg2pose.ckpt \
  --emg2pose_repo emg2pose/

# Without pretrained backbone (standalone)
python train_multilabel.py \
  --gt_csv ground_truth_labeled.csv \
  --emg_dir /path/to/emg_csvs/

# Custom filtering
python train_multilabel.py \
  --gt_csv ground_truth_labeled.csv \
  --emg_dir /path/to/emg_csvs/ \
  --filter_type hpf   # or "bpf" or "none"
```

### Inference

```bash
python predict.py \
  --model checkpoints_multilabel/best_model.pt \
  --emg_file new_recording.csv \
  --output predictions.csv
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Multi-label BCE loss** | Multiple hand zones can be active simultaneously (e.g., "all fingertips") |
| **8→16 channel expansion** | emg2pose expects 16 sEMG channels; learned linear projection adapts 8-ch input |
| **Irregular → uniform resampling** | Raw EMG has burst timestamps; interpolated to clean 500 Hz grid |
| **Pos-weight balancing** | Handles class imbalance (some zones activated more than others) |
| **First 2 Conv1d layers only** | Total stride=10x at 500Hz gives meaningful temporal features (vs 40x which would collapse) |
| **Time-based split** | Avoids temporal leakage between train/val |

## File Structure

```
train_multilabel.py    — Main training script
predict.py             — Inference script
ground_truth_labeled.csv — Voice command labels with timestamps
emg_0.csv ... emg_60.csv — EMG recordings
```

## Requirements

```
torch, numpy, pandas, scipy, scikit-learn
# For pretrained backbone:
omegaconf, pytorch-lightning, emg2pose repo
```

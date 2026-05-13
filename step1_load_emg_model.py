"""
STEP 1: Load Your Trained EMG Model
====================================
This script loads your best_model_v4.pt checkpoint and provides
a clean inference interface for the rest of the pipeline.

WHAT YOU NEED:
    - best_model_v4.pt (from your training run on the cluster)

WHAT THIS DOES:
    1. Defines the exact EMGModel architecture (copied from train_v4.py)
    2. Loads the checkpoint with thresholds and valid patterns
    3. Provides predict_zones() function for the rest of the pipeline
    4. Tests with synthetic random EMG to verify everything works

USAGE:
    python step1_load_emg_model.py --checkpoint best_model_v4.pt
    
    # Or if you don't have the checkpoint yet (uses random weights):
    python step1_load_emg_model.py --demo
"""

import numpy as np
import torch
import torch.nn as nn
import argparse
import os

# ─── Constants (must match train_v4.py exactly) ───
NUM_CH = 8        # 8 EMG channels
NUM_LABELS = 10   # 10 hand zones
ZONE_NAMES = [
    "palm", "thumb", "index_tip", "index_seg",
    "middle_tip", "middle_seg", "ring_tip", "ring_seg",
    "pinky_tip", "pinky_seg",
]


# ─── Model Architecture (exact copy from train_v4.py) ───

class AttentionPool(nn.Module):
    """Attention-weighted pooling over temporal dimension."""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, x):  # (B, d, T)
        a = torch.softmax(self.w(x.permute(0, 2, 1)), dim=1)
        return (x.permute(0, 2, 1) * a).sum(1)


class EMGModel(nn.Module):
    """
    Conv1d backbone + triple pooling (avg + max + attention) + classifier head.
    Architecture: 4 conv blocks → pool → 2 FC layers → 10 zone outputs.
    """
    def __init__(self, hidden=96, dropout=0.5, n_conv_layers=4):
        super().__init__()
        layers = []
        ch_in = NUM_CH
        channels = [32, 64, 128, 128][:n_conv_layers]
        kernels = [15, 11, 7, 5][:n_conv_layers]

        for i, (ch_out, k) in enumerate(zip(channels, kernels)):
            layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size=k, stride=2, padding=k // 2),
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
        # x: (batch, time_steps, 8_channels)
        x = x.permute(0, 2, 1)  # → (batch, 8, time_steps)
        f = self.backbone(x)
        p = torch.cat([
            self.avg_pool(f).squeeze(-1),
            self.max_pool(f).squeeze(-1),
            self.attn_pool(f),
        ], dim=1)
        return self.head(p)


# ─── Pattern Snapping ───

def snap_to_pattern(pred_binary, valid_patterns):
    """Snap a single prediction to the nearest valid pattern (Hamming distance)."""
    dists = np.abs(valid_patterns - pred_binary).sum(axis=1)
    return valid_patterns[np.argmin(dists)]


# ─── Inference Wrapper ───

class EMGPredictor:
    """
    Complete inference pipeline:
        raw EMG window → model → sigmoid → threshold → snap → zone labels
    
    Usage:
        predictor = EMGPredictor("best_model_v4.pt")
        zones = predictor.predict(emg_window)  # (500, 8) → [0,1,1,0,...]
    """

    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device,
                              weights_only=False)
            config = ckpt["config"]
            self.model = EMGModel(
                hidden=config.get("hidden", 96),
                dropout=config.get("dropout", 0.5),
                n_conv_layers=config.get("n_conv", 4),
            ).to(self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.thresholds = np.array(ckpt["thresholds"])
            self.valid_patterns = np.array(ckpt["valid_patterns"])
            self.window_size = config.get("window", 500)
            print(f"  Model loaded: F1={ckpt.get('val_f1', 'N/A')}")
            print(f"  Window size: {self.window_size}")
            print(f"  Thresholds: {self.thresholds}")
        else:
            print("No checkpoint found — using random weights (demo mode)")
            self.model = EMGModel(hidden=96, dropout=0.5, n_conv_layers=4).to(self.device)
            self.thresholds = np.full(NUM_LABELS, 0.5)
            self.valid_patterns = np.eye(NUM_LABELS)  # Dummy patterns
            self.window_size = 500

        self.model.eval()

    @torch.no_grad()
    def predict_probs(self, emg_window):
        """
        Get raw probabilities for each zone.
        
        Args:
            emg_window: numpy array of shape (time_steps, 8)
        Returns:
            numpy array of shape (10,) with probabilities
        """
        x = torch.tensor(emg_window, dtype=torch.float32).unsqueeze(0)  # (1, T, 8)
        x = x.to(self.device)
        logits = self.model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        return probs

    def predict_zones(self, emg_window, use_snap=True):
        """
        Get binary zone predictions.
        
        Args:
            emg_window: numpy array of shape (time_steps, 8)
            use_snap: whether to snap to valid patterns
        Returns:
            numpy array of shape (10,) with binary zone labels
        """
        probs = self.predict_probs(emg_window)
        binary = (probs >= self.thresholds).astype(float)
        if use_snap:
            binary = snap_to_pattern(binary, self.valid_patterns)
        return binary

    def predict_zones_named(self, emg_window, use_snap=True):
        """
        Get zone predictions as a dictionary.
        
        Returns:
            dict like {"palm": 1, "thumb": 0, "index_tip": 1, ...}
        """
        binary = self.predict_zones(emg_window, use_snap)
        return {name: int(val) for name, val in zip(ZONE_NAMES, binary)}


# ─── Synthetic EMG Generator (for testing without real data) ───

def generate_synthetic_emg(zone_pattern, window_size=500, num_channels=8):
    """
    Generate synthetic EMG-like signal for a given zone pattern.
    
    This creates realistic-looking EMG by:
    1. Base noise (always present)
    2. Higher amplitude + specific frequency content for active zones
    3. Cross-talk between neighboring channels
    
    Args:
        zone_pattern: list of 10 binary values [palm, thumb, ...]
        window_size: number of time steps (default 500 = 1 second at 500Hz)
        num_channels: number of EMG channels (8)
    Returns:
        numpy array of shape (window_size, num_channels)
    """
    # Zone-to-channel mapping (which EMG channels activate for each zone)
    # Based on forearm muscle anatomy
    ZONE_CHANNEL_MAP = {
        0: [0, 1],       # palm → channels 0,1 (flexor digitorum superficialis)
        1: [1, 2],       # thumb → channels 1,2 (flexor pollicis longus)
        2: [2, 3],       # index_tip → channels 2,3 (flexor digitorum profundus)
        3: [2, 3],       # index_seg → channels 2,3
        4: [3, 4],       # middle_tip → channels 3,4
        5: [3, 4],       # middle_seg → channels 3,4
        6: [4, 5],       # ring_tip → channels 4,5
        7: [5, 6],       # ring_seg → channels 5,6
        8: [6, 7],       # pinky_tip → channels 6,7
        9: [6, 7],       # pinky_seg → channels 6,7
    }

    t = np.linspace(0, window_size / 500.0, window_size)
    emg = np.random.randn(window_size, num_channels) * 0.1  # Base noise

    for zone_idx, active in enumerate(zone_pattern):
        if active:
            channels = ZONE_CHANNEL_MAP[zone_idx]
            for ch in channels:
                # Add muscle activation signal
                freq = np.random.uniform(20, 150)  # EMG frequency range
                amplitude = np.random.uniform(0.5, 1.5)
                emg[:, ch] += amplitude * np.sin(2 * np.pi * freq * t)
                emg[:, ch] += np.random.randn(window_size) * 0.3  # Activity noise

    # Normalize per channel
    for ch in range(num_channels):
        std = emg[:, ch].std()
        if std > 1e-6:
            emg[:, ch] = (emg[:, ch] - emg[:, ch].mean()) / std

    return emg.astype(np.float32)


# ─── Main: Test the model ───

def main():
    parser = argparse.ArgumentParser(description="Step 1: Load EMG Model")
    parser.add_argument("--checkpoint", type=str, default="best_model_v4.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode without checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 1: Load EMG Model & Test Inference")
    print("=" * 60)

    # Load model
    ckpt_path = None if args.demo else args.checkpoint
    predictor = EMGPredictor(ckpt_path)

    # Test with synthetic EMG for different zone patterns
    test_patterns = {
        "power_grasp (all zones)": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "precision_pinch (thumb+index_tip)": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "tripod (thumb+index_tip+middle_tip)": [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        "rest (no zones)": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "index_only": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    }

    print("\nTesting with synthetic EMG signals:")
    print("-" * 60)

    for name, pattern in test_patterns.items():
        emg = generate_synthetic_emg(pattern, window_size=predictor.window_size)
        probs = predictor.predict_probs(emg)
        zones = predictor.predict_zones(emg)
        named = predictor.predict_zones_named(emg)

        active_zones = [z for z, v in named.items() if v == 1]
        print(f"\n  Input: {name}")
        print(f"  Probs: {np.array2string(probs, precision=2, separator=', ')}")
        print(f"  Predicted active: {active_zones if active_zones else 'none'}")

    print("\n" + "=" * 60)
    print("  Step 1 COMPLETE — Model loaded and inference working!")
    print("=" * 60)
    print("\n  Next: Run step2_zone_to_grasp.py")

    return predictor


if __name__ == "__main__":
    main()

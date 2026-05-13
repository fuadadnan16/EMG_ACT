"""
fix_load_v1.py — Quick fix to load the v1 saved model and print results
=========================================================================
The error was: PyTorch 2.6 changed torch.load default to weights_only=True,
but the checkpoint contains numpy arrays. This script loads it with
weights_only=False.

Usage:
  python fix_load_v1.py --model checkpoints_multilabel/best_model.pt
"""

import torch
import numpy as np

import argparse
p = argparse.ArgumentParser()
p.add_argument("--model", default="checkpoints_multilabel/best_model.pt")
args = p.parse_args()

# FIX: Use weights_only=False for PyTorch 2.6+
ckpt = torch.load(args.model, map_location="cpu", weights_only=False)

print("Checkpoint loaded successfully!")
print(f"  Epoch:       {ckpt['epoch']}")
print(f"  Val F1:      {ckpt['val_f1']:.4f}")
print(f"  Val Exact:   {ckpt['val_exact_match']:.4f}")
print(f"  Val Hamming: {ckpt['val_hamming_acc']:.4f}")
print(f"  Window:      {ckpt['window_sec']}s")
print(f"  Pretrained:  {ckpt['use_pretrained']}")

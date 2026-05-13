"""
STEP 5: Imitation Learning — Teach Once, Robot Repeats
========================================================
This script:
    1. Loads demonstration trajectories from Step 4
    2. Trains a Behavioral Cloning policy
    3. Tests the policy: robot autonomously replicates learned actions
    4. Evaluates: how well does the robot reproduce the demonstration?

PREREQUISITES:
    pip install torch numpy scikit-learn matplotlib

WHAT IS BEHAVIORAL CLONING:
    The simplest form of imitation learning.
    Given: (observation, action) pairs from demonstrations
    Learn: policy(observation) → action
    It's just supervised learning on the demonstration data!

USAGE:
    python step5_imitation_learning.py
    python step5_imitation_learning.py --epochs 200 --task pick_up_mug
"""

import numpy as np
import json
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ─── Demonstration Dataset ───

class DemoDataset(Dataset):
    """Dataset of (observation, action) pairs from demonstrations."""

    def __init__(self, observations, actions):
        self.obs = torch.tensor(observations, dtype=torch.float32)
        self.act = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]


# ─── Imitation Policy Network ───

class ImitationPolicy(nn.Module):
    """
    Neural network policy: observation → action.
    
    Input: current state (joint positions + velocities + zone activations)
    Output: target actuator positions
    """

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, act_dim),
            nn.Tanh(),  # Actions bounded to [-1, 1]
        )

    def forward(self, obs):
        return self.net(obs)

    def predict(self, obs_np):
        """Predict action from numpy observation."""
        self.eval()
        with torch.no_grad():
            obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            action = self.net(obs).squeeze(0).numpy()
        return action


# ─── Data Preparation ───

def load_demonstrations(demo_path="simulation_output/all_demonstrations.json",
                        task_filter=None):
    """
    Load demonstration data and prepare for training.
    
    Returns:
        observations: (N, obs_dim) numpy array
        actions: (N, act_dim) numpy array
        task_labels: list of task names
    """
    if not os.path.exists(demo_path):
        print(f"  Demo file not found: {demo_path}")
        print(f"  Generating synthetic demonstrations...")
        return generate_synthetic_demos(task_filter)

    with open(demo_path, "r") as f:
        all_demos = json.load(f)

    observations = []
    actions = []
    task_labels = []

    tasks = [task_filter] if task_filter else list(all_demos.keys())

    for task_name in tasks:
        if task_name not in all_demos:
            continue
        trajectory = all_demos[task_name]

        for i, state in enumerate(trajectory):
            # Observation: zone activations + normalized time progress
            zones = np.array(state["zones"], dtype=np.float32)
            progress = np.array([i / max(len(trajectory) - 1, 1)],
                                dtype=np.float32)
            phase = np.array([state["phase"] / 5.0], dtype=np.float32)

            # Add previous state for temporal context
            if i > 0:
                prev_zones = np.array(trajectory[i-1]["zones"], dtype=np.float32)
            else:
                prev_zones = np.zeros(10, dtype=np.float32)

            obs = np.concatenate([zones, prev_zones, progress, phase])
            act = np.array(state["actuator_targets"], dtype=np.float32)

            observations.append(obs)
            actions.append(act)
            task_labels.append(task_name)

    observations = np.array(observations)
    actions = np.array(actions)

    print(f"  Loaded {len(observations)} demonstration steps "
          f"from {len(set(task_labels))} tasks")
    print(f"  Observation dim: {observations.shape[1]}")
    print(f"  Action dim: {actions.shape[1]}")

    return observations, actions, task_labels


def generate_synthetic_demos(task_filter=None):
    """Generate demonstrations without loading files (for testing)."""
    from step4_mujoco_hand_control import get_demo_sequences, zones_to_actuators

    demos = get_demo_sequences()
    tasks = [task_filter] if task_filter else list(demos.keys())

    observations = []
    actions = []
    task_labels = []

    for task_name in tasks:
        if task_name not in demos:
            continue
        sequence = demos[task_name]

        step_idx = 0
        total_steps = sum(dur for _, dur, _ in sequence)
        prev_zones = np.zeros(10, dtype=np.float32)

        for zones, duration, label in sequence:
            targets = zones_to_actuators(zones)
            for s in range(duration):
                progress = np.array([step_idx / max(total_steps - 1, 1)],
                                    dtype=np.float32)
                phase = np.array([0.0], dtype=np.float32)
                obs = np.concatenate([
                    np.array(zones, dtype=np.float32),
                    prev_zones,
                    progress,
                    phase,
                ])
                observations.append(obs)
                actions.append(targets)
                task_labels.append(task_name)
                prev_zones = np.array(zones, dtype=np.float32)
                step_idx += 1

    return np.array(observations), np.array(actions), task_labels


# ─── Training ───

def train_policy(observations, actions, epochs=100, batch_size=32,
                 lr=1e-3, val_split=0.2):
    """
    Train behavioral cloning policy.
    
    Returns:
        trained ImitationPolicy model
    """
    obs_dim = observations.shape[1]
    act_dim = actions.shape[1]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        observations, actions, test_size=val_split, random_state=42
    )

    train_ds = DemoDataset(X_train, y_train)
    val_ds = DemoDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    policy = ImitationPolicy(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.MSELoss()

    print(f"\n  Training Behavioral Cloning Policy")
    print(f"  {'─' * 50}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"  Obs dim: {obs_dim} | Act dim: {act_dim}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        policy.train()
        train_loss = 0
        for obs, act in train_dl:
            pred = policy(obs)
            loss = criterion(pred, act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Validate
        policy.eval()
        val_loss = 0
        with torch.no_grad():
            for obs, act in val_dl:
                pred = policy(obs)
                val_loss += criterion(pred, act).item()
        val_loss /= len(val_dl)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")

    policy.load_state_dict(best_state)
    print(f"\n  Training complete! Best val loss: {best_val_loss:.6f}")

    return policy


# ─── Evaluation ───

def evaluate_policy(policy, observations, actions, task_labels):
    """
    Evaluate how well the learned policy reproduces demonstrations.
    """
    policy.eval()
    print(f"\n  Evaluating policy reproduction quality...")
    print(f"  {'─' * 50}")

    tasks = sorted(set(task_labels))
    overall_mse = 0
    overall_count = 0

    for task in tasks:
        mask = [i for i, t in enumerate(task_labels) if t == task]
        task_obs = observations[mask]
        task_act = actions[mask]

        with torch.no_grad():
            pred = policy(torch.tensor(task_obs, dtype=torch.float32)).numpy()

        mse = np.mean((pred - task_act) ** 2)
        mae = np.mean(np.abs(pred - task_act))
        max_err = np.max(np.abs(pred - task_act))

        # Compute per-actuator accuracy (within 10% of target)
        close = np.abs(pred - task_act) < 0.1
        accuracy = close.mean() * 100

        print(f"\n  Task: {task}")
        print(f"    MSE: {mse:.6f} | MAE: {mae:.4f} | Max Error: {max_err:.4f}")
        print(f"    Accuracy (within 10%): {accuracy:.1f}%")
        print(f"    Steps: {len(mask)}")

        overall_mse += mse * len(mask)
        overall_count += len(mask)

    overall_mse /= overall_count
    print(f"\n  Overall MSE: {overall_mse:.6f}")

    return overall_mse


def replay_policy(policy, task_name, demo_sequences):
    """
    Replay the learned policy and compare to original demonstration.
    Shows the robot autonomously performing the action.
    """
    from step4_mujoco_hand_control import zones_to_actuators, ZONE_NAMES

    if task_name not in demo_sequences:
        print(f"  Unknown task: {task_name}")
        return

    sequence = demo_sequences[task_name]
    print(f"\n  Replaying learned policy for: {task_name}")
    print(f"  {'─' * 50}")

    step_idx = 0
    total_steps = sum(dur for _, dur, _ in sequence)
    prev_zones = np.zeros(10, dtype=np.float32)

    for zones, duration, label in sequence:
        original_targets = zones_to_actuators(zones)

        for s in range(0, duration, max(1, duration // 3)):
            progress = np.array([step_idx / max(total_steps - 1, 1)])
            phase = np.array([0.0])
            obs = np.concatenate([
                np.array(zones, dtype=np.float32),
                prev_zones,
                progress.astype(np.float32),
                phase.astype(np.float32),
            ])

            predicted = policy.predict(obs)
            error = np.mean(np.abs(predicted - original_targets))
            active = [ZONE_NAMES[i] for i, v in enumerate(zones) if v]

            print(f"  Step {step_idx:3d} [{label:15s}] "
                  f"zones={active or ['rest']:30s} "
                  f"error={error:.4f}")

            prev_zones = np.array(zones, dtype=np.float32)
            step_idx += s + 1


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Step 5: Imitation Learning")
    parser.add_argument("--demo_path", type=str,
                        default="simulation_output/all_demonstrations.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--task", type=str, default=None,
                        help="Train on specific task only")
    parser.add_argument("--save_model", type=str, default="imitation_policy.pt")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 5: Imitation Learning — Teach Once, Robot Repeats")
    print("=" * 60)

    # Load demonstrations
    observations, actions, task_labels = load_demonstrations(
        args.demo_path, args.task
    )

    if len(observations) == 0:
        print("  No demonstrations found!")
        return

    # Train policy
    policy = train_policy(observations, actions, epochs=args.epochs)

    # Evaluate
    evaluate_policy(policy, observations, actions, task_labels)

    # Replay
    from step4_mujoco_hand_control import get_demo_sequences
    demo_sequences = get_demo_sequences()
    tasks = [args.task] if args.task else list(demo_sequences.keys())
    for task in tasks[:2]:
        replay_policy(policy, task, demo_sequences)

    # Save model
    save_data = {
        "state_dict": policy.state_dict(),
        "obs_dim": observations.shape[1],
        "act_dim": actions.shape[1],
        "tasks": list(set(task_labels)),
        "n_demos": len(observations),
    }
    torch.save(save_data, args.save_model)
    print(f"\n  Policy saved: {args.save_model}")

    print(f"\n{'=' * 60}")
    print(f"  Step 5 COMPLETE — Robot learned to replicate your actions!")
    print(f"{'=' * 60}")
    print(f"\n  Next: Run step6_full_pipeline.py for the complete demo")


if __name__ == "__main__":
    main()

"""
run_all.py — Master Script: Run the Complete Tutorial
======================================================

This runs all 6 steps in sequence. Each step can also be run independently.

SETUP:
    # Create virtual environment (recommended)
    python -m venv emg_sim_env
    source emg_sim_env/bin/activate  # Linux/Mac
    # emg_sim_env\Scripts\activate   # Windows
    
    # Install dependencies
    pip install torch numpy scipy scikit-learn matplotlib pandas
    pip install mujoco           # For Step 4 (hand simulation)
    pip install myosuite         # For Step 3 (optional, synthetic EMG)
    pip install gymnasium        # For Step 3 (required by myosuite)

USAGE:
    # Run everything (demo mode, no checkpoint needed):
    python run_all.py
    
    # Run with your trained model:
    python run_all.py --checkpoint /path/to/best_model_v4.pt
    
    # Run individual steps:
    python step1_load_emg_model.py --demo
    python step2_zone_to_grasp.py
    python step3_myosuite_synthetic.py
    python step4_mujoco_hand_control.py
    python step5_imitation_learning.py
    python step6_full_pipeline.py
"""

import sys
import os
import argparse

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_step(step_num, description, func, **kwargs):
    """Run a single step with nice formatting."""
    print(f"\n{'█' * 60}")
    print(f"█  STEP {step_num}: {description}")
    print(f"{'█' * 60}\n")
    try:
        result = func(**kwargs)
        print(f"\n  ✓ Step {step_num} completed successfully!")
        return result
    except Exception as e:
        print(f"\n  ✗ Step {step_num} failed: {e}")
        print(f"    (This step is optional — continuing...)")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="EMG-ACT Simulation Pipeline — Complete Tutorial"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to best_model_v4.pt (optional, demo mode if not provided)")
    parser.add_argument("--skip_myosuite", action="store_true",
                        help="Skip Step 3 (MyoSuite synthetic EMG)")
    parser.add_argument("--skip_mujoco", action="store_true",
                        help="Skip Step 4 (MuJoCo hand control)")
    args = parser.parse_args()

    print("=" * 60)
    print("  EMG-ACT: From Wrist EMG to Robot Hand Control")
    print("  Complete Simulation Pipeline Tutorial")
    print("=" * 60)
    print()
    print("  This tutorial demonstrates the FULL pipeline:")
    print("  ┌──────────────────────────────────────────────┐")
    print("  │  Step 1: Load trained EMG model               │")
    print("  │  Step 2: Object & action recognition          │")
    print("  │  Step 3: Synthetic EMG from MyoSuite          │")
    print("  │  Step 4: MuJoCo Shadow Hand control           │")
    print("  │  Step 5: Imitation learning (teach & replay)  │")
    print("  │  Step 6: Full pipeline demo                   │")
    print("  └──────────────────────────────────────────────┘")
    print()
    if args.checkpoint:
        print(f"  Using checkpoint: {args.checkpoint}")
    else:
        print("  Running in DEMO MODE (no checkpoint, random weights)")
        print("  Add --checkpoint best_model_v4.pt for real predictions")

    input("\n  Press Enter to start (or Ctrl+C to cancel)...")

    # ── STEP 1 ──
    from step1_load_emg_model import EMGPredictor, generate_synthetic_emg
    predictor = run_step(
        1, "Load EMG Model",
        lambda: EMGPredictor(args.checkpoint),
    )

    # ── STEP 2 ──
    from step2_zone_to_grasp import ObjectRecognizer, ActionRecognizer
    def step2():
        obj_rec = ObjectRecognizer()
        test = obj_rec.recognize([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        print(f"  Test: all zones active → {test['grasp_type']} → {test['most_likely_object']}")
        test2 = obj_rec.recognize([0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        print(f"  Test: thumb+index → {test2['grasp_type']} → {test2['most_likely_object']}")
        return obj_rec
    run_step(2, "Object & Action Recognition", step2)

    # ── STEP 3 ──
    if not args.skip_myosuite:
        from step3_myosuite_synthetic import run_myosuite_task
        run_step(
            3, "Synthetic EMG from MyoSuite",
            lambda: run_myosuite_task(n_episodes=3, max_steps=50),
        )
    else:
        print("\n  Skipping Step 3 (--skip_myosuite)")

    # ── STEP 4 ──
    if not args.skip_mujoco:
        from step4_mujoco_hand_control import get_demo_sequences, run_hand_simulation
        def step4():
            demos = get_demo_sequences()
            traj = run_hand_simulation(
                demos["pick_up_mug"],
                save_dir="simulation_output/pick_up_mug",
            )
            return traj
        run_step(4, "MuJoCo Hand Control", step4)
    else:
        print("\n  Skipping Step 4 (--skip_mujoco)")

    # ── STEP 5 ──
    from step5_imitation_learning import (
        load_demonstrations, train_policy, evaluate_policy
    )
    def step5():
        obs, acts, labels = load_demonstrations(task_filter=None)
        if len(obs) > 0:
            policy = train_policy(obs, acts, epochs=50)
            evaluate_policy(policy, obs, acts, labels)
            return policy
        return None
    run_step(5, "Imitation Learning", step5)

    # ── STEP 6 ──
    from step6_full_pipeline import run_full_pipeline
    run_step(
        6, "Full Pipeline Demo",
        lambda: run_full_pipeline(predictor, "pick_up_mug"),
    )

    # ── FINAL SUMMARY ──
    print(f"\n{'█' * 60}")
    print(f"█  TUTORIAL COMPLETE!")
    print(f"{'█' * 60}")
    print(f"""
  You now have:
  
  1. EMG Model (step1)
     → Loads your trained v4 model
     → Predicts 10 hand zone activations from EMG
  
  2. Object/Action Recognition (step2)
     → 9 grasp types, 30+ objects
     → 6 action types with temporal recognition
  
  3. Synthetic EMG Data (step3)
     → MyoSuite muscle activations → synthetic EMG
     → Enables sim-to-real transfer experiments
  
  4. MuJoCo Hand Control (step4)
     → Zone predictions drive a simulated hand
     → 5 pre-built manipulation demos
  
  5. Imitation Learning (step5)
     → Robot learns to replicate demonstrations
     → Behavioral cloning from observation-action pairs
  
  6. Full Pipeline (step6)
     → EMG → Zones → Object → Action → Hand → Replay
     → Complete end-to-end demonstration
  
  Output files:
  ├── synthetic_data/          ← Synthetic EMG data
  ├── simulation_output/       ← MuJoCo trajectories
  │   ├── pick_up_mug/
  │   ├── turn_key/
  │   └── all_demonstrations.json
  ├── pipeline_output/         ← Full pipeline results
  └── imitation_policy.pt      ← Trained imitation model
  
  FOR YOUR MAY 12TH PRESENTATION:
  ─────────────────────────────────
  1. Show the pipeline diagram (EMG → Zones → Object → Robot)
  2. Demo step2 output (object recognition from zones)
  3. Show step4 trajectories (MuJoCo hand performing grasps)
  4. Show step5 results (robot learned to replay)
  5. Compare to FEEL paper (non-obstructive, pre-contact, scalable)
""")


if __name__ == "__main__":
    main()

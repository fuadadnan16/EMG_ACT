"""
STEP 6: Full Pipeline Demo — EMG → Object Recognition → Imitation → Replay
=============================================================================
This is the complete end-to-end demonstration:

    1. Load your trained EMG model
    2. Feed EMG signal (real or synthetic)
    3. Predict zone activations
    4. Recognize object and action
    5. Drive simulated hand
    6. Robot learns and replays autonomously

USAGE:
    python step6_full_pipeline.py
    python step6_full_pipeline.py --checkpoint best_model_v4.pt
"""

import numpy as np
import json
import os
import argparse
import time

# Import all previous steps
from step1_load_emg_model import EMGPredictor, generate_synthetic_emg, ZONE_NAMES
from step2_zone_to_grasp import ObjectRecognizer, ActionRecognizer
from step4_mujoco_hand_control import (
    zones_to_actuators, get_demo_sequences, run_hand_simulation
)


def run_full_pipeline(predictor=None, task="pick_up_mug"):
    """
    Complete pipeline demonstration.
    
    Flow:
        Synthetic EMG → EMG Model → Zone Predictions → Object/Action Recognition
        → MuJoCo Hand Control → Trajectory Recording
    """
    # Initialize components
    if predictor is None:
        predictor = EMGPredictor(None)  # Demo mode

    obj_recognizer = ObjectRecognizer()
    action_recognizer = ActionRecognizer()

    demos = get_demo_sequences()
    if task not in demos:
        task = list(demos.keys())[0]

    sequence = demos[task]

    print(f"\n{'═' * 60}")
    print(f"  FULL PIPELINE: {task}")
    print(f"{'═' * 60}")

    all_results = []

    for phase_idx, (zones, duration, label) in enumerate(sequence):
        print(f"\n  ── Phase {phase_idx + 1}: {label} ──")

        # STEP A: Generate synthetic EMG for this zone pattern
        emg_window = generate_synthetic_emg(
            zones, window_size=predictor.window_size
        )
        print(f"  [A] Generated EMG: shape={emg_window.shape}")

        # STEP B: Predict zones from EMG
        predicted_zones = predictor.predict_zones(emg_window)
        active = [ZONE_NAMES[i] for i, v in enumerate(predicted_zones) if v]
        print(f"  [B] Predicted zones: {active or ['rest']}")

        # STEP C: Recognize object
        obj_result = obj_recognizer.recognize(predicted_zones)
        print(f"  [C] Object: {obj_result['most_likely_object']} "
              f"(grasp: {obj_result['grasp_type']}, "
              f"conf: {obj_result['confidence']:.0%})")

        # STEP D: Recognize action (temporal)
        action_recognizer.add_observation(predicted_zones)
        if phase_idx >= 3:
            act_result = action_recognizer.recognize()
            print(f"  [D] Action: {act_result['action']} "
                  f"(phase: {act_result['phase']}, "
                  f"conf: {act_result['confidence']:.0%})")
        else:
            print(f"  [D] Action: buffering... ({phase_idx + 1}/4 observations)")

        # STEP E: Compute actuator targets
        actuators = zones_to_actuators(predicted_zones)
        n_active = np.count_nonzero(actuators > 0.1)
        print(f"  [E] Actuators: {n_active} active (of 17)")

        all_results.append({
            "phase": label,
            "ground_truth_zones": zones.tolist(),
            "predicted_zones": predicted_zones.tolist(),
            "object": obj_result["most_likely_object"],
            "grasp": obj_result["grasp_type"],
            "actuators": actuators.tolist(),
        })

    # STEP F: Run simulation
    print(f"\n  ── Running MuJoCo Simulation ──")
    sim_sequence = [
        (np.array(r["predicted_zones"]), 40, r["phase"])
        for r in all_results
    ]
    trajectory = run_hand_simulation(
        sim_sequence,
        save_dir=f"pipeline_output/{task}",
        visualize=False,
    )

    # STEP G: Summary
    print(f"\n{'═' * 60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'═' * 60}")
    print(f"\n  Task: {task}")
    print(f"  Phases: {len(all_results)}")
    print(f"  Trajectory: {len(trajectory)} states")

    # Accuracy check (predicted vs ground truth)
    correct = 0
    total = 0
    for r in all_results:
        gt = np.array(r["ground_truth_zones"])
        pred = np.array(r["predicted_zones"])
        correct += (gt == pred).sum()
        total += len(gt)
    zone_accuracy = correct / total * 100

    print(f"  Zone prediction accuracy: {zone_accuracy:.1f}%")
    print(f"\n  Phase-by-phase results:")
    for r in all_results:
        gt_active = [ZONE_NAMES[i] for i, v
                     in enumerate(r["ground_truth_zones"]) if v]
        pred_active = [ZONE_NAMES[i] for i, v
                       in enumerate(r["predicted_zones"]) if v]
        match = "✓" if gt_active == pred_active else "✗"
        print(f"    {match} {r['phase']:15s} → {r['object']:15s} "
              f"({r['grasp']})")

    # Save results
    os.makedirs(f"pipeline_output/{task}", exist_ok=True)
    results_path = f"pipeline_output/{task}/pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    return all_results, trajectory


def main():
    parser = argparse.ArgumentParser(description="Step 6: Full Pipeline Demo")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "pick_up_mug", "precision_pick_coin",
                                 "write_with_pen", "turn_key", "pour_water"])
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 6: Complete EMG → Simulation Pipeline")
    print("=" * 60)
    print()
    print("  Pipeline flow:")
    print("  EMG Signal → Zone Prediction → Object Recognition")
    print("  → Action Recognition → Hand Control → Imitation Learning")

    predictor = EMGPredictor(args.checkpoint)

    demos = get_demo_sequences()
    tasks = list(demos.keys()) if args.task == "all" else [args.task]

    for task in tasks:
        run_full_pipeline(predictor, task)

    print(f"\n{'═' * 60}")
    print(f"  ALL TASKS COMPLETE!")
    print(f"{'═' * 60}")
    print(f"\n  Output files in: pipeline_output/")
    print(f"  Each task folder contains:")
    print(f"    - trajectory.json (robot states for imitation learning)")
    print(f"    - pipeline_results.json (zone/object/action predictions)")
    print(f"\n  For your May 12th presentation:")
    print(f"    1. Show pipeline_results.json as proof of concept")
    print(f"    2. Show trajectory.json driving a simulated hand")
    print(f"    3. Show the FEEL comparison table from step2")
    print(f"    4. Demo the imitation learning from step5")


if __name__ == "__main__":
    main()

# EMG-ACT Simulation Pipeline
## From Wrist EMG to Robot Hand Control — Complete Tutorial

---

## Quick Start (5 Minutes)

```bash
# 1. Install dependencies
pip install torch numpy scipy scikit-learn matplotlib pandas
pip install mujoco myosuite gymnasium

# 2. Copy this folder to your machine
cp -r emg_simulation/ ~/emg_simulation/
cd ~/emg_simulation/

# 3. Run everything (demo mode, no EMG checkpoint needed)
python run_all.py

# 4. Or run with your trained model
python run_all.py --checkpoint /path/to/best_model_v4.pt
```

---

## What This Project Does

```
YOUR EMG SENSOR (8 channels, wrist-mounted)
         │
         ▼
┌─────────────────────────┐
│  STEP 1: EMG Model      │  Your trained v4 model (84.5% F1)
│  8ch EMG → 10 zones     │  Conv1d + AttentionPool + FocalLoss
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  STEP 2: Recognition    │  Zone pattern → grasp type → object
│  Object + Action ID     │  Zone sequence → action type
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  STEP 3: MyoSuite       │  Generate synthetic EMG from
│  Synthetic EMG          │  physiological muscle models
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  STEP 4: MuJoCo Hand    │  Zone predictions drive a
│  Shadow Hand Control    │  simulated dexterous hand
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  STEP 5: Imitation      │  Robot learns from your
│  Behavioral Cloning     │  demonstrations automatically
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  STEP 6: Full Pipeline  │  Complete end-to-end demo
│  EMG → Robot Replay     │  "Teach once, robot repeats"
└─────────────────────────┘
```

---

## File Structure

```
emg_simulation/
├── __init__.py                    # Package docstring
├── run_all.py                     # Master script (runs everything)
├── step1_load_emg_model.py        # Load v4 model + inference
├── step2_zone_to_grasp.py         # Object & action recognition
├── step3_myosuite_synthetic.py    # Synthetic EMG generation
├── step4_mujoco_hand_control.py   # MuJoCo hand simulation
├── step5_imitation_learning.py    # Behavioral cloning
├── step6_full_pipeline.py         # End-to-end demo
└── README.md                      # This file

Output directories (created automatically):
├── synthetic_data/                # Synthetic EMG from MyoSuite
├── simulation_output/             # MuJoCo trajectories
│   ├── pick_up_mug/
│   ├── precision_pick_coin/
│   ├── write_with_pen/
│   ├── turn_key/
│   ├── pour_water/
│   └── all_demonstrations.json
├── pipeline_output/               # Full pipeline results
└── imitation_policy.pt            # Trained imitation model
```

---

## Step-by-Step Tutorial

### Step 1: Load Your EMG Model

**What it does**: Loads your trained best_model_v4.pt and provides a
clean inference interface.

```bash
# With your checkpoint
python step1_load_emg_model.py --checkpoint best_model_v4.pt

# Without checkpoint (demo mode with random weights)
python step1_load_emg_model.py --demo
```

**Key class**: `EMGPredictor`
```python
from step1_load_emg_model import EMGPredictor

predictor = EMGPredictor("best_model_v4.pt")
zones = predictor.predict_zones(emg_window)        # [0,1,1,0,...]
named = predictor.predict_zones_named(emg_window)   # {"palm": 0, "thumb": 1, ...}
probs = predictor.predict_probs(emg_window)          # [0.12, 0.89, 0.95, ...]
```

---

### Step 2: Object & Action Recognition

**What it does**: Maps zone activation patterns to objects and actions.

```bash
python step2_zone_to_grasp.py
```

**Recognizes 9 grasp types**:
- Power grasp → mug, bottle, hammer
- Precision pinch → coin, pill, needle
- Tripod grip → pen, pencil, stylus
- Lateral pinch → key, credit card
- Hook grasp → bag handle, bucket
- Spherical grasp → ball, apple, doorknob
- Disc grasp → jar lid, bottle cap
- Platform grasp → plate, tray, book
- Rest → nothing

**Recognizes 6 action types** (temporal sequences):
- Pick up, Put down, Twist key, Write, Pour, Open jar

```python
from step2_zone_to_grasp import ObjectRecognizer, ActionRecognizer

obj_rec = ObjectRecognizer()
result = obj_rec.recognize([0, 1, 1, 0, 1, 0, 0, 0, 0, 0])
print(result["most_likely_object"])  # "pen"
print(result["grasp_type"])          # "tripod_grip"

act_rec = ActionRecognizer()
for zones in zone_sequence:
    act_rec.add_observation(zones)
result = act_rec.recognize()
print(result["action"])  # "pick_up"
```

---

### Step 3: MyoSuite Synthetic EMG

**What it does**: Generates synthetic EMG from physiologically accurate
muscle simulations. Enables sim-to-real transfer experiments.

```bash
# With MyoSuite installed
python step3_myosuite_synthetic.py --episodes 10

# If MyoSuite fails, automatically uses fallback synthetic data
python step3_myosuite_synthetic.py
```

**Output**: `synthetic_data/synthetic_*.npz`

---

### Step 4: MuJoCo Hand Control

**What it does**: Creates a simulated hand and drives it with your zone
predictions. Includes 5 pre-built manipulation demos.

```bash
# Headless (saves trajectories)
python step4_mujoco_hand_control.py

# With 3D viewer (if display available)
python step4_mujoco_hand_control.py --visualize

# Save frames for video
python step4_mujoco_hand_control.py --save_frames

# Single task
python step4_mujoco_hand_control.py --task pick_up_mug
```

**Available tasks**: pick_up_mug, precision_pick_coin, write_with_pen,
turn_key, pour_water

**Output**: `simulation_output/*/trajectory.json`

---

### Step 5: Imitation Learning

**What it does**: Trains a neural network to replicate your demonstrations.
The robot learns to perform actions autonomously.

```bash
python step5_imitation_learning.py --epochs 100

# Train on specific task
python step5_imitation_learning.py --task pick_up_mug --epochs 200
```

**Output**: `imitation_policy.pt`

**How it works**:
1. Load (observation, action) pairs from Step 4 trajectories
2. Train neural network: observation → action (supervised learning)
3. Test: given new observation, predict correct action
4. Evaluate: how closely does robot replicate the demonstration?

---

### Step 6: Full Pipeline

**What it does**: Runs the complete pipeline end-to-end.

```bash
# All tasks, demo mode
python step6_full_pipeline.py

# Specific task with your model
python step6_full_pipeline.py --checkpoint best_model_v4.pt --task turn_key
```

---

## FEEL Paper Connection

This entire pipeline demonstrates what the FEEL paper (Section 5)
recommended but couldn't achieve:

| Feature | FEEL (Force Gloves) | EMG-ACT (This Project) |
|---------|--------------------|-----------------------|
| Sensing | Force pads on fingertips | EMG on wrist |
| Obstructive? | YES (blocks fingers) | NO (wrist-only) |
| Contact detection | YES (force threshold) | YES (zone activation) |
| Pre-contact | NO (needs physical touch) | YES (muscle before contact) |
| Object recognition | NO | YES (grasp pattern → object) |
| Action recognition | Temporal contact only | Full action sequences |
| Robot control | NO | YES (MuJoCo hand control) |
| Labeling | Force calibration | Voice commands (Whisper) |
| Scalability | Expensive gloves | Cheap armband + microphone |

---

## Presentation Slides (May 12th)

Suggested order:
1. FEEL paper problem (force gloves obstruct fingers)
2. Our solution (wrist EMG, 84.5% F1)
3. Zone → Object recognition (Step 2 results)
4. Zone → Action recognition (Step 2 temporal)
5. MuJoCo hand demo (Step 4 trajectories)
6. Imitation learning (Step 5 — teach once, robot repeats)
7. Full pipeline diagram (Step 6)
8. Comparison table (FEEL vs EMG-ACT)
9. Future directions (multi-subject, real-time, CVPR)

---

## Troubleshooting

### MuJoCo won't install
```bash
# Try specific version
pip install mujoco==3.1.1

# If still failing, the pipeline works without MuJoCo
# Step 4 has a headless fallback mode
```

### MyoSuite won't install
```bash
# Requires Python 3.8-3.11
pip install myosuite gymnasium

# If failing, Step 3 automatically uses fallback synthetic data
```

### No GPU
Everything runs on CPU. No GPU required.

### "No checkpoint found"
This is fine — the pipeline runs in demo mode with random weights.
Copy your best_model_v4.pt from the cluster to use real predictions.

```bash
# From cluster
scp adnan16@umiacs.umd.edu:/nfshomes/adnan16/emg_project/files/checkpoints_v4/best_model_v4.pt .

# Then run
python run_all.py --checkpoint best_model_v4.pt
```

---

## Dependencies

| Package | Version | Required For | Install |
|---------|---------|-------------|---------|
| torch | ≥2.0 | All steps | pip install torch |
| numpy | ≥1.24 | All steps | pip install numpy |
| scipy | ≥1.10 | Step 1 (filtering) | pip install scipy |
| scikit-learn | ≥1.2 | Step 5 (train/test split) | pip install scikit-learn |
| matplotlib | ≥3.7 | Visualization | pip install matplotlib |
| pandas | ≥2.0 | Data loading | pip install pandas |
| mujoco | ≥3.0 | Step 4 (hand sim) | pip install mujoco |
| myosuite | ≥2.0 | Step 3 (synthetic) | pip install myosuite |
| gymnasium | ≥0.29 | Step 3 (MyoSuite) | pip install gymnasium |

**Minimum install** (Steps 1, 2, 5, 6 work):
```bash
pip install torch numpy scipy scikit-learn matplotlib pandas
```

**Full install** (all steps):
```bash
pip install torch numpy scipy scikit-learn matplotlib pandas mujoco myosuite gymnasium
```

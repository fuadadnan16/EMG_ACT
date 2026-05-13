"""
STEP 4: MuJoCo Shadow Hand Control from Zone Predictions
==========================================================
This script:
    1. Creates a Shadow Hand model in MuJoCo
    2. Maps your 10 zone predictions → joint angles
    3. Animates the hand performing different grasps
    4. Records the trajectory for imitation learning (Step 5)
    5. Saves frames for a presentation video

PREREQUISITES:
    pip install mujoco numpy matplotlib

USAGE:
    python step4_mujoco_hand_control.py
    python step4_mujoco_hand_control.py --visualize   # Opens 3D viewer
    python step4_mujoco_hand_control.py --save_video  # Saves frames as images

NOTE: If you don't have the Shadow Hand XML, this script includes a
      simplified hand model that works the same way.
"""

import numpy as np
import os
import json
import argparse

ZONE_NAMES = [
    "palm", "thumb", "index_tip", "index_seg",
    "middle_tip", "middle_seg", "ring_tip", "ring_seg",
    "pinky_tip", "pinky_seg",
]


# ─── Simplified Hand Model (no external XML needed) ───

SIMPLE_HAND_XML = """
<mujoco model="simple_hand">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  
  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom type="capsule" contype="1" conaffinity="1" 
          friction="1 0.005 0.0001" rgba="0.8 0.7 0.6 1"/>
  </default>
  
  <worldbody>
    <!-- Ground plane -->
    <geom type="plane" size="0.5 0.5 0.01" rgba="0.3 0.3 0.3 1"/>
    
    <!-- Objects to manipulate -->
    <body name="mug" pos="0.15 0 0.05">
      <joint type="free"/>
      <geom type="cylinder" size="0.03 0.04" rgba="0.2 0.5 0.8 1" mass="0.3"/>
    </body>
    
    <body name="ball" pos="-0.15 0 0.03">
      <joint type="free"/>
      <geom type="sphere" size="0.025" rgba="0.8 0.2 0.2 1" mass="0.1"/>
    </body>
    
    <body name="key" pos="0 0.15 0.01">
      <joint type="free"/>
      <geom type="box" size="0.03 0.005 0.001" rgba="0.8 0.8 0.2 1" mass="0.02"/>
    </body>
    
    <!-- Palm -->
    <body name="palm" pos="0 0 0.2">
      <joint name="wrist_flex" type="hinge" axis="1 0 0" range="-0.5 0.5"/>
      <joint name="wrist_dev" type="hinge" axis="0 1 0" range="-0.3 0.3"/>
      <geom type="box" size="0.04 0.05 0.01" rgba="0.8 0.7 0.6 1"/>
      
      <!-- Thumb -->
      <body name="thumb_base" pos="-0.04 -0.03 0.01">
        <joint name="thumb_abd" type="hinge" axis="0 0 1" range="-0.5 1.2"/>
        <geom fromto="0 0 0 0.02 -0.02 0" size="0.008"/>
        
        <body name="thumb_mid" pos="0.02 -0.02 0">
          <joint name="thumb_mcp" type="hinge" axis="1 0 0" range="0 1.4"/>
          <geom fromto="0 0 0 0.02 -0.015 0" size="0.007"/>
          
          <body name="thumb_tip" pos="0.02 -0.015 0">
            <joint name="thumb_ip" type="hinge" axis="1 0 0" range="0 1.2"/>
            <geom fromto="0 0 0 0.015 -0.01 0" size="0.006"/>
          </body>
        </body>
      </body>
      
      <!-- Index Finger -->
      <body name="index_base" pos="-0.02 0.05 0.01">
        <joint name="index_abd" type="hinge" axis="0 0 1" range="-0.3 0.3"/>
        <geom fromto="0 0 0 0 0.025 0" size="0.007"/>
        
        <body name="index_mid" pos="0 0.025 0">
          <joint name="index_mcp" type="hinge" axis="1 0 0" range="0 1.6"/>
          <geom fromto="0 0 0 0 0.02 0" size="0.006"/>
          
          <body name="index_pip" pos="0 0.02 0">
            <joint name="index_pip_j" type="hinge" axis="1 0 0" range="0 1.6"/>
            <geom fromto="0 0 0 0 0.018 0" size="0.005"/>
            
            <body name="index_tip" pos="0 0.018 0">
              <joint name="index_dip" type="hinge" axis="1 0 0" range="0 1.2"/>
              <geom fromto="0 0 0 0 0.012 0" size="0.005"/>
            </body>
          </body>
        </body>
      </body>
      
      <!-- Middle Finger -->
      <body name="middle_base" pos="0 0.05 0.01">
        <joint name="middle_abd" type="hinge" axis="0 0 1" range="-0.3 0.3"/>
        <geom fromto="0 0 0 0 0.028 0" size="0.007"/>
        
        <body name="middle_mid" pos="0 0.028 0">
          <joint name="middle_mcp" type="hinge" axis="1 0 0" range="0 1.6"/>
          <geom fromto="0 0 0 0 0.022 0" size="0.006"/>
          
          <body name="middle_pip" pos="0 0.022 0">
            <joint name="middle_pip_j" type="hinge" axis="1 0 0" range="0 1.6"/>
            <geom fromto="0 0 0 0 0.02 0" size="0.005"/>
            
            <body name="middle_tip" pos="0 0.02 0">
              <joint name="middle_dip" type="hinge" axis="1 0 0" range="0 1.2"/>
              <geom fromto="0 0 0 0 0.014 0" size="0.005"/>
            </body>
          </body>
        </body>
      </body>
      
      <!-- Ring Finger -->
      <body name="ring_base" pos="0.02 0.05 0.01">
        <joint name="ring_abd" type="hinge" axis="0 0 1" range="-0.3 0.3"/>
        <geom fromto="0 0 0 0 0.025 0" size="0.006"/>
        
        <body name="ring_mid" pos="0 0.025 0">
          <joint name="ring_mcp" type="hinge" axis="1 0 0" range="0 1.6"/>
          <geom fromto="0 0 0 0 0.02 0" size="0.005"/>
          
          <body name="ring_pip" pos="0 0.02 0">
            <joint name="ring_pip_j" type="hinge" axis="1 0 0" range="0 1.6"/>
            <geom fromto="0 0 0 0 0.018 0" size="0.005"/>
            
            <body name="ring_tip" pos="0 0.018 0">
              <joint name="ring_dip" type="hinge" axis="1 0 0" range="0 1.2"/>
              <geom fromto="0 0 0 0 0.012 0" size="0.005"/>
            </body>
          </body>
        </body>
      </body>
      
      <!-- Pinky Finger -->
      <body name="pinky_base" pos="0.035 0.045 0.01">
        <joint name="pinky_abd" type="hinge" axis="0 0 1" range="-0.3 0.3"/>
        <geom fromto="0 0 0 0 0.02 0" size="0.005"/>
        
        <body name="pinky_mid" pos="0 0.02 0">
          <joint name="pinky_mcp" type="hinge" axis="1 0 0" range="0 1.6"/>
          <geom fromto="0 0 0 0 0.016 0" size="0.005"/>
          
          <body name="pinky_pip" pos="0 0.016 0">
            <joint name="pinky_pip_j" type="hinge" axis="1 0 0" range="0 1.6"/>
            <geom fromto="0 0 0 0 0.014 0" size="0.004"/>
            
            <body name="pinky_tip" pos="0 0.014 0">
              <joint name="pinky_dip" type="hinge" axis="1 0 0" range="0 1.2"/>
              <geom fromto="0 0 0 0 0.01 0" size="0.004"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <position name="a_wrist_flex" joint="wrist_flex" kp="5"/>
    <position name="a_wrist_dev" joint="wrist_dev" kp="5"/>
    <position name="a_thumb_abd" joint="thumb_abd" kp="3"/>
    <position name="a_thumb_mcp" joint="thumb_mcp" kp="3"/>
    <position name="a_thumb_ip" joint="thumb_ip" kp="3"/>
    <position name="a_index_mcp" joint="index_mcp" kp="3"/>
    <position name="a_index_pip" joint="index_pip_j" kp="3"/>
    <position name="a_index_dip" joint="index_dip" kp="3"/>
    <position name="a_middle_mcp" joint="middle_mcp" kp="3"/>
    <position name="a_middle_pip" joint="middle_pip_j" kp="3"/>
    <position name="a_middle_dip" joint="middle_dip" kp="3"/>
    <position name="a_ring_mcp" joint="ring_mcp" kp="3"/>
    <position name="a_ring_pip" joint="ring_pip_j" kp="3"/>
    <position name="a_ring_dip" joint="ring_dip" kp="3"/>
    <position name="a_pinky_mcp" joint="pinky_mcp" kp="3"/>
    <position name="a_pinky_pip" joint="pinky_pip_j" kp="3"/>
    <position name="a_pinky_dip" joint="pinky_dip" kp="3"/>
  </actuator>
</mujoco>
"""

# Actuator names in the model
ACTUATOR_NAMES = [
    "a_wrist_flex", "a_wrist_dev",
    "a_thumb_abd", "a_thumb_mcp", "a_thumb_ip",
    "a_index_mcp", "a_index_pip", "a_index_dip",
    "a_middle_mcp", "a_middle_pip", "a_middle_dip",
    "a_ring_mcp", "a_ring_pip", "a_ring_dip",
    "a_pinky_mcp", "a_pinky_pip", "a_pinky_dip",
]

# Zone → Actuator mapping
# When a zone is active, which actuators engage and by how much
ZONE_ACTUATOR_MAP = {
    "palm":       {"a_wrist_flex": 0.3},
    "thumb":      {"a_thumb_abd": 0.8, "a_thumb_mcp": 1.0, "a_thumb_ip": 0.8},
    "index_tip":  {"a_index_dip": 1.0, "a_index_pip": 0.8},
    "index_seg":  {"a_index_mcp": 0.7},
    "middle_tip": {"a_middle_dip": 1.0, "a_middle_pip": 0.8},
    "middle_seg": {"a_middle_mcp": 0.7},
    "ring_tip":   {"a_ring_dip": 1.0, "a_ring_pip": 0.8},
    "ring_seg":   {"a_ring_mcp": 0.7},
    "pinky_tip":  {"a_pinky_dip": 1.0, "a_pinky_pip": 0.8},
    "pinky_seg":  {"a_pinky_mcp": 0.7},
}


def zones_to_actuators(zone_binary):
    """
    Convert 10-zone binary predictions to MuJoCo actuator targets.
    
    Args:
        zone_binary: list/array of 10 binary values
    Returns:
        numpy array of shape (17,) with actuator target positions
    """
    targets = np.zeros(len(ACTUATOR_NAMES))

    for zone_idx, (zone_name, active) in enumerate(zip(ZONE_NAMES, zone_binary)):
        if active:
            zone_map = ZONE_ACTUATOR_MAP.get(zone_name, {})
            for act_name, value in zone_map.items():
                act_idx = ACTUATOR_NAMES.index(act_name)
                targets[act_idx] = max(targets[act_idx], value)

    return targets


def run_hand_simulation(action_sequence, save_dir="simulation_output",
                        visualize=False, save_frames=False):
    """
    Run the MuJoCo hand simulation with a sequence of zone activations.
    
    Args:
        action_sequence: list of (zone_binary, duration_steps, label) tuples
        save_dir: where to save frames and trajectory
        visualize: whether to open 3D viewer
        save_frames: whether to save rendered frames
    Returns:
        trajectory: list of recorded states for imitation learning
    """
    try:
        import mujoco
    except ImportError:
        print("MuJoCo not installed. Install with: pip install mujoco")
        print("Running in headless mode with trajectory recording only...")
        return run_headless_simulation(action_sequence, save_dir)

    os.makedirs(save_dir, exist_ok=True)

    # Create model from XML string
    model = mujoco.MjModel.from_xml_string(SIMPLE_HAND_XML)
    data = mujoco.MjData(model)

    trajectory = []
    frame_idx = 0

    print(f"\n  Running simulation ({len(action_sequence)} phases)...")

    for phase_idx, (zones, duration, label) in enumerate(action_sequence):
        targets = zones_to_actuators(zones)
        active = [ZONE_NAMES[i] for i, v in enumerate(zones) if v]
        print(f"  Phase {phase_idx+1}: {label} "
              f"(zones: {active or ['rest']}, {duration} steps)")

        for step in range(duration):
            # Set actuator targets
            data.ctrl[:] = targets

            # Step physics
            mujoco.mj_step(model, data)

            # Record trajectory
            state = {
                "phase": phase_idx,
                "step": step,
                "label": label,
                "zones": zones.tolist() if hasattr(zones, 'tolist') else list(zones),
                "actuator_targets": targets.tolist(),
                "qpos": data.qpos.copy().tolist(),
                "qvel": data.qvel.copy().tolist(),
            }
            trajectory.append(state)

            # Save frame for video
            if save_frames and step % 10 == 0:
                try:
                    renderer = mujoco.Renderer(model, 480, 640)
                    renderer.update_scene(data)
                    frame = renderer.render()
                    import matplotlib.pyplot as plt
                    plt.imsave(
                        os.path.join(save_dir, f"frame_{frame_idx:04d}.png"),
                        frame
                    )
                    renderer.close()
                except:
                    pass  # Rendering might not work in headless mode
                frame_idx += 1

    # Open interactive viewer if requested
    if visualize:
        print("\n  Opening 3D viewer (close window to continue)...")
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            # Replay the trajectory
            for state in trajectory[::5]:  # Every 5th state for speed
                data.qpos[:] = state["qpos"]
                mujoco.mj_forward(model, data)
                viewer.sync()
                import time
                time.sleep(0.01)
            viewer.close()
        except Exception as e:
            print(f"  Viewer failed (headless environment?): {e}")

    # Save trajectory
    traj_path = os.path.join(save_dir, "trajectory.json")
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"\n  Trajectory saved: {traj_path} ({len(trajectory)} states)")

    return trajectory


def run_headless_simulation(action_sequence, save_dir="simulation_output"):
    """
    Record trajectory without MuJoCo (pure computation).
    Used when MuJoCo is not available.
    """
    os.makedirs(save_dir, exist_ok=True)
    trajectory = []

    for phase_idx, (zones, duration, label) in enumerate(action_sequence):
        targets = zones_to_actuators(zones)
        active = [ZONE_NAMES[i] for i, v in enumerate(zones) if v]
        print(f"  Phase {phase_idx+1}: {label} "
              f"(zones: {active or ['rest']}, {duration} steps)")

        for step in range(duration):
            # Simulate smooth interpolation
            progress = step / max(duration - 1, 1)
            current_targets = targets * min(progress * 2, 1.0)

            state = {
                "phase": phase_idx,
                "step": step,
                "label": label,
                "zones": list(zones),
                "actuator_targets": current_targets.tolist(),
                "qpos": current_targets.tolist(),  # Simplified
                "qvel": (np.zeros_like(targets)).tolist(),
            }
            trajectory.append(state)

    traj_path = os.path.join(save_dir, "trajectory.json")
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"\n  Trajectory saved: {traj_path} ({len(trajectory)} states)")

    return trajectory


# ─── Pre-defined Action Sequences ───

def get_demo_sequences():
    """Return pre-defined action sequences for common tasks."""
    return {
        "pick_up_mug": [
            (np.array([0,0,0,0,0,0,0,0,0,0]), 50, "rest"),
            (np.array([0,1,1,0,0,0,0,0,0,0]), 30, "approach"),
            (np.array([0,1,1,0,1,0,0,0,0,0]), 20, "pre_grasp"),
            (np.array([1,1,1,1,1,1,1,1,0,0]), 30, "grasp"),
            (np.array([1,1,1,1,1,1,1,1,1,1]), 50, "full_grasp"),
            (np.array([1,1,1,1,1,1,1,1,1,1]), 30, "lift"),
        ],
        "precision_pick_coin": [
            (np.array([0,0,0,0,0,0,0,0,0,0]), 50, "rest"),
            (np.array([0,1,0,0,0,0,0,0,0,0]), 30, "thumb_extend"),
            (np.array([0,1,1,0,0,0,0,0,0,0]), 40, "pinch"),
            (np.array([0,1,1,0,0,0,0,0,0,0]), 50, "hold"),
            (np.array([0,1,1,0,0,0,0,0,0,0]), 30, "lift"),
        ],
        "write_with_pen": [
            (np.array([0,0,0,0,0,0,0,0,0,0]), 30, "rest"),
            (np.array([0,1,1,0,1,0,0,0,0,0]), 40, "tripod_grip"),
            (np.array([0,1,1,0,1,0,0,0,0,0]), 60, "writing"),
            (np.array([0,1,1,0,0,0,0,0,0,0]), 20, "lift_pen"),
            (np.array([0,1,1,0,1,0,0,0,0,0]), 60, "writing_more"),
        ],
        "turn_key": [
            (np.array([0,0,0,0,0,0,0,0,0,0]), 30, "rest"),
            (np.array([0,1,0,1,0,0,0,0,0,0]), 40, "lateral_pinch"),
            (np.array([0,1,1,1,0,0,0,0,0,0]), 30, "rotate"),
            (np.array([0,1,0,1,0,0,0,0,0,0]), 30, "back"),
            (np.array([0,0,0,0,0,0,0,0,0,0]), 30, "release"),
        ],
        "pour_water": [
            (np.array([1,1,1,1,1,1,1,1,1,1]), 50, "grasp_bottle"),
            (np.array([1,1,1,1,1,1,0,0,0,0]), 40, "tilt"),
            (np.array([1,1,1,1,1,1,0,0,0,0]), 60, "pouring"),
            (np.array([1,1,1,1,1,1,1,1,1,1]), 40, "upright"),
        ],
    }


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Step 4: MuJoCo Hand Control")
    parser.add_argument("--visualize", action="store_true",
                        help="Open 3D viewer")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save rendered frames")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "pick_up_mug", "precision_pick_coin",
                                 "write_with_pen", "turn_key", "pour_water"])
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 4: MuJoCo Hand Control from Zone Predictions")
    print("=" * 60)

    demos = get_demo_sequences()
    all_trajectories = {}

    tasks = list(demos.keys()) if args.task == "all" else [args.task]

    for task_name in tasks:
        print(f"\n{'─' * 60}")
        print(f"  Task: {task_name}")
        print(f"{'─' * 60}")

        trajectory = run_hand_simulation(
            action_sequence=demos[task_name],
            save_dir=f"simulation_output/{task_name}",
            visualize=args.visualize,
            save_frames=args.save_frames,
        )
        all_trajectories[task_name] = trajectory

    # Save all trajectories for imitation learning
    combined_path = "simulation_output/all_demonstrations.json"
    os.makedirs("simulation_output", exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_trajectories, f)
    print(f"\n  All demonstrations saved: {combined_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    for task, traj in all_trajectories.items():
        phases = set(s["label"] for s in traj)
        print(f"  {task}: {len(traj)} states, {len(phases)} phases")

    print(f"\n{'=' * 60}")
    print(f"  Step 4 COMPLETE — Hand simulations recorded!")
    print(f"{'=' * 60}")
    print(f"\n  Next: Run step5_imitation_learning.py")


if __name__ == "__main__":
    main()

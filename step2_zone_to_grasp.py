"""
STEP 2: Object & Action Recognition from Zone Patterns
========================================================
This script maps zone activation patterns to:
    1. GRASP TYPE (power, precision, tripod, lateral, hook, etc.)
    2. OBJECT being held (mug, pen, key, coin, etc.)
    3. ACTION being performed (grasp, release, rotate, pour, etc.)

WHAT YOU NEED:
    - step1_load_emg_model.py (the EMGPredictor class)

HOW IT WORKS:
    - Static recognition: Single zone pattern → grasp type → object
    - Temporal recognition: Sequence of zone patterns → action

USAGE:
    python step2_zone_to_grasp.py --demo
"""

import numpy as np
from collections import deque
import time


# ─── GRASP TAXONOMY ───
# Based on Cutkosky (1989) and Feix (2016) grasp taxonomies
# Maps zone activation patterns to grasp types and likely objects

GRASP_DATABASE = {
    # ── Power Grasps (whole hand wraps around object) ──
    "power_grasp": {
        "zones": {"palm": 1, "thumb": 1, "index_tip": 1, "index_seg": 1,
                  "middle_tip": 1, "middle_seg": 1, "ring_tip": 1, "ring_seg": 1,
                  "pinky_tip": 1, "pinky_seg": 1},
        "objects": ["mug", "bottle", "can", "hammer handle", "railing"],
        "description": "Full hand wrap around cylindrical/large object",
        "force_level": "high",
    },
    "power_grasp_no_pinky": {
        "zones": {"palm": 1, "thumb": 1, "index_tip": 1, "index_seg": 1,
                  "middle_tip": 1, "middle_seg": 1, "ring_tip": 1, "ring_seg": 1,
                  "pinky_tip": 0, "pinky_seg": 0},
        "objects": ["large mug", "jar", "bottle"],
        "description": "Power grasp with pinky relaxed (common for larger objects)",
        "force_level": "high",
    },

    # ── Precision Grasps (fingertips only) ──
    "precision_pinch": {
        "zones": {"palm": 0, "thumb": 1, "index_tip": 1, "index_seg": 0,
                  "middle_tip": 0, "middle_seg": 0, "ring_tip": 0, "ring_seg": 0,
                  "pinky_tip": 0, "pinky_seg": 0},
        "objects": ["coin", "pill", "small bead", "needle", "pin"],
        "description": "Thumb and index fingertip pinch",
        "force_level": "low",
    },
    "tripod_grip": {
        "zones": {"palm": 0, "thumb": 1, "index_tip": 1, "index_seg": 0,
                  "middle_tip": 1, "middle_seg": 0, "ring_tip": 0, "ring_seg": 0,
                  "pinky_tip": 0, "pinky_seg": 0},
        "objects": ["pen", "pencil", "marker", "stylus", "chopstick"],
        "description": "Three-finger writing grip",
        "force_level": "low",
    },

    # ── Lateral Grasps ──
    "lateral_pinch": {
        "zones": {"palm": 0, "thumb": 1, "index_tip": 0, "index_seg": 1,
                  "middle_tip": 0, "middle_seg": 0, "ring_tip": 0, "ring_seg": 0,
                  "pinky_tip": 0, "pinky_seg": 0},
        "objects": ["key", "credit card", "paper sheet", "coin (flat hold)"],
        "description": "Thumb against side of index finger",
        "force_level": "medium",
    },

    # ── Hook Grasps ──
    "hook_grasp": {
        "zones": {"palm": 0, "thumb": 0, "index_tip": 1, "index_seg": 1,
                  "middle_tip": 1, "middle_seg": 1, "ring_tip": 1, "ring_seg": 1,
                  "pinky_tip": 0, "pinky_seg": 0},
        "objects": ["bag handle", "bucket", "briefcase", "door handle"],
        "description": "Fingers curled without thumb (carrying)",
        "force_level": "medium",
    },

    # ── Spherical Grasps ──
    "spherical_grasp": {
        "zones": {"palm": 1, "thumb": 1, "index_tip": 1, "index_seg": 0,
                  "middle_tip": 1, "middle_seg": 0, "ring_tip": 1, "ring_seg": 0,
                  "pinky_tip": 1, "pinky_seg": 0},
        "objects": ["ball", "apple", "orange", "doorknob", "lightbulb"],
        "description": "Fingertips spread around spherical object",
        "force_level": "medium",
    },

    # ── Disc Grasp ──
    "disc_grasp": {
        "zones": {"palm": 1, "thumb": 1, "index_tip": 1, "index_seg": 1,
                  "middle_tip": 1, "middle_seg": 1, "ring_tip": 0, "ring_seg": 0,
                  "pinky_tip": 0, "pinky_seg": 0},
        "objects": ["jar lid", "bottle cap", "plate", "frisbee"],
        "description": "Thumb and first two fingers around flat round object",
        "force_level": "medium",
    },

    # ── Platform Grasp ──
    "platform_grasp": {
        "zones": {"palm": 1, "thumb": 0, "index_tip": 0, "index_seg": 1,
                  "middle_tip": 0, "middle_seg": 1, "ring_tip": 0, "ring_seg": 1,
                  "pinky_tip": 0, "pinky_seg": 1},
        "objects": ["plate", "tray", "book (flat)", "pizza box"],
        "description": "Palm and finger segments support from below",
        "force_level": "low",
    },

    # ── Rest State ──
    "rest": {
        "zones": {"palm": 0, "thumb": 0, "index_tip": 0, "index_seg": 0,
                  "middle_tip": 0, "middle_seg": 0, "ring_tip": 0, "ring_seg": 0,
                  "pinky_tip": 0, "pinky_seg": 0},
        "objects": ["nothing (hand open/relaxed)"],
        "description": "No active zones — hand at rest",
        "force_level": "none",
    },
}

ZONE_NAMES = [
    "palm", "thumb", "index_tip", "index_seg",
    "middle_tip", "middle_seg", "ring_tip", "ring_seg",
    "pinky_tip", "pinky_seg",
]


# ─── Object Recognition ───

class ObjectRecognizer:
    """Recognize objects from zone activation patterns using grasp taxonomy."""

    def __init__(self):
        # Pre-compute pattern vectors for fast matching
        self.grasp_vectors = {}
        for name, info in GRASP_DATABASE.items():
            vec = np.array([info["zones"][z] for z in ZONE_NAMES], dtype=float)
            self.grasp_vectors[name] = vec

    def recognize(self, zone_binary):
        """
        Recognize grasp type and likely object from zone pattern.
        
        Args:
            zone_binary: numpy array of shape (10,) with binary zone values
        Returns:
            dict with grasp_type, objects, confidence, description
        """
        zone_binary = np.array(zone_binary, dtype=float)
        best_grasp = None
        best_dist = float("inf")

        for name, vec in self.grasp_vectors.items():
            dist = np.abs(vec - zone_binary).sum()  # Hamming distance
            if dist < best_dist:
                best_dist = dist
                best_grasp = name

        info = GRASP_DATABASE[best_grasp]
        confidence = max(0, 1.0 - best_dist / 10.0)  # Normalize to 0-1

        return {
            "grasp_type": best_grasp,
            "objects": info["objects"],
            "most_likely_object": info["objects"][0],
            "confidence": confidence,
            "description": info["description"],
            "force_level": info["force_level"],
            "hamming_distance": int(best_dist),
        }

    def recognize_top_k(self, zone_binary, k=3):
        """Return top-k grasp matches."""
        zone_binary = np.array(zone_binary, dtype=float)
        results = []
        for name, vec in self.grasp_vectors.items():
            dist = np.abs(vec - zone_binary).sum()
            info = GRASP_DATABASE[name]
            results.append({
                "grasp_type": name,
                "objects": info["objects"],
                "confidence": max(0, 1.0 - dist / 10.0),
                "hamming_distance": int(dist),
            })
        results.sort(key=lambda x: x["hamming_distance"])
        return results[:k]


# ─── Action Recognition ───

# Actions are defined as temporal sequences of grasp transitions
ACTION_TEMPLATES = {
    "pick_up": {
        "sequence": ["rest", "approach", "grasp", "lift"],
        "zone_transitions": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # rest: hand open
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # approach: fingers extending
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # grasp: full closure
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # lift: maintain grasp
        ],
        "description": "Reach for object, close hand around it, lift",
    },
    "put_down": {
        "sequence": ["hold", "lower", "release", "retract"],
        "zone_transitions": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # hold: grasping
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # lower: still grasping
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # release: fingers opening
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # retract: hand open
        ],
        "description": "Lower object, release grasp, retract hand",
    },
    "twist_key": {
        "sequence": ["pinch", "rotate_cw", "rotate_ccw", "release"],
        "zone_transitions": [
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # lateral pinch
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # rotate: index tip engages
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # back to pinch
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # release
        ],
        "description": "Pinch key between thumb and index, rotate",
    },
    "write": {
        "sequence": ["grip_pen", "write_stroke", "lift_pen", "write_stroke"],
        "zone_transitions": [
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # tripod grip
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # writing (same grip, slight pressure)
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # lift (middle relaxes)
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # back to writing
        ],
        "description": "Hold pen in tripod, write, lift, write again",
    },
    "pour": {
        "sequence": ["grasp_container", "tilt", "pour_steady", "upright"],
        "zone_transitions": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # power grasp on container
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # tilt: ring/pinky relax
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # pour steady
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # back to full grasp
        ],
        "description": "Grasp container, tilt to pour, return upright",
    },
    "open_jar": {
        "sequence": ["hold_jar", "grip_lid", "twist_lid", "remove_lid"],
        "zone_transitions": [
            [1, 0, 0, 1, 0, 1, 0, 1, 0, 1],  # platform hold (jar body)
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # disc grasp (lid)
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # twist
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # lift lid off
        ],
        "description": "Hold jar, grip lid, twist open, remove",
    },
}


class ActionRecognizer:
    """
    Recognize actions from temporal sequences of zone activations.
    Uses Dynamic Time Warping (DTW)-like matching.
    """

    def __init__(self, buffer_size=20):
        self.buffer = deque(maxlen=buffer_size)
        self.templates = ACTION_TEMPLATES

    def add_observation(self, zone_binary):
        """Add a zone activation observation to the temporal buffer."""
        self.buffer.append(np.array(zone_binary, dtype=float))

    def recognize(self):
        """
        Recognize the current action from the buffered sequence.
        
        Returns:
            dict with action, confidence, description, phase
        """
        if len(self.buffer) < 4:
            return {"action": "unknown", "confidence": 0.0,
                    "description": "Not enough observations", "phase": "waiting"}

        # Sample 4 evenly-spaced frames from buffer
        indices = np.linspace(0, len(self.buffer) - 1, 4, dtype=int)
        observed = [self.buffer[i] for i in indices]

        best_action = None
        best_score = float("inf")

        for name, template in self.templates.items():
            expected = template["zone_transitions"]
            # Compute total Hamming distance across 4 phases
            score = sum(
                np.abs(np.array(obs) - np.array(exp)).sum()
                for obs, exp in zip(observed, expected)
            )
            if score < best_score:
                best_score = score
                best_action = name

        # Determine current phase
        template = self.templates[best_action]
        last_obs = self.buffer[-1]
        phase_dists = [
            np.abs(last_obs - np.array(exp)).sum()
            for exp in template["zone_transitions"]
        ]
        current_phase_idx = np.argmin(phase_dists)
        current_phase = template["sequence"][current_phase_idx]

        max_possible_score = 4 * 10  # 4 phases × 10 zones
        confidence = max(0, 1.0 - best_score / max_possible_score)

        return {
            "action": best_action,
            "confidence": confidence,
            "description": template["description"],
            "phase": current_phase,
            "phase_index": current_phase_idx,
            "score": best_score,
        }

    def clear(self):
        """Clear the observation buffer."""
        self.buffer.clear()


# ─── Main Demo ───

def main():
    print("=" * 60)
    print("  STEP 2: Object & Action Recognition from Zone Patterns")
    print("=" * 60)

    obj_recognizer = ObjectRecognizer()
    action_recognizer = ActionRecognizer()

    # ── Part A: Object Recognition ──
    print("\n" + "─" * 60)
    print("  PART A: Object Recognition (Static Patterns)")
    print("─" * 60)

    test_patterns = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "all zones active"),
        ([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], "thumb + index_tip"),
        ([0, 1, 1, 0, 1, 0, 0, 0, 0, 0], "thumb + index_tip + middle_tip"),
        ([0, 1, 0, 1, 0, 0, 0, 0, 0, 0], "thumb + index_seg"),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "no zones"),
        ([1, 1, 1, 0, 1, 0, 1, 0, 1, 0], "palm + tips only"),
        ([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], "fingers without thumb"),
    ]

    for pattern, label in test_patterns:
        result = obj_recognizer.recognize(pattern)
        print(f"\n  Input: {label}")
        print(f"  Grasp: {result['grasp_type']} (conf={result['confidence']:.0%})")
        print(f"  Object: {result['most_likely_object']}")
        print(f"  Force: {result['force_level']}")

    # ── Part B: Action Recognition ──
    print("\n" + "─" * 60)
    print("  PART B: Action Recognition (Temporal Sequences)")
    print("─" * 60)

    # Simulate a "pick up" action sequence
    print("\n  Simulating: PICK UP a mug")
    pick_up_sequence = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # rest
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # still resting
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # reaching
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # more fingers engaging
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # closing
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # full grasp
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # lifting
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # holding
    ]

    action_recognizer.clear()
    for i, zones in enumerate(pick_up_sequence):
        action_recognizer.add_observation(zones)
        if i >= 3:  # Need at least 4 observations
            result = action_recognizer.recognize()
            obj_result = obj_recognizer.recognize(zones)
            active = [ZONE_NAMES[j] for j, v in enumerate(zones) if v]
            print(f"  t={i}: zones={active or ['rest']}")
            print(f"         action={result['action']} "
                  f"phase={result['phase']} "
                  f"conf={result['confidence']:.0%} "
                  f"object={obj_result['most_likely_object']}")

    # Simulate a "twist key" action sequence
    print("\n  Simulating: TWIST KEY")
    key_sequence = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # rest
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # thumb extending
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # lateral pinch
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # holding key
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # rotating
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # back
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # release
    ]

    action_recognizer.clear()
    for i, zones in enumerate(key_sequence):
        action_recognizer.add_observation(zones)
        if i >= 3:
            result = action_recognizer.recognize()
            active = [ZONE_NAMES[j] for j, v in enumerate(zones) if v]
            print(f"  t={i}: zones={active or ['rest']} → "
                  f"action={result['action']} phase={result['phase']} "
                  f"conf={result['confidence']:.0%}")

    # ── Part C: Summary ──
    print("\n" + "─" * 60)
    print("  SUMMARY: What Your EMG Model Can Recognize")
    print("─" * 60)
    print(f"\n  Grasp types:  {len(GRASP_DATABASE)}")
    print(f"  Object types: {sum(len(g['objects']) for g in GRASP_DATABASE.values())}")
    print(f"  Action types: {len(ACTION_TEMPLATES)}")
    print(f"\n  Grasp taxonomy:")
    for name in GRASP_DATABASE:
        info = GRASP_DATABASE[name]
        print(f"    {name:30s} → {', '.join(info['objects'][:3])}")

    print("\n" + "=" * 60)
    print("  Step 2 COMPLETE — Object & Action Recognition working!")
    print("=" * 60)
    print("\n  Next: Run step3_myosuite_synthetic.py")


if __name__ == "__main__":
    main()

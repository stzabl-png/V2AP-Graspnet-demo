#!/usr/bin/env python3
"""
pipeline/compose.py
====================
Step 2: candidates_local.json + T_base_mesh → candidates.json (V2AP format, base frame).

Frame math:
  Grasps from infer.py are in "base_aligned_z0" frame:
    F_z0 = F_base_aligned + z_shift (shift z_min→0 for GraspNet table)

  T_base_mesh from session corresponds to F_base_aligned:
    T_base_mesh: base_aligned_frame → robot_base_frame

  For the shifted frame:
    T_base_z0 = T_base_mesh @ T_shift_inv
    where T_shift_inv = [[I | 0,0,-z_shift]]  (undo the shift)

  Per-grasp transform:
    T_z0_grasp = [[R | grasp_point]]   (grasp pose in z0 frame)
    T_base_grasp = T_base_z0 @ T_z0_grasp

Output: output/graspnet/candidates.json — V2AP schema 1.1, base frame.
  V2AP retarget.py uses: T_base_pinch = T_base_mesh @ T_mesh_pinch
  We set T_base_mesh = T_base_z0 (stored in candidates.json).
  Then grasp_point/rotation remain in z0 frame (mesh-local).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np


def run_compose(session_dir: Path,
                candidates_local: Path | None = None) -> Path:
    """
    Transform candidates_local.json to V2AP-compatible candidates.json.
    Returns output path.
    """
    session_dir = Path(session_dir)
    base_out    = session_dir / "output" / "graspnet"

    # ── Load candidates_local ──────────────────────────────────
    local_path = candidates_local or (base_out / "candidates_local.json")
    if not local_path.exists():
        raise FileNotFoundError(f"Run infer.py first: {local_path}")
    with open(local_path) as f:
        local = json.load(f)

    z_shift    = float(local["z_shift_m"])
    candidates = local["candidates"]
    print(f"\n{'='*55}")
    print(f"  Compose Grasp Poses → Base Frame")
    print(f"  Session:  {session_dir.name}")
    print(f"  z_shift:  {z_shift*100:.2f}cm")
    print(f"  Input:    {len(candidates)} candidates")

    # ── Load T_base_mesh (base_aligned frame) ─────────────────
    tbm_path = session_dir / "output" / "register" / "T_base_mesh.json"
    if not tbm_path.exists():
        raise FileNotFoundError(
            f"T_base_mesh.json not found: {tbm_path}\n"
            "Make sure the session has completed the FoundationPose registration step."
        )
    with open(tbm_path) as f:
        tbm_data = json.load(f)
    T_base_mesh = np.array(tbm_data["T_base_mesh"], dtype=np.float64)
    print(f"  T_base_mesh translation: "
          f"[{T_base_mesh[0,3]:.3f},{T_base_mesh[1,3]:.3f},{T_base_mesh[2,3]:.3f}]m")

    # ── Compute T_base_z0 ──────────────────────────────────────
    # T_shift moves points from base_aligned frame TO z0 frame: p_z0 = p_ba + [0,0,z_shift]
    # So T_shift = [[I | 0,0,z_shift]]
    # T_base_z0 = T_base_mesh @ T_shift_inv = T_base_mesh @ [[I | 0,0,-z_shift]]
    T_shift_inv        = np.eye(4)
    T_shift_inv[2, 3]  = -z_shift         # undo the z_min shift
    T_base_z0          = T_base_mesh @ T_shift_inv

    print(f"  T_base_z0 translation:  "
          f"[{T_base_z0[0,3]:.3f},{T_base_z0[1,3]:.3f},{T_base_z0[2,3]:.3f}]m")

    # ── Compose each grasp to base frame ──────────────────────
    out_candidates = []
    for c in candidates:
        gp = np.array(c["grasp_point"], dtype=np.float64)   # in z0 frame
        R  = np.array(c["rotation"],   dtype=np.float64)    # A2G: col2=approach

        # Grasp transform in z0 frame
        T_z0_grasp       = np.eye(4)
        T_z0_grasp[:3,:3] = R
        T_z0_grasp[:3, 3] = gp

        # Grasp in base frame
        T_base_grasp     = T_base_z0 @ T_z0_grasp
        gp_base          = T_base_grasp[:3, 3]
        R_base           = T_base_grasp[:3, :3]
        approach_base    = R_base[:, 2]
        pre_grasp_base   = gp_base - approach_base * (0.105 + 0.15)

        out_candidates.append({
            "rank":              c["rank"],
            "score":             c["score"],
            # V2AP uses T_base_mesh @ T_mesh_pinch convention
            # We store the already-composed base-frame pose here
            "grasp_point":       [round(float(x), 6) for x in gp_base],
            "rotation":          [[round(float(x), 6) for x in row]
                                  for row in R_base],
            "gripper_width_m":   c["gripper_width_m"],
            "approach":          [round(float(x), 6) for x in approach_base],
            "pre_grasp_point":   [round(float(x), 6) for x in pre_grasp_base],
            # Also keep mesh-local for V2AP retarget.py if needed
            "grasp_point_local": c["grasp_point"],
            "rotation_local":    c["rotation"],
        })

    # Sanity check: verify top-1 approach makes sense in base frame
    top1 = out_candidates[0]
    app  = np.array(top1["approach"])
    print(f"\n  Top-1 in base frame:")
    print(f"    grasp_point: [{top1['grasp_point'][0]:.3f},{top1['grasp_point'][1]:.3f},{top1['grasp_point'][2]:.3f}]m")
    print(f"    approach:    [{app[0]:.2f},{app[1]:.2f},{app[2]:.2f}]")
    print(f"    pre_grasp:   [{top1['pre_grasp_point'][0]:.3f},{top1['pre_grasp_point'][1]:.3f},{top1['pre_grasp_point'][2]:.3f}]m")

    if top1['pre_grasp_point'][2] < 0:
        print(f"  ⚠️  pre_grasp z<0 ({top1['pre_grasp_point'][2]:.3f}m) — may clip floor")

    # ── Write V2AP candidates.json ─────────────────────────────
    out = {
        "schema_version": "1.1",
        "session_id":     session_dir.name,
        "mesh_frame":     "base_frame",    # Fully transformed to robot base
        "n_candidates":   len(out_candidates),
        "conventions": {
            "rotation_columns":   ["finger_open", "y_body", "approach"],
            "approach_column_index": 2,
            "pre_grasp_offset_m": 0.15,
            "lift_height_m":      0.15,
        },
        # T_base_z0 stored for reference (already baked into grasp_point/rotation)
        "T_base_mesh": [[round(float(x), 6) for x in row]
                        for row in T_base_z0.tolist()],
        "candidates": out_candidates,
    }

    out_path = base_out / "candidates.json"
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    tmp.rename(out_path)

    print(f"\n  ✅ Saved: {out_path}")
    print(f"{'='*55}\n")
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--candidates-local", default=None)
    args = p.parse_args()
    run_compose(Path(args.session),
                Path(args.candidates_local) if args.candidates_local else None)

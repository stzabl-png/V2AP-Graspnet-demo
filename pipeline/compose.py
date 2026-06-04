#!/usr/bin/env python3
"""
pipeline/compose.py
====================
Step 2: candidates_local.json + T_base_mesh → candidates.json (V2AP format).

V2AP retarget.py reads candidates.json like this:
    T_base_mesh = candidates["T_base_mesh"]
    T_base_pinch = T_base_mesh @ [[R | grasp_point]]   ← mesh-local frame

So grasp_point/rotation MUST stay in the MESH-LOCAL frame (z0 frame).
T_base_mesh is the 4x4 that maps z0 → robot base.

Output path: output/inference/candidates.json   (V2AP standard location)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np


def run_compose(session_dir: Path,
                candidates_local: Path | None = None) -> Path:
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
    print(f"  Compose → V2AP candidates.json")
    print(f"  Session:  {session_dir.name}")
    print(f"  z_shift:  {z_shift*100:.2f}cm (z_min shift applied during inference)")
    print(f"  Input:    {len(candidates)} candidates")

    # ── Load T_base_mesh (base_aligned frame) ─────────────────
    tbm_path = session_dir / "output" / "register" / "T_base_mesh.json"
    if not tbm_path.exists():
        raise FileNotFoundError(
            f"T_base_mesh.json not found: {tbm_path}\n"
            "Ensure FoundationPose registration is complete."
        )
    with open(tbm_path) as f:
        tbm_data = json.load(f)
    T_base_aligned = np.array(tbm_data["T_base_mesh"], dtype=np.float64)

    # ── T_base_z0: maps z0 frame → robot base ─────────────────
    # z0 frame = base_aligned frame shifted so z_min → 0
    # z0_pt = base_aligned_pt + [0, 0, z_shift]
    # base_pt = T_base_aligned @ base_aligned_pt
    #         = T_base_aligned @ (z0_pt - [0,0,z_shift])
    #         = (T_base_aligned @ T_shift_inv) @ z0_pt
    # where T_shift_inv = [[I | 0,0,-z_shift]]
    T_shift_inv       = np.eye(4)
    T_shift_inv[2, 3] = -z_shift
    T_base_z0         = T_base_aligned @ T_shift_inv

    print(f"  T_base_aligned t: [{T_base_aligned[0,3]:.3f},{T_base_aligned[1,3]:.3f},{T_base_aligned[2,3]:.3f}]m")
    print(f"  T_base_z0      t: [{T_base_z0[0,3]:.3f},{T_base_z0[1,3]:.3f},{T_base_z0[2,3]:.3f}]m")

    # ── Build output candidates (keep poses in z0/mesh frame) ─
    # V2AP computes: T_base_pinch = T_base_z0 @ [[R | grasp_point]]
    # So we store T_base_mesh = T_base_z0, grasp_point/rotation in z0 frame.
    mesh_span = np.array(local.get("mesh_span_m",
                                    [0.17, 0.12, 0.30]))  # XYZ extents

    # Compute mesh_span from first candidate's bbox if not stored
    # (use approximate can dimensions)
    out_candidates = []
    for c in candidates:
        gp  = [round(float(x), 6) for x in c["grasp_point"]]
        rot = [[round(float(x), 6) for x in row] for row in c["rotation"]]
        app = np.array(c["approach"])

        # Pre-grasp in z0 frame for reference (V2AP computes its own)
        pre_local = (np.array(c["grasp_point"])
                     - app * (0.105 + 0.15))

        out_candidates.append({
            "rank":            c["rank"],
            "name":            f"graspnet_rank{c['rank']}",
            "score":           c["score"],
            "grasp_point":     gp,          # z0 (mesh-local) frame ← V2AP uses this
            "rotation":        rot,          # col0=finger_open, col2=approach (A2G)
            "gripper_width_m": c["gripper_width_m"],
            "approach":        [round(float(x), 6) for x in app],
        })

    # Sanity: show top-1 after V2AP transform
    c0  = out_candidates[0]
    gp0 = np.array(c0["grasp_point"])
    R0  = np.array(c0["rotation"])
    T_z0_g = np.eye(4); T_z0_g[:3,:3] = R0; T_z0_g[:3,3] = gp0
    T_base_g = T_base_z0 @ T_z0_g
    print(f"\n  Top-1 (as V2AP will compute):")
    print(f"    grasp (z0):   [{gp0[0]:.3f},{gp0[1]:.3f},{gp0[2]:.3f}]m")
    print(f"    grasp (base): [{T_base_g[0,3]:.3f},{T_base_g[1,3]:.3f},{T_base_g[2,3]:.3f}]m")
    app_base = T_base_g[:3,2]
    pre_base = T_base_g[:3,3] - app_base * (0.105 + 0.15)
    print(f"    approach (base): [{app_base[0]:.2f},{app_base[1]:.2f},{app_base[2]:.2f}]")
    print(f"    pre_grasp (base): [{pre_base[0]:.3f},{pre_base[1]:.3f},{pre_base[2]:.3f}]m")
    if pre_base[2] < 0.0:
        print(f"  ⚠️  pre_grasp z={pre_base[2]:.3f}m < 0 — may collide with floor")

    # ── Write V2AP candidates.json ─────────────────────────────
    # Standard V2AP path: output/inference/candidates.json
    inference_dir = session_dir / "output" / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    out_path = inference_dir / "candidates.json"

    out = {
        "schema_version": "1.1",
        "session_id":     session_dir.name,
        # T_base_mesh: z0 frame → robot base
        # V2AP: T_base_pinch = T_base_mesh @ [[R | grasp_point]]
        "T_base_mesh": [[round(float(x), 6) for x in row]
                         for row in T_base_z0.tolist()],
        "mesh_span_m": [round(float(x), 4) for x in mesh_span.tolist()],
        "conventions": {
            "rotation_columns":      ["finger_open", "y_body", "approach"],
            "approach_column_index": 2,
            "pre_grasp_offset_m":    0.15,
            "lift_height_m":         0.15,
            "depth_shift_m":         local.get("shift_applied_m", 0.025),
            "note": (
                "grasp_point/rotation are in z0 frame (base_aligned mesh + z_min shift). "
                "T_base_mesh maps z0→base. V2AP computes T_base_pinch = T_base_mesh @ T_z0_pinch."
            ),
        },
        "n_candidates": len(out_candidates),
        "candidates":   out_candidates,
    }

    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    tmp.rename(out_path)

    # Also copy to graspnet/ dir for reference
    import shutil, time as _time
    shutil.copy(out_path, base_out / "candidates.json")

    # V2AP load_titan_output() checks output/status.json for success=true
    status_path = session_dir / "output" / "status.json"
    if not status_path.exists():
        status = {
            "success":    True,
            "session_id": session_dir.name,
            "source":     "graspnet_demo",
            "timestamp":  _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            "titan":      {"object_slug": session_dir.name.split("_")[-1]},
        }
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)
        print(f"  ✅ Created: {status_path}")

    print(f"\n  ✅ Saved (V2AP): {out_path}")
    print(f"  ✅ Saved (ref):  {base_out / 'candidates.json'}")
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

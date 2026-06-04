#!/usr/bin/env python3
"""
pipeline/infer.py
==================
Step 1: Session mesh → GraspNet inference → candidates in mesh-local frame.

Uses object_base_aligned.glb (Z=up, matches robot base orientation).
Shifts mesh so z_min=0 (table at origin) for GraspNet collision detection.
Applies full baseline-equivalent pipeline:
  graspgroup_to_candidates (col convention + z_approach_max=0.3)
  rerank_for_reachability
  2.5cm depth shift (same as make_shifted_graspnet_candidates.py)

Output: candidates_local.json
  grasp_point/rotation are in z0_frame (base_aligned + z_min shift)
  z_shift_m: the z offset applied (needed by compose.py)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "checkpoints" / "checkpoint-rs.tar"
SHIFT_M    = 0.025   # 2.5cm depth shift (matches baseline sim evaluation)
TCP_OFFSET = 0.105

# ── GraspNet dependency paths ──────────────────────────────────────────────
_GN_PATH = REPO / "third_party" / "graspnet-baseline"
if not _GN_PATH.exists():
    raise RuntimeError(
        f"GraspNet not found at {_GN_PATH}\n"
        "Run: git submodule update --init --recursive"
    )
for _p in [str(_GN_PATH), str(_GN_PATH / "models"),
           str(_GN_PATH / "utils"), str(_GN_PATH / "dataset")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_graspnet_model(ckpt: Path):
    import torch
    sys.path.insert(0, str(REPO / "graspnet_compat"))
    from graspnet_compat.graspnet_loader import load_model
    return load_model(str(ckpt))


def run_infer(session_dir: Path, n_top: int = 20,
              checkpoint: Path | None = None) -> Path:
    """
    Run GraspNet on session mesh → save candidates_local.json.

    Returns path to candidates_local.json.
    """
    from graspnet_compat.graspnet_loader import load_model, infer_grasps
    from graspnet_compat.convention import graspgroup_to_candidates, rerank_for_reachability

    ckpt = checkpoint or CKPT
    mesh_path = session_dir / "output" / "mesh" / "object_base_aligned.glb"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    t0 = time.time()
    print(f"\n{'='*55}")
    print(f"  GraspNet Inference")
    print(f"  Session: {session_dir.name}")
    print(f"{'='*55}")

    # ── 1. Load base_aligned mesh ──────────────────────────────
    scene   = trimesh.load(str(mesh_path))
    mesh_tm = trimesh.util.concatenate([g for g in scene.geometry.values()])
    v       = np.array(mesh_tm.vertices, dtype=np.float64)
    z_shift = float(-v[:, 2].min())          # shift needed to put z_min → 0
    mesh_tm.apply_translation([0, 0, z_shift])
    v = np.array(mesh_tm.vertices)

    print(f"\n  Mesh (base_aligned, z-shifted by {z_shift*100:.1f}cm):")
    print(f"    X=[{v[:,0].min()*100:.1f},{v[:,0].max()*100:.1f}]cm "
          f"Y=[{v[:,1].min()*100:.1f},{v[:,1].max()*100:.1f}]cm "
          f"Z=[{v[:,2].min()*100:.1f},{v[:,2].max()*100:.1f}]cm")

    # ── 2. Sample 20k surface points ──────────────────────────
    obj_pts, _ = trimesh.sample.sample_surface(mesh_tm, 20000)
    obj_pts = obj_pts.astype(np.float32)
    print(f"  Sampled: {len(obj_pts):,} surface points")

    # ── 3. GraspNet inference ──────────────────────────────────
    net = load_model(str(ckpt))
    gg  = infer_grasps(net, obj_pts, n_top=max(n_top * 15, 100))

    # ── 4. Convert to A2G convention (identical to baseline) ──
    candidates = graspgroup_to_candidates(gg, scale_factor=1.0, z_approach_max=0.3)
    candidates = rerank_for_reachability(candidates)
    candidates = candidates[:n_top]

    if not candidates:
        print("  ❌ No valid candidates after filtering.")
        return None

    # ── 5. Apply 2.5cm depth shift (identical to baseline) ────
    for c in candidates:
        approach      = np.array(c['rotation'])[:, 2]    # A2G col2 = approach
        c['position'] = np.array(c['position']) + approach * SHIFT_M

    print(f"\n  ✅ {len(candidates)} candidates (A2G + z_approach≤0.3 + {SHIFT_M*100:.1f}cm shift)")
    for i, c in enumerate(candidates[:5]):
        app = np.array(c['approach'])
        print(f"    [{i}] score={c['score']:.3f} "
              f"pt=[{c['position'][0]:.3f},{c['position'][1]:.3f},{c['position'][2]:.3f}] "
              f"approach=[{app[0]:.2f},{app[1]:.2f},{app[2]:.2f}] "
              f"w={c['gripper_width']*100:.0f}cm")

    # ── 6. Save candidates_local.json ─────────────────────────
    out_dir  = session_dir / "output" / "graspnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates_local.json"

    data = {
        "schema_version": "1.1",
        "session_id": session_dir.name,
        "frame": "base_aligned_z0",       # base_aligned mesh, shifted z_min→0
        "z_shift_m": round(z_shift, 6),   # needed by compose.py
        "shift_applied_m": SHIFT_M,
        "n_candidates": len(candidates),
        "conventions": {
            "rotation_columns": ["finger_open", "y_body", "approach"],
            "approach_column_index": 2,
            "depth_shift_m": SHIFT_M,
            "z_approach_max": 0.3,
        },
        "candidates": [
            {
                "rank":            i,
                "score":           round(float(c['score']), 5),
                "grasp_point":     [round(float(x), 6) for x in c['position']],
                "rotation":        [[round(float(x), 6) for x in row]
                                    for row in np.array(c['rotation'])],
                "gripper_width_m": round(float(c['gripper_width']), 5),
                "approach":        [round(float(x), 6) for x in c['approach']],
            }
            for i, c in enumerate(candidates)
        ],
    }

    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(out_path)

    print(f"\n  Saved: {out_path}")
    print(f"  Time:  {time.time()-t0:.1f}s")
    print(f"{'='*55}\n")
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True, help="Session directory path")
    p.add_argument("--n-top", type=int, default=20)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    run_infer(Path(args.session), n_top=args.n_top,
              checkpoint=Path(args.checkpoint) if args.checkpoint else None)

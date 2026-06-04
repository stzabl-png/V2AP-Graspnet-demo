#!/usr/bin/env python3
"""
scripts/generate_candidates.py
================================
Standalone script that runs GraspNet on a session directory and writes
V2AP-compatible output/inference/candidates.json.

Uses the local Affordance2Grasp Baseline2/graspnet code directly.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

# ── paths ──────────────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent.parent.parent  # Affordance2Grasp root
GN_DIR = PROJ / "Baseline2" / "graspnet"
GN3P = PROJ / "third_party" / "graspnet-baseline"

for _p in [str(GN_DIR), str(GN3P), str(GN3P/"models"),
           str(GN3P/"utils"), str(GN3P/"dataset")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

SHIFT_M = 0.025   # 2.5cm depth shift

# ── main ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session",    required=True, help="Session directory path")
    p.add_argument("--n-top",      type=int, default=20)
    p.add_argument("--checkpoint", default=str(GN_DIR/"checkpoints"/"checkpoint-rs.tar"))
    p.add_argument("--approach-cone-deg", type=float, default=45.0,
                   help="Allow approach XY within this many degrees of +Y (default: 45)")
    p.add_argument("--approach-y-min", type=float, default=0.0,
                   help="Minimum approach Y component (default: 0 = must have +Y)")
    args = p.parse_args()

    session   = Path(args.session)
    mesh_path = session / "output" / "mesh" / "object_base_aligned.glb"
    tbm_path  = session / "output" / "register" / "T_base_mesh.json"

    if not mesh_path.exists():
        sys.exit(f"❌ Mesh not found: {mesh_path}")
    if not tbm_path.exists():
        sys.exit(f"❌ T_base_mesh.json not found: {tbm_path}")

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  GraspNet → V2AP candidates.json")
    print(f"  Session:    {session.name}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}")

    # ── 1. Load base_aligned mesh ──────────────────────────────
    scene   = trimesh.load(str(mesh_path))
    mesh_tm = trimesh.util.concatenate([g for g in scene.geometry.values()])
    v       = np.array(mesh_tm.vertices)
    z_shift = float(-v[:, 2].min())
    mesh_tm.apply_translation([0, 0, z_shift])
    v = np.array(mesh_tm.vertices)
    mesh_span = (v.max(0) - v.min(0)).tolist()
    print(f"\n  Mesh (z-shifted +{z_shift*100:.1f}cm):")
    print(f"    X=[{v[:,0].min()*100:.1f},{v[:,0].max()*100:.1f}]cm "
          f"Y=[{v[:,1].min()*100:.1f},{v[:,1].max()*100:.1f}]cm "
          f"Z=[{v[:,2].min()*100:.1f},{v[:,2].max()*100:.1f}]cm")

    # ── 2. Sample point cloud ──────────────────────────────────
    obj_pts, _ = trimesh.sample.sample_surface(mesh_tm, 20000)
    obj_pts    = obj_pts.astype(np.float32)
    print(f"  Sampled: {len(obj_pts):,} surface points")

    # ── 3. GraspNet inference ──────────────────────────────────
    from graspnet_infer import load_model, infer_grasps
    from graspnet_to_hdf5 import graspgroup_to_candidates, rerank_for_reachability

    net        = load_model(args.checkpoint)
    gg         = infer_grasps(net, obj_pts, n_top=max(args.n_top * 15, 100))
    candidates = graspgroup_to_candidates(gg, scale_factor=1.0, z_approach_max=0.3)
    candidates = rerank_for_reachability(candidates)

    # ── Approach direction cone filter ────────────────────────
    # Keep grasps whose XY approach projection is within ±cone_deg of +Y axis.
    # approach = R[:,2] (A2G convention), in z0 frame.
    # +Y axis in XY plane = [0, 1]. Angle = arctan2(|app_x|, app_y).
    # Allow if app_y >= cos(cone_deg) * |app_xy| AND app_y > 0
    cone_rad  = np.radians(args.approach_cone_deg)
    cos_limit = np.cos(cone_rad)

    def _in_y_cone(c):
        app = np.array(c['rotation'])[:, 2]   # A2G col2 = approach
        app_xy_mag = np.linalg.norm(app[:2])
        if app_xy_mag < 0.05:                  # ~top-down: XY direction undefined → allow
            return True
        # Must have positive Y (approaching from -Y side)
        if app[1] < args.approach_y_min:
            return False
        # XY angle from +Y must be <= cone_deg
        cos_angle = app[1] / app_xy_mag        # dot([app_x,app_y], [0,1]) / mag
        return cos_angle >= cos_limit

    n_before = len(candidates)
    candidates = [c for c in candidates if _in_y_cone(c)]
    n_removed  = n_before - len(candidates)
    print(f"  Approach cone filter (±{args.approach_cone_deg:.0f}° from +Y): "
          f"removed {n_removed} → {len(candidates)} remain")

    if not candidates:
        sys.exit("❌ No candidates after approach cone filter. "
                 "Try --approach-cone-deg 90 to relax.")

    candidates = candidates[:args.n_top]

    if not candidates:
        sys.exit("❌ No valid candidates after filtering.")

    # ── 4. Apply 2.5cm depth shift ────────────────────────────
    for c in candidates:
        approach      = np.array(c['rotation'])[:, 2]
        c['position'] = np.array(c['position']) + approach * SHIFT_M
    print(f"\n  ✅ {len(candidates)} candidates (A2G + z_approach≤0.3 + {SHIFT_M*100:.1f}cm shift)")

    # ── 5. T_base_z0 ──────────────────────────────────────────
    with open(tbm_path) as f:
        T_base_mesh = np.array(json.load(f)["T_base_mesh"])
    T_shift_inv       = np.eye(4); T_shift_inv[2, 3] = -z_shift
    T_base_z0         = T_base_mesh @ T_shift_inv
    print(f"  T_base_z0 t: [{T_base_z0[0,3]:.3f},{T_base_z0[1,3]:.3f},{T_base_z0[2,3]:.3f}]m")

    # ── 6. Build V2AP candidates.json ─────────────────────────
    out_candidates = []
    for i, c in enumerate(candidates):
        gp  = c['position']
        rot = np.array(c['rotation'])
        app = np.array(c['approach'])
        # Sanity: show base-frame grasp position
        T_z0_g = np.eye(4); T_z0_g[:3,:3] = rot; T_z0_g[:3,3] = gp
        T_base_g = T_base_z0 @ T_z0_g
        if i < 3:
            print(f"  [{i}] score={c['score']:.3f} "
                  f"pt_z0=[{gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}] "
                  f"pt_base=[{T_base_g[0,3]:.3f},{T_base_g[1,3]:.3f},{T_base_g[2,3]:.3f}]")
        out_candidates.append({
            "rank":            i,
            "name":            f"graspnet_{i:03d}",
            "score":           round(float(c['score']), 4),
            "grasp_point":     [round(float(x), 6) for x in gp],
            "rotation":        [[round(float(x), 6) for x in row] for row in rot],
            "gripper_width_m": round(float(c['gripper_width']), 5),
            "approach_type":   "graspnet",
        })

    print(f"\n  ✅ {len(out_candidates)} candidates "
          f"(z_approach≤0.3 + ±{args.approach_cone_deg:.0f}°-Y cone + {SHIFT_M*100:.1f}cm shift)")

    out = {
        "schema_version": "1.1",
        "mesh_frame":     "base_aligned_z0",
        "inference_method": "graspnet",
        "T_base_mesh":    [[round(float(x), 8) for x in row] for row in T_base_z0.tolist()],
        "mesh_span_m":    [round(x, 4) for x in mesh_span],
        "conventions": {
            "rotation_columns":      ["finger_open", "y_body", "approach"],
            "approach_column_index": 2,
            "grasp_point_frame":     "base_aligned_z0",
            "ucb_tcp_offset_m":      0.105,
            "ucb_tcp_frame":         "panda_hand",
            "pre_grasp_offset_m":    0.15,
            "lift_height_m":         0.15,
            "depth_shift_m":         SHIFT_M,
            "approach_cone_deg":     args.approach_cone_deg,
            "approach_main_axis":    "+Y (from -Y side)",
        },
        "n_candidates": len(out_candidates),
        "candidates":   out_candidates,
    }

    # ── 7. Write to output/inference/candidates.json ──────────
    inference_dir = session / "output" / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    out_path = inference_dir / "candidates.json"

    # Backup existing candidates if present (don't overwrite GraspNet with GraspNet)
    if out_path.exists():
        import json as _j
        existing = _j.load(open(out_path))
        bak_name = ("candidates_pdm_backup.json"
                    if existing.get("inference_method") != "graspnet"
                    else "candidates_graspnet_prev.json")
        bak = inference_dir / bak_name
        out_path.rename(bak)
        print(f"  📦 Backed up previous candidates → {bak.name}")

    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    tmp.rename(out_path)

    # Update status.json to reflect graspnet source
    status_path = session / "output" / "status.json"
    if status_path.exists():
        with open(status_path) as f:
            status = json.load(f)
        status["titan"]["inference_method"] = "graspnet"
        status["titan"]["n_candidates"]     = len(out_candidates)
        status["titan"]["graspnet_update"]  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)

    print(f"\n  ✅ Written: {out_path}")
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"\n{'='*60}")
    print(f"  Ready for Razor. Run on Razor:")
    print(f"    python demo/phase2/run_auto_grasp.py \\")
    print(f"        --session-id {session.name} --dry-run")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

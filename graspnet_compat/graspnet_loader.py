"""
graspnet_compat/graspnet_loader.py
====================================
Thin wrapper that loads graspnet-baseline from the submodule and exposes
load_model() and infer_grasps() with the table-plane collision fix.

Requires: third_party/graspnet-baseline  (git submodule)
          checkpoints/checkpoint-rs.tar
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
_GN  = REPO / "third_party" / "graspnet-baseline"

for _p in [str(_GN), str(_GN / "models"), str(_GN / "utils"), str(_GN / "dataset")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_model(checkpoint: str):
    import torch
    from graspnet import GraspNet

    net = GraspNet(input_feature_dim=0, num_view=300,
                   num_angle=12, num_depth=4, cylinder_radius=0.05,
                   hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    net.cuda()
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    print(f"  ✅ GraspNet model loaded from {checkpoint}")
    return net


def _generate_table_plane(points: np.ndarray,
                           margin: float = 0.05,
                           n_table_points: int = 5000) -> np.ndarray:
    """Virtual table plane at z_min of the object point cloud."""
    z_min = float(points[:, 2].min())
    x_min = float(points[:, 0].min()) - margin
    x_max = float(points[:, 0].max()) + margin
    y_min = float(points[:, 1].min()) - margin
    y_max = float(points[:, 1].max()) + margin
    xs = np.random.uniform(x_min, x_max, n_table_points).astype(np.float32)
    ys = np.random.uniform(y_min, y_max, n_table_points).astype(np.float32)
    zs = np.full(n_table_points, z_min, dtype=np.float32)
    return np.column_stack([xs, ys, zs])


def infer_grasps(net, points: np.ndarray,
                 n_top: int = 50,
                 collision_thresh: float = 0.01,
                 z_approach_max: float = 0.3) -> object:
    """
    GraspNet forward → collision (object+table) → z_approach filter → NMS → top-K.

    z_approach_max=0.3: reject grasps whose approach Z > 0.3 (from below table).
    approach_dist=0.25: checks 25cm along approach to catch pre-grasp penetrations.
    """
    import torch
    from graspnet import pred_decode
    from collision_detector import ModelFreeCollisionDetector
    from graspnetAPI import GraspGroup

    ep = {"point_clouds": torch.from_numpy(
        points[np.newaxis].astype(np.float32)).cuda()}
    with torch.no_grad():
        ep = net(ep)
        preds = pred_decode(ep)

    gg = GraspGroup(preds[0].cpu().numpy())
    n_raw = len(gg)
    if n_raw == 0:
        print("  ⚠️  No grasps predicted by network"); return gg

    scene_pts = np.concatenate([points, _generate_table_plane(points)], axis=0)
    det = ModelFreeCollisionDetector(scene_pts, voxel_size=0.01)
    mask = det.detect(gg, approach_dist=0.25, collision_thresh=collision_thresh)
    gg = gg[~mask]
    n_col = len(gg)
    if n_col == 0:
        print(f"  ⚠️  All {n_raw} grasps removed by collision detection"); return gg

    # z_approach filter: GraspNet col0 = approach direction
    if z_approach_max < 1.0:
        app_z = np.array([g.rotation_matrix[2, 0] for g in gg])
        gg    = gg[app_z <= z_approach_max]
        n_rm  = n_col - len(gg)
        if n_rm:
            print(f"     z_approach filter: removed {n_rm} below-table → {len(gg)} remain")
        if len(gg) == 0:
            print("  ⚠️  All grasps removed by z_approach filter"); return gg

    try:
        gg = gg.nms().sort_by_score()
    except Exception:
        gg.sort_by_score()
    gg = gg[:n_top]

    scores = np.array([g.score for g in gg])
    print(f"  ✅ Grasps: {n_raw} raw → {n_col} post-collision → {len(gg)} NMS+top-{n_top}")
    print(f"     scores: [{scores.min():.3f}, {scores.max():.3f}] mean={scores.mean():.3f}")
    return gg

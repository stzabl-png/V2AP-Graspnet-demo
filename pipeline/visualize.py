#!/usr/bin/env python3
"""
pipeline/visualize.py
======================
Step 3 (optional): Open3D interactive visualization of grasp candidates.

Shows candidates.json (base frame) overlaid on the object mesh,
positioned in the robot base coordinate system.

Usage:
    python pipeline/visualize.py --session /path/to/session --frame base
    python pipeline/visualize.py --session /path/to/session --frame local

Mouse: Left=rotate  Right=pan  Scroll=zoom  Q=quit
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import open3d as o3d

REPO = Path(__file__).resolve().parent.parent
TCP_OFFSET = 0.105


# ─── Geometry primitives (same as vis_grasp_combined.py pattern) ─────────────

def _rot_z_to(direction):
    d = np.array(direction, dtype=float) / (np.linalg.norm(direction) + 1e-8)
    z = np.array([0., 0., 1.])
    ax = np.cross(z, d); sin_a = np.linalg.norm(ax); cos_a = float(z @ d)
    if sin_a > 1e-6:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            ax / sin_a * np.arctan2(sin_a, cos_a))
    elif cos_a < 0:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([1., 0., 0.]) * np.pi)
    else:
        R = np.eye(3)
    return R


def sphere(center, radius=0.005, color=(1, 0.5, 0)):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(center); s.paint_uniform_color(color)
    s.compute_vertex_normals(); return s


def cylinder(p1, p2, radius=0.003, color=(0.1, 0.8, 0.9)):
    d = np.array(p2) - np.array(p1); L = np.linalg.norm(d)
    if L < 1e-6: return None
    c = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L, resolution=16)
    c.rotate(_rot_z_to(d), [0, 0, 0]); c.translate((np.array(p1) + np.array(p2)) / 2)
    c.paint_uniform_color(color); c.compute_vertex_normals(); return c


def arrow(start, direction, length=0.05, radius=0.003, color=(0.9, 0.2, 0.2)):
    d = np.array(direction) / (np.linalg.norm(direction) + 1e-8)
    a = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius, cone_radius=radius*2.5,
        cylinder_height=length*0.72, cone_height=length*0.28, resolution=16)
    a.rotate(_rot_z_to(d), [0, 0, 0]); a.translate(np.array(start, dtype=float))
    a.paint_uniform_color(color); a.compute_vertex_normals(); return a


def build_gripper(pt, R, width,
                  body_color=(0.05, 0.95, 0.45),
                  tip_color=(0.10, 0.85, 0.95),
                  show_pregrasp=True):
    """Build Open3D gripper geometry. pt=grasp_point, R[:,2]=approach (A2G)."""
    geoms = []
    approach = R[:, 2]; f_open = R[:, 0]
    hw = width / 2; FLEN = 0.07; R0 = width * 0.025

    tip_L = pt + f_open * hw;  tip_R = pt - f_open * hw
    base_L = tip_L - approach * FLEN; base_R = tip_R - approach * FLEN

    geoms.append(sphere(tip_L,  width * 0.042, tip_color))
    geoms.append(sphere(tip_R,  width * 0.042, tip_color))
    for a, b in [(tip_L, tip_R), (base_L, tip_L), (base_R, tip_R), (base_L, base_R)]:
        c = cylinder(a, b, R0, tip_color if (a is tip_L and b is tip_R) else body_color)
        if c: geoms.append(c)
    a = arrow(pt, approach, length=FLEN*0.9, radius=R0*0.8, color=body_color)
    if a: geoms.append(a)
    wrist = pt - approach * TCP_OFFSET
    geoms.append(sphere(wrist, width * 0.055, (1.0, 0.9, 0.1)))
    c = cylinder(wrist, (base_L + base_R) / 2, R0 * 0.7, (0.85, 0.85, 0.2))
    if c: geoms.append(c)
    geoms.append(sphere(pt, width * 0.035, (1.0, 0.5, 0.0)))

    if show_pregrasp:
        pre = pt - approach * (TCP_OFFSET + 0.15)
        geoms.append(sphere(pre, width * 0.040, (1.0, 0.85, 0.0)))
        a = arrow(pre, approach, length=0.09, radius=R0 * 0.7, color=(1.0, 0.80, 0.0))
        if a: geoms.append(a)
    return geoms


# ─── Main ─────────────────────────────────────────────────────────────────────

BODY_COLORS = [
    (0.05, 0.95, 0.45),  # green
    (0.95, 0.60, 0.05),  # orange
    (0.55, 0.20, 0.90),  # purple
    (0.10, 0.70, 0.95),  # cyan
    (0.95, 0.15, 0.30),  # red
]


def run_vis(session_dir: Path, frame: str = "base", n_show: int = 5):
    """
    frame = "base"  → show candidates.json (base-frame poses on base-frame mesh)
    frame = "local" → show candidates_local.json (z0-frame, good for debugging)
    """
    session_dir = Path(session_dir)
    gn_dir = session_dir / "output" / "graspnet"

    # ── Load candidates ────────────────────────────────────────
    if frame == "base":
        cand_path = gn_dir / "candidates.json"
    else:
        cand_path = gn_dir / "candidates_local.json"

    if not cand_path.exists():
        raise FileNotFoundError(
            f"{cand_path.name} not found. Run infer.py and compose.py first.")

    with open(cand_path) as f:
        data = json.load(f)
    candidates = data["candidates"][:n_show]
    print(f"\nVisualizing {len(candidates)} candidates  frame={frame}")

    # ── Load mesh ──────────────────────────────────────────────
    if frame == "base":
        mesh_path = session_dir / "output" / "mesh" / "object_base_aligned.glb"
        T_base_z0 = np.array(data["T_base_mesh"])
    else:
        mesh_path = session_dir / "output" / "mesh" / "object_base_aligned.glb"
        T_base_z0 = np.eye(4)

    scene   = trimesh.load(str(mesh_path))
    mesh_tm = trimesh.util.concatenate([g for g in scene.geometry.values()])

    if frame == "base":
        # Transform mesh to base frame for display
        z_shift = float(data.get("z_shift_m", 0) or 0)
        T_shift       = np.eye(4); T_shift[2, 3] = z_shift
        T_display_mesh = T_base_z0 @ T_shift
        mesh_tm.apply_transform(T_display_mesh)
    else:
        z_shift = float(data.get("z_shift_m", 0) or 0)
        mesh_tm.apply_translation([0, 0, z_shift])

    v = np.array(mesh_tm.vertices); f = np.array(mesh_tm.faces)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices  = o3d.utility.Vector3dVector(v)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(f)
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([0.92, 0.38, 0.08])

    # ── Build scene ────────────────────────────────────────────
    geoms = [mesh_o3d]

    # Coordinate frame
    center = (v.max(0) + v.min(0)) / 2
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.08, origin=center))

    # Grippers
    for rank, c in enumerate(candidates):
        gp  = np.array(c["grasp_point"])
        R   = np.array(c["rotation"])
        w   = float(c["gripper_width_m"])
        col = BODY_COLORS[rank % len(BODY_COLORS)]
        geoms.extend(build_gripper(gp, R, w, body_color=col,
                                   show_pregrasp=(rank == 0)))
        print(f"  rank={rank} score={c['score']:.3f} "
              f"pt=[{gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}] "
              f"approach=[{R[0,2]:.2f},{R[1,2]:.2f},{R[2,2]:.2f}]")

    # ── Launch viewer ──────────────────────────────────────────
    title = f"GraspNet Candidates  [{session_dir.name}]  frame={frame}"
    print(f"\n  🖱  Left=rotate  Right=pan  Scroll=zoom  Q=quit")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=900)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color    = np.array([0.06, 0.06, 0.10])
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_zoom(0.55); ctr.set_front([-0.5, -0.7, 0.5]); ctr.set_up([0, 0, 1])
    vis.run(); vis.destroy_window()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--frame",   default="base", choices=["base", "local"])
    p.add_argument("--n-show",  type=int, default=5)
    args = p.parse_args()
    run_vis(Path(args.session), frame=args.frame, n_show=args.n_show)

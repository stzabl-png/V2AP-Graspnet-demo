#!/usr/bin/env python3
"""
run_demo.py — V2AP-GraspNet-Demo Full Pipeline
================================================
GraspNet offline inference + base-frame composition + Open3D visualization.

Usage:
    # Full pipeline (infer + compose + visualize)
    python run_demo.py --session /media/lyh/KINGSTON/20260602_192346_chips

    # Only infer (save candidates_local.json)
    python run_demo.py --session /path/to/session --only infer

    # Only compose (requires candidates_local.json)
    python run_demo.py --session /path/to/session --only compose

    # Only visualize (requires candidates.json)
    python run_demo.py --session /path/to/session --only vis

    # Visualize local (debug) frame instead of base frame
    python run_demo.py --session /path/to/session --only vis --vis-frame local

    # Top-N candidates
    python run_demo.py --session /path/to/session --n-top 10

Pipeline:
    ┌─────────────────────────────────────────────────────────────┐
    │  Session Dir                                                │
    │    output/mesh/object_base_aligned.glb   ← mesh input      │
    │    output/register/T_base_mesh.json      ← 6D pose (FP)    │
    │                                                             │
    │  Step 1: infer.py                                           │
    │    Mesh → GraspNet → A2G convention → z_approach filter     │
    │    → rerank → +2.5cm shift                                  │
    │    → output/graspnet/candidates_local.json                  │
    │                                                             │
    │  Step 2: compose.py                                         │
    │    candidates_local + T_base_mesh → base-frame poses        │
    │    → output/graspnet/candidates.json    (V2AP format)       │
    │                                                             │
    │  Step 3: visualize.py  (optional)                           │
    │    Open3D interactive: mesh + gripper poses                 │
    └─────────────────────────────────────────────────────────────┘
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from pipeline.infer     import run_infer
from pipeline.compose   import run_compose
from pipeline.visualize import run_vis


def main():
    p = argparse.ArgumentParser(
        description="V2AP-GraspNet-Demo: offline mesh inference → base-frame candidates"
    )
    p.add_argument(
        "--session", required=True,
        help="Path to V2AP session directory (must contain output/mesh/ and output/register/)"
    )
    p.add_argument(
        "--n-top", type=int, default=20,
        help="Number of top candidates to keep (default: 20)"
    )
    p.add_argument(
        "--only", choices=["infer", "compose", "vis", "all"], default="all",
        help="Run only one step (default: all)"
    )
    p.add_argument(
        "--vis-frame", choices=["base", "local"], default="base",
        help="Visualization frame: base=robot base, local=mesh z0 frame"
    )
    p.add_argument(
        "--n-vis", type=int, default=5,
        help="Number of candidates to show in visualization (default: 5)"
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to GraspNet checkpoint (default: checkpoints/checkpoint-rs.tar)"
    )
    args = p.parse_args()

    session = Path(args.session)
    if not session.exists():
        print(f"❌ Session not found: {session}"); sys.exit(1)

    ckpt = Path(args.checkpoint) if args.checkpoint else None

    only = args.only
    print(f"\n{'='*55}")
    print(f"  V2AP-GraspNet-Demo")
    print(f"  Session: {session.name}")
    print(f"  Mode:    {only}   n_top={args.n_top}")
    print(f"{'='*55}")

    if only in ("infer", "all"):
        local_path = run_infer(session, n_top=args.n_top, checkpoint=ckpt)
        if local_path is None:
            print("❌ Inference failed — no candidates found."); sys.exit(1)

    if only in ("compose", "all"):
        run_compose(session)

    if only in ("vis", "all"):
        run_vis(session, frame=args.vis_frame, n_show=args.n_vis)

    if only == "all":
        out = session / "output" / "graspnet" / "candidates.json"
        print(f"\n✅ Pipeline complete.")
        print(f"   V2AP candidates: {out}")
        print(f"   → Ready to pass to V2AP run_auto_grasp.py\n")


if __name__ == "__main__":
    main()

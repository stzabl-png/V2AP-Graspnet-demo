# V2AP-GraspNet-Demo

GraspNet offline mesh inference → 6D pose composition → V2AP-compatible candidates for Razor robot deployment.

## Pipeline Overview

```
Session Directory
  output/mesh/object_base_aligned.glb   ← SAM3D reconstructed mesh (base-aligned)
  output/register/T_base_mesh.json      ← 6D object pose (FoundationPose)
           │
           ▼
     Step 1: infer.py
       object mesh → sample 20k pts → GraspNet
       → A2G column convention
       → z_approach_max=0.3 filter (no below-table grasps)
       → rerank_for_reachability (top-down preferred)
       → +2.5cm depth shift (same as baseline sim evaluation)
       → output/graspnet/candidates_local.json
           │
           ▼
     Step 2: compose.py
       candidates_local + T_base_mesh → base-frame poses
       → output/graspnet/candidates.json  (V2AP schema 1.1)
           │
           ▼
     Step 3: visualize.py  (optional)
       Open3D interactive: mesh + gripper candidates
           │
           ▼
     → pass candidates.json to V2AP run_auto_grasp.py → Razor
```

## Session Directory Format

Each session is a directory with this layout (produced by the V2AP data collection pipeline):

```
20260602_192346_chips/
  output/
    mesh/
      object_scaled.glb            ← raw SAM3D mesh (scale embedded)
      object_base_aligned.glb      ← mesh in robot base frame (used by this demo)
    register/
      T_base_mesh.json             ← {"T_base_mesh": [[4x4 matrix]]}
      T_cam_mesh.json
    graspnet/                      ← created by this demo
      candidates_local.json        ← grasp poses in mesh-local frame
      candidates.json              ← grasp poses in robot base frame (V2AP input)
```

## Quick Start

### 1. Setup

```bash
git clone https://github.com/YOUR_ORG/V2AP-Graspnet-demo.git
cd V2AP-Graspnet-demo
git submodule update --init --recursive   # pulls graspnet-baseline

# Install dependencies
pip install open3d trimesh numpy torch graspnetAPI

# Download GraspNet checkpoint
# Place at: checkpoints/checkpoint-rs.tar
# Download from: https://graspnet.net/  (RealSense model)
```

### 2. Run Full Pipeline

```bash
# Full: infer + compose + visualize
python run_demo.py --session /path/to/session

# With custom options
python run_demo.py \
    --session /media/lyh/KINGSTON/20260602_192346_chips \
    --n-top 20 \
    --n-vis 5 \
    --vis-frame base
```

### 3. Step-by-Step

```bash
# Step 1 only: GraspNet inference
python run_demo.py --session /path/to/session --only infer --n-top 20

# Step 2 only: compose to base frame
python run_demo.py --session /path/to/session --only compose

# Step 3 only: visualize
python run_demo.py --session /path/to/session --only vis --vis-frame base

# Debug: visualize in local (mesh) frame
python run_demo.py --session /path/to/session --only vis --vis-frame local
```

### 4. Direct Module Usage

```bash
python pipeline/infer.py --session /path/to/session --n-top 20
python pipeline/compose.py --session /path/to/session
python pipeline/visualize.py --session /path/to/session --frame base
```

## candidates.json Format (V2AP Schema 1.1)

```json
{
  "schema_version": "1.1",
  "session_id": "20260602_192346_chips",
  "mesh_frame": "base_frame",
  "n_candidates": 20,
  "conventions": {
    "rotation_columns": ["finger_open", "y_body", "approach"],
    "approach_column_index": 2
  },
  "T_base_mesh": [[4x4 matrix]],
  "candidates": [
    {
      "rank": 0,
      "score": 1.677,
      "grasp_point": [x, y, z],       // in robot base frame (m)
      "rotation": [[3x3 matrix]],     // col2 = approach direction
      "gripper_width_m": 0.08,
      "approach": [ax, ay, az],
      "pre_grasp_point": [x, y, z],   // 15cm retreat along -approach
      "grasp_point_local": [...],     // in mesh-local frame (for reference)
      "rotation_local": [[...]]
    }
  ]
}
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Use `object_base_aligned.glb` | T_base_mesh rotation = identity → cleanest frame composition |
| z_approach_max = 0.3 | Matches baseline sim evaluation; rejects physically impossible below-table approaches |
| approach_dist = 0.25m | Extended from 0.05m → covers full pre-grasp reach to detect table penetrations |
| +2.5cm depth shift | Matches `make_shifted_graspnet_candidates.py` in baseline sim evaluation |
| A2G column convention | col2=approach, col0=finger_open — consistent with V2AP retarget.py |
| rerank_for_reachability | Prioritizes top-down (approach_z < -0.5) over side approaches for Franka reachability |

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA)
- `open3d >= 0.17`
- `trimesh >= 4.0`
- `graspnetAPI` (from [graspnet-baseline](https://github.com/graspnet/graspnet-baseline))
- `numpy`, `scipy`

## Submodule

`third_party/graspnet-baseline` points to the official [GraspNet-1Billion baseline](https://github.com/graspnet/graspnet-baseline).

## Related Repos

- [V2AP-demo](https://github.com/jiaka1chen/V2AP-demo) — Razor robot deployment pipeline
- [graspnet-baseline](https://github.com/graspnet/graspnet-baseline) — GraspNet model

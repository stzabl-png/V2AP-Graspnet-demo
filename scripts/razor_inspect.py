#!/usr/bin/env python3
"""
scripts/razor_inspect.py
==========================
Run this on Razor BEFORE or AFTER a grasp attempt.
Collects environment, directory structure, and grasp result.
Saves a single JSON report → copy back to USB / share with local machine.

Usage (on Razor):
    source ~/V2AP-demo/setup.sh
    python razor_inspect.py --session-id 20260602_192346_chips
    python razor_inspect.py --session-id 20260602_192346_chips --after-grasp

Output:
    razor_report_<session_id>_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ── helpers ────────────────────────────────────────────────────────────────

def _run(cmd: str, cwd=None) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=10, cwd=cwd)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"[error: {e}]"


def _tree(root: Path, max_depth: int = 3, max_files: int = 40) -> list[str]:
    """Return directory listing as list of strings."""
    lines = []
    count = [0]

    def _walk(p: Path, depth: int, prefix: str):
        if depth > max_depth or count[0] >= max_files:
            return
        try:
            children = sorted(p.iterdir())
        except PermissionError:
            return
        dirs  = [c for c in children if c.is_dir()]
        files = [c for c in children if c.is_file()]
        for i, child in enumerate(dirs + files):
            if count[0] >= max_files:
                lines.append(prefix + "  ... (truncated)")
                return
            is_last  = (i == len(dirs) + len(files) - 1)
            connector = "└── " if is_last else "├── "
            size_str  = f"  [{child.stat().st_size:,}B]" if child.is_file() else ""
            lines.append(prefix + connector + child.name + size_str)
            count[0] += 1
            if child.is_dir():
                ext = "    " if is_last else "│   "
                _walk(child, depth + 1, prefix + ext)

    lines.append(str(root))
    _walk(root, 1, "")
    return lines


def _read_json_safe(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def _find_sessions_dir() -> Path | None:
    """Try common V2AP sessions directory locations on Razor."""
    candidates = [
        Path.home() / "sessions",
        Path.home() / "V2AP-demo" / "demo" / "phase2" / "sessions",
        Path("/home/vision/sessions"),
        Path("/home/vision/V2AP-demo/demo/phase2/sessions"),
        Path("/data/sessions"),
    ]
    for p in candidates:
        if p.exists():
            return p
    # Also check env variable
    env_path = os.environ.get("V2AP_SESSIONS_DIR")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    return None


def _find_v2ap_root() -> Path | None:
    """Try to find V2AP-demo root."""
    candidates = [
        Path.home() / "V2AP-demo",
        Path("/home/vision/V2AP-demo"),
        Path("/home/razor/V2AP-demo"),
    ]
    for p in candidates:
        if (p / "demo" / "phase2" / "run_auto_grasp.py").exists():
            return p
    return None


# ── main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Razor environment inspector")
    p.add_argument("--session-id", default=None,
                   help="Session ID to inspect (e.g. 20260602_192346_chips)")
    p.add_argument("--sessions-dir", default=None,
                   help="Override V2AP sessions directory")
    p.add_argument("--after-grasp", action="store_true",
                   help="Also collect grasp execution logs/results")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: razor_report_<session>_<ts>.json)")
    args = p.parse_args()

    ts  = time.strftime("%Y%m%d_%H%M%S")
    sid = args.session_id or "no_session"
    out_path = Path(args.output or f"razor_report_{sid}_{ts}.json")

    report = {
        "generated_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id":    sid,
        "after_grasp":   args.after_grasp,
    }

    print(f"\n{'='*55}")
    print(f"  Razor Inspector — session: {sid}")
    print(f"{'='*55}")

    # ── 1. System info ─────────────────────────────────────────
    print("  [1/6] System info...")
    report["system"] = {
        "hostname":       platform.node(),
        "os":             platform.platform(),
        "python":         sys.version,
        "cpu_count":      os.cpu_count(),
        "user":           _run("whoami"),
        "gpu":            _run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU'"),
        "ros_distro":     os.environ.get("ROS_DISTRO", "not set"),
        "conda_env":      os.environ.get("CONDA_DEFAULT_ENV", "not set"),
        "v2ap_sessions":  os.environ.get("V2AP_SESSIONS_DIR", "not set"),
        "path_entries":   os.environ.get("PATH", "").split(":")[:10],
    }

    # ── 2. Python packages ─────────────────────────────────────
    print("  [2/6] Python packages...")
    pkg_out = _run("pip list --format=columns 2>/dev/null | grep -iE 'torch|open3d|trimesh|numpy|scipy|loguru|graspnet|pink|curobo|pinocchio|rbdl'")
    report["python_packages_relevant"] = pkg_out
    report["python_packages_all"]      = _run("pip list --format=json 2>/dev/null")

    # ── 3. V2AP-demo structure ─────────────────────────────────
    print("  [3/6] V2AP-demo structure...")
    v2ap_root = _find_v2ap_root()
    report["v2ap"] = {"root": str(v2ap_root) if v2ap_root else None}
    if v2ap_root:
        report["v2ap"]["tree"] = _tree(v2ap_root, max_depth=3, max_files=60)
        # Key config files
        for rel in ["demo/phase2/constants.py",
                    "demo/phase2/ee_retarget_io.py",
                    "demo/phase1/constants.py"]:
            f = v2ap_root / rel
            if f.exists():
                try:
                    report["v2ap"][rel] = f.read_text(errors="replace")[:2000]
                except Exception:
                    pass
        # Default ee_retarget yaml
        ee_yaml = v2ap_root / "demo" / "right_hand_profile.yaml"
        if ee_yaml.exists():
            report["v2ap"]["right_hand_profile.yaml"] = ee_yaml.read_text(errors="replace")[:2000]

    # ── 4. Sessions directory ──────────────────────────────────
    print("  [4/6] Sessions directory...")
    sessions_root = Path(args.sessions_dir) if args.sessions_dir else _find_sessions_dir()
    report["sessions"] = {
        "root": str(sessions_root) if sessions_root else None,
        "session_ids": [],
    }
    if sessions_root and sessions_root.exists():
        report["sessions"]["tree"] = _tree(sessions_root, max_depth=2, max_files=30)
        report["sessions"]["session_ids"] = [
            d.name for d in sorted(sessions_root.iterdir()) if d.is_dir()
        ]

    # ── 5. Target session detail ───────────────────────────────
    print("  [5/6] Target session...")
    session_dir = None
    if sessions_root and args.session_id:
        session_dir = sessions_root / args.session_id
    report["target_session"] = {
        "session_id": sid,
        "path":       str(session_dir) if session_dir else None,
        "exists":     session_dir.exists() if session_dir else False,
    }
    if session_dir and session_dir.exists():
        report["target_session"]["tree"] = _tree(session_dir, max_depth=4, max_files=60)
        # Key files
        for rel in ["output/status.json",
                    "output/inference/candidates.json",
                    "output/register/T_base_mesh.json"]:
            f = session_dir / rel
            if f.exists():
                report["target_session"][rel] = _read_json_safe(f)
        # candidates.json summary
        cand_path = session_dir / "output" / "inference" / "candidates.json"
        if cand_path.exists():
            cd = _read_json_safe(cand_path)
            report["target_session"]["candidates_summary"] = {
                "n_candidates":    cd.get("n_candidates"),
                "inference_method": cd.get("inference_method"),
                "T_base_mesh":     cd.get("T_base_mesh"),
                "top3":            cd.get("candidates", [])[:3],
            }

    # ── 6. After-grasp logs ────────────────────────────────────
    print("  [6/6] Grasp result logs...")
    if args.after_grasp:
        # Try to find recent log files
        log_paths = [
            Path.home() / ".cache" / "V2AP" / "logs",
            Path("/tmp"),
            v2ap_root / "logs" if v2ap_root else None,
        ]
        recent_logs = []
        for lp in log_paths:
            if lp and lp.exists():
                for f in sorted(lp.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                    try:
                        recent_logs.append({
                            "path": str(f),
                            "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime(f.stat().st_mtime)),
                            "tail": f.read_text(errors="replace")[-3000:],
                        })
                    except Exception:
                        pass
        report["grasp_logs"] = recent_logs

        # Terminal output hint
        report["grasp_notes"] = (
            "Run: python demo/phase2/run_auto_grasp.py --session-id <id> 2>&1 | tee grasp_run.log\n"
            "Then re-run this script to capture grasp_run.log"
        )
        # Check for grasp_run.log in cwd
        for log_name in ["grasp_run.log", "run.log"]:
            lf = Path(log_name)
            if lf.exists():
                report[log_name] = lf.read_text(errors="replace")[-5000:]

    # ── Robot / hardware ───────────────────────────────────────
    report["hardware"] = {
        "usb_drives": _run("lsblk -o NAME,SIZE,MOUNTPOINT | grep -v '^loop'"),
        "network":    _run("ip addr | grep 'inet ' | awk '{print $2}'"),
        "ros_nodes":  _run("ros2 node list 2>/dev/null || rosnode list 2>/dev/null || echo 'ROS not running'"),
    }

    # ── Write output ───────────────────────────────────────────
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  ✅ Report saved: {out_path}")
    print(f"  Size: {out_path.stat().st_size:,} bytes")
    print(f"\n  Copy to USB:")
    print(f"    cp {out_path} /media/*/  (or your USB mount path)")
    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()

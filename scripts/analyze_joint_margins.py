#!/usr/bin/env python3
"""Analyze joint limit margins and frame-to-frame jumps in GMR motion pkls."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
from general_motion_retargeting.params import ROBOT_XML_DICT


def load_qpos_trajectory(motion_path: Path, robot: str) -> tuple[np.ndarray, float]:
    with open(motion_path, "rb") as f:
        motion = pickle.load(f)

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    root_pos = np.asarray(motion["root_pos"], dtype=float)
    root_rot = np.asarray(motion["root_rot"], dtype=float)
    dof_pos = np.asarray(motion["dof_pos"], dtype=float)

    if root_rot.shape[-1] == 4 and np.abs(np.linalg.norm(root_rot[0]) - 1.0) < 0.01:
        # Saved as xyzw -> convert to MuJoCo wxyz qpos layout.
        wxyz = root_rot[:, [3, 0, 1, 2]]
    else:
        wxyz = root_rot

    qpos = np.concatenate([root_pos, wxyz, dof_pos], axis=1)
    fps = float(motion.get("fps", 30.0))
    return qpos, fps


def analyze_trajectory(
    model: mj.MjModel,
    qpos_seq: np.ndarray,
    joint_names: list[str] | None = None,
) -> dict:
    entries = GeneralMotionRetargeting.collect_limited_hinge_info(model, joint_names)
    n_frames = len(qpos_seq)
    report: dict = {"frames": n_frames, "joints": {}}

    for entry in entries:
        name = entry["name"]
        adr = entry["qpos_adr"]
        lo, hi = entry["lo"], entry["hi"]
        values = qpos_seq[:, adr]
        margins = [
            GeneralMotionRetargeting._normalized_joint_margin(float(v), lo, hi)
            for v in values
        ]
        margins_arr = np.asarray(margins, dtype=float)
        deltas = np.abs(np.diff(values)) if n_frames > 1 else np.array([])

        near_limit_ratio = float(np.mean(margins_arr < 0.08)) if n_frames else 0.0
        report["joints"][name] = {
            "min_margin": float(margins_arr.min()) if n_frames else 0.0,
            "mean_margin": float(margins_arr.mean()) if n_frames else 0.0,
            "near_limit_ratio": near_limit_ratio,
            "max_abs_delta": float(deltas.max()) if len(deltas) else 0.0,
            "p95_abs_delta": float(np.percentile(deltas, 95)) if len(deltas) else 0.0,
            "range_lo": lo,
            "range_hi": hi,
        }

    if n_frames > 1:
        all_deltas = []
        for entry in entries:
            adr = entry["qpos_adr"]
            all_deltas.extend(np.abs(np.diff(qpos_seq[:, adr])).tolist())
        report["global_max_abs_delta"] = float(max(all_deltas)) if all_deltas else 0.0
        report["global_p95_abs_delta"] = (
            float(np.percentile(all_deltas, 95)) if all_deltas else 0.0
        )
    return report


def print_report(label: str, report: dict, focus_joints: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(f"frames: {report['frames']}")
    if "global_max_abs_delta" in report:
        print(
            f"global |Δq| max={report['global_max_abs_delta']:.4f} "
            f"p95={report['global_p95_abs_delta']:.4f}"
        )

    for name in focus_joints:
        if name not in report["joints"]:
            continue
        j = report["joints"][name]
        print(
            f"  {name}: min_margin={j['min_margin']:.3f} "
            f"near_limit={100*j['near_limit_ratio']:.1f}% "
            f"max|Δq|={j['max_abs_delta']:.4f} p95|Δq|={j['p95_abs_delta']:.4f}"
        )


def compare_reports(baseline: dict, candidate: dict, focus_joints: list[str]) -> None:
    print("\n=== delta (candidate - baseline) ===")
    if "global_max_abs_delta" in baseline and "global_max_abs_delta" in candidate:
        dmax = candidate["global_max_abs_delta"] - baseline["global_max_abs_delta"]
        dp95 = candidate["global_p95_abs_delta"] - baseline["global_p95_abs_delta"]
        print(f"global max|Δq| change: {dmax:+.4f}")
        print(f"global p95|Δq| change: {dp95:+.4f}")

    for name in focus_joints:
        if name not in baseline["joints"] or name not in candidate["joints"]:
            continue
        b = baseline["joints"][name]
        c = candidate["joints"][name]
        print(
            f"  {name}: min_margin {c['min_margin'] - b['min_margin']:+.3f} "
            f"near_limit {(c['near_limit_ratio'] - b['near_limit_ratio']) * 100:+.1f}% "
            f"max|Δq| {c['max_abs_delta'] - b['max_abs_delta']:+.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="aiq11")
    parser.add_argument("--motion", type=Path, required=True, help="GMR pkl with penalty ON")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline pkl (penalty OFF) for A/B comparison",
    )
    parser.add_argument(
        "--joints",
        nargs="*",
        default=["l_arm_4", "r_arm_4", "l_arm_3", "r_arm_3"],
    )
    args = parser.parse_args()

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    qpos, fps = load_qpos_trajectory(args.motion, args.robot)
    report = analyze_trajectory(model, qpos, args.joints)
    print_report(str(args.motion), report, args.joints)
    print(f"fps: {fps}")

    if args.baseline is not None:
        qpos_base, _ = load_qpos_trajectory(args.baseline, args.robot)
        base_report = analyze_trajectory(model, qpos_base, args.joints)
        print_report(str(args.baseline), base_report, args.joints)
        compare_reports(base_report, report, args.joints)


if __name__ == "__main__":
    main()

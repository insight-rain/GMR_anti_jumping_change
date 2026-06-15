"""Multi-process frame retargeting for long offline motions."""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from .motion_retarget import GeneralMotionRetargeting


def retarget_frames_parallel(
    frames: list,
    gmr_kwargs: dict[str, Any],
    *,
    num_cpus: int = 1,
    chunk_size: int | None = None,
    show_progress: bool = True,
    apply_temporal_smooth: bool | None = None,
) -> list[np.ndarray]:
    """Retarget a motion sequence.

    For a single continuous motion, chunks are processed **sequentially** and
    each chunk warm-starts from the previous chunk's last ``qpos``. Using
    independent workers per chunk (human-root seed only) causes visible torso /
    whole-body pops at chunk boundaries (~every N/num_cpus frames).
    """
    n = len(frames)
    if n == 0:
        return []

    retargeter = GeneralMotionRetargeting(**gmr_kwargs, verbose=False)

    if num_cpus <= 1 or n == 1 or chunk_size is None:
        chunk_size = n if num_cpus <= 1 or n == 1 else max(1, (n + num_cpus - 1) // num_cpus)

    chunk_starts = list(range(0, n, chunk_size))
    iterator = chunk_starts
    if show_progress and len(chunk_starts) > 1:
        iterator = tqdm(iterator, desc="Retargeting (chunked)")
    elif show_progress:
        iterator = tqdm(frames, desc="Retargeting")

    qpos_list: list[np.ndarray] = []
    for start in chunk_starts:
        end = min(start + chunk_size, n)
        if start == 0:
            retargeter.seed_configuration_from_human(frames[start])
        else:
            retargeter.set_configuration_qpos(qpos_list[-1])
        for frame in frames[start:end]:
            qpos_list.append(retargeter.retarget(frame))

    tls = retargeter.temporal_limit_smooth
    if apply_temporal_smooth is None:
        apply_temporal_smooth = bool(tls.get("enabled", False))
    if apply_temporal_smooth and len(qpos_list) > 1:
        joint_penalty = getattr(retargeter, "joint_limit_penalty_enabled", False)
        smooth_cfg = dict(tls)
        if joint_penalty and "joints" not in smooth_cfg:
            smooth_cfg["joints"] = [
                e["name"] for e in getattr(retargeter, "_limit_penalty_entries", [])
            ]
        if "soft_margin_ratio" not in smooth_cfg:
            smooth_cfg["soft_margin_ratio"] = getattr(
                retargeter, "limit_penalty_soft_margin_ratio", 0.12
            )
        qpos_list = GeneralMotionRetargeting.smooth_qpos_sequence_limit_aware(
            retargeter.model, qpos_list, smooth_cfg
        )

    return qpos_list

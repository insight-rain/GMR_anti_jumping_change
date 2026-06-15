# GMR Anti Jumping Change

基于 [General Motion Retargeting (GMR)](https://github.com/YanjieZe/GMR) 的改进 fork，主要解决 **motion retargeting 时关节贴限进入死区、帧间突变（jumping）** 的问题。

> **Upstream:** [YanjieZe/GMR](https://github.com/YanjieZe/GMR) · MIT License  
> **This fork:** IK soft joint-limit penalty + limit-aware temporal smoothing

---

## What changed (vs upstream GMR)

| Feature | Description |
|---------|-------------|
| **Soft joint-limit penalty** | Dynamic mink `PostureTask` toward joint midpoint; cost grows exponentially near limits |
| **Limit margin on hard bounds** | Configurable `ConfigurationLimit(min_distance_from_limits=...)` |
| **Continuity decoupling** | When `arm_continuity` is enabled in IK JSON, reduce pull toward previous frame near limits |
| **Soft post-projection** | After IK, gently pull joints away from limits; skip hard clipping when soft penalty is on |
| **Batch bidirectional smooth** | Forward/backward EMA with stronger smoothing near limits (`parallel_retarget.py`) |
| **Diagnostics** | `scripts/analyze_joint_margins.py` — margin stats and \|Δq\| peaks |

Works with **any robot already supported by GMR** (Unitree G1, Booster T1, …). Enable via IK JSON blocks — no new robot assets required.

See [docs/UPSTREAM_DIFF.md](docs/UPSTREAM_DIFF.md) for file-level details.

---

## Install

```bash
git clone https://github.com/insight-rain/GMR_anti_jumping_change.git
cd GMR_anti_jumping_change

conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
conda install -c conda-forge libstdcxx-ng -y   # Linux
```

Same dependencies as upstream GMR (`mink`, `mujoco`, `smplx`, …).

---

## IK config

Add to any existing file under `general_motion_retargeting/ik_configs/`, e.g. `smplx_to_g1.json`:

```json
"joint_limit_penalty": {
  "enabled": true,
  "base_cost": 0.08,
  "exp_k": 4.0,
  "soft_margin_ratio": 0.12,
  "continuity_decay_threshold": 0.08,
  "gain": 0.95,
  "joints": ["left_elbow_joint", "right_elbow_joint"]
},
"temporal_limit_smooth": {
  "enabled": true,
  "forward_alpha": 0.35,
  "backward_alpha": 0.35,
  "limit_aware": true,
  "exp_k": 3.0
}
```

Full example: [examples/ik_joint_limit_snippet.json](examples/ik_joint_limit_snippet.json)

Set `"enabled": false` to revert to upstream-like behavior for that config.

---

## Usage

### Retarget (same as upstream)

```bash
python scripts/smplx_to_robot.py \
  --smplx_file /path/to/motion.npz \
  --robot unitree_g1 \
  --save_path output/motion.pkl
```

### Analyze margins / jumps

```bash
python scripts/analyze_joint_margins.py \
  --robot unitree_g1 \
  --motion output/motion.pkl \
  --joints left_elbow_joint right_elbow_joint
```

### Long sequences + temporal smooth

Use `general_motion_retargeting.parallel_retarget.retarget_frames_parallel()` in your batch script. When `temporal_limit_smooth.enabled` is set in IK JSON, a bidirectional limit-aware pass runs after retargeting.

---

## License

MIT License — see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

This fork is a derivative of [GMR](https://github.com/YanjieZe/GMR) by Yanjie Ze et al. You must retain the upstream copyright notice.

---

## Citation

Please cite the original GMR paper / repository: https://github.com/YanjieZe/GMR

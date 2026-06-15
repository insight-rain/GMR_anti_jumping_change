# GMR Anti Jumping Change

基于 [General Motion Retargeting (GMR)](https://github.com/YanjieZe/GMR) 的改进 fork，主要解决 **motion retargeting 时关节贴限进入死区、帧间突变（jumping）** 的问题。

> **上游：** [YanjieZe/GMR](https://github.com/YanjieZe/GMR) · MIT License  
> **本 fork：** IK 软关节限位惩罚 + 贴限感知时序平滑  
> **English:** [README.md](README.md)

**仓库：** https://github.com/insight-rain/GMR_anti_jumping_change

---

## 相对上游 GMR 的改动

| 功能 | 说明 |
|------|------|
| **软关节限位惩罚** | 动态 mink `PostureTask`，目标为关节中点；越接近限位，代价指数增大 |
| **硬限位内缩边距** | 可配置 `ConfigurationLimit(min_distance_from_limits=...)` |
| **连续性解耦** | 若 IK 配置启用 `arm_continuity`，贴限处自动衰减对上帧的拉力 |
| **软后投影** | IK 后 gently 将关节拉离限位；启用软惩罚时跳过硬截断 |
| **Batch 双向平滑** | 前向/后向 EMA，贴限处加强平滑（`parallel_retarget.py`） |
| **诊断工具** | `scripts/analyze_joint_margins.py` — 统计 margin 与 \|Δq\| 峰值 |

适用于 **GMR 已支持的所有机器人**（Unitree G1、Booster T1 等）。只需在 IK JSON 中启用配置块，**无需新增 robot 资产**。

文件级细节见 [docs/UPSTREAM_DIFF.md](docs/UPSTREAM_DIFF.md)。

---

## 安装

```bash
git clone https://github.com/insight-rain/GMR_anti_jumping_change.git
cd GMR_anti_jumping_change

conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
conda install -c conda-forge libstdcxx-ng -y   # Linux
```

依赖与上游 GMR 相同（`mink`、`mujoco`、`smplx` 等）。

---

## IK 配置

在 `general_motion_retargeting/ik_configs/` 下任意已有文件中追加，例如 `smplx_to_g1.json`：

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

完整示例：[examples/ik_joint_limit_snippet.json](examples/ik_joint_limit_snippet.json)

将 `"enabled": false` 可恢复为接近上游的行为。

---

## 使用

### 重定向（与上游相同）

```bash
python scripts/smplx_to_robot.py \
  --smplx_file /path/to/motion.npz \
  --robot unitree_g1 \
  --save_path output/motion.pkl
```

### 分析 margin / 跳变

```bash
python scripts/analyze_joint_margins.py \
  --robot unitree_g1 \
  --motion output/motion.pkl \
  --joints left_elbow_joint right_elbow_joint
```

### 长序列 + 时序平滑

在 batch 脚本中调用 `general_motion_retargeting.parallel_retarget.retarget_frames_parallel()`。若 IK JSON 中 `temporal_limit_smooth.enabled` 为 true，retarget 结束后会自动运行贴限感知双向平滑。

---

## 许可证

MIT License — 见 [LICENSE](LICENSE) 与 [NOTICE.md](NOTICE.md)。

本 fork 为 [GMR](https://github.com/YanjieZe/GMR)（Yanjie Ze 等）的衍生作品，须保留上游版权声明。

---

## 引用

请引用原始 GMR 论文 / 仓库：https://github.com/YanjieZe/GMR

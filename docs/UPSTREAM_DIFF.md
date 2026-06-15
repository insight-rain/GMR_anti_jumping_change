# 与 upstream GMR (`origin/master`) 的差异对照

基准：上游 [`YanjieZe/GMR`](https://github.com/YanjieZe/GMR) `motion_retarget.py` ≈ 313 行，仅 FrameTask + 硬 `ConfigurationLimit`，无后处理、无 batch 平滑。

以下对照 **当前本地工作区** 相对上游的全部改动，并标明 **应对外公开（本 patch）** 还是 **仅本地私有集成（勿公开）**。

---

## 应合并 — 关节限位优化（本 patch 范围）

### 1. `general_motion_retargeting/motion_retarget.py`

| 符号 / 逻辑 | 作用 |
|-------------|------|
| `_load_joint_limit_penalty` | 读 IK JSON `joint_limit_penalty` |
| `_setup_ik_limits` | `ConfigurationLimit(gain, min_distance_from_limits)` |
| `_update_limit_penalty_costs` | 动态 PostureTask cost → `q_mid` |
| `_update_continuity_costs` | 贴限时衰减 `arm_continuity`（**仅当** JSON 已启用 continuity） |
| `_apply_soft_limit_projection` | IK 后软投影 |
| `_normalized_joint_margin` / `collect_limited_hinge_info` | margin 工具 |
| `smooth_qpos_sequence_limit_aware` | 双向 limit-aware EMA |
| `_limit_aware_alpha` | 平滑强度随 margin 变化 |
| `limit_penalty_task` + `_active_ik_tasks` 扩展 | 每步 IK 注入软限位 task |
| `retarget()` 末尾 | 软投影；`joint_limit_penalty` 开启时跳过硬 clip |

**合并建议**：从本地 `motion_retarget.py` **只摘上述逻辑**；不要带上游没有的 AIQ11 专用默认关节名（如 `l_arm_4` 写死在默认参数里应改为通用或仅 JSON 配置）。

### 2. `general_motion_retargeting/parallel_retarget.py`（上游不存在，**新文件**）

- `retarget_frames_parallel()`：顺序 chunk + warm-start  
- 读取 `temporal_limit_smooth`，调用 `smooth_qpos_sequence_limit_aware`  

**合并建议**：独立新文件；上游 batch 脚本可按需 **可选** 引用（非必须改 upstream 全部 scripts）。

### 3. `scripts/analyze_joint_margins.py`（**新文件**）

- 读 GMR 标准 pkl，`--robot` 用 `params.ROBOT_XML_DICT` 已有机器人名  
- **不要**写死 `aiq11` 为唯一 robot  

### 4. IK 配置（**仅追加 JSON 块，不新增机器人**）

对 **已有** 配置文件追加字段，例如：

- `general_motion_retargeting/ik_configs/smplx_to_g1.json`  
- 或文档示例 [`examples/ik_joint_limit_snippet.json`](./examples/ik_joint_limit_snippet.json)  

字段：`joint_limit_penalty`、`temporal_limit_smooth`  

**不要**提交：`smplx_to_aiq11*.json`、`bvh_ost*.json` 等私有机器人配置。

---

## 可选合并 — 通用时序机制（非 Layer 1 必需，但与软限位常一起用）

本地 `motion_retarget.py` 还包含上游没有的 **通用、配置驱动** 能力。若希望 PR 更小，可 **不** 包含；若希望舞蹈/长序列更稳，可一并提交：

| 符号 | 说明 | 建议 |
|------|------|------|
| `arm_continuity` + `PostureTask` | 拉向上帧 qpos | 可选；与 `_update_continuity_costs` 配合 |
| `qpos_smooth` / `_smooth_qpos` | 单帧 EMA | 可选 |
| `seed_configuration_from_human` / `set_configuration_qpos` | batch warm-start | 与 `parallel_retarget` 配套则建议提交 |
| `_validate_ik_robot_frames` | 启动时校验 body 名 | 可选小改进 |
| `smooth_qpos_sequence` | 普通 EMA batch | 可选 |

---

## 勿合并 — 本地私有 / 与上游 GMR 无关

### 代码与配置

| 路径 | 原因 |
|------|------|
| `general_motion_retargeting/params.py` 中 `aiq11` / `aiq01` / `bvh_ost*` | 私有机器人路由 |
| `assets/aiq11_*` / `assets/aiq01/` | 私有模型 |
| `general_motion_retargeting/ik_configs/*aiq11*` / `*aiq01*` / `bvh_ost*` | 私有 IK |
| `general_motion_retargeting/ground_adjust.py` | EIR/贴地 CSV 专用 |
| `output/gmr_pkl_to_csv.py` | EIR CSV 导出 |
| `scripts/bvh_to_robot.py` 大改（OST/aiq11/parallel） | 光捕私有管线 |
| `scripts/vis_robot_motion.py` 大改 | AIQ11 可视化扩展 |
| `general_motion_retargeting/robot_motion_viewer.py` 大改 | 同上 |
| `general_motion_retargeting/utils/lafan1.py` OST 扩展 | `ost_take019` 等 |
| `scripts/amass_retarget_pipeline.py`、sim2sim、train_mujoco* | 部署/RL 私有 |

### 数据目录

| 路径 | 原因 |
|------|------|
| `import/` | 原始 BVH / AMASS |
| `output/` | pkl / csv 产物 |
| `wandb/` | 训练日志 |

---

## 上游已有、本地 **未改** 的部分

以下保持与 [YanjieZe/GMR](https://github.com/YanjieZe/GMR) 一致即可：

- `setup.py` 依赖  
- 上游已有机器人 `assets/unitree_g1/` 等  
- 上游 `scripts/smplx_to_robot.py` 基本流程（除非你要 **可选** 接 `parallel_retarget`）  
- 上游 `ik_configs/smplx_to_g1.json` 等（仅 **追加** 软限位 JSON 块）

---

## 当前 git 状态说明（便于 cherry-pick）

相对 `origin/master` **已修改且混入私有内容** 的 tracked 文件：

```
general_motion_retargeting/motion_retarget.py   ← 需手工拆出 patch 部分
general_motion_retargeting/params.py            ← 勿整文件提交
general_motion_retargeting/robot_motion_viewer.py
general_motion_retargeting/kinematics_model.py
general_motion_retargeting/utils/lafan1.py
scripts/bvh_to_robot.py
scripts/vis_robot_motion.py
scripts/smplx_to_robot.py
scripts/smplx_to_robot_dataset.py
scripts/batch_gmr_pkl_to_csv.py
...
```

**推荐流程**：

1. 从 upstream GMR 新 fork 一支 `feature/ik-soft-joint-limits`  
2. 只复制 [应合并](#应合并--关节限位优化本-patch-范围) 中的文件/代码段  
3. 在一个 upstream 机器人（如 `unitree_g1`）的 IK JSON 加示例配置  
4. 用 `analyze_joint_margins.py` 在 G1 上验证  

---

## 验证（仅用上游机器人）

```bash
conda activate gmr
pip install -e .

python scripts/smplx_to_robot.py \
  --smplx_file <amass_sample.npz> \
  --robot unitree_g1 \
  --save_path /tmp/g1_test.pkl

python scripts/analyze_joint_margins.py \
  --robot unitree_g1 \
  --motion /tmp/g1_test.pkl
```

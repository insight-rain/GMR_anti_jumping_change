# 公开仓库中 **不要** 包含的内容

本文档列出本地 GMR 工作区里存在、但 **不应** 出现在「仅 upstream + 关节限位优化」公开仓库中的路径。

---

## 数据与产物

```
import/                 # 原始 BVH、AMASS npz 等
output/                 # 全部 pkl、csv、视频
wandb/
*.pkl
*.csv                   # 重定向产物（文档示例除外）
```

---

## 私有机器人与资产

```
assets/aiq11_description/
assets/aiq11_gmr/
assets/aiq11_*.xml
assets/aiq01/
assets/simplified_meshes/    # 若仅服务 aiq11
import/aiq01/
```

---

## 私有 IK 与路由

```
general_motion_retargeting/ik_configs/smplx_to_aiq11*.json
general_motion_retargeting/ik_configs/bvh_ost*.json
general_motion_retargeting/ik_configs/bvh_ost_rollin_to_aiq01.json
general_motion_retargeting/ik_configs/smplx_to_aiq01.json

# params.py 中以下条目不应出现在公开 patch PR：
#   aiq11, aiq11_bvh_ost, aiq01
#   IK_CONFIG_DICT 的 bvh_ost, bvh_ost_take019, bvh_ost_rollin
```

---

## 私有脚本与管线

```
output/gmr_pkl_to_csv.py
scripts/amass_retarget_pipeline.py
scripts/bvh_to_robot.py          # 若含 OST/aiq11 专用逻辑，勿整文件提交
scripts/vis_aiq11_csv.py
scripts/gmr_csv_to_beyondmimic.py
scripts/mujoco_sim2sim*.py
scripts/train_mujoco*.py
scripts/s2s*.py
scripts/aiq11_*.py
configs/                         # 本地 AMASS 批处理配置
agent须知/
github_release/                  # 本地 AIQ11 全栈发布说明（可保留在私有仓，勿与 patch 混淆）
```

---

## 建议 `.gitignore`（公开 patch 仓）

```gitignore
import/
output/
wandb/
__pycache__/
*.pkl
*.egg-info/
.conda/
```

---

## 两个说明文件夹的区别

| 文件夹 | 用途 |
|--------|------|
| **`gmr_joint_limit_patch/`**（本目录） | 只对 upstream GMR 的 **通用 IK 软限位** patch |
| **`github_release/`** | 本地 **AIQ11 全栈** 集成说明（模型、BVH、CSV）；**不要**当作 upstream PR 内容 |

两者都保留，互不删除。

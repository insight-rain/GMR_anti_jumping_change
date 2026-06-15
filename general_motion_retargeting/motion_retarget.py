
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
        qpos_smooth_override: float | None = None,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        self.qpos_smooth = float(ik_config.get("qpos_smooth", 0.0))
        if qpos_smooth_override is not None:
            self.qpos_smooth = float(qpos_smooth_override)
        self.qpos_smooth_arm = float(
            ik_config.get("qpos_smooth_arm", self.qpos_smooth)
        )
        self._prev_qpos = None
        self.posture_task = None
        self.limit_penalty_task = None
        self._continuity_base_costs = None
        self._arm_hinge_qpos_adrs = self._collect_arm_hinge_qpos_adrs()
        self._load_elbow_constraint(ik_config.get("elbow_constraint", {}))
        self._load_arm_step_limit(ik_config.get("arm_step_limit", {}))
        self.temporal_limit_smooth = ik_config.get("temporal_limit_smooth", {})
        self._load_joint_limit_penalty(ik_config.get("joint_limit_penalty", {}))

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self._setup_ik_limits(ik_config.get("joint_limit_penalty", {}))
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))
            
        self.setup_retarget_configuration(ik_config)
        self._validate_ik_robot_frames()
        
        self.ground_offset = 0.0

    def _validate_ik_robot_frames(self):
        for table_name, table in (
            ("ik_match_table1", self.ik_match_table1),
            ("ik_match_table2", self.ik_match_table2),
        ):
            for frame_name, entry in table.items():
                pos_weight, rot_weight = entry[1], entry[2]
                if pos_weight == 0 and rot_weight == 0:
                    continue
                if frame_name not in self.robot_body_names:
                    available = sorted(self.robot_body_names.keys())
                    raise ValueError(
                        f"{table_name}: robot body '{frame_name}' not in model "
                        f"({self.xml_file}). Available: {available}"
                    )

    def _collect_arm_hinge_qpos_adrs(self):
        adrs = []
        for jnt_id in range(self.model.njnt):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
            if (
                name
                and name.startswith(("l_arm_", "r_arm_"))
                and self.model.jnt_type[jnt_id] == mj.mjtJoint.mjJNT_HINGE
            ):
                adrs.append(int(self.model.jnt_qposadr[jnt_id]))
        return adrs

    def _load_elbow_constraint(self, elbow_constraint):
        self.elbow_constraint_enabled = bool(elbow_constraint.get("enabled", False))
        self.elbow_joint_names = list(
            elbow_constraint.get("joints", ["l_arm_4", "r_arm_4"])
        )
        self.elbow_max_step = float(elbow_constraint.get("max_step_rad", 0.4))
        self.elbow_flex_upper_bound = float(
            elbow_constraint.get("flex_upper_bound", 0.05)
        )

    def _load_arm_step_limit(self, arm_step_limit):
        self.arm_step_limit_enabled = bool(arm_step_limit.get("enabled", False))
        self.arm_step_limit_joints = list(
            arm_step_limit.get(
                "joints",
                [
                    "l_arm_1",
                    "l_arm_2",
                    "l_arm_3",
                    "l_arm_4",
                    "r_arm_1",
                    "r_arm_2",
                    "r_arm_3",
                    "r_arm_4",
                ],
            )
        )
        self.arm_step_limit_rad = float(arm_step_limit.get("max_step_rad", 0.55))

    @staticmethod
    def _normalized_joint_margin(q, lo, hi):
        travel = float(hi - lo)
        if travel <= 1e-9:
            return 0.5
        return float(np.clip(min(q - lo, hi - q) / travel, 0.0, 0.5))

    def _setup_ik_limits(self, limit_cfg):
        gain = float(limit_cfg.get("gain", 0.95))
        soft_margin_ratio = float(limit_cfg.get("soft_margin_ratio", 0.0))
        min_distance = 0.0
        if limit_cfg.get("enabled", False) and soft_margin_ratio > 0.0:
            margins = []
            for entry in self._limit_penalty_entries:
                lo, hi = entry["lo"], entry["hi"]
                margins.append(soft_margin_ratio * (hi - lo))
            if margins:
                min_distance = float(min(margins))
        self.ik_limits = [
            mink.ConfigurationLimit(
                self.model,
                gain=gain,
                min_distance_from_limits=min_distance,
            )
        ]

    def _load_joint_limit_penalty(self, limit_cfg):
        self.joint_limit_penalty_enabled = bool(limit_cfg.get("enabled", False))
        self.limit_penalty_base_cost = float(limit_cfg.get("base_cost", 0.08))
        self.limit_penalty_exp_k = float(limit_cfg.get("exp_k", 4.0))
        self.limit_penalty_soft_margin_ratio = float(
            limit_cfg.get("soft_margin_ratio", 0.12)
        )
        self.continuity_decay_threshold = float(
            limit_cfg.get("continuity_decay_threshold", 0.08)
        )
        joint_cost_override = limit_cfg.get("joint_cost_override", {})
        configured_joints = limit_cfg.get("joints")

        self._limit_penalty_entries = []
        self._limit_penalty_target_q = self.model.qpos0.copy().astype(float)
        self._limit_penalty_costs = np.zeros(self.model.nv, dtype=float)

        if not self.joint_limit_penalty_enabled:
            return

        for jnt_id in range(self.model.njnt):
            jnt_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
            if not jnt_name:
                continue
            if configured_joints is not None and jnt_name not in configured_joints:
                continue
            if self.model.jnt_type[jnt_id] not in (
                mj.mjtJoint.mjJNT_HINGE,
                mj.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            if not self.model.jnt_limited[jnt_id]:
                continue

            lo, hi = self.model.jnt_range[jnt_id]
            qpos_adr = int(self.model.jnt_qposadr[jnt_id])
            dof_adr = int(self.model.jnt_dofadr[jnt_id])
            base_cost = float(joint_cost_override.get(jnt_name, self.limit_penalty_base_cost))
            q_mid = 0.5 * (lo + hi)
            self._limit_penalty_target_q[qpos_adr] = q_mid
            self._limit_penalty_entries.append(
                {
                    "name": jnt_name,
                    "qpos_adr": qpos_adr,
                    "dof_adr": dof_adr,
                    "lo": float(lo),
                    "hi": float(hi),
                    "mid": float(q_mid),
                    "base_cost": base_cost,
                }
            )

    def _update_limit_penalty_costs(self, qpos):
        costs = np.zeros(self.model.nv, dtype=float)
        exp_k = self.limit_penalty_exp_k
        for entry in self._limit_penalty_entries:
            q = float(qpos[entry["qpos_adr"]])
            margin = self._normalized_joint_margin(q, entry["lo"], entry["hi"])
            weight = entry["base_cost"] * np.exp(exp_k * (1.0 - 2.0 * margin))
            costs[entry["dof_adr"]] = weight
        self._limit_penalty_costs = costs
        if self.limit_penalty_task is not None:
            self.limit_penalty_task.set_cost(costs)
            self.limit_penalty_task.set_target(self._limit_penalty_target_q)

    def _update_continuity_costs(self, qpos):
        if (
            self.posture_task is None
            or self._continuity_base_costs is None
            or not self.joint_limit_penalty_enabled
        ):
            return

        costs = self._continuity_base_costs.copy()
        threshold = self.continuity_decay_threshold
        for entry in self._limit_penalty_entries:
            dof_adr = entry["dof_adr"]
            if costs[dof_adr] <= 0.0:
                continue
            q = float(qpos[entry["qpos_adr"]])
            margin = self._normalized_joint_margin(q, entry["lo"], entry["hi"])
            if margin < threshold:
                costs[dof_adr] *= margin / threshold
        self.posture_task.set_cost(costs)

    def _apply_soft_limit_projection(self, qpos):
        if not self.joint_limit_penalty_enabled:
            return qpos

        out = np.asarray(qpos, dtype=float).copy()
        soft_ratio = self.limit_penalty_soft_margin_ratio
        for entry in self._limit_penalty_entries:
            adr = entry["qpos_adr"]
            lo, hi, q_mid = entry["lo"], entry["hi"], entry["mid"]
            val = float(out[adr])
            margin = self._normalized_joint_margin(val, lo, hi)
            soft_margin = soft_ratio * 0.5
            if margin < soft_margin:
                beta = 1.0 - margin / soft_margin
                val = val + beta * 0.35 * (q_mid - val)
            out[adr] = np.clip(val, lo, hi)
        return out

    @classmethod
    def collect_limited_hinge_info(cls, model, joint_names=None):
        """Return per-joint limit metadata for margin analysis / smoothing."""
        entries = []
        for jnt_id in range(model.njnt):
            jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
            if not jnt_name:
                continue
            if joint_names is not None and jnt_name not in joint_names:
                continue
            if model.jnt_type[jnt_id] != mj.mjtJoint.mjJNT_HINGE:
                continue
            if not model.jnt_limited[jnt_id]:
                continue
            lo, hi = model.jnt_range[jnt_id]
            entries.append(
                {
                    "name": jnt_name,
                    "qpos_adr": int(model.jnt_qposadr[jnt_id]),
                    "lo": float(lo),
                    "hi": float(hi),
                    "mid": float(0.5 * (lo + hi)),
                }
            )
        return entries

    def _hinge_qpos_adr(self, joint_name):
        jnt_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if jnt_id < 0:
            return None
        return int(self.model.jnt_qposadr[jnt_id])

    def _apply_elbow_constraint(self, qpos):
        if not self.elbow_constraint_enabled or self._prev_qpos is None:
            return qpos

        out = np.asarray(qpos, dtype=float).copy()
        for joint_name in self.elbow_joint_names:
            adr = self._hinge_qpos_adr(joint_name)
            if adr is None:
                continue
            prev = float(self._prev_qpos[adr])
            val = float(out[adr])
            lo, hi = self.model.jnt_range[
                mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            ]
            hi = min(hi, self.elbow_flex_upper_bound)
            val = np.clip(val, lo, hi)
            if self.elbow_max_step > 0.0:
                val = prev + np.clip(val - prev, -self.elbow_max_step, self.elbow_max_step)
            out[adr] = np.clip(val, lo, hi)
        return out

    def _apply_arm_step_limit(self, qpos):
        if not self.arm_step_limit_enabled or self._prev_qpos is None:
            return qpos

        out = np.asarray(qpos, dtype=float).copy()
        for joint_name in self.arm_step_limit_joints:
            adr = self._hinge_qpos_adr(joint_name)
            if adr is None:
                continue
            jnt_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            prev = float(self._prev_qpos[adr])
            val = float(out[adr])
            if self.model.jnt_limited[jnt_id]:
                lo, hi = self.model.jnt_range[jnt_id]
                val = prev + np.clip(
                    val - prev, -self.arm_step_limit_rad, self.arm_step_limit_rad
                )
                out[adr] = np.clip(val, lo, hi)
            else:
                out[adr] = prev + np.clip(
                    val - prev, -self.arm_step_limit_rad, self.arm_step_limit_rad
                )
        return out

    def _build_joint_continuity_cost(self, arm_continuity):
        default_cost = float(arm_continuity.get("default_cost", 0.0))
        arm_cost = float(arm_continuity.get("arm_cost", 0.0))
        joint_cost = arm_continuity.get("joint_cost", {})
        costs = np.full(self.model.nv, default_cost, dtype=float)
        for i in range(self.model.nv):
            jnt_name = mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i]
            )
            if jnt_name in joint_cost:
                costs[i] = float(joint_cost[jnt_name])
            elif jnt_name.startswith(("l_arm_", "r_arm_")):
                costs[i] = arm_cost
        return costs

    def setup_retarget_configuration(self, ik_config):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        self.posture_task = None
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

        arm_continuity = ik_config.get("arm_continuity", {})
        if arm_continuity.get("enabled", False):
            costs = self._build_joint_continuity_cost(arm_continuity)
            if np.any(costs > 0.0):
                self._continuity_base_costs = costs.copy()
                self.posture_task = mink.PostureTask(self.model, cost=costs)

        if self.joint_limit_penalty_enabled and self._limit_penalty_entries:
            self.limit_penalty_task = mink.PostureTask(
                self.model, cost=self._limit_penalty_costs
            )
            self.limit_penalty_task.set_target(self._limit_penalty_target_q)

    def _active_ik_tasks(self, base_tasks):
        tasks = list(base_tasks)
        qpos = self.configuration.data.qpos.copy()

        if self.limit_penalty_task is not None:
            self._update_limit_penalty_costs(qpos)
            tasks.append(self.limit_penalty_task)

        if self.posture_task is not None and self._prev_qpos is not None:
            self.posture_task.set_target(self._prev_qpos)
            if self.joint_limit_penalty_enabled:
                self._update_continuity_costs(qpos)
            elif self._continuity_base_costs is not None:
                self.posture_task.set_cost(self._continuity_base_costs)
            tasks.append(self.posture_task)
        return tasks

    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            
            
    def retarget(self, human_data, offset_to_ground=False):
        self.update_targets(human_data, offset_to_ground)

        if self.use_ik_match_table1:
            tasks1 = self._active_ik_tasks(self.tasks1)
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            tasks2 = self._active_ik_tasks(self.tasks2)
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                next_error = self.error2()
                num_iter += 1

        qpos = self.configuration.data.qpos.copy()
        qpos = self._apply_soft_limit_projection(qpos)
        if not self.joint_limit_penalty_enabled:
            qpos = self._apply_elbow_constraint(qpos)
            qpos = self._apply_arm_step_limit(qpos)
        self.configuration.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.configuration.data)

        if self.qpos_smooth > 0.0:
            qpos = self._smooth_qpos(qpos)
            self.configuration.data.qpos[:] = qpos
            mj.mj_forward(self.model, self.configuration.data)
        else:
            self._prev_qpos = qpos.copy()
        return qpos

    def _hinge_smooth_alpha(self, qpos_adr):
        if self.qpos_smooth <= 0.0:
            return 1.0
        if qpos_adr in self._arm_hinge_qpos_adrs and self.qpos_smooth_arm > 0.0:
            return self.qpos_smooth_arm
        return self.qpos_smooth

    def _smooth_qpos(self, qpos_new):
        if self._prev_qpos is None:
            self._prev_qpos = qpos_new.copy()
            return qpos_new

        out = qpos_new.copy()
        for jnt_id in range(self.model.njnt):
            adr = self.model.jnt_qposadr[jnt_id]
            jnt_type = self.model.jnt_type[jnt_id]
            if jnt_type == mj.mjtJoint.mjJNT_FREE:
                alpha = self.qpos_smooth
                out[adr : adr + 3] = (
                    alpha * qpos_new[adr : adr + 3]
                    + (1.0 - alpha) * self._prev_qpos[adr : adr + 3]
                )
                q_prev = self._prev_qpos[adr + 3 : adr + 7]
                q_new = qpos_new[adr + 3 : adr + 7]
                out[adr + 3 : adr + 7] = self._slerp_quat_wxyz(q_prev, q_new, alpha)
            elif jnt_type in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                alpha = self._hinge_smooth_alpha(int(adr))
                out[adr] = (
                    alpha * qpos_new[adr] + (1.0 - alpha) * self._prev_qpos[adr]
                )

        self._prev_qpos = out.copy()
        return out

    @staticmethod
    def _slerp_quat_wxyz(q0, q1, t):
        from scipy.spatial.transform import Slerp

        rots = R.from_quat(np.stack([q0, q1]), scalar_first=True)
        return Slerp([0.0, 1.0], rots)([t]).as_quat(scalar_first=True)[0]

    @classmethod
    def smooth_qpos_sequence(cls, model, qpos_sequence, alpha):
        """Temporal EMA over a qpos trajectory (for parallel post-pass)."""
        if alpha <= 0.0 or len(qpos_sequence) == 0:
            return list(qpos_sequence)

        helper = cls.__new__(cls)
        helper.model = model
        helper.qpos_smooth = alpha
        helper._prev_qpos = None
        helper._arm_hinge_qpos_adrs = []

        smoothed = []
        for qpos in qpos_sequence:
            smoothed.append(helper._smooth_qpos(np.asarray(qpos, dtype=float)))
        return smoothed

    @classmethod
    def _limit_aware_alpha(cls, base_alpha, margin, exp_k, limit_aware):
        if not limit_aware or base_alpha <= 0.0:
            return base_alpha
        boost = np.exp(exp_k * (1.0 - 2.0 * margin))
        return float(base_alpha / (1.0 + 0.5 * (boost - 1.0)))

    @classmethod
    def smooth_qpos_sequence_limit_aware(cls, model, qpos_sequence, cfg):
        """Forward-backward EMA with stronger smoothing near joint limits."""
        if len(qpos_sequence) == 0:
            return []

        forward_alpha = float(cfg.get("forward_alpha", 0.35))
        backward_alpha = float(cfg.get("backward_alpha", 0.35))
        limit_aware = bool(cfg.get("limit_aware", True))
        exp_k = float(cfg.get("exp_k", 3.0))
        soft_margin_ratio = float(cfg.get("soft_margin_ratio", 0.12))
        joint_names = cfg.get("joints")
        limit_entries = cls.collect_limited_hinge_info(model, joint_names)

        seq = [np.asarray(q, dtype=float).copy() for q in qpos_sequence]
        n = len(seq)

        fwd = [seq[0].copy()]
        for t in range(1, n):
            out = seq[t].copy()
            prev = fwd[t - 1]
            for jnt_id in range(model.njnt):
                adr = model.jnt_qposadr[jnt_id]
                jnt_type = model.jnt_type[jnt_id]
                if jnt_type == mj.mjtJoint.mjJNT_FREE:
                    alpha = forward_alpha
                    out[adr : adr + 3] = (
                        alpha * seq[t][adr : adr + 3]
                        + (1.0 - alpha) * prev[adr : adr + 3]
                    )
                    out[adr + 3 : adr + 7] = cls._slerp_quat_wxyz(
                        prev[adr + 3 : adr + 7],
                        seq[t][adr + 3 : adr + 7],
                        alpha,
                    )
                elif jnt_type in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                    alpha = forward_alpha
                    for entry in limit_entries:
                        if entry["qpos_adr"] == adr:
                            margin = cls._normalized_joint_margin(
                                seq[t][adr], entry["lo"], entry["hi"]
                            )
                            alpha = cls._limit_aware_alpha(
                                forward_alpha, margin, exp_k, limit_aware
                            )
                            break
                    out[adr] = alpha * seq[t][adr] + (1.0 - alpha) * prev[adr]
            fwd.append(out)

        bwd = [fwd[-1].copy()]
        for t in range(n - 2, -1, -1):
            nxt = bwd[0]
            out = fwd[t].copy()
            for jnt_id in range(model.njnt):
                adr = model.jnt_qposadr[jnt_id]
                jnt_type = model.jnt_type[jnt_id]
                if jnt_type == mj.mjtJoint.mjJNT_FREE:
                    alpha = backward_alpha
                    out[adr : adr + 3] = (
                        alpha * fwd[t][adr : adr + 3]
                        + (1.0 - alpha) * nxt[adr : adr + 3]
                    )
                    out[adr + 3 : adr + 7] = cls._slerp_quat_wxyz(
                        fwd[t][adr + 3 : adr + 7],
                        nxt[adr + 3 : adr + 7],
                        alpha,
                    )
                elif jnt_type in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                    alpha = backward_alpha
                    for entry in limit_entries:
                        if entry["qpos_adr"] == adr:
                            margin = cls._normalized_joint_margin(
                                fwd[t][adr], entry["lo"], entry["hi"]
                            )
                            alpha = cls._limit_aware_alpha(
                                backward_alpha, margin, exp_k, limit_aware
                            )
                            break
                    out[adr] = alpha * fwd[t][adr] + (1.0 - alpha) * nxt[adr]
            bwd.insert(0, out)

        if limit_aware and limit_entries:
            soft_margin = soft_margin_ratio * 0.5
            for out in bwd:
                for entry in limit_entries:
                    adr = entry["qpos_adr"]
                    val = float(out[adr])
                    margin = cls._normalized_joint_margin(
                        val, entry["lo"], entry["hi"]
                    )
                    if margin < soft_margin:
                        beta = 1.0 - margin / soft_margin
                        val = val + beta * 0.25 * (entry["mid"] - val)
                        out[adr] = np.clip(val, entry["lo"], entry["hi"])

        return bwd

    def set_configuration_qpos(self, qpos):
        self.configuration.data.qpos[:] = qpos
        self._prev_qpos = np.asarray(qpos, dtype=float).copy()
        mj.mj_forward(self.model, self.configuration.data)

    def seed_configuration_from_human(self, human_data):
        """Initialize IK from scaled human root pose (for parallel chunk warm-start)."""
        self._prev_qpos = None
        self.configuration.data.qpos[:] = self.model.qpos0
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(
            human_data, self.human_root_name, self.human_scale_table
        )
        root_pos, root_quat = human_data[self.human_root_name]
        jnt_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if jnt_id >= 0 and self.model.jnt_type[jnt_id] == mj.mjtJoint.mjJNT_FREE:
            adr = self.model.jnt_qposadr[jnt_id]
            self.configuration.data.qpos[adr : adr + 3] = root_pos
            self.configuration.data.qpos[adr + 3 : adr + 7] = root_quat
        mj.mj_forward(self.model, self.configuration.data)

    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if body_name not in rot_offsets:
                offset_human_data[body_name] = [pos, quat]
                continue
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data


class HumanMotionPreprocessor:
    """Apply GMR human scale + ik_match_table1 offsets (no robot IK)."""

    def __init__(self, ik_config, actual_human_height=None) -> None:
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0

        self.human_root_name = ik_config["human_root_name"]
        self.human_scale_table = {
            key: ik_config["human_scale_table"][key] * ratio
            for key in ik_config["human_scale_table"].keys()
        }
        self.ground_offset = 0.0
        ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        for entry in ik_config["ik_match_table1"].values():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight == 0 and rot_weight == 0:
                continue
            self.pos_offsets1[body_name] = np.array(pos_offset) - ground
            self.rot_offsets1[body_name] = R.from_quat(rot_offset, scalar_first=True)

        self._helper = GeneralMotionRetargeting.__new__(GeneralMotionRetargeting)

    def __call__(self, human_data, offset_to_ground=False):
        human_data = self._helper.to_numpy(human_data)
        human_data = self._helper.scale_human_data(
            human_data, self.human_root_name, self.human_scale_table
        )
        human_data = self._helper.offset_human_data(
            human_data, self.pos_offsets1, self.rot_offsets1
        )
        self._helper.ground_offset = self.ground_offset
        human_data = self._helper.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self._helper.offset_human_data_to_ground(human_data)
        return human_data

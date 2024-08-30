from __future__ import annotations

from dataclasses import MISSING

import torch, numpy as np

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg, EventTermCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns 
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab_tasks.manager_based.manipulation.unstructured import env_tools as tools

from .flip_env_cfg import UnstructuredFlipEnvCfg
from .config.franka.flip.ik_abs_env_cfg import FrankaFlipObjectEnvCfg
from .config.franka.grasp.ik_abs_env_cfg import FrankaGraspObjectEnvCfg
from omni.isaac.lab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
from . import mdp

##
# Scene definition
##

LOW_LEVEL_FLIP_ENV_CFG = FrankaFlipObjectEnvCfg()
LOW_LEVEL_GRASP_ENV_CFG = FrankaGraspObjectEnvCfg()


##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        grasp_policy_path=f"/home/kjs-dt/RL/orbit/logs/rsl_rl/franka_grasp/2024-08-08_15-51-12/exported/policy.pt",
        flip_policy_path=f"/home/kjs-dt/RL/orbit/logs/rsl_rl/franka_flip/2024-08-07_15-49-56/exported/policy.pt",
        low_level_decimation=2,
        low_level_body_action=LOW_LEVEL_FLIP_ENV_CFG.actions.arm_action,
        low_level_finger_action=LOW_LEVEL_FLIP_ENV_CFG.actions.gripper_action,
        low_level_flip_observations=LOW_LEVEL_FLIP_ENV_CFG.observations.policy,
        low_level_grasp_observations=LOW_LEVEL_GRASP_ENV_CFG.observations.policy,
    )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # eef_pos = ObsTerm(func=mdp.eef_pose_in_robot_root_frame)
        # book_pose = ObsTerm(func=mdp.object_pose_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("book_01")})
        # object_pose = ObsTerm(func=mdp.object_pose_in_robot_root_frame)
        # flip_pose = ObsTerm(func=mdp.book_flip_point_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    object_reach = RewTerm(
        func=mdp.flip_rewards,
        params={},
        weight=1.0 #1.0, 2.0
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""



##
# Environment configuration
##


@configclass
class HighLevelEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: SceneEntityCfg = LOW_LEVEL_FLIP_ENV_CFG.scene
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = LOW_LEVEL_FLIP_ENV_CFG.commands
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = LOW_LEVEL_FLIP_ENV_CFG.events
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4 * 16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 16
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**20
        self.sim.physx.friction_correlation_distance = 0.00625        
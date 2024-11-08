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

from .config.ur5e.reaching.joint_pos_env_cfg import UR5eShelfReachingEnvCfg
from .config.ur5e.sweeping.joint_pos_env_cfg import UR5eShelfEnvCfg
from . import mdp

##
# Scene definition
##

LOW_LEVEL_REACHING_ENV_CFG = UR5eShelfReachingEnvCfg()
LOW_LEVEL_SWEEPING_ENV_CFG = UR5eShelfEnvCfg()


##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        sweep_policy_path=f"/home/irol/IsaacLab/logs/rsl_rl/Shelf_sweep/2024-06-18_23-08-43/exported/policy.pt",
        reach_policy_path=f"/home/irol/IsaacLab/logs/rsl_rl/Shelf_reach/2024-06-19_16-17-41/exported/policy.pt",
        low_level_decimation=2,
        low_level_body_action=LOW_LEVEL_REACHING_ENV_CFG.actions.body_joint_pos,
        low_level_finger_action=LOW_LEVEL_REACHING_ENV_CFG.actions.finger_joint_pos,
        low_level_reach_observations=LOW_LEVEL_REACHING_ENV_CFG.observations.policy,
        low_level_sweep_observations=LOW_LEVEL_SWEEPING_ENV_CFG.observations.policy,
    )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


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
    scene: SceneEntityCfg = LOW_LEVEL_REACHING_ENV_CFG.scene
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = LOW_LEVEL_REACHING_ENV_CFG.events
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
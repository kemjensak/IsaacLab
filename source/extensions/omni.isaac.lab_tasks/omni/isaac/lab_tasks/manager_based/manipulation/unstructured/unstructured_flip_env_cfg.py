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
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns 
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab_tasks.manager_based.manipulation.unstructured import unstructured_env_tools as tools

from . import mdp

##
# Scene definition
##


@configclass
class UnstructuredTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the unstructured scene with a robot and a objects.
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # apple_01 = tools.SetRigidObjectCfgFromUsdFile("Apple_01")
    book_01 = tools.SetRigidObjectCfgFromUsdFile("Book_01_with_grasping_points")
    # kiwi01 = tools.SetRigidObjectCfgFromUsdFile("Kiwi01")
    # lemon_01 = tools.SetRigidObjectCfgFromUsdFile("Lemon_01")
    # NaturalBostonRoundBottle_A01_PR_NVD_01 = tools.SetRigidObjectCfgFromUsdFile("NaturalBostonRoundBottle_A01_PR_NVD_01")
    # rubix_cube = tools.SetRigidObjectCfgFromUsdFile("RubixCube")
    salt_box = tools.SetRigidObjectCfgFromUsdFile("salt_box")

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.75, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                         scale=(1.5, 1.5, 1.0),),
    )
    
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Sensor
    # Sensor = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Sensor",
    #     init_state=AssetBaseCfg.InitialStateCfg(),
    #     spawn=UsdFileCfg(usd_path=f"omniverse://localhost/Library/usd/unstructured/objects/top_rgbd.usd"),
    # )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.0, 0.5), pos_y=(-0.6, -0.4), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pose = ObsTerm(func=mdp.eef_pose_in_robot_root_frame)
        object_pose = ObsTerm(func=mdp.object_pose_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        book_pose = ObsTerm(func=mdp.object_pose_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("book_01")})
        flip_pose = ObsTerm(func=mdp.book_flip_point_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.2), "y": (-0.15, 0.15), "z": (0.0, 0.0),
                           "roll": (-90.0, 90.0), "pitch": (-90.0, 90.0), "yaw": (-180.0, 180.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_book_01_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (0.07, 0.13), "z": (0.03, 0.03),
                            "roll": (-0.0, 0.0), "pitch": (-0.0, 0.0), "yaw": (89.0, 91.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("book_01"),
        },
    )

    # reset_apple_01_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.2, 0.1), "y": (-0.15, 0.15), "z": (0.0, 0.0),
    #                        "roll": (-90.0, 90.0), "pitch": (-90.0, 90.0), "yaw": (-180.0, 180.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("apple_01"),
    #     },
    # )

    # reset_kiwi01_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.2, 0.1), "y": (-0.15, 0.15), "z": (0.0, 0.0),
    #                        "roll": (-90.0, 90.0), "pitch": (-90.0, 90.0), "yaw": (-180.0, 180.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("kiwi01"),
    #     },
    # )

    # reset_lemon_01_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.2, 0.1), "y": (-0.15, 0.15), "z": (0.0, 0.0),
    #                        "roll": (-90.0, 90.0), "pitch": (-90.0, 90.0), "yaw": (-180.0, 180.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("lemon_01"),
    #     },
    # )

    # reset_NaturalBostonRoundBottle_A01_PR_NVD_01_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.2, 0.1), "y": (-0.15, 0.15), "z": (0.0, 0.0),
    #                        "roll": (-90.0, -90.0), "pitch": (-10.0, 10.0), "yaw": (-180.0, 180.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("NaturalBostonRoundBottle_A01_PR_NVD_01"),
    #     },
    # )

    # reset_RubixCube_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.2, 0.1), "y": (-0.15, 0.15), "z": (0.0, 0.0),
    #                        "roll": (-90.0, 90.0), "pitch": (-90.0, 90.0), "yaw": (-180.0, 180.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("rubix_cube"),
    #     },
    # )

    reset_Saltbox_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.13, -0.10), "y": (-0.0, 0.0), "z": (0.0, 0.0),
                           "roll": (-90.0, -90.0), "pitch": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("salt_box"),
        },
    )

    # load_object_pose = EventTerm(
    #     func=mdp.reset_root_state_from_file,
    #     mode="reset",
    #     params={
    #         "object_cfg": [SceneEntityCfg("object"),
    #                        SceneEntityCfg("apple_01"),
    #                        SceneEntityCfg("book_01"),
    #                        SceneEntityCfg("kiwi01"),
    #                        SceneEntityCfg("lemon_01"),
    #                        SceneEntityCfg("NaturalBostonRoundBottle_A01_PR_NVD_01"),
    #                        SceneEntityCfg("rubix_cube"),
    #                        SceneEntityCfg("salt_box")],
    #         "loaded_object_poses": torch.from_numpy(np.load("/home/kjs-dt/RL/objcet_pose/object_poses_grasp.npy")).to('cuda')
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # reaching_object = RewTerm(
    #     func=mdp.object_ee_distance,
    #     params={"std": 0.1, "object_cfg": SceneEntityCfg("book_01")},
    #     weight=1.0
    # )

    # object_rotation = RewTerm(
    #     func=mdp.target_object_rotation,
    #     params={},
    #     weight=1.0 # 1.0
    # )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted_from_initial,
        params={"minimal_height": 0.02, "asset_cfg": SceneEntityCfg("book_01")},
        weight=10.0
    )

    object_reach = RewTerm(
        func=mdp.flip_rewards,
        params={},
        weight=1.0 #1.0, 2.0
    )

    object_flip = RewTerm(
        func=mdp.object_is_flipped,
        params={"object_cfg": SceneEntityCfg("book_01")},
        weight=10.0 # 10.0
    )

    home_after_flip = RewTerm(
            func=mdp.home_after_flip,
            params={},
            weight=20.0,
    )

    # ee_vel = RewTerm(func=mdp.ee_velocity, weight=-1e-3)

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # touching_other_object = RewTerm(
    #     func=mdp.touching_other_object,
    #     weight=-1e-3,
    #     params={"asset_cfg_list": [SceneEntityCfg("apple_01"),
    #                             SceneEntityCfg("book_01"),
    #                             SceneEntityCfg("kiwi01"),
    #                             SceneEntityCfg("lemon_01"),
    #                             SceneEntityCfg("NaturalBostonRoundBottle_A01_PR_NVD_01"),
    #                             SceneEntityCfg("rubix_cube"),
    #                             SceneEntityCfg("salt_box")],
    #             },
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # max_consecutive_success = DoneTerm(
    #     func=mdp.max_consecutive_success, params={"num_success": 50, "command_name": "object_pose"}
    # )

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    # apple_01_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("apple_01")}
    # )

    book_01_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("book_01")}
    )

    # kiwi01_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("kiwi01")}
    # )

    # lemon_01_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("lemon_01")}
    # )

    # NaturalBostonRoundBottle_A01_PR_NVD_01_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("NaturalBostonRoundBottle_A01_PR_NVD_01")}
    # )

    # rubix_cube_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("rubix_cube")}
    # )

    salt_box_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("salt_box")}
    )

        


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )

    # ee_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "ee_vel", "weight": -1, "num_steps": 10000}
    # ) # 10000

    # object_rotation = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "object_rotation", "weight": 4.0, "num_steps": 10000}
    # )
    
    # object_flip = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "object_flip", "weight": 4.0, "num_steps": 10000}
    # )
    
    # lifting_object = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "lifting_object", "weight": 1e-1, "num_steps": 10000}
    # )


##
# Environment configuration
##


@configclass
class UnstructuredFlipEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: UnstructuredTableSceneCfg = UnstructuredTableSceneCfg(num_envs=4096, env_spacing=2.5) # 4096
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 4.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 2.0
        # self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4 * 16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 16
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**20
        self.sim.physx.friction_correlation_distance = 0.00625        
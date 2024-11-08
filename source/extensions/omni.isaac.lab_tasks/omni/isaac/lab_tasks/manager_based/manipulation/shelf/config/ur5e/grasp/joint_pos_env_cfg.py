# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim.schemas.schemas_cfg import MassPropertiesCfg

from omni.isaac.lab_tasks.manager_based.manipulation.shelf import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.shelf.shelf_grasping_env_cfg import ShelfGraspingEnvCfg
import torch
##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.ur5e import UR5e_CFG

@configclass
class UR5eShelfGraspingEnvCfg(ShelfGraspingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = UR5e_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint"], 
            scale=0.5, 
            use_default_offset=True
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_outer_knuckle_joint", "right_outer_knuckle_joint"],
            open_command_expr={"left_outer_knuckle_joint": 0.0, "right_outer_knuckle_joint": 0.0},
            close_command_expr={"left_outer_knuckle_joint": 0.4, "right_outer_knuckle_joint": 0.4}, # 0.5
        )

        # Set Cube as object
        self.scene.cup2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.88, 0.0, 0.66], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"omniverse://localhost/Library/usd/Object/SM_PlasticCup.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.5),
            ),
        )

        self.scene.cup = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.72, -0.2, 0.66], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"omniverse://localhost/Library/usd/Object/SM_Cup_empty.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.5),
            ),
        )

        # self.scene.cup2 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Cup2",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.8, 0.0, 0.71], rot=[1, 0, 0, 0]), #[0.8, 0.0, 0.72], [1, 0, 0, 0]
        #     spawn=UsdFileCfg(
        #         usd_path=f"omniverse://localhost/Library/usd/Object/Cylinder.usd", #f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"     f"omniverse://localhost/Library/usd/Object/Cylinder.usd"
        #         scale=(1.0, 1.0, 1.0), #(0.8, 0.8, 2.0)
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=100.0, # 5.0
        #             disable_gravity=False,
        #         ),
        #         mass_props=MassPropertiesCfg(mass=0.5),
        #     ),
        # )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/robotiq_arg2f_base_link_01",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.146],
                    ),
                ),
            ],
        )

        self.scene.wrist_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/robotiq_arg2f_base_link_01",
                    name="wrist",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, -0.15),
                    ),
                ),
            ],
        )

        self.scene.wrist_upper_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_1_link",
                    name="wrist_upper",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )

        self.rewards.grasp_handle.params["open_joint_pos"] = 0.0
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]

@configclass
class UR5eShelfGraspingEnvCfg_PLAY(UR5eShelfGraspingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

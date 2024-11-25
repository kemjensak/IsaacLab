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

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.shelf.shelf_test_env_cfg import TestEnvCfg
import torch
##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets import UR3_CFG

@configclass
class TestEnvCfg(TestEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
       

        # Set actions for the specific robot type
        self.actions.arm_action = mdp.JointPositionActionCfg(
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
        
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_outer_knuckle_joint","right_outer_knuckle_joint"],
            open_command_expr={"left_outer_knuckle_joint": 0.0,"right_outer_knuckle_joint": 0.0},
            close_command_expr={"left_outer_knuckle_joint":0.0,"right_outer_knuckle_joint": 0.5},
        )
        
        # Set Cup as object
        self.scene.cup = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.55, 0.0, 0.97], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=f"omniverse://localhost/Library/Shelf/Object/SM_Cup_empty.usd",
                scale=(0.9, 0.9, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.3),
            ),
        )
        
        

        # Set Cube as object
        self.scene.cup2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.7, 0.0, 0.97], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"omniverse://localhost/Library/Shelf/Object/SM_PlasticCup.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.3),
            ),
        )
        


@configclass
class TestEnvCfg_PLAY(TestEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
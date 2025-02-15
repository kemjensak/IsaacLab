# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##

from omni.isaac.lab_assets import UR3_CFG


@configclass
class UR3ShelfEnvCfg(joint_pos_env_cfg.UR3ShelfEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set the robot config
        self.scene.robot = UR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (UR5e)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint",], 
            body_name="flange",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.14]),
            scale=0.5
        )


@configclass
class UR3ShelfEnvCfg_PLAY(joint_pos_env_cfg.UR3ShelfEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

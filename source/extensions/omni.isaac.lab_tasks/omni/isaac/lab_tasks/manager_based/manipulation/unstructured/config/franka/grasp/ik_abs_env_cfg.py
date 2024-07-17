# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets.articulation import ArticulationCfg

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaGraspObjectEnvCfg(joint_pos_env_cfg.FrankaGraspObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        # FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_joint[1-4]"].velocity_limit = 2.175/2
        # FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_joint[5-7]"].velocity_limit = 2.61/2
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                                    init_state=ArticulationCfg.InitialStateCfg(
                                                        joint_pos={
                                                            "panda_joint1": -1.5708,
                                                            "panda_joint2": 0.0,
                                                            "panda_joint3": 0.785398,
                                                            "panda_joint4": -3.05433,
                                                            "panda_joint5": 2.04204, # 0.0
                                                            "panda_joint6": 1.67552,
                                                            "panda_joint7": 0.837758,
                                                            "panda_finger_joint.*": 0.04,
                                                        },
                                                    ),
                                                    )
        

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaGraspObjectEnvCfg_PLAY(FrankaGraspObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

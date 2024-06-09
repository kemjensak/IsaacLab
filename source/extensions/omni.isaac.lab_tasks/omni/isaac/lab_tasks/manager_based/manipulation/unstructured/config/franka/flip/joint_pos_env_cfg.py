# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, CameraCfg, ContactSensorCfg, RayCasterCfg, patterns 
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.unstructured import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.unstructured.unstructured_flip_env_cfg import UnstructuredFlipEnvCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip

@configclass
class FrankaFlipObjectEnvCfg(UnstructuredFlipEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
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
                                                    actuators={
                                                        "panda_shoulder": ImplicitActuatorCfg(
                                                            joint_names_expr=["panda_joint[1-4]"],
                                                            effort_limit=87.0,
                                                            velocity_limit=2.175,
                                                            stiffness=80.0,
                                                            damping=4.0,
                                                        ),
                                                        "panda_forearm": ImplicitActuatorCfg(
                                                            joint_names_expr=["panda_joint[5-7]"],
                                                            effort_limit=12.0,
                                                            velocity_limit=2.61,
                                                            stiffness=80.0,
                                                            damping=4.0,
                                                        ),
                                                        "panda_hand": ImplicitActuatorCfg(
                                                            joint_names_expr=["panda_finger_joint.*"],
                                                            effort_limit=200.0,
                                                            velocity_limit=0.2,
                                                            stiffness=2e3,
                                                            damping=1e2,
                                                        ),
                                                    }
                                                    )

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Set Cube as object
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        #         scale=(0.8, 0.8, 0.8),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )


@configclass
class FrankaFlipObjectEnvCfg_PLAY(FrankaFlipObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import RigidObjectCfg
from omni.isaac.orbit.sensors import FrameTransformerCfg, CameraCfg, ContactSensorCfg, RayCasterCfg, patterns 
from omni.isaac.orbit.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.orbit.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

from omni.isaac.orbit_tasks.manipulation.unstructed import mdp
from omni.isaac.orbit_tasks.manipulation.unstructed.unstructed_env_cfg import UnstructedEnvCfg

##
# Pre-defined configs
##
from omni.isaac.orbit.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.orbit_assets.franka import FRANKA_PANDA_CFG  # isort: skip

@configclass
class FrankaGraspObjectEnvCfg(UnstructedEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Top-down RGB-D camera
        self.scene.camera_topDown = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/topDown_cam",
            update_period=0.01,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )

        # Set writst attached RGB-D camera
        # self.scene.camera_wrist = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/wrist,
        #     update_period=0.001,
        #     height=480,
        #     width=640,
        #     data_types=["rgb", "distance_to_image_plane"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        #     ),
        #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        # )

        # Set contact-force sensors on finger link
        # self.scene.contact_finger = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT", update_period=0.0, history_length=6, debug_vis=True
        # )

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

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
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaGraspObjectEnvCfg_PLAY(FrankaGraspObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

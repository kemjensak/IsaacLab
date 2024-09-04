from __future__ import annotations

from dataclasses import MISSING

import torch, numpy as np

import omni.isaac.lab.sim as sim_utils


from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, ContactSensorCfg, RayCasterCfg, patterns 
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from . import mdp
from .grasp_env_cfg import UnstructuredTableSceneCfg, UnstructuredGraspEnvCfg

##
# Scene definition
##
@configclass
class UnstructuredTableSceneRGBCameraCfg(UnstructuredTableSceneCfg):
    """Configuration for the unstructured scene with a robot and a objects.
    """

    # Top-Down Camera
    camera_topDown: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/topDown_cam",
            update_period=0.05, # rgb 30hz / depth 90hz
            height=480,
            width=640,
            data_types=["rgb"], # rgb or depth
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

#TODO: Add depth cam
@configclass
# class UnstructuredTableSceneDepthCameraCfg(grasp_env_cfg.UnstructuredTableSceneCfg):
#     """Configuration for the unstructured scene with a robot and a objects.
#     """

#     # Top-Down Camera
#     camera_topDown: TiledCameraCfg = TiledCameraCfg(
#         prim_path="{ENV_REGEX_NS}/Robot/topDown_cam",
#             update_period=0.01, # rgb 30hz / depth 90hz
#             height=480,
#             width=640,
#             data_types=["rgb", "distance_to_image_plane"], # rgb or depth
#             spawn=sim_utils.PinholeCameraCfg(
#                 focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
#             ),
#             offset=TiledCameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
#     )

##
# MDP settings
##

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        rgb_cam = ObsTerm(func=mdp.rgb_cam)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

##
# Environment configuration
##

@configclass
class UnstructuredGraspRGBCameraEnvCfg(UnstructuredGraspEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: UnstructuredTableSceneRGBCameraCfg = UnstructuredTableSceneRGBCameraCfg(num_envs=128, env_spacing=2.5) # 4096
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()

#TODO: Add depth cam
# @configclass
# class UnstructuredGraspDepthCameraEnvCfg(UnstructuredGraspEnvCfg):
#     """Configuration for the lifting environment."""

#     # Scene settings
#     scene: UnstructuredTableSceneDepthCameraCfg = UnstructuredTableSceneDepthCameraCfg(num_envs=4096, env_spacing=2.5) # 4096
#     # Basic settings
#     observations: RGBObservationsCfg = RGBObservationsCfg()     
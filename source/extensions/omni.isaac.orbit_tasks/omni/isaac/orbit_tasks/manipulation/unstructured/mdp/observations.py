# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import subtract_frame_transforms
from omni.isaac.orbit.sensors import FrameTransformerData

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def object_position_in_robot_root_frame(
    env: RLTaskEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def object_pose_in_robot_root_frame(
    env: RLTaskEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The pose of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_quat_w = object.data.root_quat_w[:, :4]
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w, object_quat_w
    )
    return torch.concat((object_pos_b, object_quat_b), dim=1)

def eef_pose_in_robot_root_frame(
    env: RLTaskEnv,
) -> torch.Tensor:
    """The pose of the end effector in the robot's root frame."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    
    return ee_tf_data.target_pos_w[..., 0, :]
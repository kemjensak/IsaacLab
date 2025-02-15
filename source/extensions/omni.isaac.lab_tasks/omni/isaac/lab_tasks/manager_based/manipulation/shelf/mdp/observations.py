# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_unique
from omni.isaac.lab.sensors import FrameTransformerData, ContactSensorData
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    
    return object_pos_b

def target_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup2"),
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
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:

    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )

    return torch.concat((object_pos_b, object_quat_b), dim=1)

def object_init_pos_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_init_pos_w = object.data.default_root_state[:, :3]
    
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_init_pos_w
    )
    # print(object_pos_b)
    return object_pos_b

def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos

def ee_pos_r(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos_w = ee_tf_data.target_pos_w[..., 0, :]
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w
    )

    # print(f"ee_pos_w: {ee_tf_data.target_pos_w[..., 0, :]}")
    # print(f"ee_pos_r: {ee_pos_r}")

    return ee_pos_b

def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return quat_unique(ee_quat) if make_quat_unique else ee_quat

def ee_quat_r(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    ee_pos_r, ee_quat_r = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_tf_data.target_pos_w[..., 0, :], ee_quat
    )

    return quat_unique(ee_quat_r) if make_quat_unique else ee_quat_r


def target_goal_pose(
        env: ManagerBasedRLEnv, 
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("cup")) -> torch.Tensor:
    
    robot: RigidObject = env.scene[robot_cfg.name]

    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3]
    des_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_w
    )
    
    # print(f"des_pos_w:{des_pos_w}")
    # print(f"des_pos_r: {des_pos_r}")
    return des_pos_b

def ee_goal_pos(
        env: ManagerBasedRLEnv, 
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:

    robot:  Articulation = env.scene[robot_cfg.name]

    # command = env.command_manager.get_command(command_name)
    # des_pos_w = command[:, :3]
    # des_pos_b, _ = subtract_frame_transforms(
    #     robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_w
    # )
    
    des_joint_pos = robot.data.default_joint_pos[:, :6]

    # print(f"des_pos_w:{des_pos_w}")
    # print(f"des_pos_r: {des_pos_b}")
    return des_joint_pos

def rl_joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, : 8] - asset.data.default_joint_pos[:, : 8]

def rl_joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, : 8] - asset.data.default_joint_vel[:, : 8]
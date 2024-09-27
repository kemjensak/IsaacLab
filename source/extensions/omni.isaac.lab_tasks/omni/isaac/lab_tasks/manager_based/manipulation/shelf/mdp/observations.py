# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
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

def object_vel(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object_lin_vel = object.data.body_lin_vel_w[...,0, :]
    object_ang_vel = object.data.body_ang_vel_w[...,0, :]

    return torch.concat((object_lin_vel, object_ang_vel), dim=1)

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
    ee_pos_r, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_tf_data.target_pos_w[..., 0, :]
    )

    return ee_pos_r

def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return quat_unique(ee_quat) if make_quat_unique else ee_quat

def ee_quat_r(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the robot base frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    _, ee_quat_r = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_tf_data.target_pos_w[..., 0, :],  ee_tf_data.target_quat_w[..., 0, :]
    )
    # make first element of quaternion positive
    return quat_unique(ee_quat_r) if make_quat_unique else ee_quat_r

def Contact_sensor(env: ManagerBasedRLEnv, 
            robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    
    contact_data: ContactSensorData = env.scene["contact_sensor"].data


    force_net = contact_data.net_forces_w
    norm_force_net = torch.norm(force_net,dim=2)

    return norm_force_net

class target_goal_pos(ManagerTermBase):

    def __init__(self, cfg: ObsTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        shelf_cfg = SceneEntityCfg("shelf")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._shelf: RigidObject = env.scene[shelf_cfg.name]

        self.__initial_object_pos = self._target.data.root_pos_w.clone()
        self._shelf_pos = self._shelf.data.root_pos_w.clone()
        self._shelf_quat = self._shelf.data.root_quat_w.clone()

        reset_mask = env.common_step_counter

    
    def __call__(self, env: ManagerBasedRLEnv,):
        
        return self._target_goal_pos_(env)
    
    def _target_goal_pos_(self, env: ManagerBasedRLEnv,) -> torch.Tensor:
        
        self._shelf_pos = self._shelf.data.root_pos_w[:, :3]
        self._shelf_quat = self._shelf.data.root_quat_w[:, :4]

        self._target_goal_pos, _ = subtract_frame_transforms(self._shelf_pos, self._shelf_quat, self.__initial_object_pos[:, :])
        self._target_goal_pos[:, 1] = self._target_goal_pos[:, 1] - 0.2

        

        return self._target_goal_pos
    


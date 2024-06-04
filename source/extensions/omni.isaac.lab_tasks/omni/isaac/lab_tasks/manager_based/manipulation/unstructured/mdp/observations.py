# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_mul, transform_points
from omni.isaac.lab.sensors import FrameTransformerData
from omni.isaac.lab.managers import ObservationTermCfg as ObsTrem

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
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
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The pose of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_quat_w = object.data.root_quat_w[:, :4]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return torch.concat((object_pos_b, object_quat_w), dim=1)

class book_flip_point_in_robot_root_frame(ManagerTermBase):

    def __init__(self, cfg: ObsTrem, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg = SceneEntityCfg("book_01")
        self._asset: RigidObject = env.scene[asset_cfg.name]

        # y-up z-center is for grasp-to-flip, x-up z-center is for below-to-flip

        self._top_offset = torch.zeros((env.num_envs, 7), device=env.device)
        self._top_offset[:,:3] = torch.tensor([0.0, -0.16123+0.03, 0.0127 - 0.03])
        # self._top_offset[:,3:7] = torch.tensor([0.0, 0.0, -0.70711, -0.70711]) # y-up z-center
        self._top_offset[:,3:7] = torch.tensor([0.5, -0.5, 0.5, 0.5]) # x-down z-center
        # self._top_offset[:,3:7] = torch.tensor([0, 0, 90/180*3.141592])

        self._bottom_offset = torch.zeros((env.num_envs, 7), device=env.device)
        self._bottom_offset[:,:3] = torch.tensor([0.0, 0.16123-0.03, 0.0127 - 0.03])
        # self._bottom_offset[:,3:7] = torch.tensor([0.70711, 0.70711, 0.0, 0.0]) # y-up z-center
        self._bottom_offset[:,3:7] = torch.tensor([0.5, 0.5, 0.5, -0.5]) # x-down z-center
        # self._bottom_offset[:,3:7] = torch.tensor([0, 0, -90/180*3.141592])

        self._left_offset = torch.zeros((env.num_envs, 7), device=env.device)
        self._left_offset[:,:3] = torch.tensor([0.116881-0.03, 0.0, 0.0127 - 0.03])
        # self._left_offset[:,3:7] = torch.tensor([0.5, 0.5, -0.5, -0.5]) # y-up z-center
        self._left_offset[:,3:7] = torch.tensor([0.0, -0.70711, 0.0, 0.70711])  # x-down z-center
        # self._left_offset[:,3:7] = torch.tensor([0, 0, 3.141592])

        self._right_offset = torch.zeros((env.num_envs, 7), device=env.device)
        self._right_offset[:,:3] = torch.tensor([-0.116881+0.03, 0.0, 0.0127 - 0.03])
        # self._right_offset[:,3:7] = torch.tensor([0.5, 0.5, 0.5, 0.5]) # y-up z-center
        self._right_offset[:,3:7] = torch.tensor([-0.70711, 0.0, -0.70711, 0.0])  # x-down z-center
        # self._right_offset[:,3:7] = torch.tensor([0, 0, 0])

    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
    ):
        
        return self._calc_grasping_pos(env)
    def _calc_grasping_pos(self,
                           env: ManagerBasedRLEnv
    ) -> torch.Tensor:
        # 네 모서리 point의 offset을 book frame에서 simulation world frame 으로 변환하는 quat 적용

        top_pos = torch.cat((transform_points(self._top_offset[:, :3],
                                              self._asset.data.root_pos_w,
                                              self._asset.data.root_state_w[:, 3:7])[..., 0 , :],
                            quat_mul(self._asset.data.root_state_w[:, 3:7],
                                     self._top_offset[:, 3:7])),
                            dim=1)
        bottom_pos = torch.cat((transform_points(self._bottom_offset[:, :3],
                                                 self._asset.data.root_pos_w,
                                                 self._asset.data.root_state_w[:, 3:7])[..., 0 , :],
                                quat_mul(self._asset.data.root_state_w[:, 3:7],
                                         self._bottom_offset[:, 3:7])),
                                dim=1)
        left_pos = torch.cat((transform_points(self._left_offset[:, :3],
                                               self._asset.data.root_pos_w,
                                               self._asset.data.root_state_w[:, 3:7])[..., 0, :],
                            quat_mul(self._asset.data.root_state_w[:, 3:7],
                                     self._left_offset[:, 3:7])),
                            dim=1)
        right_pos = torch.cat((transform_points(self._right_offset[:, :3],
                                                self._asset.data.root_pos_w,
                                                self._asset.data.root_state_w[:, 3:7])[..., 0, :],
                            quat_mul(self._asset.data.root_state_w[:, 3:7],
                                     self._right_offset[:, 3:7])),
                            dim=1)


        grasp_poses = torch.zeros((env.num_envs, 7), device=env.device)

        poses = torch.stack((top_pos, bottom_pos, left_pos, right_pos), dim=1)

        # Find the index of the maximum z-value for each environment
        max_indices = torch.argmax(poses[:, :, 2], dim=1)

        # Gather the corresponding poses based on the indices
        grasp_poses = poses[torch.arange(env.num_envs), max_indices]
        return grasp_poses
    
def eef_pos_in_robot_root_frame(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """The pose of the end effector in the robot's root frame."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    return ee_tf_data.target_pos_w[..., 0, :]

def eef_quat_in_robot_root_frame(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """The pose of the end effector in the robot's root frame."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    return ee_tf_data.target_quat_w[..., 0, :]
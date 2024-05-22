# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject, Articulation
from omni.isaac.orbit.managers import SceneEntityCfg, ManagerTermBase, SceneEntityCfg
from omni.isaac.orbit.sensors import FrameTransformer
from omni.isaac.orbit.utils.math import combine_frame_transforms, quat_error_magnitude, axis_angle_from_quat, quat_box_minus, quat_inv, quat_mul
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

# TODO: 완성 후 정상작동 확인 필요
class target_object_rotation(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: RLTaskEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("book_01"))
        self._asset: RigidObject = env.scene[asset_cfg.name]


        # store initial target object pose
        self._initial_object_quat = self._asset.data.root_state_w[:, 3:7].clone()

    def __call__(
        self,
        env: RLTaskEnv,
        std: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("book_01"),
        minimal_rotation: float = 0.1,
    ):
        
        # if the env's step is 1, store the initial object pose
        for idx in range(env.num_envs):
            if env.episode_length_buf[idx] == 1:
                self._initial_object_quat[idx, :4] = self._asset.data.root_state_w[idx, 3:7].clone()

        # TODO: quat_err와 rel_axis_angle중 어떤 것을 사용할지 결정
        # initial quternion의 inverse와 현재 quaternion의 곱을 통해 둘 사이의 상대적 quaternion을 구함
        # rel_quat = quat_mul(quat_inv(self._initial_object_quat[:, 3:7]), self._asset.data.root_state_w[:, 3:7])

        # quaternion을 axis-angle로 변환
        # rel_axis_angle = axis_angle_from_quat(rel_quat)

        # 두 quaternion 사이의 차이를 rad로 계산
        quat_err = quat_error_magnitude(self._initial_object_quat[:, :4], self._asset.data.root_state_w[:, 3:7])
        

        return torch.abs(quat_err/std)


# TODO: 정상작동 확인 필요
class object_is_lifted_from_initial(ManagerTermBase):

    def __init__(self, cfg: RewTerm, env: RLTaskEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._asset: RigidObject = env.scene[asset_cfg.name]

        # store initial target object position
        self._initial_object_height = self._asset.data.root_pos_w[:, 2].clone()

    def __call__(
        self,
        env: RLTaskEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        minimal_height: float = 0.06,
    ):
        # if the env's step is 1, store the initial object position
        for idx in range(env.num_envs):
            if env.episode_length_buf[idx] == 1:
                # print("init position set")
                self._initial_object_height[idx] = self._asset.data.root_pos_w[idx, 2].clone()

        # return the reward
        return torch.where(self._asset.data.root_pos_w[:, 2] > (self._initial_object_height + minimal_height), 1.0, 0.0)


def object_is_lifted(
    env: RLTaskEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height from initial position."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_ee_quat_diff(
    env: RLTaskEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("book_01"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_quat_w = object.data.root_quat_w

    ee_q = ee_frame.data.target_quat_w[..., 0, :]

    return quat_error_magnitude(object_quat_w, ee_q)

def object_ee_distance(
    env: RLTaskEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: RLTaskEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


# TODO: 정상 작동 여부 확인 필요
def touching_other_object(
    env: RLTaskEnv,
    asset_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("robot")],
) -> torch.Tensor:
    """Penalize the agent for touching the other object."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg_list[0].name]
    penalty = torch.zeros(env.num_envs, device=env.device)
    for asset_cfg in asset_cfg_list:
        asset = env.scene[asset_cfg.name]
        penalty += torch.sum(torch.square(asset.data.root_lin_vel_b[:, :3]), dim=1)
    return penalty
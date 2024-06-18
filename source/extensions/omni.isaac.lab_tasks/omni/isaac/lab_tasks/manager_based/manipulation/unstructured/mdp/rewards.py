# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, transform_points, quat_apply, quat_from_euler_xyz, axis_angle_from_quat, quat_box_minus, quat_inv, quat_mul
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG # isort: skip
from omni.isaac.lab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

class target_object_rotation(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("book_01"))
        self._asset: RigidObject = env.scene[asset_cfg.name]


        # store initial target object pose
        self._initial_object_quat = self._asset.data.root_state_w[:, 3:7].clone()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
    ):        
        # if the env's step is 1, store the initial object pose
        reset_mask = env.episode_length_buf == 1
        self._initial_object_quat[reset_mask, :4] = self._asset.data.root_state_w[reset_mask, 3:7].clone()

        # TODO: quat_err와 rel_axis_angle중 어떤 것을 사용할지 결정
        # initial quternion의 inverse와 현재 quaternion의 곱을 통해 둘 사이의 상대적 quaternion을 구함
        # rel_quat = quat_mul(quat_inv(self._initial_object_quat[:, 3:7]), self._asset.data.root_state_w[:, 3:7])

        # quaternion을 axis-angle로 변환
        # rel_axis_angle = axis_angle_from_quat(rel_quat)

        # 두 quaternion 사이의 차이를 rad로 계산
        quat_err = torch.abs(quat_error_magnitude(self._initial_object_quat[:, :4], self._asset.data.root_state_w[:, 3:7]))
        reward = torch.where(quat_err > 1.0, 1.0, torch.tanh(quat_err/1.57))
        # print(torch.tanh(quat_err/1.57))
        return torch.tanh(quat_err/1.57)

class object_is_lifted_from_initial(ManagerTermBase):

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._asset: RigidObject = env.scene[asset_cfg.name]

        # store initial target object position
        self._initial_object_height = self._asset.data.root_pos_w[:, 2].clone()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        minimal_height: float = 0.04,
    ):
        # Update initial object height where the environment's step is 1
        reset_mask = env.episode_length_buf == 1
        self._initial_object_height[reset_mask] = self._asset.data.root_pos_w[reset_mask, 2].clone()

        # return the reward
        return torch.where(self._asset.data.root_pos_w[:, 2] > (self._initial_object_height + minimal_height), 1.0, 0.0)

class object_goal_distance_from_initial(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._asset: RigidObject = env.scene[asset_cfg.name]

        # store initial target object position
        self._initial_object_height = self._asset.data.root_pos_w[:, 2].clone()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
    ):
        # Update initial object height where the environment's step is 1
        reset_mask = env.episode_length_buf == 1
        self._initial_object_height[reset_mask] = self._asset.data.root_pos_w[reset_mask, 2].clone()

        return (self._object_goal_distance(env=env, std=0.3, minimal_height=0.04, command_name="object_pose") * 16+
                self._object_goal_distance(env=env, std=0.05, minimal_height=0.04, command_name="object_pose") * 5)
                

    def _object_goal_distance(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimal_height: float,
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ):
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
        return (object.data.root_pos_w[:, 2] > (self._initial_object_height + minimal_height)) * (1 - torch.tanh(distance / std))

# TODO: 정상작동 확인 필요
class flip_rewards(ManagerTermBase):

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg = SceneEntityCfg("book_01")
        self._asset: RigidObject = env.scene[asset_cfg.name]
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer2"
        self._marker = VisualizationMarkers(marker_cfg)

        # y-up z-center is for grasp-to-flip, x-up z-center is for below-to-flip

        self._top_offset = torch.zeros((env.num_envs, 7), device=env.device)
        self._top_offset[:,:3] = torch.tensor([0.0, -0.16123+0.03, 0.0127 - 0.04])
        # self._top_offset[:,3:7] = torch.tensor([0.0, 0.0, -0.70711, -0.70711]) # y-up z-center
        self._top_offset[:,3:7] = torch.tensor([0.5, -0.5, 0.5, 0.5]) # x-down z-center
        # self._top_offset[:,3:7] = torch.tensor([0, 0, 90/180*3.141592])

        self._bottom_offset = torch.zeros((env.num_envs, 7), device=env.device)
        self._bottom_offset[:,:3] = torch.tensor([0.0, 0.16123-0.03, 0.0127 - 0.04])
        # self._bottom_offset[:,3:7] = torch.tensor([0.70711, 0.70711, 0.0, 0.0]) # y-up z-center
        self._bottom_offset[:,3:7] = torch.tensor([0.5, 0.5, 0.5, -0.5]) # x-down z-center
        # self._bottom_offset[:,3:7] = torch.tensor([0, 0, -90/180*3.141592])

        self._left_offset = torch.zeros((env.num_envs, 7), device=env.device)
        self._left_offset[:,:3] = torch.tensor([0.116881-0.03, 0.0, 0.0127 - 0.04])
        # self._left_offset[:,3:7] = torch.tensor([0.5, 0.5, -0.5, -0.5]) # y-up z-center
        self._left_offset[:,3:7] = torch.tensor([0.0, -0.70711, 0.0, 0.70711])  # x-down z-center
        # self._left_offset[:,3:7] = torch.tensor([0, 0, 3.141592])

        self._right_offset = torch.zeros((env.num_envs, 7), device=env.device)
        self._right_offset[:,:3] = torch.tensor([-0.116881+0.03, 0.0, 0.0127 - 0.04])
        # self._right_offset[:,3:7] = torch.tensor([0.5, 0.5, 0.5, 0.5]) # y-up z-center
        self._right_offset[:,3:7] = torch.tensor([-0.70711, 0.0, -0.70711, 0.0])  # x-down z-center
        # self._right_offset[:,3:7] = torch.tensor([0, 0, 0])

        self._initial_object_poses = self._asset.data.root_state_w[:, :7].clone()
        self._initial_grasp_poses = self._asset.data.root_state_w[:, :7].clone()

        self._initial_distance = torch.zeros(env.num_envs, device=env.device)
        self._last_distance = torch.zeros(env.num_envs, device=env.device)
        self._last_ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :].clone()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
    ):
        self._grasp_poses = grasp_poses = self._calc_grasping_pose(env).clone()

        reset_mask = env.episode_length_buf == 1
        self._initial_object_poses[reset_mask] = self._asset.data.root_state_w[reset_mask, :7].clone()
        self._initial_grasp_poses[reset_mask] = grasp_poses[reset_mask, :7].clone()

        # touching_before_grasp = self._touching_book_before_grasp(env, grasp_poses[:, :3]) * -0.002
        # approach_gripper = self._approach_gripper_handle(env, grasp_poses[:, :3], 0.04) * 5.0
        # align_gripper = self._align_grasp_around_handle(env, grasp_poses[:, :3]) * 0.125
        # grasp_point = self._grasp_target_point(env, 0.03, grasp_poses[:, :3]) * 0.5

        self._marker.visualize(translations=grasp_poses[:, :3], orientations=grasp_poses[:, 3:7])

        return (
                torch.where(self._is_flipped(env) == True,
                            1.5,
                            self._approach_grasp_point(env, 0.1) * 0.5 +
                            self._align_ee_grasp_point(env, grasp_poses[:, 3:7]) * 0.5)
                # self._touching_book_before_grasp(env, grasp_poses[:, :3]) * -0.002
                )
    
    def _calc_grasping_pose(self,
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

        poses = torch.stack((top_pos, bottom_pos, left_pos, right_pos), dim=1)

        # Find the index of the maximum z-value for each environment
        max_indices = torch.argmax(poses[:, :, 2], dim=1)

        # Gather the corresponding poses based on the indices
        grasp_poses = poses[torch.arange(env.num_envs), max_indices]
        
        # print(grasp_poses) # 16, 7
        return grasp_poses

    def _approach_grasp_point(
            self,
            env: ManagerBasedRLEnv,
            threshold: float,
    ) -> torch.Tensor:
        """Reward the agent for approaching the grasp point."""

        ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
        # Compute the distance of the end-effector to the handle
        distance = torch.norm(self._grasp_poses[:, :3] - ee_tcp_pos, dim=-1, p=2)
        # print(distance) 

        # Reward the robot for reaching the handle
        reward = 1.0 / (1.0 + distance**2)
        reward = torch.pow(reward, 2)
        reward = torch.exp(-1.2 * distance)
        return torch.where(distance <= threshold, 2 * reward, reward)
    
    def _align_ee_grasp_point(
            self,
            env: ManagerBasedRLEnv,
            grasp_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Reward the agent for aligning the end-effector with the grasp point."""
        ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]

        # Compute the quaternion error between the end-effector and the handle
        quat_err = quat_error_magnitude(grasp_quat, ee_tcp_quat)

        # Reward the robot for aligning the end-effector with the handle
        return 1.0 - torch.tanh(quat_err)

    def _is_flipped(
            self,
            env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("book_01")
    ) -> torch.Tensor:
        """Reward the agent for flipping the object."""
        object: RigidObject = env.scene[object_cfg.name]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
        z_component = quat_apply(object.data.root_quat_w, z_axis)[:, 2]
        return torch.where(z_component < 0.0, True, False)
    
    def _approach_gripper_handle(self,
                                env: ManagerBasedRLEnv,
                                grasp_pos: torch.Tensor,
                                offset: float = 0.04
    ) -> torch.Tensor:
        """Reward the robot's gripper reaching the drawer handle with the right pose.

        This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
        (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
        """
        # Fingertips position: (num_envs, n_fingertips, 3)
        ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
        lfinger_pos = ee_fingertips_w[..., 0, :]
        rfinger_pos = ee_fingertips_w[..., 1, :]

        # Compute the distance of each finger from the handle
        lfinger_dist = torch.abs(lfinger_pos[:, 2] - grasp_pos[:, 2])
        rfinger_dist = torch.abs(rfinger_pos[:, 2] - grasp_pos[:, 2])

        # Check if hand is in a graspable pose
        is_graspable = (rfinger_pos[:, 2] < grasp_pos[:, 2]) & (lfinger_pos[:, 2] > grasp_pos[:, 2])

        return is_graspable * ((offset - lfinger_dist) + (offset - rfinger_dist))
    
    def _align_grasp_around_handle(self,
                                  env: ManagerBasedRLEnv,
                                  grasp_pos: torch.Tensor
    ) -> torch.Tensor:
        """Bonus for correct hand orientation around the handle.

        The correct hand orientation is when the left finger is above the handle and the right finger is below the handle.
        """
        # Fingertips position: (num_envs, n_fingertips, 3)
        ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
        lfinger_pos = ee_fingertips_w[..., 0, :]
        rfinger_pos = ee_fingertips_w[..., 1, :]

        # Check if hand is in a graspable pose
        is_graspable = (rfinger_pos[:, 2] < grasp_pos[:, 2]) & (lfinger_pos[:, 2] > grasp_pos[:, 2])

        # bonus if left finger is above the drawer handle and right below
        return is_graspable
        
    def _grasp_target_point(self,
                        env: ManagerBasedRLEnv,
                        threshold: float,
                        grasp_pos: torch.Tensor,
                        open_joint_pos: float = 0.04,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_.*"])
    ) -> torch.Tensor:
        """Reward for closing the fingers when being close to the handle.

        The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
        The :attr:`open_joint_pos` is the joint position when the fingers are open.

        Note:
            It is assumed that zero joint position corresponds to the fingers being closed.
        """
        ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
        gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

        distance = torch.norm(grasp_pos - ee_tcp_pos, dim=-1, p=2)
        is_close = distance <= threshold

        return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)
    
    def _touching_book_before_grasp(
            self,
            env: ManagerBasedRLEnv,
            grasp_pos: torch.Tensor,
            threshold: float = 0.1
    ) -> torch.Tensor:
        """Reward the agent for touching the book before grasping."""
        # End-effector position: (num_envs, 3)
        ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
        # Compute the distance of the end-effector to the handle
        distance = torch.norm(grasp_pos - ee_tcp_pos, dim=-1, p=2)
        # Reward the robot for reaching the handle
        asset: RigidObject = env.scene["book_01"]
        return torch.where((distance >= threshold) &
                           (torch.sum(torch.abs(asset.data.root_lin_vel_b[:, :3]), dim=1)>0.05),
                            1, 0)
    
def home_after_flip(
        env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Reward the agent for returning to the home position after flipping."""
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    robot: Articulation = env.scene[asset_cfg.name]
    object_cfg: SceneEntityCfg = SceneEntityCfg("book_01")
    object: RigidObject = env.scene[object_cfg.name]
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    z_component = quat_apply(object.data.root_quat_w, z_axis)[:, 2]
    grasp_ready_position = torch.tensor([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04], device=env.device).repeat(env.num_envs, 1)
    joint_pos_error = torch.sum(torch.abs(robot.data.joint_pos[:, asset_cfg.joint_ids] - grasp_ready_position), dim=1)
    # print(z_component)
    # print(joint_pos_error)
    reward_for_home_pose = 1.0 - torch.tanh(joint_pos_error / 2.0)
    
    return torch.where(z_component < -0.05, reward_for_home_pose, 0)

def ee_velocity(
    env: ManagerBasedRLEnv,
    threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for slower ee velocity."""
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[asset_cfg.name]
    # Target object position: (num_envs, 3)
    ee_vel_w = robot.data.body_lin_vel_w[:, 8, :] # 8 is the index of the end-effector(panda_hand link) in the articulation
    ee_velocity = torch.norm(ee_vel_w, dim=1)
    # print(ee_velocity)
    # print(robot.data.soft_joint_vel_limits)
    return torch.where(ee_velocity > threshold, torch.tanh(ee_velocity), 0)
    # return torch.where(ee_velocity > threshold, 0.7, 0)
    
def object_is_flipped(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("book_01")
) -> torch.Tensor:
    """Reward the agent for flipping the object."""
    object: RigidObject = env.scene[object_cfg.name]
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    z_component = quat_apply(object.data.root_quat_w, z_axis)[:, 2]

    return torch.where(z_component < 0.0, torch.tanh(-z_component*2), torch.tanh(-z_component))

def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height from initial position."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_ee_quat_diff(
    env: ManagerBasedRLEnv,
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
    env: ManagerBasedRLEnv,
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
    env: ManagerBasedRLEnv,
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

def touching_other_object(
    env: ManagerBasedRLEnv,
    asset_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("robot")],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize the agent for touching the other object."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg_list[0].name]
    penalty = torch.zeros(env.num_envs, device=env.device)
    for asset_cfg in asset_cfg_list:
        asset = env.scene[asset_cfg.name]
        penalty += torch.where(torch.sum(torch.abs(asset.data.root_lin_vel_b[:, :3]), dim=1)>threshold, 1.0, 0.0)
        # was 
        # penalty += torch.sum(torch.square(asset.data.root_lin_vel_b[:, :3]), dim=1)
    return penalty
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms, matrix_from_quat, euler_xyz_from_quat, quat_mul, transform_points
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class shelf_Grasp_Reaching(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup2")
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        shelf_cfg = SceneEntityCfg("shelf")
        robot_cfg = SceneEntityCfg("robot")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self._shelf: RigidObject = env.scene[shelf_cfg.name]
        self.__initial_object_pos = self._target.data.root_pos_w.clone()
        self._initial_shelf_pos = self._shelf.data.root_pos_w.clone()
        self._initial_ee_quat = self._ee.data.target_quat_w.clone()
        self._robot: RigidObject = env.scene[robot_cfg.name]

        self._initial_distance = torch.zeros(env.num_envs, device=env.device)

        self._ee_pos_last_w = self._ee.data.target_pos_w[..., 0, :].clone()

        self._target_last_w = self._target.data.root_pos_w.clone()
        self._reach_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._reach_offset[:, :3] = torch.tensor([0.0, 0., 0.03]) #  cup : 0.0 0. 0.03  /  cube : 0.0 0.0 -0.03  /  cylinder : 0.0 0.0 -0.02
    
    def __call__(self, env: ManagerBasedRLEnv,):

        reach = self.object_ee_distance(env)
        align = self.align_ee_target(env)

        return reach + align
    
    def object_ee_distance(self, env:ManagerBasedRLEnv) -> torch.Tensor:
        # ee target position
        offset_pos = transform_points(self._reach_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]
        # object_pos_w = self._target.data.root_pos_w.clone()


        distance = torch.norm(offset_pos - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)
        # reset_mask = env.episode_length_buf == 1
        
        # self._initial_distance[reset_mask] = distance[reset_mask].clone()
        # self._ee_pos_last_w = self._ee.data.target_pos_w[..., 0, :].clone()
        

        # print("distance: {}".format(distance))
        # print("ee: {}".format(self._ee.data.target_pos_w[..., 0, :]))
        # print("target: {}".format(offset_pos))

        zeta_m = torch.where(distance < 0.015 , 0, 1)

        reward = zeta_m * torch.exp(-1.2 * distance) + (1 - zeta_m)
        # print("reward: {}".format(reward))
        
        return reward

    def align_ee_target(self, env: ManagerBasedRLEnv,) -> torch.Tensor:
        # offset_pos = transform_points(self._reach_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]
        

        # reset_mask = env.episode_length_buf == 1
        # self._initial_ee_quat[reset_mask] = self._ee.data.target_quat_w[reset_mask, :].clone()
        offset_pos = transform_points(self._reach_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]
        # object_pos_w = self._target.data.root_pos_w.clone()


        distance = torch.norm(offset_pos - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)

        zeta_m = torch.where(distance < 0.01 , 0, 1)

        robot_pos_quat = self._robot.data.root_state_w[:, 3:7]

        ee_tcp_quat = self._ee.data.target_quat_w[..., 0, :]
        

        ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
        rot_mat = matrix_from_quat(robot_pos_quat)
        # init_rot_mat = matrix_from_quat(self._initial_ee_quat[..., 0, :])

        init_ee_x = rot_mat[..., 2]
        ee_tcp_x = ee_tcp_rot_mat [..., 0]


        align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -1*init_ee_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        return zeta_m * (torch.sign(align_x) * align_x**2)
    
class shelf_Collision(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup2")
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        shelf_cfg = SceneEntityCfg("shelf")
        wrist_frame_cfg= SceneEntityCfg("wrist_frame")
        wrist_upper_frame_cfg = SceneEntityCfg("wrist_upper_frame")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self._shelf: RigidObject = env.scene[shelf_cfg.name]
        self._wrist: FrameTransformer = env.scene[wrist_frame_cfg.name]
        self._wrist_upper: FrameTransformer = env.scene[wrist_upper_frame_cfg.name]

        self._initial_shelf_pos = self._shelf.data.root_pos_w.clone()

        self._target_last_w = self._target.data.root_pos_w.clone()

    
    def __call__(self, env: ManagerBasedRLEnv,):

        collision = self.shelf_collision_pentaly(env)
        collision_dynamic = self.shelf_dynamic_penalty(env)
        collision_dynamic_upper = self.shelf_dynamic_penalty_upper(env)
        # print(self._shelf.)
        return collision + collision_dynamic

    def shelf_collision_pentaly(self,env: ManagerBasedRLEnv,) -> torch.Tensor:
        
        shelf_vel = self._shelf.data.root_lin_vel_w
        shelf_delta = self._shelf.data.root_pos_w - self._initial_shelf_pos
        moved = torch.where(torch.norm(shelf_delta , dim=-1, p=2) + torch.norm(shelf_vel , dim=-1, p=2)> 0.05, 1.0, 0.0)

        return moved

    def shelf_dynamic_penalty(self, env: ManagerBasedRLEnv,) -> torch.Tensor:
        object_pos_w = self._target.data.root_pos_w.clone()
        object_pos_w[:, 2] = object_pos_w[:, 2] + 0.09
        object_pos_w[:, 1] = object_pos_w[:, 1] + 0.08

        distance = torch.norm(object_pos_w - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)

        zeta = torch.where(distance < 0.2, 1, 0)


        dst_ee_shelf = self._ee.data.target_pos_w[..., 0, 2] - (self._shelf.data.root_pos_w[:, 2] + 0.66)

        dst_wrist_shelf = self._wrist.data.target_pos_w[..., 0, 2] - (self._shelf.data.root_pos_w[:, 2] + 0.66)

        reward_ee = 1 - dst_ee_shelf / 0.04
        reward_wrist = 1 - dst_wrist_shelf / 0.04

        reward_ee = torch.clamp(reward_ee, 0, 1)
        reward_wrist = torch.clamp(reward_wrist, 0, 1)

        R = 5 * zeta * (reward_ee + reward_wrist)


        return R
    
    def shelf_dynamic_penalty_upper(self, env: ManagerBasedRLEnv,) -> torch.Tensor:
        object_pos_w = self._target.data.root_pos_w.clone()
        object_pos_w[:, 2] = object_pos_w[:, 2] + 0.09
        object_pos_w[:, 1] = object_pos_w[:, 1] + 0.08

        distance = torch.norm(object_pos_w - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)

        zeta = torch.where(distance < 0.2, 1, 0)


        dst_ee_shelf = self._ee.data.target_pos_w[..., 0, 2] - (self._shelf.data.root_pos_w[:, 2] + 0.96)

        dst_wrist_shelf = self._wrist_upper.data.target_pos_w[..., 0, 2] - (self._shelf.data.root_pos_w[:, 2] + 0.96)

        reward_ee = 1 - dst_ee_shelf / 0.05
        reward_wrist = 1 - dst_wrist_shelf / 0.0

        reward_ee = torch.clamp(reward_ee, 0, 1)
        reward_wrist = torch.clamp(reward_wrist, 0, 1)

        R = 5 * zeta * (reward_ee + reward_wrist)

        return R


class Object_drop(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self._target_last_w = self._target.data.root_pos_w.clone()

        self._top_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._top_offset[:, :3] = torch.tensor([0.0, 0.0, 0.07]) #0.0 0.0 0.07  

    def __call__(self, env:ManagerBasedRLEnv,):
        drop = self.object_drop(env)
        vel = self.object_velocity(env)
        return drop + vel

    def object_drop(self, env: ManagerBasedRLEnv,)-> torch.Tensor:

        offset_pos = transform_points(self._top_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]
        object_vel = self._target.data.root_lin_vel_w

        delta_z = 0.73 - offset_pos[:, 2] #0.73

        penalty_object = torch.tanh(5 * torch.abs(delta_z) / 0.01)
        return penalty_object
    
    def object_velocity(self, env: ManagerBasedRLEnv,)-> torch.Tensor:
        object_lin_vel_w = self._target.data.root_lin_vel_w.clone()
        object_lin_vel_norm = torch.norm(object_lin_vel_w, dim=-1, p=2)
        penalty = torch.where(object_lin_vel_norm > 1, 1, 0)
        return penalty
    
class Object2_drop(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup2")
        ee_frame_cfg = SceneEntityCfg("ee_frame")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self._target_last_w = self._target.data.root_pos_w.clone()

        self._top_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._top_offset[:, :3] = torch.tensor([0.0, 0.0, 0.07]) #0.0 0.0 0.07  

    def __call__(self, env:ManagerBasedRLEnv,):
        drop = self.object_drop(env)
        vel = self.object_velocity(env)
        return drop + vel

    def object_drop(self, env: ManagerBasedRLEnv,)-> torch.Tensor:

        offset_pos = transform_points(self._top_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]
        object_vel = self._target.data.root_lin_vel_w

        delta_z = 0.73 - offset_pos[:, 2] #0.73

        penalty_object = torch.tanh(5 * torch.abs(delta_z) / 0.01)
        return penalty_object
    
    def object_velocity(self, env: ManagerBasedRLEnv,)-> torch.Tensor:
        object_lin_vel_w = self._target.data.root_lin_vel_w.clone()
        object_lin_vel_norm = torch.norm(object_lin_vel_w, dim=-1, p=2)
        penalty = torch.where(object_lin_vel_norm > 1, 1, 0)
        return penalty
    
class Home_pose(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.object_cfg = SceneEntityCfg("cup2")
        self.asset_cfg = SceneEntityCfg("robot")        

        self._target: RigidObject = env.scene[self.object_cfg.name]
        self._robot: Articulation = env.scene[self.asset_cfg.name]

        self._initial_object_pos = self._target.data.root_pos_w.clone()

        self._initial_distance = torch.zeros(env.num_envs, device=env.device)

        self._target_last_w = self._target.data.root_pos_w.clone()

        self._top_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._top_offset[:, :3] = torch.tensor([0.0, 0.0, 0.07]) #0.0 0.0 0.07

        self.home_position = torch.tensor([-1.6, -1.9, 1.9, 0.05, 1.57, 1.57], device=env.device).repeat(env.num_envs, 1)
    
    def __call__(self, env: ManagerBasedRLEnv,):

        homing = self.home_pose(env)

        return homing
    
    def home_pose(self, env:ManagerBasedRLEnv) -> torch.Tensor:

        ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
        object_pos = env.scene["cup2"].data.root_pos_w.clone()
        object_quat = env.scene["cup2"].data.root_quat_w.clone()

        offset = torch.zeros((env.num_envs, 3), device=env.device)
        offset[:,:3] = torch.tensor([0.0, 0.0, -0.03])

        object_offset = transform_points(offset, object_pos, object_quat)[..., 0, :]
        distance = torch.norm(object_offset - ee_tcp_pos, dim=-1, p=2)
        dis_obj=torch.where(distance < 0.015, 1, 0)

        # current object state
        object_pos_w = self._target.data.root_pos_w.clone()

        # initial object & ee state
        reset_mask = env.episode_length_buf == 1
        self._initial_object_pos[reset_mask] = self._target.data.root_pos_w[reset_mask, :].clone()

        # ee target position
        offset_pos = transform_points(self._top_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]
       

        # Displacement of cup from initial state
        delta_z = (object_pos_w[:, 2] - self._initial_object_pos[:, 2])

        # indicator factor
        zeta_s = torch.where(torch.abs(delta_z) > 0.04, 1, 0)

        delta_z_D = 0.73 - offset_pos[:, 2] #0.73
        
        drop_con = torch.where(delta_z_D < 0.01, 1, 0)

        joint_pos_error = torch.sum(torch.abs(self._robot.data.joint_pos[:, :6] - self.home_position), dim=1)

        reward_for_home_pose = zeta_s * (1 - torch.tanh(joint_pos_error / 3.0))
        # reward_for_home_pose = dis_obj * zeta_s * drop_con * (1 - torch.tanh(joint_pos_error / 2.0))
        # reward_for_home_pose = drop_con * (1 - torch.tanh(joint_pos_error / 2.0))

        return reward_for_home_pose

def grasp_handle(
    env: ManagerBasedRLEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle.

    The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["cup2"].data.root_pos_w.clone()
    object_quat = env.scene["cup2"].data.root_quat_w.clone()
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


    # print("gripper: {}".format(gripper_joint_pos))
    offset = torch.zeros((env.num_envs, 3), device=env.device)
    offset[:,:3] = torch.tensor([0.0, 0.0, 0.03]) #0.0 0.0 0.03

    object_offset = transform_points(offset, object_pos, object_quat)[..., 0, :]
    distance = torch.norm(object_offset - ee_tcp_pos, dim=-1, p=2)

    is_close = distance <= threshold

    return is_close * torch.sum(gripper_joint_pos - open_joint_pos, dim=-1)

def object_lift( env: ManagerBasedRLEnv, threshold: float, object_cfg: SceneEntityCfg = SceneEntityCfg("cup2"), ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    offset = torch.zeros((env.num_envs, 3), device=env.device)
    offset[:,:3] = torch.tensor([0.0, 0.0, 0.03]) #0.0 0.0 0.03

    object_offset = transform_points(offset, obj.data.root_pos_w, obj.data.root_state_w[:, 3:7])[..., 0, :]
    distance = torch.norm(object_offset - ee_frame.data.target_pos_w[..., 0, :], dim=-1, p=2)
    zeta_m = torch.where(distance < 0.02 , 1, 0)
    # print(zeta_m)
    
    return torch.where((zeta_m * object_offset[:, 2])> threshold, 1.0, 0.0)
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


class ee_Reaching(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        shelf_cfg = SceneEntityCfg("shelf")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self._shelf: RigidObject = env.scene[shelf_cfg.name]
        self._initial_object_pos = self._target.data.root_pos_w.clone()
        self._initial_shelf_pos = self._shelf.data.root_pos_w.clone()
        self._initial_ee_quat = self._ee.data.target_quat_w.clone()

        self._initial_distance = torch.zeros(env.num_envs, device=env.device)

        self._ee_pos_last_w = self._ee.data.target_pos_w[..., 0, :].clone()

        self._target_last_w = self._target.data.root_pos_w.clone()
        self._reach_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._reach_offset[:, :3] = torch.tensor([0.06, 0.12, 0.04])
    
    def __call__(self, env: ManagerBasedRLEnv,):

        reach = self.object_ee_distance(env)


        return reach
    
    def object_ee_distance(self, env:ManagerBasedRLEnv) -> torch.Tensor:
        # ee target position

        object_pos_w = self._target.data.root_pos_w.clone()

        offset_pos = self._target.data.root_pos_w.clone()
        offset_pos[:,0] = offset_pos[:, 0] + 0.02
        offset_pos[:,1] = offset_pos[:, 1] + 0.08
        offset_pos[:,2] = offset_pos[:, 2] + 0.08 

        # initial object & ee state
        reset_mask = env.episode_length_buf == 1
        self._initial_object_pos[reset_mask] = self._target.data.root_pos_w[reset_mask, :].clone()
        distance = torch.norm(offset_pos - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)
        zeta_s = torch.where(torch.abs(object_pos_w[:, 1] - self._initial_object_pos[:, 1]) > 0.19, 0, 1)
        reward = zeta_s * torch.exp(-1.2 * distance) + (1 - zeta_s)

        # print("distance: {}".format(distance))
        # print("ee: {}".format(self._ee.data.target_pos_w[..., 0, :]))
        # print("target: {}".format(offset_pos))
        
        return reward
    

class ee_Align(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        shelf_cfg = SceneEntityCfg("shelf")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self.__initial_object_pos = self._target.data.root_pos_w.clone()
        self._initial_ee_quat = self._ee.data.target_quat_w.clone()

        self._initial_distance = torch.zeros(env.num_envs, device=env.device)

        self._ee_pos_last_w = self._ee.data.target_pos_w[..., 0, :].clone()

        self._target_last_w = self._target.data.root_pos_w.clone()
        self._reach_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._reach_offset[:, :3] = torch.tensor([0.06, 0.13, 0.03])
    
    def __call__(self, env: ManagerBasedRLEnv,):

        align = self.align_ee_target(env)


        return align

    def align_ee_target(self, env: ManagerBasedRLEnv,) -> torch.Tensor:      

        reset_mask = env.episode_length_buf == 1
        self._initial_ee_quat[reset_mask] = self._ee.data.target_quat_w[reset_mask, :].clone()


        ee_tcp_quat = self._ee.data.target_quat_w[..., 0, :]
        
        ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
        init_rot_mat = matrix_from_quat(self._initial_ee_quat[..., 0, :])

        init_ee_x = init_rot_mat[..., 0]
        ee_tcp_x = ee_tcp_rot_mat[..., 0]

        init_ee_y = init_rot_mat[..., 1]
        ee_tcp_y = ee_tcp_rot_mat[..., 1]

        align_x = torch.bmm(ee_tcp_x.unsqueeze(1), init_ee_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        align_y = torch.bmm(ee_tcp_y.unsqueeze(1), init_ee_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return torch.sign(align_y) * align_y**2
    

class shelf_Pushing(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self._initial_object_pos = self._target.data.root_pos_w.clone() 
        self._initial_ee_pos = self._ee.data.target_pos_w.clone() 

        self._initial_distance = torch.zeros(env.num_envs, device=env.device)

        self._ee_pos_last_w = self._ee.data.target_pos_w[..., 0, :].clone()

        self._target_last_w = self._target.data.root_pos_w.clone()

        self._push_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._push_offset[:, :3] = torch.tensor([0.06, 0.12, 0.04])
    
    def __call__(self, env: ManagerBasedRLEnv,):

        push = self.push_object(env)

        return push
    
    def push_object(self, env:ManagerBasedRLEnv) -> torch.Tensor:

        # current object state
        object_pos_w = self._target.data.root_pos_w.clone()
         
        # current ee state
        ee_pos_w = self._ee.data.target_pos_w.clone()

        # initial object & ee state
        reset_mask = env.episode_length_buf == 1
        self._initial_object_pos[reset_mask] = self._target.data.root_pos_w[reset_mask, :].clone()
        self._initial_ee_pos[reset_mask] = self._ee.data.target_pos_w[reset_mask, :].clone()

        # ee target position
        offset_pos = self._target.data.root_pos_w.clone()
        offset_pos[:,0] = offset_pos[:, 0] + 0.02
        offset_pos[:,1] = offset_pos[:, 1] + 0.08
        offset_pos[:,2] = offset_pos[:, 2] + 0.08

        # distance between ee and object
        distance = torch.norm(offset_pos - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)

        # Displacement of end-effector from initial state
        D_x_ee = (ee_pos_w[..., 0, 0]- self._initial_object_pos[:, 0])

        # Velocity of end-effector 
        v_y_ee = -1 * (ee_pos_w[..., 0, 1] - self._ee_pos_last_w[..., 0, 1])/env.step_dt


        # Displacement of cup from initial state
        delta_y = -1*(object_pos_w[:, 1] - self._initial_object_pos[:, 1])
        delta_x = (object_pos_w[:, 0] - self._initial_object_pos[:, 0])
        # delta_z = torch.where(torch.norm(object_pos_w[:, 2] - self._target_last_w[:, 2])>0.1, 1, 0)

        # indicator factor
        zeta_s = torch.where(torch.abs(object_pos_w[:, 1] - self._initial_object_pos[:, 1]) > 0.21, 0, 1)
        zeta_m = torch.where(distance < 0.03 , 1, 0)

        velocity_ee_reward = torch.where(v_y_ee < 0.3, v_y_ee * 2, -3 * v_y_ee)

        pushing_reward = zeta_s * zeta_m * ((4*torch.tanh(3*delta_y/0.2)) - 0.15 * (torch.tanh(2 * D_x_ee/0.1)) + velocity_ee_reward ) + 4 * (1 - zeta_s)
        # pushing_reward = zeta_s * zeta_m * ((4*torch.tanh(3*delta_y/0.2)) - 0.15 * (torch.tanh(2 * D_x_ee/0.1)) - 0.05 * (torch.tanh(2 * delta_x/0.1)) + 0.5 * (velocity_ee_reward + velocity_obj_reward))
        pushing_reward = torch.clamp(pushing_reward, -4, 4)

        R = pushing_reward
        # print("Distance: {}".format(distance))
        # print("delta_y: {}".format(delta_y))
        # print("delta_x: {}".format(D_x_ee))
        # print((1-zeta_s))
        # print("ee: {}".format( self._ee.data.target_pos_w[..., 0, :]))
        # print("object: {}".format(offset_pos))
        # print("zeta_m: {}".format(zeta_m))
        # print("zeta_s: {}".format(zeta_s))
        # print("reward: {}".format(R))
        # print("vel rew: {}".format(velocity_reward))
        # print("dt: {}".format(v_y_ee))
        # self._target_last_w = object_pos_w.clone()

        self._ee_pos_last_w = self._ee.data.target_pos_w.clone()
        
        return R


class shelf_Collision(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        shelf_cfg = SceneEntityCfg("shelf")
        wrist_frame_cfg= SceneEntityCfg("wrist_frame")
        # wrist_upper_frame_cfg = SceneEntityCfg("wrist_upper_frame")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self._shelf: RigidObject = env.scene[shelf_cfg.name]
        self._wrist: FrameTransformer = env.scene[wrist_frame_cfg.name]
        # self._wrist_upper: FrameTransformer = env.scene[wrist_upper_frame_cfg.name]

        self._initial_shelf_pos = self._shelf.data.root_pos_w.clone()

        self._target_last_w = self._target.data.root_pos_w.clone()

    
    def __call__(self, env: ManagerBasedRLEnv,):

        collision = self.shelf_collision_pentaly(env)
        collision_dynamic = self.shelf_dynamic_penalty(env)
        return collision + collision_dynamic

    def shelf_collision_pentaly(self,env: ManagerBasedRLEnv,) -> torch.Tensor:
        
        shelf_vel = self._shelf.data.root_lin_vel_w
        shelf_delta = self._shelf.data.root_pos_w - self._initial_shelf_pos
        moved = torch.where(torch.norm(shelf_delta , dim=-1, p=2) + torch.norm(shelf_vel , dim=-1, p=2)> 0.005, 1.0, 0.0)

        return moved

    def shelf_dynamic_penalty(self, env: ManagerBasedRLEnv,) -> torch.Tensor:
        object_pos_w = self._target.data.root_pos_w.clone()
        object_pos_w[:,0] = object_pos_w[:, 0] + 0.02
        object_pos_w[:,1] = object_pos_w[:, 1] + 0.08
        object_pos_w[:,2] = object_pos_w[:, 2] + 0.08

        distance = torch.norm(object_pos_w - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)

        zeta = torch.where(distance < 0.2, 1, 0)


        dst_ee_shelf = self._ee.data.target_pos_w[..., 0, 2] - (self._shelf.data.root_pos_w[:, 2] + 0.66)

        dst_wrist_shelf = self._wrist.data.target_pos_w[..., 0, 2] - (self._shelf.data.root_pos_w[:, 2] + 0.66)

        reward_ee = 1 - dst_ee_shelf / 0.06
        reward_wrist = 1 - dst_wrist_shelf / 0.06

        reward_ee = torch.clamp(reward_ee, 0, 1)
        reward_wrist = torch.clamp(reward_wrist, 0, 1)

        R = 5 * zeta * (reward_ee + reward_wrist)


        return R
    

class Object_drop(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        object2_cfg = SceneEntityCfg("cup2")
        ee_frame_cfg = SceneEntityCfg("ee_frame")


        self._target: RigidObject = env.scene[object_cfg.name]
        self._target2: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self._target_last_w = self._target.data.root_pos_w.clone()

        self._top_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._top_offset[:, :3] = torch.tensor([0.0, 0.0, 0.07]) #0.0 0.0 0.07

    def __call__(self, env:ManagerBasedRLEnv,):
        drop = self.object_drop(env)
        drop2 = self.object_drop2(env)
        vel = self.object_velocity(env)
        return drop + drop2 + vel

    def object_drop(self, env: ManagerBasedRLEnv,)-> torch.Tensor:

        offset_pos = transform_points(self._top_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]
        delta_z = 0.73 - offset_pos[:, 2] #0.73

        penalty_object = torch.tanh(5 * torch.abs(delta_z) / 0.01)
        return penalty_object
    
    def object_drop2(self, env: ManagerBasedRLEnv,)-> torch.Tensor:

        offset_pos = transform_points(self._top_offset,self._target2.data.root_pos_w, self._target2.data.root_state_w[:, 3:7] )[..., 0 , :]
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
        self.object_cfg = SceneEntityCfg("cup")
        self.asset_cfg = SceneEntityCfg("robot")
        ee_frame_cfg = SceneEntityCfg("ee_frame")          

        self._target: RigidObject = env.scene[self.object_cfg.name]
        self._robot: Articulation = env.scene[self.asset_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self._initial_object_pos = self._target.data.root_pos_w.clone()
        self._initial_ee_pos = self._ee.data.target_pos_w.clone() 

        self._initial_distance = torch.zeros(env.num_envs, device=env.device)

        self._target_last_w = self._target.data.root_pos_w.clone()

        self._top_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._top_offset[:, :3] = torch.tensor([0.0, 0.0, 0.07]) #0.0 0.0 0.07

        self.home_position = torch.tensor([-1.6, -1.9, 1.9, 0.05, 1.57, 2.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    
    def __call__(self, env: ManagerBasedRLEnv,):

        homing = self.home_pose(env)

        return homing
    
    def home_pose(self, env:ManagerBasedRLEnv) -> torch.Tensor:

        # current object state
        object_pos_w = self._target.data.root_pos_w.clone()

        # initial object & ee state
        reset_mask = env.episode_length_buf == 1
        self._initial_object_pos[reset_mask] = self._target.data.root_pos_w[reset_mask, :].clone()
        self._initial_ee_pos[reset_mask] = self._ee.data.target_pos_w[reset_mask, :].clone()

        # ee target position
        offset_pos = transform_points(self._top_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]
      

        # Displacement of cup from initial state
        delta_y = -1*(object_pos_w[:, 1] - self._initial_object_pos[:, 1])

        # indicator factor
        zeta_s = torch.where(torch.abs(delta_y) > 0.19, 1, 0)

        delta_z = 0.73 - offset_pos[:, 2]


        drop_con = torch.where(delta_z < 0.02, 1, 0)

        joint_pos_error = torch.sum(torch.abs(self._robot.data.joint_pos[:, :8] - self.home_position), dim=1)

        reward_for_home_pose = zeta_s * (1 - torch.tanh(joint_pos_error/3.0))

        return reward_for_home_pose

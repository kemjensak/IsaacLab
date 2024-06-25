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

class shelf_Reaching(ManagerTermBase):
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
        offset_pos[:,0] = offset_pos[:, 0] + 0.06
        offset_pos[:,1] = offset_pos[:, 1] + 0.11
        offset_pos[:,2] = offset_pos[:, 2] + 0.05 

        # initial object & ee state
        reset_mask = env.episode_length_buf == 1
        self._initial_object_pos[reset_mask] = self._target.data.root_pos_w[reset_mask, :].clone()

        distance = torch.norm(offset_pos - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)


        zeta_s = torch.where(torch.abs(object_pos_w[:, 1] - self._initial_object_pos[:, 1]) > 0.19, 0, 1)
        # print(zeta_s)
        reward = zeta_s * torch.exp(-1.2 * distance) + (1 - zeta_s)

        # reset_mask = env.episode_length_buf == 1
        
        # self._initial_distance[reset_mask] = distance[reset_mask].clone()
        # self._ee_pos_last_w = self._ee.data.target_pos_w[..., 0, :].clone()
        

        # print("distance: {}".format(distance))
        # print("ee: {}".format(self._ee.data.target_pos_w[..., 0, :]))
        # print("target: {}".format(offset_pos))
        
        return reward
    

class shelf_Align(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        shelf_cfg = SceneEntityCfg("shelf")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self._shelf: RigidObject = env.scene[shelf_cfg.name]

        self.__initial_object_pos = self._target.data.root_pos_w.clone()
        self._initial_shelf_pos = self._shelf.data.root_pos_w.clone()
        self._initial_ee_quat = self._ee.data.target_quat_w.clone()

        self._initial_distance = torch.zeros(env.num_envs, device=env.device)

        self._ee_pos_last_w = self._ee.data.target_pos_w[..., 0, :].clone()

        self._target_last_w = self._target.data.root_pos_w.clone()
        self._reach_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._reach_offset[:, :3] = torch.tensor([0.06, 0.12, 0.04])
    
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
        return (torch.sign(align_x) * align_x**2)
    
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
        offset_pos[:,0] = offset_pos[:, 0] + 0.06
        offset_pos[:,1] = offset_pos[:, 1] + 0.11
        offset_pos[:,2] = offset_pos[:, 2] + 0.05 

        # distance between ee and object
        distance = torch.norm(offset_pos - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)
        distance_f = torch.norm(self._initial_ee_pos[..., 0, :] - self._ee.data.target_pos_w[..., 0, :],  dim=-1, p=2)   

        # Displacement of end-effector from initial state
        D_y_ee = (ee_pos_w[..., 0, 1] - self._initial_ee_pos[..., 0, 1])
        D_x_ee = (ee_pos_w[..., 0, 0]- self._initial_object_pos[:, 0])
        D_z_ee = (ee_pos_w[..., 0, 2] - self._initial_ee_pos[..., 0, 2])

        # Velocity of end-effector 
        v_y_ee = -1 * (ee_pos_w[..., 0, 1] - self._ee_pos_last_w[..., 0, 1])/env.step_dt


        # Displacement of cup from initial state
        delta_y = -1*(object_pos_w[:, 1] - self._initial_object_pos[:, 1])
        delta_x = (object_pos_w[:, 0] - self._initial_object_pos[:, 0])
        # delta_z = torch.where(torch.norm(object_pos_w[:, 2] - self._target_last_w[:, 2])>0.1, 1, 0)

        # indicator factor
        zeta_s = torch.where(torch.abs(object_pos_w[:, 1] - self._initial_object_pos[:, 1]) > 0.19, 0, 1)
        zeta_m = torch.where(torch.bitwise_or(distance < 0.03, torch.abs(object_pos_w[:, 1] - self._initial_object_pos[:, 1]) > 0.19) , 1, 0)
        # zeta_m = torch.where(distance < 0.03 , 1, 0)

        velocity_reward = zeta_s * torch.where(v_y_ee < 0.4, v_y_ee * 5, -5 * v_y_ee)
        pushing_reward = zeta_m * ((3*torch.tanh(2*delta_y/0.2)) - 0.1 * (torch.tanh(2 * D_x_ee/0.1)) + velocity_reward)
        pushing_reward = torch.clamp(pushing_reward, -4, 4)

        # R = pushing_reward + (1-zeta_s)*3*torch.exp(-1.2 * distance_f)
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

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self._shelf: RigidObject = env.scene[shelf_cfg.name]
        self._wrist: FrameTransformer = env.scene[wrist_frame_cfg.name]

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


class Object_drop(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self._target_last_w = self._target.data.root_pos_w.clone()

        self._top_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._top_offset[:, :3] = torch.tensor([0.0, 0.0, 0.07])

    def __call__(self, env:ManagerBasedRLEnv,):
        drop = self.object_drop(env)
        vel = self.object_velocity(env)
        return drop + vel

    def object_drop(self, env: ManagerBasedRLEnv,)-> torch.Tensor:

        offset_pos = transform_points(self._top_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]
        object_vel = self._target.data.root_lin_vel_w

        delta_z = 0.73 - offset_pos[:, 2]

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

        self._target: RigidObject = env.scene[self.object_cfg.name]
        self._robot: Articulation = env.scene[self.asset_cfg.name]

        self._initial_object_pos = self._target.data.root_pos_w.clone()

        self._initial_distance = torch.zeros(env.num_envs, device=env.device)

        self._target_last_w = self._target.data.root_pos_w.clone()

        self._top_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._top_offset[:, :3] = torch.tensor([0.0, 0.0, 0.07])

        self.home_position = torch.tensor([-1.6, -1.9, 1.9, 0.05, 1.57, 2.1, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    
    def __call__(self, env: ManagerBasedRLEnv,):

        homing = self.home_pose(env)

        return homing
    
    def home_pose(self, env:ManagerBasedRLEnv) -> torch.Tensor:

        # current object state
        object_pos_w = self._target.data.root_pos_w.clone()

        # initial object & ee state
        reset_mask = env.episode_length_buf == 1
        self._initial_object_pos[reset_mask] = self._target.data.root_pos_w[reset_mask, :].clone()

        # ee target position
        offset_pos = transform_points(self._top_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]
       

        # Displacement of cup from initial state
        delta_y = -1*(object_pos_w[:, 1] - self._initial_object_pos[:, 1])

        # indicator factor
        zeta_s = torch.where(torch.abs(delta_y) > 0.19, 1, 0)

        delta_z = 0.73 - offset_pos[:, 2]

        # add-haneul
        # object_lin_vel_w = self._target.data.root_lin_vel_w.clone()
        # object_lin_vel_norm = torch.norm(object_lin_vel_w, dim=-1, p=2)
        # object_vel = torch.where(object_lin_vel_norm < 0.1, 1, 0)

        drop_con = torch.where(delta_z < 0.01, 1, 0)

        joint_pos_error = torch.sum(torch.abs(self._robot.data.joint_pos[:, :8] - self.home_position), dim=1)

        reward_for_home_pose = zeta_s * drop_con * (1 - torch.tanh(joint_pos_error / 2.0))

        return reward_for_home_pose
    


        





    
def object_lift( env: ManagerBasedRLEnv, threshold: float, object_cfg: SceneEntityCfg = SceneEntityCfg("cup")) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2]> threshold, 1.0, 0.0)

def shelf_collision_pentaly( env: ManagerBasedRLEnv, threshold: float, shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),) -> torch.Tensor:
    shelf: RigidObject = env.scene[shelf_cfg.name]

    shelf_vel = shelf.data.root_lin_vel_w
    shelf_ang_vel = shelf.data.root_ang_vel_w
    
    moved = torch.where( torch.norm(shelf_vel , dim=-1, p=2) + torch.norm(shelf_ang_vel , dim=-1, p=2)> threshold, 1.0, 0.0)

    return moved

def object_collision_pentaly( 
        env: ManagerBasedRLEnv, 
        object_cfg: SceneEntityCfg = SceneEntityCfg("cup"), 
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    
    object_pos_w = object.data.root_pos_w

    distance = torch.norm(object_pos_w - ee_w[..., 0, :], dim=-1, p=2)
    object_vel = object.data.root_lin_vel_w
    object_ang_vel = object.data.root_ang_vel_w
    
    zeta = torch.where(distance > 0.05, 0, 1)
    mu_v = torch.where(torch.norm(object_vel, dim=-1, p=2) < 0.02, 0, 1)
    mu_w = torch.where(torch.norm(object_ang_vel, dim=-1, p=2)< 0.1, 0, 1)

    R = 0.5*(mu_v*torch.tanh(torch.norm(object_vel, dim=-1, p=2)/0.01))
    return R
    

def shelf_dynamic_collision_penalty(
        env: ManagerBasedRLEnv,
        threshold: float,
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        wrist_frame_cfg: SceneEntityCfg = SceneEntityCfg("wrist_frame"),
        shelf_frame_cfg : SceneEntityCfg = SceneEntityCfg("shelf")
) -> torch.Tensor:
    def point_to_line_distance_tensor(point, line_start, line_end):

        point = torch.tensor(point, dtype=torch.float32).cuda() if not isinstance(point, torch.Tensor) else point.cuda()
        line_start = torch.tensor(line_start, dtype=torch.float32).cuda() if not isinstance(line_start, torch.Tensor) else line_start.cuda()
        line_end = torch.tensor(line_end, dtype=torch.float32).cuda() if not isinstance(line_end, torch.Tensor) else line_end.cuda()

        # Vector from line start to point
        vec_start_to_point = point - line_start
        # Vector from line start to line end
        vec_start_to_end = line_end - line_start
        

        # Project vector onto the line
        projection = torch.sum(vec_start_to_point * vec_start_to_end, dim=-1, keepdim=True) / torch.sum(vec_start_to_end * vec_start_to_end, dim=-1, keepdim=True)        
        # Clamp the projection between 0 and 1 to find the nearest point on the line segment
        projection = torch.clamp(projection, 0, 1)
        # Calculate the nearest point on the line segment
        nearest_point = line_start + projection * vec_start_to_end
        # Distance from the point to the nearest point on the line segment
        # print(nearest_point)
        distance = torch.norm(point - nearest_point, dim=-1)
        return distance

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    wrist_frame: FrameTransformer = env.scene[wrist_frame_cfg.name]
    wrist_w = wrist_frame.data.target_pos_w[..., 0, :]

    

    shelf_frame = FrameTransformer = env.scene[shelf_frame_cfg.name]
    shelf_w = shelf_frame.data.root_pos_w

    

    lower_x_bounds = shelf_w[..., :, 0] - 0.25
    upper_x_bounds = shelf_w[..., :, 0] + 0.25

    lower_y_bounds = shelf_w[..., :, 1] - 0.6
    upper_y_bounds = shelf_w[..., :, 1] + 0.6

    lower_plane_z = shelf_w[..., :, 2] + 0.66
    upper_plane_z = shelf_w[..., :, 2] + 0.96

    
    

    # Check if both G and W are within x bounds
    outside_x_bounds = (ee_w[:, 0] < lower_x_bounds ) & (ee_w[:, 0] > upper_x_bounds) & (wrist_w[:, 0] < lower_x_bounds ) & (wrist_w[:, 0] > upper_x_bounds)
    
    # # Check boundaries in y directions
    outside_y_bounds = (ee_w[:, 1] < lower_y_bounds) & (ee_w[:, 1] > upper_y_bounds) & (wrist_w[:, 1] < lower_y_bounds) & (wrist_w[:, 1] > upper_y_bounds)

    # # Combine x and y bound conditions
    outside_bounds = outside_x_bounds & outside_y_bounds
    points = ee_w[..., :].clone()
    points[..., :, 2] = lower_plane_z


    
    lower_plane_distance = point_to_line_distance_tensor(points, ee_w[...,:,:], wrist_w[...,:,:])


    condition = torch.logical_or(lower_plane_distance > threshold, outside_bounds)
    return torch.where(condition, torch.tensor(0.0, dtype=torch.float32).cuda(), torch.tanh(threshold / lower_plane_distance))
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms, matrix_from_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

class shelf_class(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self.__initial_object_pos = self._target.data.root_pos_w.clone()

        self._initial_distance = torch.zeros(env.num_envs, device=env.device)

        self._ee_pos_last_w = self._ee.data.target_pos_w[..., 0, :].clone() 
    
    def __call__(self, env: ManagerBasedRLEnv,):

        reach = self.object_ee_distance(env)
        align = self.align_ee_target(env)

        return reach 


    def object_ee_distance(self, env:ManagerBasedRLEnv) -> torch.Tensor:
        ready_point_pos_w = self._target.data.root_pos_w.clone()
        ready_point_pos_w[:, 2] = ready_point_pos_w[:, 2] + 0.05
        ready_point_pos_w[:, 1] = ready_point_pos_w[:, 1] + 0.1

        vec_p = (ready_point_pos_w - self._ee.data.target_pos_w[..., 0, :])/torch.norm((ready_point_pos_w - self._ee.data.target_pos_w[..., 0, :]),dim=-1, p=2, keepdim=True)
  
        vec_u = (self._ee.data.target_pos_w[..., 0, :] - self._ee_pos_last_w)/torch.norm((self._ee.data.target_pos_w[..., 0, :] - self._ee_pos_last_w),dim=-1, p=2, keepdim=True)

        kappa = torch.sum(vec_p * vec_u, dim=-1, keepdim=True) * 2
        sign = torch.sign(torch.sum(vec_p * vec_u, dim=-1, keepdim=True))
        epsilon = torch.where(ready_point_pos_w[:, 0] - self._ee.data.target_pos_w[..., 0, 0] < 0.005, 1.0 + 0.1*(50 - torch.norm(ready_point_pos_w - self._ee.data.target_pos_w)*100), 1.0)
        
        distance = torch.norm(ready_point_pos_w - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)
        reset_mask = env.episode_length_buf == 1
        
        self._initial_distance[reset_mask] = distance[reset_mask].clone()
        zeta = torch.where(torch.norm(ready_point_pos_w - self._ee.data.target_pos_w)<0.005, 1/(sign.squeeze()*kappa.squeeze()*(1-distance/self._initial_distance)), 1)
        self._ee_pos_last_w = self._ee.data.target_pos_w[..., 0, :].clone()

        R = epsilon * torch.tanh(zeta*sign.squeeze()*kappa.squeeze()*(1-distance/self._initial_distance))
        
        
        return R




    # def object_ee_distance(self, env: ManagerBasedRLEnv,) -> torch.Tensor:

    #     object_pos_w = self._target.data.root_pos_w.clone()
    #     object_pos_w[:, 2] = object_pos_w[:, 2] + 0.05
    #     ee_w = self._ee.data.target_pos_w[..., 0, :]

    #     distance = torch.norm(object_pos_w - ee_w, dim=-1, p=2)

    #     reset_mask = env.episode_length_buf == 1
    #     self._initial_distance[reset_mask] = distance[reset_mask].clone()
    #     distance_ratio = 1 - distance / self._initial_distance
    #     print(torch.clamp(distance_ratio, 0, 1))
    #     return torch.clamp(distance_ratio, 0, 1)
    
    def align_ee_target(self, env: ManagerBasedRLEnv,) -> torch.Tensor:
        ee_tcp_quat = self._ee.data.target_quat_w[..., 0, :]
        world_quat = env.scene["robot"].data.root_quat_w

        ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
        world_rot_mat = matrix_from_quat(world_quat)

        world_z = world_rot_mat[..., 2]
        ee_tcp_x = ee_tcp_rot_mat [..., 0]

        align_z = torch.bmm(ee_tcp_x.unsqueeze(1), -world_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return 0.5 * (torch.sign(align_z) * align_z**2)
    
def object_lift( env: ManagerBasedRLEnv, threshold: float, object_cfg: SceneEntityCfg = SceneEntityCfg("cup")) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2]> threshold, 1.0, 0.0)

def shelf_collision_pentaly( env: ManagerBasedRLEnv, threshold: float, shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),) -> torch.Tensor:
    shelf: RigidObject = env.scene[shelf_cfg.name]

    shelf_vel = shelf.data.root_lin_vel_w
    shelf_ang_vel = shelf.data.root_ang_vel_w
    
    moved = torch.where( torch.norm(shelf_vel , dim=-1, p=2) + torch.norm(shelf_ang_vel , dim=-1, p=2)> threshold, 1.0, 0.0)

    return moved
    
    

def shelf_dynamic_collision_penalty(
        env: ManagerBasedRLEnv,
        threshold: float,
        x_bounds: torch.Tensor,
        y_bounds: torch.Tensor,
        z_bounds: torch.Tensor,
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
    outside_x_bounds = (ee_w[:, 0] < lower_x_bounds ) | (ee_w[:, 0] > upper_x_bounds) | (wrist_w[:, 0] < lower_x_bounds ) | (wrist_w[:, 0] > upper_x_bounds)
    
    # # Check boundaries in y directions
    outside_y_bounds = (ee_w[:, 1] < lower_y_bounds) | (ee_w[:, 1] > upper_y_bounds) | (wrist_w[:, 1] < lower_y_bounds) | (wrist_w[:, 1] > upper_y_bounds)

    # # Combine x and y bound conditions
    outside_bounds = outside_x_bounds & outside_y_bounds
    points = ee_w[..., :].clone()
    points[..., :, 2] = lower_plane_z


    
    lower_plane_distance = point_to_line_distance_tensor(points, ee_w[...,:,:], wrist_w[...,:,:])
    points[..., :, 2] = upper_plane_z
    upper_plane_distance = point_to_line_distance_tensor(points, ee_w, wrist_w)


    near_plane_distance = torch.min(lower_plane_distance, upper_plane_distance)
    condition = torch.logical_or(near_plane_distance > threshold, outside_bounds)
    # print(torch.where(condition, torch.tensor(0.0, dtype=torch.float32).cuda(), torch.tanh(threshold / near_plane_distance)))
    # return 
    # return torch.where(outside_bounds, torch.tensor(0.0, dtype=torch.float32).cuda(), torch.tanh(threshold/near_plane_distance) if near_plane_distance < threshold else torch.tensor(0, dtype=torch.float32).cuda())
    return torch.where(condition, torch.tensor(0.0, dtype=torch.float32).cuda(), torch.tanh(threshold / near_plane_distance))
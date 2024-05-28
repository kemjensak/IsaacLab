from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import FrameTransformer
from omni.isaac.orbit.utils.math import combine_frame_transforms
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def object_is_grasped(
    env: RLTaskEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""

    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    target_pos = env.scene["cup"].data.root_pos_w[..., 0, :]
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    distance = torch.norm(target_pos - ee_tcp_pos, dim=-1, p=2)
    is_close = distance <= threshold
    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)

# def align_ee_target(env: RLTaskEnv) -> torch.Tensor:
#     """
#     Reward for aligning the end-effector with the target object
#     """
#     ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
#     world_quat = env.scene["shelf_frame"].data.target_quat_w[..., 0, :]
    
#     ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
#     world_rot_mat = matrix_from_quat(world_quat)

#     world_x, world_y = world_rot_mat[..., 0], world_rot_mat[..., 1]
#     ee_tcp_y, ee_tcp_z = ee_tcp_rot_mat[..., 1], ee_tcp_rot_mat[..., 2]
    
#     align_z = torch.bmm(ee_tcp_z.unsqueeze(1), world_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
#     align_y = torch.bmm(ee_tcp_y.unsqueeze(1), world_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
#     return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_y) * align_y**2)

def object_ee_distance(
    env: RLTaskEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    target_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(target_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)

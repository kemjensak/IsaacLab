from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, Articulation
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import FrameTransformer

import numpy as np

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv, BaseEnv


def save_object_pose(
    env: RLTaskEnv,
    object_cfg: list[SceneEntityCfg],
    save_path: str = "/home/kjs-dt/RL/objcet_pose/object_poses_new.npy",
    minimal_height: float = -0.06,
):
    asset: RigidObject | Articulation = env.scene[object_cfg[0].name]
    # get default root state
    root_states = asset.data.default_root_state.clone()

    """Save pose of object on the table ."""
    if env.episode_length_buf[0] != 250:
        return
    loaded_object_poses = torch.empty(1,8,7, device=env.device)
    try:
        loaded_object_poses = torch.from_numpy(np.load("/home/kjs-dt/RL/objcet_pose/object_poses.npy")).to('cuda')
    except:
        print("No file")

    all_object_poses = torch.stack([env.scene[object_cfg_idx.name].data.root_state_w[:, :7]
                                            for object_cfg_idx in object_cfg], dim=1)
    
    
    all_object_poses[:, :, :3] -= (env.scene.env_origins + root_states[:, 0:3]).unsqueeze(1)
    
    for env_idx in all_object_poses:
        object_dropped = False
        for object_idx in env_idx:
            if object_idx[2] < -minimal_height:
                object_dropped = True
                break
        if object_dropped:
            continue
        loaded_object_poses = torch.concat((loaded_object_poses, env_idx.unsqueeze(0)), dim=0)

    print(loaded_object_poses.shape)
    object_poses_np = loaded_object_poses.cpu().numpy()
    np.save("/home/kjs-dt/RL/objcet_pose/object_poses.npy", object_poses_np)

def reset_root_state_from_file(
    env: BaseEnv,
    env_ids: torch.Tensor,
    object_cfg: list[SceneEntityCfg],
    loaded_object_poses: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[object_cfg[0].name]
    # get default root state
    root_states = asset.data.default_root_state.clone()[:len(env_ids)]

    # episode(env) 갯수만큼 랜덤하게 뽑고, idx 저장
    env_indices = torch.randperm(len(loaded_object_poses))[:len(env_ids)]

    # 각 env에 대해 object pose를 적용
    for object_idx in range(len(object_cfg)):
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + loaded_object_poses[env_indices, object_idx, 0:3] 
        orientations = loaded_object_poses[env_indices, object_idx, 3:7]
        asset = env.scene[object_cfg[object_idx].name]
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        
        velocities = root_states[:, 7:13]
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    


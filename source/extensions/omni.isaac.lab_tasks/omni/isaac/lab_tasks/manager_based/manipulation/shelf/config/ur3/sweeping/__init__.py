# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Shelf-UR3-Sweep-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.UR3ShelfEnvCfg,
    #    "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ShelfSweepPPORunnerCfg,
    #    "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml"
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Shelf-UR3-Sweep-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.UR3ShelfEnvCfg_PLAY,
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ShelfSweepPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml"
    },
    disable_env_checker=True,
)

# ##
# # Inverse Kinematics - Absolute Pose Control
# ##

# gym.register(
#     id="Isaac-Shelf-UR5e-Sweep-IK-Abs-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": ik_abs_env_cfg.UR5eShelfEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ShelfSweepPPORunnerCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Shelf-UR5e-Sweep-IK-Abs-v0-Play-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": ik_abs_env_cfg.UR5eShelfEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ShelfSweepPPORunnerCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
#     disable_env_checker=True,
# )
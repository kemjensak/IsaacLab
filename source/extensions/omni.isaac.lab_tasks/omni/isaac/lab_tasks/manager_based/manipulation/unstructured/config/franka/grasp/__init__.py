# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import agents, rgb_ik_abs_env_cfg
from . import ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg


##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Grasp-Object-Franka-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaGraspObjectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Grasp-Object-Franka-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaGraspObjectEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Grasp-Object-Franka-IK-Abs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.FrankaGraspObjectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Grasp-Object-Franka-IK-Abs-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.FrankaGraspObjectEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Grasp-Object-Franka-IK-Rel-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.FrankaGraspObjectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Grasp-Object-Franka-IK-Rel-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.FrankaGraspObjectEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control + RGB cam
##

gym.register(
    id="Isaac-Grasp-Object-Franka-RGB-Camera-IK-Abs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": rgb_ik_abs_env_cfg.FrankaGraspObjectEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Grasp-Object-Franka-RGB-Camera-IK-Abs-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": rgb_ik_abs_env_cfg.FrankaGraspObjectEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)


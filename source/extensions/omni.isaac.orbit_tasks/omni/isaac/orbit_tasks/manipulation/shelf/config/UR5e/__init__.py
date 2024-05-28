import gymnasium as gym
import os

from . import agents
from .joint_pos_env_cfg import UR5eShelfEnvCfg, UR5eShelfEnvCfg_PLAY


##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Grasp-Object-UR5e-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UR5eShelfEnvCfg,
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftCubePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-Object-UR5e-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UR5eShelfEnvCfg_PLAY,
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftCubePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
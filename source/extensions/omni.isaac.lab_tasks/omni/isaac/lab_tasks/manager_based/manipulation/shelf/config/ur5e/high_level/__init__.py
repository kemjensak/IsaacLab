import gymnasium as gym
import os

from . import agents
from omni.isaac.lab_tasks.manager_based.manipulation.shelf.shelf_high_level_env_cfg import HighLevelEnvCfg


##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-High-Level-Shelf-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": HighLevelEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ShelfHighLevelPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-High-Level-Shelf-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": HighLevelEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ShelfHighLevelPPORunnerCfg,
    },
    disable_env_checker=True,
)

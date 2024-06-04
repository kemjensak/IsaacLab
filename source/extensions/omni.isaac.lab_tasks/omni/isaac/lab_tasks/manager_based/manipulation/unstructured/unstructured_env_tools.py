from __future__ import annotations

from dataclasses import MISSING


from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from . import mdp

##
# Scene definition
##s

def SetRigidObjectCfgFromUsdFile(usd_file_name: str):
    return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/"+(usd_file_name),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"/home/kjs-dt/isaac_save/2023.1.1/obejcts/"+(usd_file_name)+".usd",
                scale=(0.01, 0.01, 0.01),
                rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
                )
            )
        )


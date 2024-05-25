from __future__ import annotations

from dataclasses import MISSING


from omni.isaac.orbit.assets import RigidObjectCfg
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg


from . import mdp

##
# Scene definition
##

def SetRigidObjectCfgFromUsdFile(usd_file_name: str):
    return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/"+(usd_file_name),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, 0.0, 0.405), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"/home/irol/KTH_dt/usd/Object/"+(usd_file_name)+".usd", #usd_path -> local directory
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
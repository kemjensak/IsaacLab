import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import RigidObjectCfg
from omni.isaac.orbit.sensors import FrameTransformerCfg, CameraCfg, ContactSensorCfg, RayCasterCfg, patterns 
from omni.isaac.orbit.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.orbit.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

from omni.isaac.orbit_tasks.manipulation.shelf import mdp
from omni.isaac.orbit_tasks.manipulation.shelf.shelf_grasp_env_cfg import ShelfGraspEnvCfg

##
# Pre-defined configs
##
from omni.isaac.orbit.markers.config import FRAME_MARKER_CFG 
from omni.isaac.orbit_assets.ur5e import UR5e_CFG

@configclass
class UR5eTargetGraspEnvCfg(ShelfGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UR5e_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        # self.rewards.

        # override actions
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )

        # override command generator body
        # end-effector is along x-direction
        self.commands.object_pose.body_name =  "ee_link"

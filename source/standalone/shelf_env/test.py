# import numpy as np

<<<<<<< Updated upstream
"""
This script demonstrates the environment for a quadruped robot with height-scan sensor.

In this example, we use a locomotion policy to control the robot. The robot is commanded to
move forward at a constant velocity. The height-scan sensor is used to detect the height of
the terrain.
=======
# import matplotlib.pyplot as plt
 
# image_data = np.load('source/standalone/shelf_env/output/camera/distance_to_image_plane_1_0.npy')
>>>>>>> Stashed changes

# depth_image_normalized = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

<<<<<<< Updated upstream
    # Run the script
    ./isaaclab.sh -p source/standalone/tutorials/04_envs/quadruped_base_env.py --num_envs 32
=======
# # Plot the depth image with a grayscale colormap
# plt.imshow(depth_image_normalized, cmap='gray')
# plt.axis('off')  # Hide axis for a cleaner look
# plt.show()
>>>>>>> Stashed changes

# import os

<<<<<<< Updated upstream
"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a quadruped base environment.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from omni.isaac.lab_assets import UR3_CFG   # isort: skip


##
# Custom observation terms
##


def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # add terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # add robot
    robot: ArticulationCfg = UR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
=======
# def delete_npy_files(folder_path):
#     # Check if the provided folder path exists
#     if not os.path.exists(folder_path):
#         print(f"The folder '{folder_path}' does not exist.")
#         return
    
#     # Iterate through the files in the folder
#     for file_name in os.listdir(folder_path):
#         # Build the full file path
#         file_path = os.path.join(folder_path, file_name)
        
#         # Check if it's a file and ends with .npy
#         if os.path.isfile(file_path) and file_name.endswith('.npy'):
#             try:
#                 os.remove(file_path)
#                 print(f"Deleted: {file_path}")
#             except Exception as e:
#                 print(f"Error deleting file {file_path}: {e}")

# # Specify the folder to clean
# folder_to_clean = "/home/haneul/IsaacLab/source/standalone/shelf_env/output/camera"
# delete_npy_files(folder_to_clean)


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
import cv2

def combine_all_images(folder_path, output_file="merged_depth_image.npy", sigma=5):
    """
    Combine all semantic segmentation and depth images in a folder to create a merged depth map.

    Parameters:
        folder_path (str): Path to the folder containing image files.
        output_file (str): Path to save the final merged depth image.
        sigma (float): Standard deviation for Gaussian blur.

    Returns:
        np.ndarray: The merged depth image.
    """
    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Separate semantic segmentation files and depth files
    segmentation_files = sorted([f for f in all_files if f.startswith("semantic_segmentation") and f.endswith(".png")])
    depth_files = sorted([f for f in all_files if f.startswith("distance_to_image_plane") and f.endswith(".npy")])

    if len(segmentation_files) != len(depth_files):
        raise ValueError("Number of semantic segmentation files and depth files do not match.")

    # Initialize a list to hold masked depth images
    masked_depth_images = []
>>>>>>> Stashed changes

    # Process each pair of segmentation and depth images
    for seg_file, depth_file in zip(segmentation_files, depth_files):
        # Load the segmentation image
        seg_path = os.path.join(folder_path, seg_file)
        segmentation = np.array(Image.open(seg_path))

<<<<<<< Updated upstream
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
=======
        # Convert RGBA to a binary mask (assume green indicates the cup)
        green_channel = segmentation[..., 1]  # Extract green channel
        cup_mask = (green_channel > 0).astype(np.float32)  # Create binary mask

        # Load the depth image
        depth_path = os.path.join(folder_path, depth_file)
        depth_image = np.load(depth_path).squeeze()
>>>>>>> Stashed changes

        # Mask the depth image (set non-cup regions to 0)
        masked_depth = depth_image * cup_mask
        masked_depth_images.append(masked_depth)

<<<<<<< Updated upstream
##
# MDP settings
##
=======
    # Stack masked depth images into a 3D array
    masked_depth_stack = np.array(masked_depth_images)

    # Identify pixels that have constant values across all frames and set them to 0
    constant_mask = np.all(masked_depth_stack == masked_depth_stack[0], axis=0)
    merged_depth = np.mean(masked_depth_stack, axis=0)
    merged_depth[constant_mask] = 0
>>>>>>> Stashed changes

    # Apply Gaussian blur for smoothing
    blurred_merged_depth = gaussian_filter(merged_depth, sigma=sigma)

<<<<<<< Updated upstream
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


##
# Environment configuration
##


@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)



def main():
    """Main function."""
    # setup base environment
    env_cfg = QuadrupedEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    # load level policy
    policy_path = ISAACLAB_NUCLEUS_DIR + "/Policies/ANYmal-C/HeightScan/policy.pt"
    # check if policy file exists
    if not check_file_path(policy_path):
        raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
    file_bytes = read_file(policy_path)
    # jit load the policy
    policy = torch.jit.load(file_bytes).to(env.device).eval()

    # simulate physics
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # infer action
            action = policy(obs["policy"])
            # step env
            obs, _ = env.step(action)
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
=======
    # Save the final merged depth map
    # output_path = os.path.join(folder_path, output_file)
    # np.save(output_path, blurred_merged_depth)
    # print(f"Merged depth image saved to {output_path}")

     # Save the final merged depth map
    # output_path = os.path.join(folder_path, output_file)
    # np.save(output_path, blurred_merged_depth)
    # print(f"Merged depth image saved to {output_path}")

    # Normalize the depth image to uint8 for OpenCV display
    normalized_depth = cv2.normalize(blurred_merged_depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = normalized_depth.astype(np.uint8)

    # Display the image using OpenCV
    cv2.imshow("Merged Depth Map", blurred_merged_depth)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

    return blurred_merged_depth

# Example usage
folder_to_process = "/home/haneul/IsaacLab/source/standalone/shelf_env/output/camera"
combine_all_images(folder_to_process, output_file="merged_depth_image.npy", sigma=5)
>>>>>>> Stashed changes

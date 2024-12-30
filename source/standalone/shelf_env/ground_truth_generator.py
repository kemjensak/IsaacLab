# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the camera sensor from the Isaac Lab framework.

The camera sensor is created and interfaced through the Omniverse Replicator API. However, instead of using
the simulator or OpenGL convention for the camera, we use the robotics or ROS convention.

.. code-block:: bash

    # Usage with GUI
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/run_usd_camera.py --enable_cameras

    # Usage with headless
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/run_usd_camera.py --headless --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import random
import torch

import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.utils import convert_dict_to_backend, math


def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    # In contrast to the ray-cast camera, we spawn the prim at these locations.
    # This means the camera sensor will be attached to these prims.
    prim_utils.create_prim("/World/Origin_00", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "semantic_segmentation",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene() -> dict:
    """Design the scene."""
    # Populate scene
    # -- Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- Lights
    # spawn distant light
    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(1.0, 1.0, 1.0),
    )
    
    cfg_light_dome.func("/World/lightDistant", cfg_light_dome, translation=(-5, 0, 10))

    # Create a dictionary for the scene entities
    scene_entities = {}
    
    # spawn a usd file of a shelf into the scene
    rack_cfg = RigidObjectCfg(
        prim_path="/World/Rack",
        spawn=sim_utils.UsdFileCfg(usd_path=f"omniverse://localhost/Library/Shelf/Arena/test_rack.usd",),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        debug_vis=False,
    )
    
    rack = RigidObject(cfg=rack_cfg)

    # Xform to hold objects
    prim_utils.create_prim("/World/Objects", "Xform", translation=(-0.24, -0.3, 0.66))
    
     

    
    scene_entities = obj_spawn(scene_entities=scene_entities)

    # Sensors
    camera = define_sensor()

    # return the scene information
    scene_entities["camera"] = camera
    return scene_entities

def obj_spawn(scene_entities:dict, 
              prim_path = "/World/Objects/target", 
              common_properties = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
            "mass_props": sim_utils.MassPropertiesCfg(mass=1.0),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0)) -> dict:
    obj = RigidObject(cfg=RigidObjectCfg(
                                prim_path = prim_path,
                                spawn=sim_utils.UsdFileCfg(
                                    usd_path=f"omniverse://localhost/Library/Shelf/Object/Can_1.usd",
                                    scale=(1.0, 1.0, 1.0),
                                    semantic_tags=[("class", "mug"),("color", "red")],
                                    **common_properties
                                ),
                                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                            ))

    scene_entities["target"]=obj
    scene_entities["target_pos"]=torch.tensor(pos)
    
    return scene_entities

def scene_update(sim: sim_utils.SimulationContext, scene_entities: dict, scene_count: int):
    target: RigidObject = scene_entities["target"]
    state_w = torch.zeros(7)
    
    quat_w = math.quat_from_euler_xyz(torch.tensor([0]), torch.tensor([0]), torch.tensor([scene_count * 0.79]))
    
    state_w[0:3] = scene_entities["target_pos"]
    state_w[3:7] = quat_w
    
    if scene_count == 0:
        if state_w[1] < 0.6:
            state_w[1] = state_w[1] + 0.01
        elif state_w[1] >= 0.6:
            state_w[0] = state_w[0] + 0.12
            state_w[1] = 0.0 
    
    for key in list(scene_entities.keys()):
            if (key == "target"):
                prim_utils.delete_prim(scene_entities[key].cfg.prim_path)


    scene_entities = obj_spawn(scene_entities=scene_entities, pos=state_w[0:3], rot=state_w[3:7])
    
    return scene_entities
    
    


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    scene_count = 0
    # extract entities for simplified notation
    camera: Camera = scene_entities["camera"]

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    # Camera positions, targets, orientations
    camera_positions = torch.tensor([[1.1, 0.0, 0.8]],  device = sim.device)
    camera_targets = torch.tensor([[0.0, 0.0, 0.8]], device=sim.device)

    # Set pose: There are two ways to set the pose of the camera.
    # -- Option-1: Set pose using view
    camera.set_world_poses_from_view(camera_positions, camera_targets)

    # Index of the camera to use for visualization and saving
    camera_index = args_cli.camera_id

    # Create the markers for the --draw option outside of is_running() loop
    if sim.has_gui() and args_cli.draw:
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)

    # Simulate physics
    while simulation_app.is_running():
        # Step simulation
        sim.step()
        # Update camera data
        camera.update(dt=sim.get_physics_dt())
        
        # update step count
        count += 1

        if count % 50 == 0:
        
            # Extract camera data
            if args_cli.save:
                # Save images from camera at camera_index
                # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
                # tensordict allows easy indexing of tensors in the dictionary
                single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )

                # Extract the other information
                single_cam_info = camera.data.info[camera_index]

                # Pack data back into replicator format to save them using its writer
                rep_output = {"annotators": {}}
                for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                    if info is not None:
                        rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                    else:
                        rep_output["annotators"][key] = {"render_product": {"data": data}}
                # Save images
                # Note: We need to provide On-time data for Replicator to save the images.
                rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
                rep_writer.write(rep_output)
            
            if (count > 1000) & (scene_entities["target_pos"][0] >= 0.24):
                raise RuntimeError
                    
            
            if scene_count == 8:
                scene_count = 0

            scene_entities = scene_update(sim=sim,scene_entities=scene_entities, scene_count=scene_count)
            scene_count += 1


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
    # design the scene
    scene_entities = design_scene()
    # Play simulator
    sim.reset() 
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

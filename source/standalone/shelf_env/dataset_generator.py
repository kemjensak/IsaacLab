import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ''--camera_id''."
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ''--camera_id''.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        "The viewport will always initialize with the perspective of camera 0."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import os
import random
import torch

import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep
from omni.isaac.lab.sim import SimulationContext
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg, AssetBaseCfg, AssetBase
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.sim.schemas.schemas_cfg import MassPropertiesCfg
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.utils import convert_dict_to_backend


class ENV_Cfg:
    def __init__(self):
        self.first_row = [[0.12, -0.2, 0.66], [0.12, 0.0, 0.66], [0.12, 0.2, 0.66]]
        self.second_row = [[0.0, -0.2, 0.66], [0.0, 0.0, 0.66], [0.0, 0.2, 0.66]]
        self.third_row = [[-0.12, -0.2, 0.66], [-0.12, 0.0, 0.66], [-0.12, 0.2, 0.66]]
        self.items = ["mug", "plastic_cup", "pencil_cup"]
        
    def design_scene(self):
        """Designs the scene by spawning ground plane, light, objects and meshes from usd files"""
        # Ground-plane
        cfg_ground = sim_utils.GroundPlaneCfg()
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
        
        # spawn distant light
        cfg_light_dome = sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(1.0, 1.0, 1.0),
        )
        
        cfg_light_dome.func("/World/lightDistant", cfg_light_dome, translation=(-5, 0, 10))

        # spawn a usd file of a shelf into the scene
        rack_cfg = RigidObjectCfg(
            prim_path="/World/Rack",
            spawn=sim_utils.UsdFileCfg(usd_path=f"omniverse://localhost/Library/Shelf/Arena/gorilla_rack.usd", mass_props=MassPropertiesCfg(mass=50)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            debug_vis=False,
        )
        
        rack = RigidObject(cfg=rack_cfg)

        for i, origin in enumerate(self.first_row):
            prim_utils.create_prim(f"/World/f_r{i}", "Xform", translation=origin)

        for i, origin in enumerate(self.second_row):
            prim_utils.create_prim(f"/World/s_r{i}", "Xform", translation=origin)

        for i, origin in enumerate(self.third_row):
            prim_utils.create_prim(f"/World/t_r{i}", "Xform", translation=origin)

        
        scene_entities = {}
        
        scene_entities = self.obj_spawn()

        camera = self.define_sensor()

        scene_entities["camera"] = camera
        
        return scene_entities

    def obj_spawn(self) -> dict:
        scene_entities = {}
        target_idx = np.random.randint(0, 3)
        common_properties = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
            "mass_props": sim_utils.MassPropertiesCfg(mass=1.0),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            }    

        for i, origin in enumerate(self.first_row):
            item = np.random.choice(self.items)
            
            prim_path = f"/World/f_r{i}/{item}"
            position = np.random.randn(3) * 0.02
            position[2] = 0.0 

            if item == "mug":
                obj = RigidObject(cfg=RigidObjectCfg(
                                prim_path = prim_path,
                                spawn=sim_utils.UsdFileCfg(
                                    usd_path=f"omniverse://localhost/Library/Shelf/Object/SM_Cup_empty.usd",
                                    scale=(1.0, 1.0, 1.0),
                                    semantic_tags=[("class", item)],
                                    **common_properties
                                ),
                                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                            ))
                
            elif item == "plastic_cup":
                obj = RigidObject(cfg=RigidObjectCfg(
                                prim_path = prim_path,
                                spawn=sim_utils.UsdFileCfg(
                                    usd_path=f"omniverse://localhost/Library/Shelf/Object/PlasticCup.usd",
                                    scale=(1.0, 1.0, 1.0),
                                    semantic_tags=[("class", item)],
                                    **common_properties                 
                                ),
                                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                            ))
                
            elif item == "pencil_cup":
                obj = RigidObject(cfg=RigidObjectCfg(
                                prim_path = prim_path,
                                spawn=sim_utils.UsdFileCfg(
                                    usd_path=f"omniverse://localhost/Library/Shelf/Object/PencilCup.usd",
                                    scale=(1.0, 1.0, 1.0),
                                    semantic_tags=[("class", item)],
                                    **common_properties                  
                                ),
                                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                            ))
                
            scene_entities[f"f_r{i}"]=obj

        for i, origin in enumerate(self.second_row):
            item = np.random.choice(self.items)
            prim_path = f"/World/s_r{i}/{item}"
            position = np.random.randn(3) * 0.02
            position[2] = 0.0 

            if item == "mug":
                obj = RigidObject(cfg=RigidObjectCfg(
                                prim_path=prim_path,
                                spawn=sim_utils.UsdFileCfg(
                                    usd_path=f"omniverse://localhost/Library/Shelf/Object/SM_Cup_empty.usd",
                                    scale=(1.0, 1.0, 1.0),
                                    semantic_tags=[("class", item)],
                                    **common_properties
                                ),
                                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                            ))
                
            elif item == "plastic_cup":
                obj = RigidObject(cfg=RigidObjectCfg(
                                prim_path=prim_path,
                                spawn=sim_utils.UsdFileCfg(
                                    usd_path=f"omniverse://localhost/Library/Shelf/Object/PlasticCup.usd",
                                    scale=(1.0, 1.0, 1.0),
                                    semantic_tags=[("class", item)],
                                    **common_properties                 
                                ),
                                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                            ))
            
            elif item == "pencil_cup":
                obj = RigidObject(cfg=RigidObjectCfg(
                                prim_path=prim_path,
                                spawn=sim_utils.UsdFileCfg(
                                    usd_path=f"omniverse://localhost/Library/Shelf/Object/PencilCup.usd",
                                    scale=(1.0, 1.0, 1.0),
                                    semantic_tags=[("class", item)],
                                    **common_properties                  
                                ),
                                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                            ))
                
            scene_entities[f"s_r{i}"]=obj

        for i, origin in enumerate(self.third_row):
            item = np.random.choice(self.items)
            
            if i ==  target_idx:
                prim_path = f"/World/t_r{i}/target"
                common_properties["visual_material"] = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.5)
            else:
                prim_path = f"/World/t_r{i}/{item}"

            position = np.random.randn(3) * 0.02
            position[2] = 0.0 

            if item == "mug":
                obj = RigidObject(cfg=RigidObjectCfg(
                                prim_path = prim_path,
                                spawn=sim_utils.UsdFileCfg(
                                    usd_path=f"omniverse://localhost/Library/Shelf/Object/SM_Cup_empty.usd",
                                    scale=(1.0, 1.0, 1.0),
                                    semantic_tags=[("class", item)],
                                    **common_properties
                                ),
                                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                            ))
                
            elif item == "plastic_cup":
                obj = RigidObject(cfg=RigidObjectCfg(
                                prim_path = prim_path,
                                spawn=sim_utils.UsdFileCfg(
                                    usd_path=f"omniverse://localhost/Library/Shelf/Object/PlasticCup.usd",
                                    scale=(1.0, 1.0, 1.0),
                                    semantic_tags=[("class", item)],
                                    **common_properties                 
                                ),
                                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                            ))
                
            elif item == "pencil_cup":
                obj = RigidObject(cfg=RigidObjectCfg(
                                prim_path = prim_path,
                                spawn=sim_utils.UsdFileCfg(
                                    usd_path=f"omniverse://localhost/Library/Shelf/Object/PencilCup.usd",
                                    scale=(1.0, 1.0, 1.0),
                                    semantic_tags=[("class", item)],
                                    **common_properties                  
                                ),
                                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                            ))
                
            scene_entities[f"t_r{i}"]=obj    

        return scene_entities
    
    def reset_scene(self,entities: dict[str, RigidObject]):
        """Reset the scene configuration"""

        for key in entities:
            prim_utils.delete_prim(entities[key].cfg.prim_path)
        
        entities = self.obj_spawn()

        return entities
    
    def define_sensor(self,) -> Camera:
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
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
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

        

def run_simulator(sim: sim_utils.SimulationContext, entities: dict, cfg: ENV_Cfg):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # extract entities for simplified notation
    camera: Camera = entities["camera"]

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
    )

    # Camera positons, targets, orientations
    camera_positions = torch.tensor([[1.0, 0.0, 0.68]],  device = sim.device)
    camera_targets = torch.tensor([[0.0, 0.0, 0.68]], device=sim.device)
    # These orientatiosn are in ROS-convention, and will position the cameras to view the origin
    camera_orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=sim.device)

    camera.set_world_poses_from_view(camera_positions, camera_targets)

    # Index of the camera to use for visualization and saving
    camera_index = args_cli.camera_id



    # Simulate physics
    while simulation_app.is_running():


        # perform step
        sim.step()

        # Update camera data
        camera.update(dt=sim.get_physics_dt())


        # update sim-time
        sim_time += sim_dt
        count += 1

        if count % 200 == 0:

            # Extract camera data
            if args_cli.save:
                # Save images from camera at camera_index
                # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
                # tensordict allows easy indexing of tensors in the dictionary
                single_cam_data = convert_dict_to_backend(camera.data.output[camera_index], backend="numpy")

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

            entities = cfg.reset_scene(entities)


def main():

    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    env = ENV_Cfg()
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])

    # Design scene
    scene_entities = env.design_scene()
    # scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, env)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


            

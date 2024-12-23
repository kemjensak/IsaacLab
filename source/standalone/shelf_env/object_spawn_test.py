import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by `--camera_id.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by `--camera_id.",
)
parser.add_argument(
    "--target_object",
    type=str,
    default="cup_1",
    help="Name of the target object",
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

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import numpy as np
import os
import random

import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep
import numpy as np

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.utils import convert_dict_to_backend

from scipy.spatial.transform import Rotation as R

target_row_index = 5 # 타겟 선반 위치 (1~5)

class ENV_Cfg:
    def __init__(self):
        self.target_row_index = target_row_index
        self.shelf = [
            [[0.24, 0.3, 0.66], [0.24, 0.15, 0.66], [0.24, 0.0, 0.66], [0.24, -0.15, 0.66], [0.24, -0.3, 0.66]],
            [[0.12, 0.3, 0.66], [0.12, 0.15, 0.66], [0.12, 0.0, 0.66], [0.12, -0.15, 0.66], [0.12, -0.3, 0.66]],
            [[0.0, 0.3, 0.66], [0.0, 0.15, 0.66], [0.0, 0.0, 0.66], [0.0, -0.15, 0.66], [0.0, -0.3, 0.66]],
            [[-0.12, 0.3, 0.66], [-0.12, 0.15, 0.66], [-0.12, 0.0, 0.66], [-0.12, -0.15, 0.66], [-0.12, -0.3, 0.66]],
            [[-0.24, 0.3, 0.66], [-0.24, 0.15, 0.66], [-0.24, 0.0, 0.66], [-0.24, -0.15, 0.66], [-0.24, -0.3, 0.66]]
        ]
        self.items = ["cup_1", "cup_2", "cup_3", "cup_4", "cup_5", "mug_1", "mug_2", "mug_3", "mug_4", "mug_5", "bottle_1", "bottle_2", "bottle_3", "bottle_4", "bottle_5", "can_1", "can_2", "can_3", "can_4", "can_5"]
        # 카테고리별 아이템 분류
        self.category_mapping = {
            "cup": ["cup_1", "cup_2", "cup_3", "cup_4", "cup_5"],
            "mug": ["mug_1", "mug_2", "mug_3", "mug_4", "mug_5"],
            "bottle": ["bottle_1", "bottle_2", "bottle_3", "bottle_4", "bottle_5"],
            "can": ["can_1", "can_2", "can_3", "can_4", "can_5"]
        }
        
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
            spawn=sim_utils.UsdFileCfg(usd_path=f"omniverse://localhost/Library/Shelf/Arena/test_rack.usd", mass_props=MassPropertiesCfg(mass=500)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            debug_vis=False,
        )
        rack = RigidObject(cfg=rack_cfg)

        # Create prims for each position in the shelf
        for shelf_idx, shelf_row in enumerate(self.shelf):
            for col_idx, position in enumerate(shelf_row):
                prim_utils.create_prim(f"/World/shelf_{shelf_idx}_col_{col_idx}", "Xform", translation=position)

        # Spawn objects in the scene
        scene_entities = self.obj_spawn()
        
        # Define camera sensor and add it to the scene
        camera = self.define_sensor()
        scene_entities["camera"] = camera
        
        return scene_entities

    def get_category(self, item_name):
        for category, items in self.category_mapping.items():
            if item_name in items:
                return category
        return None

    def obj_spawn(self) -> dict:
        scene_entities = {}

        # 사용자 입력 기준의 target_row_index를 배열 인덱스로 변환
        adjusted_target_row_index = self.target_row_index - 1  # 사람이 1~5로 입력한 값을 0~4로 변환

        # 타겟 위치 및 객체 설정
        self.target_position = (adjusted_target_row_index, np.random.randint(0, len(self.shelf[adjusted_target_row_index])))  # (row_idx, col_idx)
        target_object = args_cli.target_object  # argparse에서 받은 타겟 객체 이름

        # 타겟 객체에 대한 USD 파일 경로 설정
        usd_path_mapping = {
            "cup_1": "omniverse://localhost/Library/Shelf/Object/Cup_1.usd",
            "cup_2": "omniverse://localhost/Library/Shelf/Object/Cup_2.usd",
            "cup_3": "omniverse://localhost/Library/Shelf/Object/Cup_3.usd",
            "cup_4": "omniverse://localhost/Library/Shelf/Object/Cup_4.usd",
            "cup_5": "omniverse://localhost/Library/Shelf/Object/Cup_5.usd",
            "mug_1": "omniverse://localhost/Library/Shelf/Object/Mug_1.usd",
            "mug_2": "omniverse://localhost/Library/Shelf/Object/Mug_2.usd",
            "mug_3": "omniverse://localhost/Library/Shelf/Object/Mug_3.usd",
            "mug_4": "omniverse://localhost/Library/Shelf/Object/Mug_4.usd",
            "mug_5": "omniverse://localhost/Library/Shelf/Object/Mug_5.usd",
            "bottle_1": "omniverse://localhost/Library/Shelf/Object/Bottle_1.usd",
            "bottle_2": "omniverse://localhost/Library/Shelf/Object/Bottle_2.usd",
            "bottle_3": "omniverse://localhost/Library/Shelf/Object/Bottle_3.usd",
            "bottle_4": "omniverse://localhost/Library/Shelf/Object/Bottle_4.usd",
            "bottle_5": "omniverse://localhost/Library/Shelf/Object/Bottle_5.usd",
            "can_1": "omniverse://localhost/Library/Shelf/Object/Can_1.usd",
            "can_2": "omniverse://localhost/Library/Shelf/Object/Can_2.usd",
            "can_3": "omniverse://localhost/Library/Shelf/Object/Can_3.usd",
            "can_4": "omniverse://localhost/Library/Shelf/Object/Can_4.usd",
            "can_5": "omniverse://localhost/Library/Shelf/Object/Can_5.usd",
        }

        # 타겟 위치의 Prim 경로 생성
        target_row_idx, target_col_idx = self.target_position
        target_prim_path = f"/World/shelf_{target_row_idx}_col_{target_col_idx}"

        if not prim_utils.is_prim_path_valid(target_prim_path):
            prim_utils.create_prim(target_prim_path, "Xform")  # Prim 생성만 수행 (좌표계 없음)

        # 타겟 오브젝트 배치 (노이즈 추가)
        target_noise = np.random.uniform(-0.01, 0.01, size=2)  # x와 y 축에 각각 최대 1cm 노이즈 추가
        target_rotation_z = np.random.uniform(0, 360)  # Z축 회전 (0~360도)
        target_position_offset = [
            target_noise[0],  # 노이즈 값만 추가
            target_noise[1],
            0.0,  # z 좌표를 0으로 고정
        ]
        target_rotation_quaternion = R.from_euler('z', target_rotation_z, degrees=True).as_quat()  # 쿼터니언 계산
        target_rotation_quaternion = [target_rotation_quaternion[3],  # w
                                  target_rotation_quaternion[0],  # x
                                  target_rotation_quaternion[1],  # y
                                  target_rotation_quaternion[2]]  # z

        obj = RigidObject(cfg=RigidObjectCfg(
            prim_path=f"{target_prim_path}/{target_object}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path_mapping[target_object],
                scale=(1.0, 1.0, 1.0),
                semantic_tags=[("class", target_object)],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=target_position_offset, rot=target_rotation_quaternion),
        ))
        scene_entities[f"shelf_{target_row_idx}_col_{target_col_idx}"] = obj

        # 카테고리 구분
        target_category = self.get_category(target_object)
        same_category_items = self.category_mapping[target_category].copy()
        same_category_items.remove(target_object)

        # 유사한 카테고리 (0.5): 다른 카테고리 중에서 유사한 카테고리 선택
        similar_category = None
        if target_category in ["cup", "mug"]:
            similar_category = "mug" if target_category == "cup" else "cup"
        elif target_category in ["bottle", "can"]:
            similar_category = "can" if target_category == "bottle" else "bottle"
        similar_category_items = self.category_mapping[similar_category].copy()

        # 다른 카테고리 (0.1): 나머지 카테고리
        other_categories = set(self.category_mapping.keys()) - {target_category, similar_category}
        other_category_items = []
        for cat in other_categories:
            other_category_items.extend(self.category_mapping[cat])

        # 전체 위치 리스트 생성
        all_positions = [(row_idx, col_idx) for row_idx in range(len(self.shelf)) for col_idx in range(len(self.shelf[0]))]
        all_positions.remove((target_row_idx, target_col_idx))  # 타겟 위치 제외

        # 위치별로 배치할 오브젝트 리스트 생성
        placement_list = []

        # 이미 사용된 위치를 추적하기 위한 집합
        used_positions = {self.target_position}  # 타겟 위치 추가

        def place_items_with_weights(items, candidate_positions, position_weights):
            """아이템을 가중치 기반으로 배치하고 중복 발생 시 다른 유효한 자리를 재탐색."""
            while items and candidate_positions:
                # 가중치 기반으로 위치 선택
                weighted_pos = random.choices(
                    population=candidate_positions,
                    weights=position_weights,
                    k=1
                )[0]

                if weighted_pos not in used_positions:
                    # 중복되지 않은 경우 배치
                    item = items.pop(0)
                    placement_list.append((weighted_pos, item))
                    used_positions.add(weighted_pos)
                    
                    # 선택된 위치를 후보와 가중치에서 제거
                    idx = candidate_positions.index(weighted_pos)
                    candidate_positions.pop(idx)
                    position_weights.pop(idx)
                else:
                    # 중복된 경우 후보와 가중치에서 해당 위치만 제거
                    idx = candidate_positions.index(weighted_pos)
                    candidate_positions.pop(idx)
                    position_weights.pop(idx)

        # 같은 카테고리 (0.8) 배치
        same_category_positions = []
        for row_idx in range(adjusted_target_row_index - 1, -1, -1):  # 타겟보다 뒤쪽(행 번호가 작은 방향)
            for col_offset in [-1, 0, 1]:  # 타겟 열 주변의 좌(-1), 정면(0), 우(1)
                col_idx = target_col_idx + col_offset  # 열 계산
                if 0 <= col_idx < len(self.shelf[0]):  # 유효한 열인지 확인
                    same_category_positions.append((row_idx, col_idx))  # 위치 저장

        # 중심 열에 더 높은 가중치를 부여
        position_weights = [5.0 if pos[1] == target_col_idx else 1.0 for pos in same_category_positions]
        place_items_with_weights(same_category_items, same_category_positions, position_weights)


        # 유사한 카테고리 (0.5) 배치
        similar_category_positions = []
        similar_cols = [target_col_idx - 1, target_col_idx + 1]
        position_weights = []

        for col_idx in similar_cols:
            if 0 <= col_idx < len(self.shelf[0]):
                for row_idx in range(len(self.shelf)):
                    similar_category_positions.append((row_idx, col_idx))
                    position_weights.append(5.0)

                    # 좌, 우로 확장
                    adj_col_idx = col_idx + (1 if col_idx == target_col_idx - 1 else -1)
                    if 0 <= adj_col_idx < len(self.shelf[0]):
                        similar_category_positions.append((row_idx, adj_col_idx))
                        position_weights.append(1.0)

        place_items_with_weights(similar_category_items, similar_category_positions, position_weights)

        # 카테고리 0.8과 0.5에서 사용된 열 추적
        used_columns = {target_col_idx}  # 타겟 열 포함
        used_columns.update([pos[1] for pos in same_category_positions])  # 0.8에서 사용된 열 추가
        used_columns.update([pos[1] for pos in similar_category_positions])  # 0.5에서 사용된 열 추가

        # 다른 카테고리 (0.1) 배치
        other_category_positions = []
        available_columns = [col_idx for col_idx in range(len(self.shelf[0])) if col_idx not in used_columns]

        for col_idx in available_columns:  # 사용되지 않은 열에서만 선택
            for row_idx in range(len(self.shelf)):
                other_category_positions.append((row_idx, col_idx))

        position_weights = [1.0] * len(other_category_positions)  # 균등 가중치
        place_items_with_weights(other_category_items, other_category_positions, position_weights)


        # 실제 오브젝트 배치
        for (row_idx, col_idx), object_name in placement_list:
            prim_path = f"/World/shelf_{row_idx}_col_{col_idx}"

            if not prim_utils.is_prim_path_valid(prim_path):
                prim_utils.create_prim(prim_path, "Xform")  # 일반 Prim 생성

            # 오브젝트 배치 (노이즈 추가)
            noise = np.random.uniform(-0.01, 0.01, size=2)  # x와 y 축에 각각 최대 1cm 노이즈 추가
            rotation_z = np.random.uniform(0, 360)  # Z축 회전 (0~360도)
            position_offset = [
                noise[0],  # 노이즈 값만 추가
                noise[1],  
                0.0,  # z 좌표를 0으로 고정
            ]
            rotation_quaternion = R.from_euler('z', rotation_z, degrees=True).as_quat()
            rotation_quaternion = [rotation_quaternion[3],  # w
                                rotation_quaternion[0],  # x
                                rotation_quaternion[1],  # y
                                rotation_quaternion[2]]  # z

            usd_path = usd_path_mapping[object_name]

            # 오브젝트 생성
            obj = RigidObject(cfg=RigidObjectCfg(
                prim_path=f"{prim_path}/{object_name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    scale=(1.0, 1.0, 1.0),
                    semantic_tags=[("class", object_name)],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=position_offset, rot=rotation_quaternion),
            ))
            scene_entities[f"shelf_{row_idx}_col_{col_idx}"] = obj

        return scene_entities

    def reset_scene(self, entities: dict):
        """Reset the scene configuration"""

        for key in list(entities.keys()):
            if key != "camera":
                prim_utils.delete_prim(entities[key].cfg.prim_path)
                del entities[key]

        new_entities = self.obj_spawn()
        entities.update(new_entities)

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

    # Camera positions, targets, orientations
    camera_positions = torch.tensor([[1.1, 0.0, 0.8]],  device = sim.device)
    camera_targets = torch.tensor([[0.0, 0.0, 0.8]], device=sim.device)
    # These orientations are in ROS-convention, and will position the cameras to view the origin
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

            entities = cfg.reset_scene(entities)


def main():

    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    env = ENV_Cfg()
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.3], target=[0.0, 0.0, 0.8])

    # Design scene
    scene_entities = env.design_scene()
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
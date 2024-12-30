import os
import numpy as np
from PIL import Image
import cv2
import argparse

parser = argparse.ArgumentParser(description="This script generates datasets for the FCN network to assist in high-level planning for target object search.")
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--target_object",
    type=str,
    default="cup_1",
    help="Name of the target object",
)

target_id={"cup": "1",
           "mug": "2",
           "bottle": "3",
           "can": "4"}


def combine_all_images(folder_path: str, obj_type: str, sigma=5):
    """
    Combine all semantic segmentation and depth images in a folder to create a merged depth map.
    
    Parameters:
        folder_path (str): Path to the folder containing image files.
        output_folder (str): Path to save the final merged depth image.
        sigma (float): Standard deviation for Gaussian blur.
    """
    segmentation_path = os.path.join(folder_path, args_cli.target_object, obj_type,"semantic_seg_data")
    depth_path = os.path.join(folder_path, args_cli.target_object, obj_type,"dis_to_img_plane")

    segmentation_files = sorted([f for f in os.listdir(segmentation_path) if f.endswith(".png")])
    depth_files = sorted([f for f in os.listdir(depth_path) if f.endswith(".npy")])

    output_folder = os.path.join(folder_path, args_cli.target_object, obj_type, "processed_depth")

    if len(segmentation_files) != len(depth_files):
        raise ValueError("Number of semantic segmentation files and depth files do not match.")

    masked_depth_images = []

    # Process each pair of segmentation and depth images
    for seg_file, depth_file in zip(segmentation_files, depth_files):
        # Load the segmentation image
        seg_path = os.path.join(segmentation_path, seg_file)
        segmentation = np.array(Image.open(seg_path))

        # Check if the image has multiple channels (e.g., RGBA or RGB)
        if segmentation.ndim == 3:
            # Extract green channel if it is RGBA
            green_channel = segmentation[..., 1]
            cup_mask = (green_channel > 0).astype(np.float32)
        else:
            # If the segmentation image is grayscale, consider it as binary mask
            cup_mask = (segmentation > 0).astype(np.float32)

        # Load the depth image
        i_depth_path = os.path.join(depth_path, depth_file)
        depth_image = np.load(i_depth_path).squeeze()

        # Mask the depth image (set non-cup regions to 0)
        masked_depth = depth_image * cup_mask
        
        # Handle NaN values by replacing them with 0
        masked_depth = np.nan_to_num(masked_depth, nan=0)
        
        # Extract the base name without extension and adjust the file naming
        base_name = os.path.splitext(depth_file)[0]  # Extract base name without extension
        
        # Get the parts of the name (e.g., "distance_to_image_plane_1_0.npy" -> ["distance_to_image_plane", "1", "0"])
        name_parts = base_name.split('_')
        
        # Construct the new name: e.g., "depth_1_0.npy"
        new_file_name = f"depth_{name_parts[4]}.npy"
        
        # Set output path for the masked depth image
        output_path = os.path.join(output_folder, new_file_name)

        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the masked depth image with the new name
        np.save(output_path, masked_depth)
        print(f"Saved masked depth image: {output_path}")
        


def create_depth_distribution_map(folder_path: str, visualization: bool, save: bool, occlusion_threshold=0.3):
    """
    Create a depth distribution map by comparing the depth of each pixel in the target image with the corresponding pixel
    in the scene image. A smaller depth value in the scene indicates that the object is closer to the camera.
    
    Parameters:
        folder_path (str): Path to the folder containing the image files.
        occlusion_threshold (float): Threshold for occlusion detection.
    """
    scene_path = os.path.join(folder_path, args_cli.target_object, "scene","processed_depth")
    target_path = os.path.join(folder_path, args_cli.target_object, "target", "processed_depth")
    output_folder = os.path.join(folder_path, args_cli.target_object, "scene", "depth_dis_map")
    rgb_path = os.path.join(folder_path, args_cli.target_object, "scene", "rgb")

    scene_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".npy")])
    target_files = sorted([f for f in os.listdir(target_path) if f.endswith(".npy")])

    occluded_target_images = []

    for scene_file in scene_files:
        occluded_target_images.clear()
        scene_file_path = os.path.join(scene_path, scene_file)
        # Load the scene depth image as a NumPy array
        scene = np.load(scene_file_path)
        
        
        if visualization: 
            # Extract the corresponding RGB image filename based on the depth image
            base_name = scene_file.replace("depth_", "").replace(".npy", "")  # Extract the base name like "1", "2", etc.
            rgb_file_name = f"rgb_{base_name}_0.png"  # Generate the corresponding RGB filename
            rgb_scene_path = os.path.join(rgb_path, rgb_file_name)  # Full path for RGB image
            
            # Load the RGB image
            rgb_scene = cv2.imread(rgb_scene_path)

        for target_file in target_files:
            target_file_path = os.path.join(target_path, target_file)
            target = np.load(target_file_path)
            
            if scene.shape != target.shape:
                raise ValueError("Scene and target depth images must have the same dimensions.")
                        
            # Create a mask for the non-zero values (target object area)
            object_mask = target != 0  # mask where target has non-zero values (the object part)

            # Apply the mask to the scene and target images to compare only the object part
            scene_object_part = scene[object_mask]  # only the part of scene where target is non-zero
            target_object_part = target[object_mask]  # only the part of target where the object is

            # **Exclude pixels where the scene value is 0** (no object in the scene at that pixel)
            valid_mask = (scene_object_part != 0)  # Only consider non-zero scene parts (where the object is in the scene)

            # Apply this mask to exclude invalid scene pixels (scene = 0)
            scene_object_part = scene_object_part[valid_mask]
            target_object_part = target_object_part[valid_mask]

            # Compare the scene and target only where the target is non-zero (i.e., object part)
            occluded_mask = scene_object_part < target_object_part            
            
            # Check if at least occlusion_threshold (80%) of object part is occluded
            occlusion_ratio = np.sum(occluded_mask) / np.sum(target[object_mask])  # Calculate the ratio of occluded pixels

            if occlusion_ratio >= occlusion_threshold:  # If 80% or more of the object part is occluded
                occluded_target_images.append(target)
            
                
        if not occluded_target_images:
            raise ValueError("No target images with full occlusion were found.")
        
        
        # Stack all occluded target images into a single array (e.g., stack them along a new axis)
        merged_target_images = np.stack(occluded_target_images, axis=0)
        
        # Sum up all the images
        summed_target_images = np.sum(merged_target_images, axis=0)
        
        # Normalize the summed values by the number of images to get the average depth value
        average_target_image = summed_target_images / len(occluded_target_images)
        
        # Map pixels based on their occurrence
        # Pixels that are 0 in all target images remain 0
        # Pixels that appeared once will be lighter, multiple appearances will make them darker
        result_map = np.zeros_like(scene, dtype=np.float32)
        
        
        result_map[average_target_image > 0] = average_target_image[average_target_image > 0]
        
        # Normalize result_map to 0-255 range based on its min and max values
        min_val = np.min(result_map)
        max_val = np.max(result_map)

        # Apply linear scaling to map the values to the 0-255 range
        result_map_normalized = 255 * (result_map - min_val) / (max_val - min_val)

        # Ensure the values are in the valid range of [0, 255]
        result_map_normalized = np.clip(result_map_normalized, 0, 255).astype(np.uint8)
        
        if save:
            # Extract the base name without extension and adjust the file naming
            base_name = os.path.splitext(scene_file)[0]  # Extract base name without extension
            
            # Get the parts of the name (e.g., "distance_to_image_plane_1_0.npy" -> ["distance_to_image_plane", "1", "0"])
            name_parts = base_name.split('_')
            
            # Construct the new name: e.g., "depth_1_0.npy"
            new_file_name = f"01_{name_parts[1]}.npy"
            
            # Set output path for the masked depth image
            output_path = os.path.join(output_folder, new_file_name)
            
            # Ensure the output folder exists
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            np.save(output_path, result_map_normalized)
            print(f"Saved masked depth image: {output_path}")
        
        if visualization:
            # Resize result_map to# Set output path for the masked depth image
            
            # Convert result_map to a color map for visualization (use a colormap)
            result_map_colored = cv2.applyColorMap(np.uint8(result_map_normalized * 255), cv2.COLORMAP_JET)

            # Overlay the result map on the RGB scene (you can adjust transparency as needed)
            overlayed_image = cv2.addWeighted(rgb_scene, 0.7, result_map_colored, 0.3, 0)

            # Optionally display the final overlayed image
            cv2.imshow("Final Overlayed Image", overlayed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            

    print("Depth distribution map creation complete.")
    

def process_file(depth_file, similarity_file, output_file):
    """
    단일 depth map과 similarity map을 결합하여 distribution map 생성 후 저장.

    Args:
        depth_file (str): depth map 파일 경로 (.npy).
        similarity_file (str): similarity map 파일 경로 (.png).
        output_file (str): 저장할 distribution map 파일 경로 (.png).
    """
    # depth map 로드
    depth_map = np.load(depth_file)

    # similarity map 로드
    similarity_map = cv2.imread(similarity_file, cv2.IMREAD_GRAYSCALE)

    # depth map 정규화 (0~255)
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 두 맵 결합 (50%씩 반영)
    combined_map = cv2.addWeighted(depth_map_norm, 0.5, similarity_map, 0.5, 0)
    

    # 결과 저장
    cv2.imwrite(output_file, combined_map)


def process_all_maps(folder_path: str):
    """
    모든 depth map과 similarity map을 순서에 맞게 처리하여 distribution map 생성.

    Args:
        depth_folder (str): depth map 폴더 경로.
        similarity_folder (str): similarity map 폴더 경로.
        output_folder (str): 저장할 distribution map 폴더 경로.
        num_files (int): 처리할 파일 수 (기본값: 1000).
    """
    depth_distribution_path = os.path.join(folder_path, args_cli.target_object, "scene","depth_dis_map")
    similarity_map_path = os.path.join(folder_path, args_cli.target_object, "scene", "mask")
    output_folder = os.path.join(folder_path, args_cli.target_object, "scene", "distribution_map")
    
    depth_distribution_files = sorted([f for f in os.listdir(depth_distribution_path) if f.endswith(".npy")])
    similarity_map_files = sorted([f for f in os.listdir(similarity_map_path) if f.endswith(".png")])
    
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    if len(depth_distribution_files) != len(similarity_map_files):
        raise ValueError("Number of depth distribution map files and similarity map files do not match.")

    for i in range(1, len(depth_distribution_files)+1):
        # 파일 이름 생성
        depth_file = f"{depth_distribution_path}/01_{i}.npy"
        similarity_file = f"{similarity_map_path}/mask_cup_1_frame_{i}.png"
        output_file = f"{output_folder}/01_{i}.png"

        # 파일 처리
        process_file(depth_file, similarity_file, output_file)

    print("Distribution map 생성 완료!")
    
def labeling(folder_path: str):
    base_name = args_cli.target_object
    name_parts = base_name.split('_')
    target_type = name_parts[0]
    object_id = name_parts[1]
    
    rgb_path = os.path.join(folder_path, args_cli.target_object,'scene', 'rgb')
    rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith(".png")])
    rgb_output_path = os.path.join(folder_path, args_cli.target_object, 'scene', 'labeled_rgb')
    # Ensure the output folder exists
    if not os.path.exists(rgb_output_path):
        os.makedirs(rgb_output_path)
    
    for rgb_file in rgb_files:
        rgb_scene_path = os.path.join(rgb_path, rgb_file)
        
        # Load the RGB image
        rgb_scene = cv2.imread(rgb_scene_path)
        
        rgb_name_parts = rgb_file.split('_')
        img_num = '{:05d}'.format(int(rgb_name_parts[1]))
        rgb_new_name = target_id[target_type] + '_' + object_id + '_' + img_num + ".png"
        
        output_path = os.path.join(rgb_output_path, rgb_new_name)
        cv2.imwrite(output_path, rgb_scene)
        
        
    mask_path = os.path.join(folder_path, args_cli.target_object,'scene', 'mask')
    mask_files = sorted([f for f in os.listdir(mask_path) if f.endswith(".png")])
    mask_output_path = os.path.join(folder_path, args_cli.target_object, 'scene', 'labeled_mask')
    # Ensure the output folder exists
    if not os.path.exists(mask_output_path):
        os.makedirs(mask_output_path)
    
    for mask_file in mask_files:
        mask_scene_path = os.path.join(mask_path, mask_file)
        
        # Load the RGB image
        mask_scene = cv2.imread(mask_scene_path)
        
        mask_name_parts = mask_file.split('_')
        img_num = '{:05d}'.format(int(mask_name_parts[4].split('.')[0]))
        mask_new_name = target_id[target_type] + '_' + object_id + '_' + img_num + ".png"
        
        output_path = os.path.join(mask_output_path, mask_new_name)
        cv2.imwrite(output_path, mask_scene)
        
    depth_map_path = os.path.join(folder_path, args_cli.target_object,'scene', 'depth_dis_map')
    depth_map_files = sorted([f for f in os.listdir(depth_map_path) if f.endswith(".npy")])
    depth_map_output_path = os.path.join(folder_path, args_cli.target_object, 'scene', 'labeled_depth_distribution_map')
    # Ensure the output folder exists
    if not os.path.exists(depth_map_output_path):
        os.makedirs(depth_map_output_path)
    
    for depth_map_file in depth_map_files:
        depth_map_scene_path = os.path.join(depth_map_path, depth_map_file)
        
        # Load the RGB image
        depth_map_scene = np.load(depth_map_scene_path)
        
        depth_map_name_parts = depth_map_file.split('_')
        img_num = '{:05d}'.format(int(depth_map_name_parts[1].split('.')[0]))
        depth_map_new_name = target_id[target_type] + '_' + object_id + '_' + img_num + ".npy"
        
        output_path = os.path.join(depth_map_output_path, depth_map_new_name)
        np.save(output_path, depth_map_scene)
        
    
    distribution_path = os.path.join(folder_path, args_cli.target_object,'scene', 'mask')
    distribution_files = sorted([f for f in os.listdir(mask_path) if f.endswith(".png")])
    distribution_output_path = os.path.join(folder_path, args_cli.target_object, 'scene', 'labeled_mask')
    # Ensure the output folder exists
    if not os.path.exists(distribution_output_path):
        os.makedirs(distribution_output_path)
    
    for distribution_file in distribution_files:
        distribution_scene_path = os.path.join(distribution_path, distribution_file)
        
        # Load the RGB image
        distribution_scene = cv2.imread(distribution_scene_path)
        
        distribution_name_parts = rgb_file.split('_')
        img_num = '{:05d}'.format(int(distribution_name_parts[1].split('.')[0]))
        distribution_new_name = target_id[target_type] + '_' + object_id + '_' + img_num + ".png"
        
        output_path = os.path.join(distribution_output_path, distribution_new_name)
        cv2.imwrite(output_path, distribution_scene)
    
    
    
    
    
    
    


if __name__ == "__main__":
    args_cli = parser.parse_args()
    # 이미지 폴더 경로
    folder_path = "/home/irol/IsaacLab/source/standalone/shelf_env/output/camera"
    # combine_all_images(folder_path=folder_path, obj_type="target")
    # combine_all_images(folder_path=folder_path, obj_type="scene")
    # create_depth_distribution_map(folder_path=folder_path, visualization=False, save=True)
    # process_all_maps(folder_path=folder_path)
    labeling(folder_path=folder_path)
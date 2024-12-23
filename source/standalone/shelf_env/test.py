# import matplotlib.pyplot as plt
 
# image_data = np.load('source/standalone/shelf_env/output/camera/distance_to_image_plane_1_0.npy')


# depth_image_normalized = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))



# # Plot the depth image with a grayscale colormap
# plt.imshow(depth_image_normalized, cmap='gray')
# plt.axis('off')  # Hide axis for a cleaner look
# plt.show()


# import os


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


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from scipy.ndimage import gaussian_filter
# import cv2

# def combine_all_images(folder_path, output_file="merged_depth_image.npy", sigma=5):
#     """
#     Combine all semantic segmentation and depth images in a folder to create a merged depth map.

#     Parameters:
#         folder_path (str): Path to the folder containing image files.
#         output_file (str): Path to save the final merged depth image.
#         sigma (float): Standard deviation for Gaussian blur.

#     Returns:
#         np.ndarray: The merged depth image.
#     """
#     # List all files in the folder
    
#     segmentation_path = folder_path + "/target_image/semantic"
#     depth_path = folder_path + "/target_image/depth"
#     segmentation_files = os.listdir(segmentation_path)
#     depth_files = os.listdir(depth_path)
#     # Separate semantic segmentation files and depth files
#     segmentation_file = sorted([f for f in segmentation_files if f.startswith("semantic_segmentation") and f.endswith(".png")])
#     depth_file = sorted([f for f in depth_files if f.startswith("distance_to_image_plane") and f.endswith(".npy")])

#     if len(segmentation_file) != len(depth_file):
#         raise ValueError("Number of semantic segmentation files and depth files do not match.")

#     # Initialize a list to hold masked depth images
#     masked_depth_images = []


#     # Process each pair of segmentation and depth images
#     for seg_file, depth_file in zip(segmentation_file, depth_file):
#         # Load the segmentation image
#         seg_path = os.path.join(folder_path, seg_file)
#         segmentation = np.array(Image.open(seg_path))



#         # Convert RGBA to a binary mask (assume green indicates the cup)
#         green_channel = segmentation[..., 1]  # Extract green channel
#         cup_mask = (green_channel > 0).astype(np.float32)  # Create binary mask

#         # Load the depth image
#         depth_path = os.path.join(folder_path, depth_file)
#         depth_image = np.load(depth_path).squeeze()


#         # Mask the depth image (set non-cup regions to 0)
#         masked_depth = depth_image * cup_mask



##
# MDP settings
##

    # # Stack masked depth images into a 3D array
    # masked_depth_stack = np.array(masked_depth_images)

    # # Identify pixels that have constant values across all frames and set them to 0
    # constant_mask = np.all(masked_depth_stack == masked_depth_stack[0], axis=0)
    # merged_depth = np.mean(masked_depth_stack, axis=0)
    # merged_depth[constant_mask] = 0


    # # Apply Gaussian blur for smoothing
    # blurred_merged_depth = gaussian_filter(merged_depth, sigma=sigma)


# # Example usage
# folder_to_process = "/home/irol/IsaacLab/source/standalone/shelf_env/output/camera"
# combine_all_images(folder_to_process, output_file="merged_depth_image.npy", sigma=5)


import os
import numpy as np
from PIL import Image
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
    # Define paths for semantic segmentation and depth images
    segmentation_path = os.path.join(folder_path, "cup_1", "semantic_seg_data")
    depth_path = os.path.join(folder_path, "cup_1", "dis_to_img_plane")
    
    # Get all the segmentation and depth files
    segmentation_files = sorted([f for f in os.listdir(segmentation_path) if f.endswith(".png")])
    depth_files = sorted([f for f in os.listdir(depth_path) if f.endswith(".npy")])

    if len(segmentation_files) != len(depth_files):
        raise ValueError("Number of semantic segmentation files and depth files do not match.")

    # Initialize a list to hold masked depth images
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
        
        # cv2.imshow("Image", masked_depth)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
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
        

    print("Processing complete.")
        
        
def create_depth_distribution_map(folder_path, occlusion_threshold=0.3):
    """
    Create a depth distribution map by comparing the depth of each pixel in the target image with the corresponding pixel
    in the scene image. A smaller depth value in the scene indicates that the object is closer to the camera.

    Parameters:
        scene_depth_image (np.ndarray): The depth image of the scene containing multiple objects.
        target_depth_image (np.ndarray): The depth image of the target object.
        output_path (str): The path to save the generated depth distribution map.

    Returns:
        np.ndarray: The depth distribution map where each pixel indicates if the target object is in front (lower depth) or behind (higher depth).
    """
    # Define paths for semantic segmentation and depth images
    scene_path = os.path.join(folder_path, "cup_1", "processed_depth")
    target_path = os.path.join(folder_path, "target_image", "processed_depth")
    output_folder = os.path.join(folder_path, "cup_1", "depth_dis_map")
    rgb_path = os.path.join(folder_path, "cup_1", "rgb")
    
    # Get all the segmentation and depth files
    scene_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".npy")])
    target_files = sorted([f for f in os.listdir(target_path) if f.endswith(".npy")])
    
    occluded_target_images = []

    for scene_file in scene_files:
        occluded_target_images.clear()
        scene_file_path = os.path.join(scene_path, scene_file)
        # Load the scene depth image as a NumPy array
        scene = np.load(scene_file_path)
        
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
            
            # print(scene_object_part)


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
        
        # # Extract the base name without extension and adjust the file naming
        # base_name = os.path.splitext(scene_file)[0]  # Extract base name without extension
        
        # # Get the parts of the name (e.g., "distance_to_image_plane_1_0.npy" -> ["distance_to_image_plane", "1", "0"])
        # name_parts = base_name.split('_')
        
        # # Construct the new name: e.g., "depth_1_0.npy"
        # new_file_name = f"01_{name_parts[1]}.npy"
        
        # # Set output path for the masked depth image
        # output_path = os.path.join(output_folder, new_file_name)
        
        # # Ensure the output folder exists
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)
        
        # np.save(output_path, result_map_normalized)
        # print(f"Saved masked depth image: {output_path}")
        
        
        # Resize result_map to match RGB scene size
        result_map_resized = cv2.resize(result_map_normalized, (rgb_scene.shape[1], rgb_scene.shape[0]))

        # Convert result_map to a color map for visualization (use a colormap)
        result_map_colored = cv2.applyColorMap(np.uint8(result_map_resized * 255), cv2.COLORMAP_JET)

        # Overlay the result map on the RGB scene (you can adjust the transparency here)
        overlayed_image = cv2.addWeighted(rgb_scene, 0.2, result_map_colored, 0.8, 0)
        
        
        # # print(scene_file)
        
        cv2.imshow("Image", overlayed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
            
def check_and_handle_nan(folder_path,  replace_nan_with=None):
    
    depth_path = os.path.join(folder_path, "scene_image", "processed_depth")
    
    depth_files = sorted([f for f in os.listdir(depth_path) if f.endswith(".npy")])
    
    
    for file in depth_files:
        target_file_path = os.path.join(depth_path, file)
        target = np.load(target_file_path)
    
        nan_mask = np.isnan(target)
        
        # If there are NaN values, print their locations
        if np.any(nan_mask):
            print("NaN values are present in the image.")
            # Optionally print the indices and values of the NaNs
            indices_of_nan = np.where(nan_mask)
            print("Indices of NaN values:")
            for i, j in zip(*indices_of_nan):
                print(f"Index: ({i}, {j}), Value: {target[i, j]}")
        else:
            print("No NaN values in the image.")
            
        raise RuntimeError
    
# Global variable to store the coordinates where the mouse is clicked
pixel_value = None

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function to display pixel value when clicked on the image.
    """
    global pixel_value
    if event == cv2.EVENT_LBUTTONDOWN:  # When left button is clicked
        # Get the pixel value at the (x, y) position
        pixel_value = param[y, x]
        print(f"Pixel Value at ({x}, {y}): {pixel_value}")  # Print pixel value at the clicked point

        # Optionally, display the pixel value on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(param, f"Value: {pixel_value}", (x+10, y-10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

def display_image(folder_path):
    """
    Function to load depth image and display pixel value when mouse is clicked.
    """
    depth_path = os.path.join(folder_path, "cup_1", "dis_to_img_plane")
    depth_files = sorted([f for f in os.listdir(depth_path) if f.endswith(".npy")])
    
    for file in depth_files:
        target_file_path = os.path.join(depth_path, file)
        target = np.load(target_file_path)
        
        # Display the image and set the mouse callback
        cv2.imshow("Image", target)
        cv2.setMouseCallback("Image", mouse_callback, target)  # Pass the image as the parameter to the callback

        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 폴더 경로 설정
DEPTH_FOLDER = "/home/irol/IsaacLab/source/standalone/shelf_env/output/camera/cup_1/depth_dis_map"
SIMILARITY_FOLDER = "/home/irol/IsaacLab/source/standalone/shelf_env/output/camera/cup_1/mask"
OUTPUT_FOLDER = "/home/irol/IsaacLab/source/standalone/shelf_env/output/camera/cup_1/final_distribution"


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


def process_all_maps(depth_folder, similarity_folder, output_folder, num_files=1000):
    """
    모든 depth map과 similarity map을 순서에 맞게 처리하여 distribution map 생성.

    Args:
        depth_folder (str): depth map 폴더 경로.
        similarity_folder (str): similarity map 폴더 경로.
        output_folder (str): 저장할 distribution map 폴더 경로.
        num_files (int): 처리할 파일 수 (기본값: 1000).
    """
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    for i in range(1, num_files + 1):
        # 파일 이름 생성
        depth_file = f"{depth_folder}/01_{i}.npy"
        similarity_file = f"{similarity_folder}/mask_cup_1_frame_{i}.png"
        output_file = f"{output_folder}/01_{i}.png"

        # 파일 처리
        process_file(depth_file, similarity_file, output_file)

    print("Distribution map 생성 완료!")


# 함수 호출
# process_all_maps(DEPTH_FOLDER, SIMILARITY_FOLDER, OUTPUT_FOLDER)




# 이미지 폴더 경로
folder_path = "/home/irol/IsaacLab/source/standalone/shelf_env/output/camera"
# semantic_folder = "/home/irol/IsaacLab/source/standalone/shelf_env/output/camera/target_image/semantic"

# 출력 폴더
output_folder = "/home/irol/IsaacLab/source/standalone/shelf_env/output/camera/cup_1/processed_depth"

# combine_all_images(folder_path=folder_path)

# create_depth_distribution_map(folder_path=folder_path)

display_image(folder_path=folder_path)

# check_and_handle_nan(folder_path=folder_path)
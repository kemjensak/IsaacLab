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


    # Process each pair of segmentation and depth images
    for seg_file, depth_file in zip(segmentation_files, depth_files):
        # Load the segmentation image
        seg_path = os.path.join(folder_path, seg_file)
        segmentation = np.array(Image.open(seg_path))



        # Convert RGBA to a binary mask (assume green indicates the cup)
        green_channel = segmentation[..., 1]  # Extract green channel
        cup_mask = (green_channel > 0).astype(np.float32)  # Create binary mask

        # Load the depth image
        depth_path = os.path.join(folder_path, depth_file)
        depth_image = np.load(depth_path).squeeze()


        # Mask the depth image (set non-cup regions to 0)
        masked_depth = depth_image * cup_mask
        masked_depth_images.append(masked_depth)


##
# MDP settings
##

    # Stack masked depth images into a 3D array
    masked_depth_stack = np.array(masked_depth_images)

    # Identify pixels that have constant values across all frames and set them to 0
    constant_mask = np.all(masked_depth_stack == masked_depth_stack[0], axis=0)
    merged_depth = np.mean(masked_depth_stack, axis=0)
    merged_depth[constant_mask] = 0


    # Apply Gaussian blur for smoothing
    blurred_merged_depth = gaussian_filter(merged_depth, sigma=sigma)


# Example usage
folder_to_process = "/home/haneul/IsaacLab/source/standalone/shelf_env/output/camera"
combine_all_images(folder_to_process, output_file="merged_depth_image.npy", sigma=5)


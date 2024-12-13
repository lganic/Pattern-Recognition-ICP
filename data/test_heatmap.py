import os
import numpy as np
import cv2
import random
from glob import glob
import matplotlib.pyplot as plt
import json

# Configuration
TEST_DIR = 'onlybody'  # Replace with your actual test directory path
HEATMAPS_SUBDIR = 'heatmaps'             # Subdirectory where heatmaps are stored
JOINT_ALIAS = 'shoulder'                  # Specify the joint alias you want to visualize

# Define the PATH_ALIAS mapping (same as previous script)
PATH_ALIAS = {
    'l_shoulder': 'shoulder',
    'r_shoulder': 'shoulder',
    'l_elbow': 'elbow',
    'r_elbow': 'elbow',
    'l_wrist': 'wrist',
    'r_wrist': 'wrist',
    'l_hip': 'hip',
    'r_hip': 'hip',
    'l_knee': 'knee',
    'r_knee': 'knee',
    'l_ankle': 'ankle',
    'r_ankle': 'ankle'
}

def get_alias(keypoint_name):
    """
    Map a keypoint to its alias using PATH_ALIAS.
    If the keypoint is not in PATH_ALIAS, return its original name.
    """
    return PATH_ALIAS.get(keypoint_name, keypoint_name)

def load_aliases(test_dir, heatmaps_subdir):
    """
    Load all aliases from the heatmaps by inspecting one heatmap file.
    Assumes that all heatmaps have the same channel order.
    """
    heatmap_files = glob(os.path.join(heatmaps_subdir, '*.npy'))
    if not heatmap_files:
        raise ValueError(f"No heatmap files found in {os.path.join(heatmaps_subdir)}")
    
    # Load the first heatmap to determine aliases
    first_heatmap = heatmap_files[0]
    # Extract UUID from filename
    uuid = os.path.splitext(os.path.basename(first_heatmap))[0].replace('_heatmap', '')
    
    # Locate the corresponding JSON to get the aliases
    json_file = os.path.join(test_dir, f"{uuid}.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file {json_file} not found for heatmap {first_heatmap}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Collect aliases
    aliases = set(PATH_ALIAS.values())
    for key in data.keys():
        if key == 'file_name':
            continue
        alias = get_alias(key)
        aliases.add(alias)
    
    aliases = sorted(list(aliases))
    return aliases

def overlay_heatmap_on_image(image, heatmap, colormap=cv2.COLORMAP_JET, alpha=0):
    """
    Overlay a single-channel heatmap on an image using a colormap and alpha blending.
    
    Parameters:
    - image: Original image as a NumPy array (BGR).
    - heatmap: Single-channel heatmap as a NumPy array.
    - colormap: OpenCV colormap to apply.
    - alpha: Transparency factor for the heatmap.
    
    Returns:
    - Composite image with heatmap overlay.
    """
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    # Convert image to BGR if it's in RGB
    if image.shape[2] == 3:
        image_bgr = image.copy()
    else:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Blend the heatmap with the image
    composite = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    return composite

def main():
    # Load all aliases
    try:
        aliases = load_aliases(TEST_DIR, HEATMAPS_SUBDIR)
    except Exception as e:
        print(f"Error loading aliases: {e}")
        return
    
    print(f"Available aliases (heatmap channels): {aliases}")
    
    if JOINT_ALIAS not in aliases:
        print(f"Error: Specified joint alias '{JOINT_ALIAS}' not found in aliases.")
        print(f"Available aliases: {aliases}")
        return
    
    # Get the index of the specified joint alias
    joint_idx = aliases.index(JOINT_ALIAS)
    
    # Get all heatmap files
    heatmap_files = glob(os.path.join(HEATMAPS_SUBDIR, '*.npy'))
    if not heatmap_files:
        print(f"No heatmap files found in {os.path.join(HEATMAPS_SUBDIR)}")
        return
    
    # Select a random heatmap file
    selected_heatmap_file = random.choice(heatmap_files)
    print(f"Selected heatmap file: {selected_heatmap_file}")
    
    # Extract UUID from heatmap filename
    base_filename = os.path.basename(selected_heatmap_file)
    uuid = base_filename.replace('_heatmap.npy', '')
    
    # Locate the corresponding image file (assuming .png extension)
    with open(os.path.join(TEST_DIR, f'{uuid}.json'), 'r') as f:
        data = json.load(f)

        image_file = os.path.join(TEST_DIR, data['file_name'])
    if not os.path.exists(image_file):
        print(f"Image file {image_file} does not exist. Skipping.")
        return
    
    # Load the image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Failed to load image {image_file}. Skipping.")
        return
    
    # Load the heatmap
    heatmaps = np.load(selected_heatmap_file)  # Shape: (height, width, num_aliases)
    
    heatmaps = cv2.resize(
        heatmaps, 
        (256, 192), 
        interpolation=cv2.INTER_AREA
    )

    if joint_idx >= heatmaps.shape[2]:
        print(f"Joint index {joint_idx} out of bounds for heatmaps with shape {heatmaps.shape}")
        return
    
    # Extract the heatmap for the specified joint
    joint_heatmap = heatmaps[:, :, joint_idx]
    
    # Normalize the heatmap to range [0, 255]
    joint_heatmap_norm = cv2.normalize(joint_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    joint_heatmap_uint8 = joint_heatmap_norm.astype(np.uint8)
    print(joint_heatmap_uint8)
    
    # Overlay the heatmap on the image
    composite_image = overlay_heatmap_on_image(image, joint_heatmap_uint8, alpha=1)
    
    # Convert BGR to RGB for displaying with matplotlib
    composite_image_rgb = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)
    original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the original image and the composite image
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(composite_image_rgb)
    plt.title(f'Heatmap Overlay: {JOINT_ALIAS}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Optionally, save the composite image to disk
    output_filename = os.path.join(HEATMAPS_SUBDIR, f"{uuid}_heatmap_{JOINT_ALIAS}.png")
    cv2.imwrite(output_filename, composite_image)
    print(f"Composite image saved to {output_filename}")

if __name__ == "__main__":
    main()

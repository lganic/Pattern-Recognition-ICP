import os
import json
import numpy as np
import cv2
from glob import glob
import shutil

# Define the PATH_ALIAS mapping
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
    'r_ankle': 'ankle',
    'nose': 'nose'
}

def get_alias(keypoint_name):
    """
    Map a keypoint to its alias using PATH_ALIAS.
    If the keypoint is not in PATH_ALIAS, return its original name.
    """
    return PATH_ALIAS.get(keypoint_name, keypoint_name)

def generate_heatmaps(image_shape, keypoints, aliases, sigma=10):
    """
    Generate a multi-channel heatmap for all aliases.

    Parameters:
    - image_shape: Tuple representing the shape of the image (height, width, channels).
    - keypoints: Dictionary of keypoints with their details.
    - aliases: List of unique aliases.
    - sigma: Standard deviation for the Gaussian.

    Returns:
    - heatmaps: NumPy array of shape (height, width, num_aliases).
    """
    height, width = image_shape[:2]

    # Create a mapping from alias to channel index for quick access

    alias_to_idx = {}

    seen_alias = []
    for alias in aliases:
        if alias not in seen_alias:
            seen_alias.append(alias)
        
        alias_to_idx[alias] = seen_alias.index(alias)

    heatmaps = np.zeros((height, width, len(seen_alias)), dtype=np.float32)

    if len(seen_alias) != 7:
        raise ValueError('Unexpected number of classes')

    for keypoint_name, value in keypoints.items():
        if keypoint_name == 'file_name':
            continue
        if not all(k in value for k in ('x', 'y', 'visible')):
            continue
        if not value['visible']:
            continue

        x, y = int(value['x']), int(value['y'])
        if not (0 <= x < width and 0 <= y < height):
            continue

        alias = get_alias(keypoint_name)
        if alias not in alias_to_idx:
            # If a new alias is found that's not in the aliases list, skip it
            print(f"Warning: Alias '{alias}' not found in aliases list. Skipping keypoint '{keypoint_name}'.")
            continue

        channel_idx = alias_to_idx[alias]

        # Generate a Gaussian heatmap for the keypoint
        # Define the range for the Gaussian
        tmp_size = sigma * 3
        ul = [int(x - tmp_size), int(y - tmp_size)]
        br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]

        # Ensure the Gaussian is within image boundaries
        if ul[0] < 0 or ul[1] < 0 or br[0] > width or br[1] > height:
            # Adjust the Gaussian to fit within the image
            size = 2 * tmp_size + 1
            x0 = tmp_size
            y0 = tmp_size
            g = np.exp(-((np.arange(size) - x0) ** 2 + (np.arange(size)[:, None] - y0) ** 2) / (2 * sigma ** 2))
            g = g / g.max()

            # Determine the bounds of the Gaussian on the image
            g_x_start = max(0, -ul[0])
            g_y_start = max(0, -ul[1])
            g_x_end = min(br[0], width) - ul[0]
            g_y_end = min(br[1], height) - ul[1]

            # Determine the bounds of the Gaussian on the canvas
            img_x_start = max(ul[0], 0)
            img_y_start = max(ul[1], 0)
            img_x_end = min(br[0], width)
            img_y_end = min(br[1], height)

            heatmaps[img_y_start:img_y_end, img_x_start:img_x_end, channel_idx] += g[g_y_start:g_y_end, g_x_start:g_x_end]
        else:
            # If the Gaussian is fully within the image boundaries
            size = 2 * tmp_size + 1
            x0 = tmp_size
            y0 = tmp_size
            g = np.exp(-((np.arange(size) - x0) ** 2 + (np.arange(size)[:, None] - y0) ** 2) / (2 * sigma ** 2))
            g = g / g.max()
            heatmaps[ul[1]:br[1], ul[0]:br[0], channel_idx] += g

    # Clip the heatmaps to a maximum of 1.0 to prevent overflow
    heatmaps = np.clip(heatmaps, 0, 1)

    return heatmaps

def main():
    # Define your data directory
    data_dir = 'onlybody'  # Replace with your actual data directory path
    json_files = glob(os.path.join(data_dir, '*.json'))
    output_dir = 'heatmaps'
    os.makedirs(output_dir, exist_ok=True)

    # Define all possible aliases (from PATH_ALIAS and additional keypoints like 'nose')
    # You can add more aliases here if your dataset includes other keypoints
    predefined_aliases = set(PATH_ALIAS.values())
    additional_aliases = set()

    # First pass: Determine additional aliases not in PATH_ALIAS
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        for key in data.keys():
            if key == 'file_name':
                continue
            alias = get_alias(key)
            if alias not in predefined_aliases:
                additional_aliases.add(alias)

    # Combine all aliases
    aliases = sorted(list(predefined_aliases.union(additional_aliases)))
    print(f"Aliases to be used as heatmap channels: {aliases}")

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract UUID from the JSON filename
        uuid = os.path.splitext(os.path.basename(json_file))[0]
        # Assume the corresponding image is a PNG with the same UUID
        image_file = os.path.join(data_dir,data['file_name'])
        if not os.path.exists(image_file):
            print(f"Warning: Image file {image_file} does not exist. Skipping.")
            continue

        # Load the image to get its dimensions
        image = cv2.imread(image_file)
        if image is None:
            print(f"Error: Failed to load image {image_file}. Skipping.")
            continue
        height, width = image.shape[:2]

        # Extract keypoints from the JSON
        keypoints = {}
        for key, value in data.items():
            if key == 'file_name':
                continue
            if all(k in value for k in ('x', 'y', 'visible', 'in_frame')):
                keypoints[key] = {
                    'x': value['x'],
                    'y': value['y'],
                    'visible': value['in_frame']
                }

        if not keypoints:
            print(f"Warning: No valid keypoints found in {json_file}. Skipping.")
            continue

        # Generate heatmaps
        heatmaps = generate_heatmaps(image.shape, keypoints, aliases, sigma=10)

        # **Rescale Heatmaps to 128 x 96**
        desired_width, desired_height = 128, 96  # (width, height)
        # OpenCV uses (width, height) for resizing
        heatmaps_resized = cv2.resize(
            heatmaps, 
            (desired_width, desired_height), 
            interpolation=cv2.INTER_AREA
        )

        # Optionally normalize heatmaps to the range [0, 255]
        heatmaps_normalized = (heatmaps_resized * 255).astype(np.uint8)

        # Save the heatmaps as a NumPy array
        heatmap_file = os.path.join(output_dir, f"{uuid}_heatmap.npy")
        np.save(heatmap_file, heatmaps_normalized)
        print(f"Saved heatmap for {uuid} to {heatmap_file}")

        shutil.copy(image_file, output_dir)

        # Alternatively, save each heatmap channel as a separate image
        # Uncomment the following lines if you prefer saving as images
        """
        for idx, alias in enumerate(aliases):
            single_heatmap = heatmaps_normalized[:, :, idx]
            single_heatmap_file = os.path.join(output_dir, f"{uuid}_heatmap_{alias}.png")
            cv2.imwrite(single_heatmap_file, single_heatmap)
        print(f"Saved individual heatmaps for {uuid} as separate images.")
        """

if __name__ == "__main__":
    main()

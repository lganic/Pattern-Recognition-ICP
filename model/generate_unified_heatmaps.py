import numpy as np

from pose_estimation.heatmap_class import HeatmapGenerator

model_dictionary = {
    "nose": 0,
    "r_shoulder": 2,
    "r_elbow": 3,
    "r_wrist": 4,
    "l_shoulder": 5,
    "l_elbow": 6,
    "l_wrist": 7,
    "r_hip": 8,
    "r_knee": 9,
    "r_ankle": 10,
    "l_hip": 11,
    "l_knee": 12,
    "l_ankle": 13
}

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

meta_items = ['nose', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']

def norm(ar):
    # Find the minimum and maximum values
    min_value = np.min(ar)
    max_value = np.max(ar)

    # Perform linear remapping to [0, 1]
    normalized_array = (ar - min_value) / (max_value - min_value)

    return normalized_array

def create_from_image(generator: HeatmapGenerator, image):
    heatmaps = generator.create_heatmap(image)

    output_heatmaps = {}

    for keypoint in model_dictionary:
        kp_index = model_dictionary[keypoint]

        heatmap_for_item = heatmaps[:, :, kp_index]

        belongs_to = PATH_ALIAS.get(keypoint, keypoint)

        if not belongs_to in output_heatmaps:
            output_heatmaps[belongs_to] = heatmap_for_item
        
        else:
            output_heatmaps[belongs_to] = output_heatmaps[belongs_to] + heatmap_for_item

    for keypoint in output_heatmaps:
        output_heatmaps[keypoint] = norm(output_heatmaps[keypoint])

    return output_heatmaps
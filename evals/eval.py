import numpy as np
import os
import json
import cv2

from matplotlib import pyplot as plt

from model_filter import key_filter

# Specify the limb indices and names
LIMBS = {
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
    "l_ankle": 13,
}

from model_1 import detect_joints as model_1_predict
from full_pose_recognition import stupid_pose as model_2_predict

MODELS = {
    "MediaPipe Pose": model_1_predict,
    "My Model": model_2_predict,
}


def calculate_mse(true_points, predicted_points):
    """Calculate Mean Squared Error (MSE) between true and predicted points."""
    return np.mean([(true["x"] - pred["x"])**2 + (true["y"] - pred["y"])**2
                    for true, pred in zip(true_points, predicted_points)])

def main(data_dir):
    errors = {model_name: {limb: [] for limb in LIMBS} for model_name in MODELS}

    # Loop over all JSON files
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            # Load JSON file
            json_path = os.path.join(data_dir, file_name)
            with open(json_path, "r") as f:
                annotation = json.load(f)

            # Load the associated image
            img_path = os.path.join(data_dir, annotation["file_name"])
            image = cv2.imread(img_path)

            if image is None:
                print(f"Image not found: {img_path}")
                continue

            # Get ground truth for visible/in-frame limbs
            true_points = {
                limb: annotation[limb]
                for limb in LIMBS
                if annotation[limb]["in_frame"] or annotation[limb]["visible"]
            }

            # Run all models and calculate errors
            for model_name, model_predict in MODELS.items():
                predictions = model_predict(image)

                if not predictions:
                    continue

                for limb, true_point in true_points.items():
                    pred_point = predictions[limb]
                    error = (true_point["x"] - pred_point[0])**2 + (true_point["y"] - pred_point[1])**2
                    errors[model_name][limb].append(error)

    # Calculate average MSE for each limb and model
    avg_errors = {
        model_name: {limb: np.mean(err_list) if err_list else 0 for limb, err_list in limbs.items()}
        for model_name, limbs in errors.items()
    }

    # Plot the results
    plt.figure(figsize=(12, 6))
    for model_name, limb_errors in avg_errors.items():
        plt.plot(LIMBS.keys(), [limb_errors[limb] for limb in LIMBS], label=model_name, marker='o')

    plt.title("Mean Squared Error by Model and Limb")
    plt.xlabel("Limb")
    plt.ylabel("Average MSE")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_directory = "/home/logan/icp/test_dataset"  # Replace with your directory path
    main(data_directory)

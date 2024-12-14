import cv2
import mediapipe as mp

def detect_joints(image):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    
    # Landmark names corresponding to MediaPipe's indices
    landmark_names = [
        "nose", "l_eye_inner", "l_eye", "l_eye_outer", "r_eye_inner", "r_eye", "r_eye_outer",
        "l_ear", "r_ear", "mouth_left", "mouth_right", "l_shoulder", "r_shoulder", "l_elbow",
        "r_elbow", "l_wrist", "r_wrist", "l_pinky", "r_pinky", "l_index", "r_index",
        "l_thumb", "r_thumb", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
        "l_heel", "r_heel", "l_foot_index", "r_foot_index"
    ]
    
    # # Load the image
    # image = cv2.imread(image_path)
    # if image is None:
    #     raise ValueError(f"Image at {image_path} could not be loaded.")
    
    # Convert BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = pose.process(rgb_image)
    
    # Check if keypoints are detected
    if not results.pose_landmarks:
        return {}
    
    # Extract keypoints with titles
    keypoints = {}
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        if idx < len(landmark_names):
            keypoints[landmark_names[idx]] = (
                int(landmark.x * image.shape[1]),  # Scale x to image width
                int(landmark.y * image.shape[0])   # Scale y to image height
            )
    
    return keypoints

# Example Usage
if __name__ == "__main__":
    image_path = "test.jpg"  # Replace with the actual image path

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    keypoints = detect_joints(image)
    for name, coords in keypoints.items():
        print(f"{name}: {coords}")

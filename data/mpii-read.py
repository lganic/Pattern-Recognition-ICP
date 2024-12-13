import pandas as pd
import os
# from PIL import Image, ImageDraw
from metadata_formatting import process_and_move, generate_part_data_packet
from tqdm import tqdm

df = pd.read_csv('mpii_dataset.csv')

files_location = 'mpii_human_pose_v1/images'

def generate_part(x, y):
    # Assume part is invisible (safer)
    visible = False

    in_frame = True

    if x <= 0 or y <= 0:
        # Part not in frame
        in_frame = False

    return generate_part_data_packet(x, y, visible, in_frame)

def midpoint(c1, c2):
    return ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)

def intify(c):
    return (int(c[0]), int(c[1]))

output_location = 'mpi_dataset'

# Iterate through each row and convert it to a dictionary
for index, row in tqdm(df.iterrows(), total = len(df)):
    row_dict = row.to_dict()

    image_name = row_dict['NAME']

    r_ankle = (row_dict['r ankle_X'], row_dict['r ankle_Y'])
    r_ankle_data = generate_part(*r_ankle)

    r_knee = (row_dict['r knee_X'], row_dict['r knee_Y'])
    r_knee_data = generate_part(*r_knee)

    r_hip = (row_dict['r hip_X'], row_dict['r hip_Y'])
    r_hip_data = generate_part(*r_hip)

    l_hip = (row_dict['l hip_X'], row_dict['l hip_Y'])
    l_hip_data = generate_part(*l_hip)

    l_knee = (row_dict['l knee_X'], row_dict['l knee_Y'])
    l_knee_data = generate_part(*l_knee)

    l_ankle = (row_dict['l ankle_X'], row_dict['l ankle_Y'])
    l_ankle_data = generate_part(*l_ankle)

    pelvis = (row_dict['pelvis_X'], row_dict['pelvis_Y'])
    pelvis_data = generate_part(*pelvis)

    thorax = (row_dict['thorax_X'], row_dict['thorax_Y'])
    thorax_data = generate_part(*thorax)

    upper_neck = (row_dict['upper neck_X'], row_dict['upper neck_Y'])
    upper_neck_data = generate_part(*upper_neck)

    head_top = (row_dict['head top_X'], row_dict['head top_Y'])
    head_top_data = generate_part(*head_top)

    r_wrist = (row_dict['r wrist_X'], row_dict['r wrist_Y'])
    r_wrist_data = generate_part(*r_wrist)

    r_elbow = (row_dict['r elbow_X'], row_dict['r elbow_Y'])
    r_elbow_data = generate_part(*r_elbow)

    r_shoulder = (row_dict['r shoulder_X'], row_dict['r shoulder_Y'])
    r_shoulder_data = generate_part(*r_shoulder)

    l_shoulder = (row_dict['l shoulder_X'], row_dict['l shoulder_Y'])
    l_shoulder_data = generate_part(*l_shoulder)

    l_elbow = (row_dict['l elbow_X'], row_dict['l elbow_Y'])
    l_elbow_data = generate_part(*l_elbow)

    l_wrist = (row_dict['l wrist_X'], row_dict['l wrist_Y'])
    l_wrist_data = generate_part(*l_wrist)

    # Load the image
    image_name = row_dict['NAME']
    image_path = os.path.join(files_location, image_name)

    nose_location = intify(midpoint(midpoint(l_shoulder, r_shoulder), head_top))
    nose_data = generate_part(*nose_location)

    try:
        process_and_move(image_path, output_location, nose_data, l_shoulder_data, r_shoulder_data, l_elbow_data, r_elbow_data, l_wrist_data, r_wrist_data, l_hip_data, r_hip_data, l_knee_data, r_knee_data, l_ankle_data, r_ankle_data)
    except:
        print('An error occurred')
        pass


    # image = Image.open(os.path.join(files_location, image_name))
    # draw = ImageDraw.Draw(image)

    # # Draw lines connecting the body parts
    # for part1, part2 in connections:
    #     draw.line([body_parts[part1], body_parts[part2]], fill="red", width=3)

    # # Save the resulting image
    # image.save('annotated_' + image_name)
    # image.show()





import json
import os
from metadata_formatting import process_and_move, generate_part_data_packet

year = '2017'
dataset = 'train'

with open(f'G:\\data\\annotations_trainval{year}\\annotations\\person_keypoints_{dataset}{year}.json', 'r') as f:
    json_object = json.loads(f.read())

files_location = f'G:\\data\\{dataset}{year}\\{dataset}{year}'

images = {}
for image_object in json_object['images']:
    image_id = image_object['id']

    images[image_id] = image_object['file_name']


output_location = 'G:\\dataset'

segmenetation_keys = {}

for segmentation_info in json_object['annotations']:

    image_name = images[segmentation_info['image_id']]

    parse_location = lambda k: (segmentation_info['keypoints'][3 * k], segmentation_info['keypoints'][3 * k + 1])
    parse_bools = lambda k: (segmentation_info['keypoints'][3 * k + 2] == 2, segmentation_info['keypoints'][3 * k + 2] >= 1)

    nose_location = parse_location(0)
    visible, inframe = parse_bools(0)
    nose_data = generate_part_data_packet(*nose_location, inframe, visible)

    left_eye_location = parse_location(1)
    visible, inframe = parse_bools(1)
    left_eye_data = generate_part_data_packet(*left_eye_location, inframe, visible)

    right_eye_location = parse_location(2)
    visible, inframe = parse_bools(2)
    right_eye_data = generate_part_data_packet(*right_eye_location, inframe, visible)

    left_ear_location = parse_location(3)
    visible, inframe = parse_bools(3)
    left_ear_data = generate_part_data_packet(*left_ear_location, inframe, visible)

    right_ear_location = parse_location(4)
    visible, inframe = parse_bools(4)
    right_ear_data = generate_part_data_packet(*right_ear_location, inframe, visible)

    left_shoulder_location = parse_location(5)
    visible, inframe = parse_bools(5)
    left_shoulder_data = generate_part_data_packet(*left_shoulder_location, inframe, visible)

    right_shoulder_location = parse_location(6)
    visible, inframe = parse_bools(6)
    right_shoulder_data = generate_part_data_packet(*right_shoulder_location, inframe, visible)

    left_elbow_location = parse_location(7)
    visible, inframe = parse_bools(7)
    left_elbow_data = generate_part_data_packet(*left_elbow_location, inframe, visible)

    right_elbow_location = parse_location(8)
    visible, inframe = parse_bools(8)
    right_elbow_data = generate_part_data_packet(*right_elbow_location, inframe, visible)

    left_wrist_location = parse_location(9)
    visible, inframe = parse_bools(9)
    left_wrist_data = generate_part_data_packet(*left_wrist_location, inframe, visible)

    right_wrist_location = parse_location(10)
    visible, inframe = parse_bools(10)
    right_wrist_data = generate_part_data_packet(*right_wrist_location, inframe, visible)

    left_hip_location = parse_location(11)
    visible, inframe = parse_bools(11)
    left_hip_data = generate_part_data_packet(*left_hip_location, inframe, visible)

    right_hip_location = parse_location(12)
    visible, inframe = parse_bools(12)
    right_hip_data = generate_part_data_packet(*right_hip_location, inframe, visible)

    left_knee_location = parse_location(13)
    visible, inframe = parse_bools(13)
    left_knee_data = generate_part_data_packet(*left_knee_location, inframe, visible)

    right_knee_location = parse_location(14)
    visible, inframe = parse_bools(14)
    right_knee_data = generate_part_data_packet(*right_knee_location, inframe, visible)

    left_ankle_location = parse_location(15)
    visible, inframe = parse_bools(15)
    left_ankle_data = generate_part_data_packet(*left_ankle_location, inframe, visible)

    right_ankle_location = parse_location(16)
    visible, inframe = parse_bools(16)
    right_ankle_data = generate_part_data_packet(*right_ankle_location, inframe, visible)

    image_path = os.path.join(files_location, image_name)

    try:
        process_and_move(image_path, output_location, nose_data, left_shoulder_data, right_shoulder_data, left_elbow_data, right_elbow_data, left_wrist_data, right_wrist_data, left_hip_data, right_hip_data, left_knee_data, right_knee_data, left_ankle_data, right_ankle_data)
    except:
        print('An error occurred')

    # # Extract coordinates for the body parts
    # body_parts = {
    #     'r ankle': right_ankle_location,
    #     'r knee': right_knee_location,
    #     'r hip': right_hip_location,
    #     'l hip': left_hip_location,
    #     'l knee': left_knee_location,
    #     'l ankle': left_ankle_location,
    #     'upper neck': nose_location,
    #     'r wrist': right_wrist_location,
    #     'r elbow': right_elbow_location,
    #     'r shoulder': right_shoulder_location,
    #     'l shoulder': left_shoulder_location,
    #     'l elbow': left_elbow_location,
    #     'l wrist': left_wrist_location
    # }

    # # Pairs of body parts to connect with lines
    # connections = [
    #     ('r-ankle', 'r-knee'), ('r-knee', 'r-hip'), ('r-hip', 'r-shoulder'), ('r-shoulder', 'r-elbow'), ('r-elbow', 'r-wrist'),
    #     ('l-ankle', 'l-knee'), ('l-knee', 'l-hip'), ('l-hip', 'l-shoulder'), ('l-shoulder', 'l-elbow'), ('l-elbow', 'l-wrist'),
    #     ('r-shoulder', 'l-shoulder'), ('r-hip', 'l-hip'), ('l-shoulder', 'nose'), ('r-shoulder', 'nose')]

    # ]

    # image = Image.open(os.path.join(files_location, image_name))
    # draw = ImageDraw.Draw(image)

    # # Draw lines connecting the body parts
    # for part1, part2 in connections:
    #     draw.line([body_parts[part1], body_parts[part2]], fill="red", width=3)

    # # Save the resulting image
    # image.save('annotated_' + image_name)
    # image.show()


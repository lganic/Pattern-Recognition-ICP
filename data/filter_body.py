import os
import json
from metadata_formatting import list_data, ordering
from tqdm import tqdm
from PIL import Image as image

connections = [
('r_ankle', 'r_knee'), ('r_knee', 'r_hip'), ('r_hip', 'r_shoulder'), ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'),
('l_ankle', 'l_knee'), ('l_knee', 'l_hip'), ('l_hip', 'l_shoulder'), ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
('r_shoulder', 'l_shoulder'), ('r_hip', 'l_hip'), ('l_shoulder', 'nose'), ('r_shoulder', 'nose')]

path = 'mpi_dataset'
output_location = 'onlybody'

n_removed = 0

files = os.listdir(path)

TARGET_WIDTH = 256
TARGET_HEIGHT = 192

def avg_pos(p1, p2):
    if type(p1) == tuple:
        p1 = {'x': p1[0], 'y': p1[1]}

    if type(p2) == tuple:
        p2 = {'x': p2[0], 'y': p2[1]}

    return ((p1['x'] + p2['x']) // 2, (p1['y'] + p2['y']) // 2)

def subtract(p1, p2):
    if type(p1) == tuple:
        p1 = {'x': p1[0], 'y': p1[1]}

    if type(p2) == tuple:
        p2 = {'x': p2[0], 'y': p2[1]}

    return ((p1['x'] - p2['x']), (p1['y'] - p2['y']))

def mag(p1):
    if type(p1) == tuple:
        p1 = {'x': p1[0], 'y': p1[1]}

    return (p1['x'] ** 2 + p1['y'] ** 2) ** .5

def transform(point, center, rect_l, rect_u, twidth, theight):
    offset = subtract(point, center)

    p_x = int(twidth * (offset[0] + rect_l) / (2 * rect_l))
    p_y = int(theight * (offset[1] + rect_u) / (2 * rect_u))

    return p_x, p_y

with tqdm(total=len(files)) as pbar:

    for item in files:

        if not item.endswith('json'):
            continue

        pbar.update(2)

        file_path = os.path.join(path, item)

        with open(file_path, 'r') as json_file:
            metadata = json.loads(json_file.read())
        
        if metadata['r_hip']['in_frame'] and metadata['l_hip']['in_frame'] and metadata['nose']['in_frame']:
            # data = list_data(metadata)

            square_center = avg_pos(metadata['r_hip'], metadata['l_hip'])

            square_size = 2.2 * mag(subtract(metadata['nose'], square_center))

            square_size *= (225/137) # Thanks leo 

            rect_left_offset = int((TARGET_WIDTH / TARGET_HEIGHT) * (square_size / 2))
            rect_up_offset = int(square_size / 2)

            new_data = metadata.copy()

            frame_valid = True

            for name in metadata:

                if name == 'file_name':
                    continue

                item = metadata[name]

                if item['in_frame']:
                    t_cord = transform(item, square_center, rect_left_offset, rect_up_offset, TARGET_WIDTH, TARGET_HEIGHT)

                    new_data[name]['x'] = t_cord[0]
                    new_data[name]['y'] = t_cord[1]

                    if t_cord[0] < 0 or t_cord[1] < 0 or t_cord[0] >= TARGET_WIDTH or t_cord[1] >= TARGET_HEIGHT:
                        frame_valid = False

                        break
            
            if not frame_valid:
                continue

            if square_center[0] - rect_left_offset < 0:
                continue # Out of bounds

            if square_center[1] - rect_up_offset < 0:
                continue # Out of bounds

            # Load image
            im = image.open(os.path.join(path, metadata['file_name']))

            width, height = im.size

            if square_center[0] + rect_left_offset >= width:
                continue # Out of bounds

            if square_center[1] + rect_up_offset >= height:
                continue # Out of bounds

            crop = im.crop((square_center[0] - rect_left_offset, square_center[1] - rect_up_offset, square_center[0] + rect_left_offset, square_center[1] + rect_up_offset))

            crop = crop.resize((TARGET_WIDTH, TARGET_HEIGHT))

            os.makedirs(output_location, exist_ok=True)

            crop.save(os.path.join(output_location, metadata['file_name']))

            s = json.dumps(new_data)

            json_name = metadata['file_name'].partition('.')[0] + '.json'
            f = open(os.path.join(output_location, json_name), 'w')
            f.write(s)
            f.close()


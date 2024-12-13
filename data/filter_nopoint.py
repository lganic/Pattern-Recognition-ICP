import os
import json
from metadata_formatting import list_data
from tqdm import tqdm

connections = [
('r_ankle', 'r_knee'), ('r_knee', 'r_hip'), ('r_hip', 'r_shoulder'), ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'),
('l_ankle', 'l_knee'), ('l_knee', 'l_hip'), ('l_hip', 'l_shoulder'), ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
('r_shoulder', 'l_shoulder'), ('r_hip', 'l_hip'), ('l_shoulder', 'nose'), ('r_shoulder', 'nose')]

path = 'G:\\dataset'

n_removed = 0

files = os.listdir(path)

with tqdm(total=len(files)) as pbar:
    for item in files:

        if not item.endswith('json'):
            continue

        file_path = os.path.join(path, item)

        with open(file_path, 'r') as json_file:
            metadata = json.loads(json_file.read())
        
        data = list_data(metadata)

        for item in data:
            if item['in_frame']:
                break
        else:
            os.remove(file_path)
            os.remove(os.path.join(path, metadata['file_name']))

            n_removed += 2
            pbar.set_postfix({"Removed": n_removed})

            pass

        pbar.update(2)

from pose_estimation.heatmap_class import HeatmapGenerator
from generate_unified_heatmaps import create_from_image, meta_items
from k_means_mat import decompose, dual_clustering
import numpy as np

def pose_from_image(generator: HeatmapGenerator, image, threshold = .7):
    import time

    t = time.time()
    heatmaps = create_from_image(generator, image)
    print('TIME', time.time() - t)


    locations = {}
    t = time.time()
    for item in meta_items:

        heatmap_decomposed = decompose(heatmaps[item], threshold)

        if item == 'nose':
            # Special case! We only have one of these!

            # Perform k-means on one class (average positions)
            locations['nose'] = np.mean(heatmap_decomposed, axis=0)

            continue

        cluster_centroids = dual_clustering(heatmap_decomposed)

        if cluster_centroids[0][0] < cluster_centroids[1][0]:
            # The first point is on the left. Assign it as such
            left_point = cluster_centroids[0]
            right_point = cluster_centroids[1]
        else:
            # The first point is on the right. Assign it as such
            left_point = cluster_centroids[1]
            right_point = cluster_centroids[0]

        locations[f'l_{item}'] = left_point
        locations[f'r_{item}'] = right_point


    print('TIME', time.time() - t)
    return locations

if __name__ == '__main__':
    import cv2 
    from PIL import Image, ImageDraw

    gen = HeatmapGenerator()

    imname = 'test.png'

    image = cv2.imread(imname)

    locations = pose_from_image(gen, image)
    print(locations)

    # # Pairs of body parts to connect with lines
    connections = [
        ('r_ankle', 'r_knee'), ('r_knee', 'r_hip'), ('r_hip', 'r_shoulder'), ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'),
        ('l_ankle', 'l_knee'), ('l_knee', 'l_hip'), ('l_hip', 'l_shoulder'), ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
        ('r_shoulder', 'l_shoulder'), ('r_hip', 'l_hip'), ('l_shoulder', 'nose'), ('r_shoulder', 'nose')]


    image = Image.open(imname)
    draw = ImageDraw.Draw(image)

    # Draw lines connecting the body parts
    for part1, part2 in connections:

        c1 = (locations[part1][0], locations[part1][1])
        c2 = (locations[part2][0], locations[part2][1])

        draw.line([c1, c2], fill="red", width=3)

    # Save the resulting image
    image.save('annotated.png')

    image.show()
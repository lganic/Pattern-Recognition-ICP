from sklearn.cluster import KMeans
import numpy as np

def decompose(array, threshold):
    # List the coordinates i.e. [(x1,y1), (x2,y2) ...] for all points in the 2D array > threshold

    # Get the indices of elements greater than the threshold
    coordinates = np.argwhere(array > threshold)
    
    # Convert indices to a list of tuples
    coordinates_list = [tuple(coord)[::-1] for coord in coordinates]
    
    return coordinates_list

def dual_clustering(points):
    # Perform K-Means Clustering on 2 groups of points, and return the resulting centroids.

    # Perform K-Means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(points)

    # Return the centroids
    return kmeans.cluster_centers_

if __name__ == '__main__':
    # Example usage
    points = np.array([[1, 2], [2, 3], [3, 4], [8, 9], [9, 10], [10, 11]])
    centroids = dual_clustering(points)
    print("Centroids:", centroids)
import numpy as np
from matplotlib import pyplot as plt

def generate_random_geometric_graph(num_sensor, radius, AREA_WIDTH):
    """
    Generate a random geometric graph with sensors placed in a square area.
    
    Parameters:
    - num_sensor: Number of sensors to place in the area.
    - radius: Communication radius of each sensor.
    - AREA_WIDTH: Width of the square area where sensors are placed.
    
    Returns:
    - A tuple containing the positions of the sensors and the adjacency matrix of the graph.
    """
    positions = np.random.rand(num_sensor, 2) * AREA_WIDTH
    adjacency_matrix = np.zeros((num_sensor, num_sensor), dtype=int)

    for i in range(num_sensor):
        for j in range(i + 1, num_sensor):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance <= radius:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return positions, adjacency_matrix
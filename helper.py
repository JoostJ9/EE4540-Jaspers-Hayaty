"""
This helper module provides different functions used in EE4540.ipynb.
"""

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

def min_sensors_for_radius(desired_radius, area_width, dimension):
    """
    Calculate the minimum number of sensors required to ensure connectivity
    in a random geometric graph with a given communication radius and area width.
    Using the rearranged formula: n >= 2 * log(n) / (r^d).
    This will be solved iteratively since n appears on both sides of the equation.
    Args:
        desired_radius (float): The communication radius of each sensor.
        area_width (float): The width of the square area where sensors are placed.
        dimension (int): The dimension of the space (2 for 2D, 3 for 3D, etc.).

    Prints:
        - The minimum number of sensors required.
        - The probability of connectivity with the calculated number of sensors and radius.
        
    Returns:
        int: The minimum number of sensors required.
    """
    unit_radius = desired_radius / area_width
    radius_term = 0.5*unit_radius ** dimension

    # Use a simple iterative approach
    n_guess = 2
    while True:
        n_term = np.log(n_guess) / n_guess
        if radius_term >= n_term:
            break
        n_guess += 1

    print(f"Minimum number of sensors for radius {desired_radius} m: {n_guess}")
    probability_of_connectivity = (1 - 1 / n_guess**2) * 100  # Simplified probability of connectivity
    print(f"Probability of connectivity with {n_guess} sensors and radius {desired_radius} m: {probability_of_connectivity:.4f}%")    
    return n_guess

def min_radius_for_sensors(num_sensors, dimension, size=1):
    """
    Calculate the minimum radius required for connectivity with high probability.
    Args:
        num_sensors (int): Number of sensors.
        dimension (int): Dimension of the space (2 for 2D, 3 for 3D, etc.).
        size (float): Size of the area (default is 1 for a unit square).
    Prints:
        The minimum required radius for connectivity with high probability.
    Returns:    
        float: The minimum required radius for connectivity.
    """
    unit_cube_radius = np.power(2 * np.log(num_sensors) / num_sensors, 1 / dimension)
    required_radius = unit_cube_radius * size
    print(f"Minimum required radius for connectivity with high probability (n={num_sensors}, area={size}x{size}): {required_radius:.2f} m")
    return required_radius

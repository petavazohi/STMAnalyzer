import numpy as np
from typing import Tuple
import distinctipy

def generate_colors_with_yellow_last(n_spectra):
    # Generate initial n_spectra colors
    colors = distinctipy.get_colors(n_spectra)
    
    # Check if yellow is in the list
    if (1.0, 1.0, 0.0) in colors:
        # Remove yellow from its current position
        colors.remove((1.0, 1.0, 0.0))
        # Append yellow as the last color
        colors.append((1.0, 1.0, 0.0))
    return colors

def adjust_dimensions2d(hit_histogram, dimensions: Tuple = None):
    if hit_histogram.ndim == 2: 
        return hit_histogram
    elif dimensions is None:
        print('Dimensions not provided. Guessing ...')
        n = np.sqrt(max(hit_histogram.shape))
        print(hit_histogram.shape)
        if n.is_integer():
            n = int(n)
            dimensions= (n, n)
        else:
            raise ValueError(
                f"Hit histogram is not squarable. Length: {len(hit_histogram)}. "
                f"Expected a perfect square."
            )
        print(f"SOM dimensions selected as {n}x{n}")
    hit_histogram = np.reshape(hit_histogram, (dimensions[0], dimensions[1]))
    return hit_histogram

def get_neighbors(coordinate: list, dimension: tuple, grid_type:str = 'hex'):
    max_x, max_y = dimension
    neighbors = []
    if grid_type == 'hex':
        q, r = coordinate  # Current position
        # Directions based on row parity
        directions_even = [
            [1, 0],  # east
            [0, 1],  # north-east
            [-1, 1], # north-west
            [-1, 0], # west
            [-1, -1], # south-west
            [0, -1],  # south-east
        ]

        directions_odd = [
            [1, 0],  # east
            [1, 1],  # north-east
            [0, 1],  # north-west
            [-1, 0], # west
            [0, -1], # south-west
            [1, -1], # south-east
        ]
        
        # Use row parity to select directions
        directions = directions_even if r % 2 == 0 else directions_odd

        for dq, dr in directions:
            neighbor = [q + dq, r + dr]
            if 0 <= neighbor[0] < max_x and 0 <= neighbor[1] < max_y:
                neighbors.append(neighbor)
    elif grid_type == 'rect':
        x, y = coordinate
        # Directions for a rectangular grid
        directions = [
            [0, 1],   # north
            [1, 0],   # east
            [0, -1],  # south
            [-1, 0],  # west
        ]
        for dx, dy in directions:
            neighbor = [x + dx, y + dy]
            if 0 <= neighbor[0] < max_x and 0 <= neighbor[1] < max_y:
                neighbors.append(neighbor)
    else:
        raise ValueError("Unsupported grid type")
    return neighbors

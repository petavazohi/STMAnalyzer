import numpy as np
from typing import Tuple

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
    hit_histogram = np.reshape(hit_histogram, dimensions)
    return hit_histogram

def get_neighbors(coordinate: list, dimension: tuple, grid_type:str = 'hex'):
    max_x, max_y = dimension
    neighbors = []
    if grid_type == 'hex':
        q, r = coordinate
        # Axial coordinate system for hex grid neighbors

        directions = [
            [1, 0],  # east
            [0, 1],  # southeast
            [-1, 1], # southwest
            [-1, 0], # west
            [0, -1], # northwest
            [1, -1]  # northeast
        ]
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

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



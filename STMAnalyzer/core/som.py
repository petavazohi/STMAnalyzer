from ..utils.arrays import adjust_dimensions2d
import numpy as np
from ..core.scan import STMScan


class SelfOrganizingMap:
    def __init__(self, hit_histogram: np.ndarray, weights: np.ndarray, cluster_index: np.ndarray, topology: str, stm_scan: STMScan) -> None:
        if hit_histogram.ndim == 1:
            hit_histogram = adjust_dimensions2d(hit_histogram)
        dimensions = hit_histogram.shape
        if som_weights is not None and som_weights.ndim != 3:
            som_weights = som_weights.reshape(dimensions+(-1,))
        if cluster_index.ndim != 2:
            cluster_index = cluster_index.reshape(stm_scan.nx, stm_scan.ny)
        self.hit_histogram = hit_histogram
        self.weights = weights
        self.cluster_index = cluster_index
        self.topology = topology

    
    
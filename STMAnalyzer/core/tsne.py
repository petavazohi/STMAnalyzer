from sklearn.manifold import TSNE
import matplotlib as mpl
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pylab as plt
from typing import Union, List, Dict, Tuple


class T_SNE:
    def __init__(self,
                 data,
                 n_components=2,
                 random_state=23,
                 perplexity=4,
                 max_iter=1000,
                 method='exact',
                 **kwargs):
        self.data = data
        self.tsne = TSNE(n_components=n_components,
                         random_state=random_state,
                         perplexity=perplexity,
                         max_iter=max_iter,
                         method=method,
                         **kwargs)
        self.data_embedded = self.tsne.fit_transform(self.data)
        self.distances = pdist(self.data_embedded)
        self.distance_matrix = squareform(self.distances)
        
    def plot_embeding(self,
                      ax: plt.Axes = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        trans_data = self.data_embedded.T
        ax.scatter(trans_data[0], trans_data[1])
        ax.axis('off')
        return ax
        
        
    def plot_embedded_weights(self,
                       ax: plt.Axes = None,
                       patch_shape: str = "Circle",
                       n_polygon_vertices: int = None,
                       linestyle: str = "-",
                       linewidth: float = None,
                       hatch: str = None,
                       alpha: float = 1.0) -> plt.Axes:
        global_max = np.max(self.data)
        global_min = np.min(self.data)
        global_range = global_max - global_min
        patches = []
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()
        radius = np.min(self.distances)/2
        for i, datum in enumerate(self.data_embedded):
            x, y = datum
            weights = self.data[i]
            if patch_shape.lower() == 'circle':
                patch = mpl.patches.Circle((x, y),
                                           radius=radius,
                                           facecolor='none',
                                           edgecolor='black')
            else:
                orientation = np.pi / 4 if n_polygon_vertices == 4 else 0
                patch = mpl.patches.RegularPolygon((x, y),
                                                numVertices=self.n_polygon_vertices,
                                                radius=radius)
            patches.append(patch)
            ax.add_patch(patch)
            half_box_size = (radius / np.sqrt(2)) if n_polygon_vertices == 4 else radius*0.92
            if global_range > 0:
                weights = (weights - global_min) / global_range * (half_box_size) - half_box_size/1.5
            x_spectrum = np.linspace(-half_box_size, half_box_size, len(weights))
            ax.plot(x + x_spectrum, y + weights, color='black')
        ax.set_aspect('equal')
        ax.axis('off')
        return patches, ax

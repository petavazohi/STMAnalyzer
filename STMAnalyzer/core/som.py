import numpy as np
from typing import Tuple, Dict, Union, List, Any
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import distance # import distance.pdist, distance.squareform, distance.euclidean
import warnings
import distinctipy
from .. import io
from .. import utils

# import STMAnalyzer.io as io
# import STMAnalyzer.utils as utils

class Node:
    def __init__(self, weight: np.ndarray, grid_index: Union[Tuple, List], topology: str = "hexagonal") -> None:
        """
        Initialize a Node in the Self-Organizing Map.

        Parameters
        ----------
        weight : np.ndarray
            The weight vector associated with this node.
        grid_index : Union[Tuple, List]
            The index of this node in the SOM grid.
        topology : str
            The topology of the SOM grid (e.g., "hexagonal", "rectangular"). Default is "hexagonal".
        """
        self.weight = weight
        self.grid_index = grid_index
        self.topology = topology
        self.n_polygon_vertices = self._get_polygon_vertices()
        if self.topology == "hexagonal":
            self.separation_factor = 1.8
        elif self.topology == "rectangular":
            self.separation_factor = 1.93
        self.radius = self.separation_factor / (4 * np.cos(np.pi / self.n_polygon_vertices)) if self.n_polygon_vertices else None
        self.color: np.ndarray = None

    def _get_polygon_vertices(self) -> Union[int, None]:
        """
        Determine the number of polygon vertices based on topology.

        Returns
        -------
        int or None
            The number of vertices for the polygon, or None if not applicable.
        """
        if self.topology == "hexagonal":
            return 6
        elif self.topology == "rectangular":
            return 4
        elif self.topology == "triangular":
            return 3
        else:
            return None

    def get_position(self) -> Tuple:
        """
        Calculate the position of the node in the SOM grid.

        Returns
        -------
        Tuple
            The (x, y) position of the node in the grid.
        """
        x, y = self.grid_index
        if self.topology == "hexagonal":
            y *= self.separation_factor/2
            if self.grid_index[1] % 2 != 0:
                x += 0.5
        return x, y

    def create_patch(self, radius: float = None, facecolor: str = "none", edgecolor: str = "black", orientation: float = None,
                     linestyle: str = None, linewidth: float = None, hatch: str = None, alpha: float = 1.0, shape: str = "polygon") -> mpl.patches.Patch:
        """
        Create a patch for the node with customizable properties. Supports polygon and circle shapes.

        Parameters
        ----------
        radius : float, optional
            The radius of the polygon or circle. If None, the default radius is used.
        facecolor : str, optional
            The face color of the patch.
        edgecolor : str, optional
            The edge color of the patch.
        orientation : float, optional
            The orientation of the polygon in radians. Default is based on topology. Ignored for circles.
        linestyle : str, optional
            The line style of the patch's edges.
        linewidth : float, optional
            The line width of the patch's edges.
        hatch : str, optional
            The hatch pattern for the patch. Supported patterns include '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'.
        alpha : float, optional
            The transparency level of the patch. Default is 1.0 (opaque).
        shape : str, optional
            The shape of the patch. Supported values are "polygon" (default) and "circle".

        Returns
        -------
        mpl.patches.Patch
            The created patch.
        """
        radius = radius if radius is not None else self.radius
        scale = 0.92 if self.topology == "hexagonal" else 1/np.sqrt(2)
        x, y = self.get_position()
        if shape == "circle":
            patch = mpl.patches.Circle((x, y),
                                    radius=radius*scale,
                                    edgecolor=edgecolor,
                                    facecolor=facecolor,
                                    alpha=alpha)
        else:
            if orientation is None:
                orientation = np.pi / 4 if self.topology == "rectangular" else 0
            patch = mpl.patches.RegularPolygon((x, y),
                                            numVertices=self.n_polygon_vertices,
                                            radius=radius,
                                            orientation=orientation,
                                            edgecolor=edgecolor,
                                            facecolor=facecolor,
                                            alpha=alpha)
        if linestyle:
            patch.set_linestyle(linestyle)
        if linewidth:
            patch.set_linewidth(linewidth)
        if hatch:
            patch.set_hatch(hatch)
        return patch

class SelfOrganizingMap:
    def __init__(self, weights: np.ndarray, topology: str, dimensions: Tuple[int, int], distance_metric: str = 'euclidean', metadata: Dict = {}) -> None:
        """
        Initialize the Self-Organizing Map (SOM).

        Parameters
        ----------
        weights : np.ndarray
            Initial weights of the SOM.
        topology : str
            Topology of the SOM (e.g., "hexagonal", "random", "rectangular", "triangular", "hextop", "randtop", "gridtop", "tritop", "hex", "rand", "rect", "grid", "tri").
        dimensions : Tuple[int, int]
            Dimensions of the SOM grid.
        metadata : Dict
            Metadata associated with the SOM (e.g., labels or additional information).
        distance_metric : str, optional
            The metric to use when calculating pairwise distances between weights. Default is 'distance.euclidean'.
            Supported metrics include 'distance.euclidean', 'manhattan', 'cosine', and others supported by scipy.spatial.distance.distance.pdist.
        """
        self.metadata: Dict = metadata
        if weights.ndim != 3:
            weights = weights.reshape(dimensions[0], dimensions[1], -1)
        self.weights: np.ndarray = weights
        self.topology: str = self._validate_topology(topology)
        self.distance_metric: str = self._validate_distance_metric(distance_metric)
        self.dimensions: Tuple[int, int] = dimensions
        # Create and store nodes
        self.nodes: List[Node] = [
            Node(weight=self.weights[i, j], grid_index=(i, j), topology=self.topology)
            for i in range(dimensions[0])
            for j in range(dimensions[1])
        ]

        # Validate the dimensions of the weights
        if weights.shape[:2] != dimensions:
            raise ValueError("The shape of the weights array does not match the specified dimensions.")
        flattened_weights = self.weights.reshape(-1, self.weights.shape[-1])
        self.distances: np.ndarray = distance.pdist(flattened_weights, metric=self.distance_metric)
        self.distance_matrix: np.ndarray = distance.squareform(self.distances)
        self.merge_dict: Dict[Tuple[int, int], List[Tuple[int, int]]] = None
        self._calculate_u_matrix()

    def __iter__(self):
        """
        Iterate over all nodes in the container.

        Yields
        ------
        Node
            Each node in the container.
        """
        for node in self.nodes:
            yield node

    def _validate_topology(self, topology: str) -> str:
        """
        Validate the topology argument and map it to a standard form.

        Parameters
        ----------
        topology : str
            The input topology (e.g., "hexagonal", "hextop", "hex").

        Returns
        -------
        str
            The validated and standardized topology string.
        """
        mapping = {
            "hextop": "hexagonal",
            "hex": "hexagonal",
            "randtop": "random",
            "rand": "random",
            "gridtop": "rectangular",
            "rect": "rectangular",
            "grid": "rectangular",
            "tritop": "triangular",
            "tri": "triangular"
        }
        topology_lower = topology.lower()
        if topology_lower in mapping:
            return mapping[topology_lower]
        valid_topologies = ["hexagonal", "random", "rectangular", "triangular"]
        if topology_lower not in valid_topologies:
            raise ValueError(f"Invalid topology: {topology}. Choose from 'hexagonal', 'random', 'rectangular', or 'triangular'.")
        return topology_lower

    def _validate_distance_metric(self, distance_metric: str) -> str:
        """
        Validate the distance metric argument and map it to a distance.pdist-compatible option.

        Parameters
        ----------
        distance_metric : str
            The input distance metric.

        Returns
        -------
        str
            The validated and standardized distance metric string.
        """
        valid_metrics = {
            "linkdist": "braycurtis",
            "dist": "euclidean",
            "mandist": "cityblock"
        }
        return valid_metrics[distance_metric.lower()]

    def _calculate_u_matrix(self) -> None:
        """
        Calculate the U-Matrix for the SOM.

        The U-Matrix represents the average distance between the weight vector of each node and its neighbors.

        Returns
        -------
        np.ndarray
            A 2D array representing the U-Matrix.
        """
        u_matrix = np.zeros(self.dimensions)

        for i, j in np.ndindex(self.dimensions):
            neighbors = self.get_neighbors((i, j))
            if neighbors:
                distances = [self.get_weight_distance((i, j), neighbor) for neighbor in neighbors]
                u_matrix[i, j] = np.mean(distances)
            else:
                u_matrix[i, j] = 0  # No neighbors
        self.u_matrix = u_matrix
        return

    def create_grid(self, ax: plt.Axes = None, patch_shape: Union[str, None] = "polygon", linestyle: str = "-",
                    linewidth: float = None, hatch: str = None, alpha: float = 1.0) -> plt.Axes:
        """
        Create a grid for the SOM nodes, optionally coloring and adding patches based on parameters.

        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib Axes object to draw on. If None, a new Axes is created.
        patch_shape : Union[str, None], optional
            Shape of the patch for each node (e.g., "polygon"). If None, no patches are created.
        linestyle : str, optional
            Line style for the patch edges.
        linewidth : float, optional
            Line width for the patch edges.
        hatch : str, optional
            Hatch pattern for the patches.
        alpha : float, optional
            Alpha transparency for the patches.
        color : Union[str, tuple, np.ndarray, None], optional
            Color for the patches. Can be:
                - A single color (e.g., "red" or (1.0, 0.0, 0.0)).
                - A 2D array matching SOM dimensions with RGB values.
                - "u_matrix" to use the U-Matrix colors.
                - None for default colors.

        colormap : str, optional
            Name of the colormap to use when `color` is set to "u_matrix". Ignored if `color` is not "u_matrix" or a 2d array.

        Returns
        -------
        plt.Axes
            The Matplotlib Axes with the SOM grid.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(self.dimensions[0], self.dimensions[1]))
        else:
            fig = ax.get_figure()
        patches = [[[] for _ in range(self.dimensions[1])] for _ in range(self.dimensions[0])]
        for node in self:
            if patch_shape is not None:
                i, j = node.grid_index
                patch = node.create_patch(
                    shape=patch_shape,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    hatch=hatch,
                    alpha=alpha
                )
                ax.add_patch(patch)
                patches[i][j] = patch

        if self.topology == "hexagonal":
            ax.set_xlim(-1, self.dimensions[0] + 0.5)
            ax.set_ylim(-1, self.dimensions[1] * node.separation_factor / 2)
        else:
            ax.set_xlim(-1, self.dimensions[0])
            ax.set_ylim(-1, self.dimensions[1])

        ax.set_aspect('equal')
        ax.axis("off")
        return patches, ax

    def color_patches(self,
                      ax: plt.Axes = None,
                      patches: List[List[mpl.patches.Patch]] = None,
                      color: Union[str, tuple, np.ndarray, None] = None,
                      colormap: str = 'viridis', **kwargs) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(self.dimensions[0], self.dimensions[1]))
        else:
            fig = ax.get_figure()
        if patches is None:
            patches, _ = self.create_grid(ax=ax)

        if isinstance(color, str) and color in ['u_matrix', 'umatrix', 'u-matrix']:
            cmap = plt.get_cmap(colormap)
            u_matrix = self.u_matrix
            u_matrix = (u_matrix - u_matrix.min()) / (u_matrix.max() - u_matrix.min())
            color_matrix = cmap(u_matrix)
        elif isinstance(color, str) and 'merg' in color.lower():
            if self.merge_dict is None:
                if 'threshold' in kwargs:
                    threshold = kwargs['threshold']
                else:
                    threshold = 'loose'
                self.merge_nodes(threshold=threshold)
            n_spectra = len(self.merge_dict)
            colors = utils.arrays.generate_colors_with_yellow_last(n_spectra)
            color_matrix = np.zeros((self.dimensions[0], self.dimensions[1], 3))
            for k, (_, value) in enumerate(self.merge_dict.items()):
                for i, j in value:
                    color_matrix[i, j] = colors[k]
        elif isinstance(color, np.ndarray) and color.shape in [self.dimensions + (3,), self.dimensions + (4,)]:
            color_matrix = color
        elif isinstance(color, (tuple, list)):
            color_matrix = np.tile(color, (self.dimensions[0], self.dimensions[1], 1))
        else:
            color_matrix = np.zeros((self.dimensions[0], self.dimensions[1], 4))
        for node in self:
            i, j = node.grid_index
            patch = patches[i][j]
            node_color = color_matrix[i, j]
            patch.set_facecolor(node_color)
            node.color = node_color
        node = self[0, 0]
        if self.topology == "hexagonal":
            ax.set_xlim(-1, self.dimensions[0] + 0.5)
            ax.set_ylim(-1, self.dimensions[1] * node.separation_factor / 2)
        else:
            ax.set_xlim(-1, self.dimensions[0])
            ax.set_ylim(-1, self.dimensions[1])

        ax.set_aspect('equal')
        ax.axis("off")
        if isinstance(color, str) and color.lower() in ["u_matrix", "umatrix", "u-matrix"]:
            cbar_ax = fig.add_axes([0.89, 0.15, 0.03, 0.7])
            norm = mpl.colors.Normalize(vmin=self.u_matrix.min(), vmax=self.u_matrix.max())
            mpl.colorbar.ColorbarBase(cbar_ax, cmap=plt.get_cmap(colormap), norm=norm)
        return ax

    def plot_weights_on_grid(self, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot the weights as a spectrum inside each node's polygon.

        Parameters
        ----------
        ax : plt.Axes, optional
            The matplotlib Axes object to draw the plot on. If None, a new Axes is created, but a warning will be displayed to create the grid first.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(self.dimensions[0], self.dimensions[1]))
            print("No grid was provided. If you want to see the patches, create the grid first using 'create_grid'.")
        
        global_max = np.max(self.weights)
        global_min = np.min(self.weights)
        global_range = global_max - global_min
    
        for node in self:
            x, y = node.get_position()
            weights = node.weight
            radius = node.radius * 0.9
            half_box_size = (radius / np.sqrt(2)) if self.topology == "rectangular" else radius*0.92
            # half_box_size = radius*0.95
            if global_range > 0:
                weights = (weights - global_min) / global_range * (half_box_size) - half_box_size/1.5
            x_spectrum = np.linspace(-half_box_size, half_box_size, len(weights))
            if node.color is not None:
                color = distinctipy.get_text_color(node.color)
            else:
                color = 'black'
            ax.plot(x + x_spectrum, y + weights, color=color)
            # ax.axhline(y=y- half_box_size/1.5, xmin=x - half_box_size, xmax=x + half_box_size, color='black', linestyle='--')
                

        if self.topology == "hexagonal":
            ax.set_xlim(-1, self.dimensions[0] + 0.5)
            ax.set_ylim(-1, self.dimensions[1] * node.separation_factor / 2)
        else:
            ax.set_xlim(-1, self.dimensions[0])
            ax.set_ylim(-1, self.dimensions[1])

        ax.set_aspect('equal')
        ax.axis("off") 
        return ax

    def plot_weights(self,
                     indices: Union[List[Tuple[int, int]], str] = None,
                     offset: float = 0.0,
                     ax: plt.Axes = None,
                     linewidth: float = None,
                     savefig: Union[str, None] = None,
                     x_values: np.ndarray | List = None) -> plt.Axes:
        
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if indices is None:
            indices = np.ndindex(self.dimensions)
        elif isinstance(indices, str) and 'merg' in indices.lower():
            if self.merge_dict is None:
                self.merge_nodes()
            indices = self.merge_dict.keys() 
        if x_values is None:
            ax.set_xticklabels([])
            x_values = np.arange(self.weights.shape[2])
        else:
            x_values = np.array(x_values)
        for k, (i, j) in enumerate(indices):
            node = self[i, j]
            weights = node.weight
            ax.plot(x_values,
                    weights+k*offset,
                    linewidth=linewidth,
                    color=node.color)
        ax.set_xlim(x_values.min(), x_values.max())
        ax.set_ylim(0,)
        ax.set_xlabel("Bias (mV)")
        ax.set_ylabel("dI/dV (a.u.)")
        ax.tick_params(axis='both', which='major', direction='inout')
        ax.tick_params(axis='both', which='minor',direction='in')
        ax.minorticks_on()  # Enable minor ticks
        if savefig:
            ax.figure.savefig(savefig)
        return ax

    def merge_nodes(self,
                    threshold: float = None,
                    start_coordinates: List[Union[List, np.ndarray]] = [[0, 0]]) -> None:
        n, m = self.dimensions
        visited = set()
        merged_nodes = set()
        merge_dict = {node.grid_index: [node.grid_index] for node in self}
        if threshold is None:
            threshold = 'loose'
        if isinstance(threshold, str):
            thresholds, _, _ = self.get_thresholds(plot_histogram=True)
            threshold = thresholds[threshold]
        def find_parent(coordinate):
            """Find the parent node or cluster in merge_dict."""
            for parent, merged in merge_dict.items():
                if tuple(coordinate) in merged:
                    return parent
            return tuple(coordinate)  # If not found, treat it as its own parent
        
        def average_weight(coordinate):
            """Compute the average weight for a given coordinate."""
            parent = find_parent(coordinate)
            coordinates = merge_dict[parent]  # Use parent to get the cluster
            return sum(self.weights[c] for c in coordinates) / len(coordinates)
        
        def merge(coordinate):
            stack = [coordinate]
            while stack:
                current = stack.pop()
                visited.add(tuple(current))
                neighbors = self.get_neighbors(current)
                neighbors = sorted(neighbors, key=lambda x: self.get_weight_distance(current, x))
                for neighbor in neighbors:
                    if tuple(neighbor) in visited:
                        continue
                    avg_weight_current = average_weight(current)
                    avg_weight_neighbor = average_weight(neighbor)
                    dist = distance.euclidean(avg_weight_current, avg_weight_neighbor)
                    if dist < threshold:
                        parent_current = find_parent(current)
                        parent_neighbor = find_parent(neighbor)
                        if parent_neighbor != parent_current:  # Avoid self-merge
                            merged_nodes.update(merge_dict[parent_neighbor])
                            merged_nodes.update(merge_dict[parent_current])
                            merge_dict[parent_current].extend(merge_dict.pop(parent_neighbor))
                        stack.append(neighbor)
        for coordinate in start_coordinates:
            if tuple(coordinate) not in visited:
                merge(coordinate)
                
        for i, j in np.ndindex(self.dimensions):
            coordinate = (i, j)
            if coordinate not in visited and coordinate not in merged_nodes:
                merge([i, j])
        self.merge_dict = merge_dict
        return merge_dict

    def get_neighbors(self, coordinate: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get the neighboring indices of a node based on the grid topology.

        Parameters
        ----------
        coordinate : Tuple[int, int]
            The (row, column) index of the node.

        Returns
        -------
        List[Tuple[int, int]]
            A list of neighboring indices.
        """
        max_x, max_y = self.dimensions
        neighbors = []
        if self.topology == 'hexagonal':
            q, r = coordinate  # Current position
            # Directions based on row parity
            directions_even = [
                (1, 0),  # east
                (0, 1),  # north-east
                (-1, 1), # north-west
                (-1, 0), # west
                (-1, -1), # south-west
                (0, -1),  # south-east
            ]

            directions_odd = [
                (1, 0),  # east
                (1, 1),  # north-east
                (0, 1),  # north-west
                (-1, 0), # west
                (0, -1), # south-west
                (1, -1), # south-east
            ]
            
            # Use row parity to select directions
            directions = directions_even if r % 2 == 0 else directions_odd

            for dq, dr in directions:
                neighbor = (q + dq, r + dr)
                if 0 <= neighbor[0] < max_x and 0 <= neighbor[1] < max_y:
                    neighbors.append(neighbor)
        elif self.topology == 'rectangular':
            x, y = coordinate
            # Directions for a rectangular grid
            #TODO: check if you need to add the diagonal neighbors
            directions = [
                (0, 1),   # north
                (1, 0),   # east
                (0, -1),  # south
                (-1, 0),  # west
            ]
            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                if 0 <= neighbor[0] < max_x and 0 <= neighbor[1] < max_y:
                    neighbors.append(neighbor)
        else:
            raise ValueError("Unsupported grid type")
        return neighbors

    def overlay_hit_histogram(self, hit_histogram: Union[np.ndarray, List[List[int]]], ax: plt.Axes, location = 'center') -> None:
        """
        Overlay a hit histogram as text onto the SOM grid.

        Parameters
        ----------
        hit_histogram : Union[np.ndarray, List[List[int]]]
            A 2D array or list representing the hit counts for each node in the SOM grid.
        ax : plt.Axes
            The matplotlib Axes object where the text annotations will be added.
        """
        shift = {'upper':+1, 'center':0, 'lower':-1}[location]
        # fig_width, fig_height = ax.figure.get_size_inches()
        # avg_dim = (self.dimensions[0] + self.dimensions[1]) / 2
        # txt_size = max(5, min(15, int(0.4 * min(fig_width, fig_height) / avg_dim * 10)))
        radius = self[0, 0].radius * 0.9
        half_box_size = (radius / np.sqrt(2)) if self.topology == "rectangular" else radius*0.92
        for node in self:
            x, y = node.get_position()
            num_hits = hit_histogram[node.grid_index]
            if node.color is not None:
                text_color = distinctipy.get_text_color(node.color)
            else:
                text_color = 'black'
            ax.text(x, y+shift*half_box_size/2, str(num_hits), ha='center', va='center', color=text_color)

    def get_weight_distance(self, index1: Tuple[int, int], index2: Tuple[int, int]) -> float:
        """
        Calculate the distance.euclidean distance between the weights of two nodes in the SOM grid.

        Parameters
        ----------
        index1 : Tuple[int, int]
            The (row, column) index of the first node.
        index2 : Tuple[int, int]
            The (row, column) index of the second node.

        Returns
        -------
        float
            The distance.euclidean distance between the weights of the two specified nodes.
        """
        flat_index1 = np.ravel_multi_index(index1, self.dimensions)
        flat_index2 = np.ravel_multi_index(index2, self.dimensions)
        return self.distance_matrix[flat_index1, flat_index2]

    def plot_distance_histogram(self, bins: Union[int, str] = 'auto', ax: plt.Axes = None, savefig: Union[str, None] = None) -> plt.Axes:
        """
        Plot a histogram of pairwise distances between the SOM weights.

        Parameters
        ----------
        bins : Union[int, str], optional
            The number of bins or binning strategy for the histogram. Default is 'auto'.
        ax : plt.Axes, optional
            The Matplotlib Axes object on which to draw the histogram. If None, a new Axes is created.
        savefig : Union[str, None], optional
            The file path to save the figure. If None, the figure is not saved.

        Returns
        -------
        plt.Axes
            The Matplotlib Axes object with the plotted histogram.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        counts, _, _ = ax.hist(self.distances, bins=bins, alpha=0.75, label="Pairwise Distances")
        ax.set_xlabel("Pairwise Distance")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of SOM Pairwise Distances")
        ax.legend()

        if savefig:
            ax.figure.savefig(savefig)

        return ax

    def get_thresholds(self,
                       plot_histogram=True,
                       savefig=None):    
        mean_distance = np.mean(self.distances)#       ) np.mean(self.distances)
        std_distance = np.std(self.distances)

        aggressive_threshold = mean_distance - 0.0 * std_distance  # Aggressive
        medium_threshold = mean_distance - 0.5 * std_distance       # Medium
        loose_threshold = mean_distance - 1.0 * std_distance        # Loose
        thresholds = {
            'aggressive': aggressive_threshold,
            'medium': medium_threshold,
            'loose': loose_threshold
            }
        if plot_histogram:
            fig, ax = plt.subplots(1,1)
            ax.hist(self.distances, bins='auto', alpha=0.75, label="Pairwise Distances")
            
            aggressive_line =plt.axvline(aggressive_threshold, color='red', linestyle='--', linewidth=1, label='Aggressive')
            medium_line = plt.axvline(medium_threshold, color='orange', linestyle='--', linewidth=1, label='Medium')
            loose_line = plt.axvline(loose_threshold, color='green', linestyle='--', linewidth=1, label='Loose')

            ymin, ymax = plt.ylim()  
            plt.text(aggressive_threshold, ymax * 0.5, f'{aggressive_threshold:.2f}', color='red', fontsize=10, ha='right', rotation=90)
            plt.text(medium_threshold, ymax * 0.3, f'{medium_threshold:.2f}', color='orange', fontsize=10, ha='right', rotation=90)
            plt.text(loose_threshold, ymax * 0.1, f'{loose_threshold:.2f}', color='green', fontsize=10, ha='right', rotation=90)

            ax.set_xlabel("Pairwise Distance")
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram of SOM Pairwise Distances")
            ax.legend()
            if savefig:
                modified_savefig = savefig.with_name(savefig.stem + "_hist_som_distances" + savefig.suffix)
                plt.savefig(modified_savefig)
        
        return thresholds, mean_distance, std_distance

    def __getitem__(self, index: Tuple[int, int]) -> Node:
        """
        Get the node at the specified index.

        Parameters
        ----------
        index : Tuple[int, int]
            The (row, column) index of the node.

        Returns
        -------
        Node
            The node at the specified index.
        """
        flat_index = np.ravel_multi_index(index, self.dimensions)
        return self.nodes[flat_index]


    def __setitem__(self, index: Tuple[int, int], value: Node) -> None:
        """
        Set a node at the specified index.

        Parameters
        ----------
        index : Tuple[int, int]
            The (row, column) index of the node.
        value : Node
            The new node to set at the index.
        """
        flat_index = np.ravel_multi_index(index, self.dimensions)
        self.nodes[flat_index] = value

    @property
    def shape(self) -> Tuple[int, int]:
        return self.dimensions
    
    @classmethod
    def from_hdf5(cls, path: Union[Path, str], dimensions: Tuple[int, int] = None, topology: str = 'hexagonal', distance_metric: str = 'euclidean') -> 'SelfOrganizingMap':
        """
        Create a SelfOrganizingMap instance from an HDF5 file.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the HDF5 file.
        dimensions : Tuple[int, int], optional
            Dimensions of the SOM grid. If None, they will be inferred from the weights.
        topology : str, optional
            The topology of the SOM grid. Default is 'hexagonal'.
        distance_metric : str, optional
            Metric for calculating distances. Default is 'euclidean'.

        Returns
        -------
        SelfOrganizingMap
            A new SelfOrganizingMap instance initialized with data from the HDF5 file.

        Raises
        ------
        FileNotFoundError
            If the specified HDF5 file does not exist.
        KeyError
            If the key 'weights' is not found in the HDF5 file.
        """
        path = Path(path)
        output_path = next(
            (rf for rf in [path.with_suffix(ext) for ext in ['.h5', '.hdf5']] if rf.exists()),
            None
        )
        weights = None
        if not output_path:
            raise FileNotFoundError(f"No HDF5 file found at {path} with .h5 or .hdf5 extensions")
        data, attribs = io.read_hdf5(output_path)
        print(f"Loaded Self-Organizing Map from {output_path}.")
        weights = next((
            v for k, v in data.items() if k == 'weights' or isinstance(v, dict) and 'weights' in v), None
                       )['weights']
        if "dimensions" in attribs:
            dimensions = attribs["dimensions"]
        elif "dimensions" in data:
            dimensions = data["dimensions"]
        if "platform" in attribs and attribs["platform"].lower() == "matlab":
            dimensions = tuple(reversed(dimensions))
        for key in attribs:
            if 'topo' in key.lower():
                topology = attribs[key]
                print(f"  Topology: {topology}")
            if 'dist' in key.lower():
                distance_metric = attribs[key]
                print(f"  Distance Metric: {distance_metric}")
                
        if weights is None:
            raise KeyError("Key 'weights' not found in the HDF5 file.")
        if dimensions is None:
            warnings.warn(
                "Warning: Dimensions not provided. Assuming a square SOM grid. This may cause incorrect results if √n_features is an integer.",
                category=UserWarning
            )
            for dim in weights.shape:
                if np.sqrt(dim).is_integer():
                    n = int(np.sqrt(dim))
                    dimensions = (n, n)
                    break
        print(f"  Dimensions: {dimensions}")
        return cls(weights=weights, topology=topology, dimensions=dimensions, distance_metric=distance_metric, metadata=attribs)        

class SOMDataMapper:
    def __init__(self,
                 som: SelfOrganizingMap,
                 data: np.ndarray) -> None:
        self.som = som
        self.data = data
        
        if self.data.shape[-1] != self.som.weights.shape[-1]:
            raise ValueError(
                "The dimension of the input data does not match the dimension of the SOM weights."
                )
        
        self.flattened_data = self.data.reshape(-1, self.data.shape[-1])
        self.flattened_weights = self.som.weights.reshape(-1, self.som.weights.shape[-1])
        self.distances = distance.cdist(self.flattened_data,
                                        self.flattened_weights,
                                        metric=som.distance_metric)
        self.cluster_index = self.get_best_matching_unit()
        self.hit_histogram = self._generate_hit_histogram()
        
    @property
    def cluster_index_2d(self):
        return self.cluster_index.reshape(self.data.shape[:-1])
    
    def get_best_matching_unit(self, output_dim: int = 1, index_dim: int = 1, order: int = 1,) -> np.ndarray:
        """
        Get the best matching unit (BMU) for each data point.

        Returns
        -------
        np.ndarray
            An array of BMU indices for each data point.
        """
        bmus_flat = self.distances.argsort(axis=1)[:, order - 1]
        if index_dim == 2:
            bmus = np.array([np.unravel_index(bmu, self.som.shape) for bmu in bmus_flat])
        else:
            bmus = bmus_flat
        if output_dim == 2:
            bmus = bmus.reshape(self.data.shape[:-1] + (2,) if index_dim == 2 else self.data.shape[:-1])
        return bmus
    
    def _generate_hit_histogram(self):
        """
        Generate the hit histogram for the SOM.
        
        Returns
        -------
        np.ndarray
            An array of the same shape as the SOM grid, where each element represents
            the number of data points mapped to the corresponding node.
        """
        hit_histogram = np.zeros(self.som.shape, dtype=int)
        bmus = self.get_best_matching_unit(index_dim=2)
        
        for bmu in bmus:
            hit_histogram[tuple(bmu)] += 1
        return hit_histogram
    
    def plot_cluster_map(self,
                         colormap: str ='viridis',
                         ax: plt.Axes = None) -> plt.Axes:
        if self.data.ndim != 3:
            raise ValueError("The input data must be a 3D array. With the last dimension being the number of features.")
        if ax is None:
            fig_size = np.array(self.data.shape[:-1])/20
            fig, ax = plt.subplots(1, 1, figsize=fig_size)
        
        # norm = mpl.colors.Normalize(vmin=self.cluster_index.min(), vmax=self.cluster_index.max())
        
        ax.imshow(self.cluster_index_2d, cmap=colormap)
        # plt.colorbar(mpl.cm.ScalarMappable(cmap=colormap), ax=ax)
        plt.tight_layout()
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        return ax
    
    def plot_cluster_map_and_som(self,
                                 colormap: str ='viridis',
                                 overlay_hit_histogram: bool = True,
                                 plot_weights_on_grid: bool = True,
                                 axes: Tuple[plt.Axes, plt.Axes] = None) -> plt.Axes:
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        norm = mpl.colors.Normalize(vmin=0, vmax=self.som.shape[0] * self.som.shape[1])
        cm = mpl.cm.get_cmap(colormap)
        color_matrix = cm(norm(np.arange(self.som.shape[0] * self.som.shape[1])).reshape(self.som.shape[0], self.som.shape[1]))
        patches, _ = self.som.create_grid(patch_shape='hexagon', ax=axes[0])
        self.som.color_patches(patches=patches, color=color_matrix, ax=axes[0])
        if overlay_hit_histogram:
            self.som.overlay_hit_histogram(self.hit_histogram, ax=axes[0], location='upper')
        if plot_weights_on_grid:
            self.som.plot_weights_on_grid(ax=axes[0])
        self.plot_cluster_map(ax=axes[1], colormap=colormap)
        return axes
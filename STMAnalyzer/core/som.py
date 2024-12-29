import numpy as np
from typing import Tuple, Dict, Union, List, Any
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import distance # import distance.pdist, distance.squareform, distance.euclidean
import warnings
# from .. import io
# from .. import utils
import STMAnalyzer.io as io
import STMAnalyzer.utils as utils

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
        self.color = None

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
        x, y = self.grid_index[0], self.grid_index[1]
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
        self.nodes: List[List[Node]] = [
            [Node(weight=self.weights[i, j], grid_index=(i, j), topology=self.topology) for j in range(dimensions[1])]
            for i in range(dimensions[0])
        ]

        # Validate the dimensions of the weights
        if weights.shape[:2] != dimensions:
            raise ValueError("The shape of the weights array does not match the specified dimensions.")
        flattened_weights = self.weights.reshape(-1, self.weights.shape[-1])
        self.distances: np.ndarray = distance.pdist(flattened_weights, metric=self.distance_metric)
        self.distance_matrix: np.ndarray = distance.squareform(self.distances)
        self.merge_dict: Dict[Tuple[int, int], List[Tuple[int, int]]] = None
        self._calculate_u_matrix()

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

        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                neighbors = self.get_neighbors((i, j))
                if neighbors:
                    distances = [self.get_weight_distance((i, j), neighbor) for neighbor in neighbors]
                    u_matrix[i, j] = np.mean(distances)
                else:
                    u_matrix[i, j] = 0  # No neighbors
        self.u_matrix = (u_matrix - u_matrix.min()) / (u_matrix.max() - u_matrix.min())
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
        patches = []
        for i in range(self.dimensions[0]):
            patches.append([])
            for j in range(self.dimensions[1]):
                if patch_shape is not None:
                    node = self[i, j]
                    patch = node.create_patch(
                        shape=patch_shape,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        hatch=hatch,
                        alpha=alpha,
                        facecolor=node.color,
                    )
                    ax.add_patch(patch)
                    patches[-1].append(patch)

        if self.topology == "hexagonal":
            ax.set_xlim(-1, self.dimensions[0] + 0.5)
            ax.set_ylim(-1, self.dimensions[1] * node.separation_factor / 2)
        else:
            ax.set_xlim(-1, self.dimensions[0])
            ax.set_ylim(-1, self.dimensions[1])

        ax.set_aspect('equal')
        ax.axis("off")
        return patches, ax

    def assign_color(self, method: str, colormap: str = 'viridis') -> np.ndarray:
        if method in ['u_matrix', 'umatrix', 'u-matrix']:
            cmap = plt.get_cmap(colormap)
            color_matrix = cmap(self.u_matrix)
            for i in range(self.dimensions[0]):
                for j in range(self.dimensions[1]):
                    self[i, j].color = color_matrix[i, j]
        elif 'merg' in method:
            if self.merge_nodes is None:
                self.merge_nodes()
            n_spectra = len(self.merge_dict)
            colors = utils.arrays.generate_colors_with_yellow_last(n_spectra)
            color_matrix = np.zeros((self.dimensions[0], self.dimensions[1], 3))
            for i, (_, value) in enumerate(self.merge_dict.items()):
                for v in value:
                    color_matrix[v[0], v[1]] = colors[i]
                    self[v[0], v[1]].color = colors[i]
        return color_matrix

    def color_patches(self,
                      ax: plt.Axes = None,
                      patches: List[List[mpl.patches.Patch]] = None,
                      color: Union[str, tuple, np.ndarray, None] = None,
                      colormap: str = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(self.dimensions[0], self.dimensions[1]))
        else:
            fig = ax.get_figure()
        if patches is None:
            patches, _ = self.create_grid(ax=ax)
        if isinstance(color, str):
            color_matrix = self.assign_color(method=color, colormap=colormap)
        elif isinstance(color, np.ndarray) and color.shape in [self.dimensions + (3,), self.dimensions + (4,)]:
            color_matrix = color
        elif isinstance(color, (tuple, list)):
            color_matrix = np.tile(color, (self.dimensions[0], self.dimensions[1], 1))
        else:
            color_matrix = np.zeros((self.dimensions[0], self.dimensions[1], 4))
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                patch = patches[i][j]
                node_color = color_matrix[i, j]
                patch.set_facecolor(node_color)
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
            print("Adding colorbar")
            cbar_ax = fig.add_axes([0.89, 0.15, 0.03, 0.7])
            norm = mpl.colors.Normalize(vmin=self.u_matrix.min(), vmax=self.u_matrix.max())
            # Create the colorbar without ScalarMappable
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
    
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                node = self.nodes[i][j]
                x, y = node.get_position()
                weights = node.weight
                radius = node.radius * 0.9
                half_box_size = (radius / np.sqrt(2)) if self.topology == "rectangular" else radius*0.92
                # half_box_size = radius*0.95
                if global_range > 0:
                    weights = (weights - global_min) / global_range * (half_box_size) - half_box_size/1.5
                x_spectrum = np.linspace(-half_box_size, half_box_size, len(weights))
                ax.plot(x + x_spectrum, y + weights, color='black')
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

    def merge_nodes(self,
                    threshold: float = None,
                    start_coordinates: List[Union[List, np.ndarray]] = [[0, 0]]) -> None:
        n, m = self.dimensions
        visited = set()
        merged_nodes = set()
        merge_dict = {tuple((i, j)): [tuple((i, j))] for i in range(n) for j in range(m)}
        if threshold is None:
            threshold = 'loose'
        if isinstance(threshold, str):
            thresholds, _, _ = self.get_thresholds(plot_histogram=False)
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
                
        for i in range(n):
            for j in range(m):
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
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                node =self.nodes[i][j]
                x, y = node.get_position()
                num_hits = hit_histogram[i, j]
                radius = node.radius * 0.9
                
                half_box_size = (radius / np.sqrt(2)) if self.topology == "rectangular" else radius*0.92

                txt_size = max(5, 20 - max(self.dimensions) // 2)
                ax.text(x, y+shift*half_box_size/2, str(num_hits), ha='center', va='center', size=txt_size, color='black')

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
        mean_distance = np.mean(self.distances)
        std_distance = np.std(self.distances)

        aggressive_threshold = mean_distance - 0 * std_distance  # Aggressive
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
        return self.nodes[index[0]][index[1]]

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
        self.nodes[index[0]][index[1]] = value

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.nodes), len(self.nodes[0]))
    
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

    
if __name__ == "__main__":

    # # Initialize the Self-Organizing Map
    # som = SelfOrganizingMap(weights=np.random.rand(*dimensions, 3), topology='hex', dimensions=dimensions)

    # # # Test weight distance
    # # index1 = (2, 1)
    # # index2 = (2, 1)
    # # distance = som.get_weight_distance(index1, index2)
    # # print(f"Distance between {index1} and {index2}: {distance}")
    
    # # # Plot distance histogram
    # # ax= som.plot_distance_histogram()

    # hit_histogram = np.random.randint(0, 100, dimensions)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # som.create_grid(ax=ax)
    # som.overlay_hit_histogram(hit_histogram, ax=ax)
    # plt.show()

    # som = SelfOrganizingMap(weights=np.random.rand(*dimensions, 3), topology='rect', dimensions=dimensions, metadata={})

    # som[2, 3].patch.set_facecolor('orange')
    # som[4, 5].patch.set_edgecolor('blue')
    # som[4, 5].patch.set_linewidth(3)
    # som[6, 7].radius = 0.4
    # som[8, 2].patch.orientation = np.pi / 6
    # som[5, 5].patch.set_linestyle('--')
    # som[3, 4].patch.set_linewidth(2)
    # som[9, 9].patch.set_hatch('x')

    # central_node = (np.random.randint(dimensions[0]), np.random.randint(dimensions[1]))
    # som[central_node].patch.set_facecolor('red')

    # hit_histogram = np.random.randint(0, 100, dimensions)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # som.create_grid(ax=ax)
    # som.overlay_hit_histogram(hit_histogram, ax=ax)
    
    # plt.show()
    path = Path('.')
    # dimensions = (6, 12)
    # path = path.resolve() /'tests'/f'output{dimensions[0]}x{dimensions[1]}.hdf5'
    # path = Path(r"G:\My Drive\_projects\ML-STS\Matlab\20241227-FeSeTe_paper-7x10-hextop-0Ep-dist-covStp10000-iniNei10\output.hdf5")
    # path = Path(r"G:\My Drive\_projects\ML-STS\Matlab\20241226-FeSeTe_paper-4x4-hextop-0Ep-dist-covStp300-iniNei3\output.hdf5")
    path = Path(r"G:\My Drive\_projects\ML-STS\Matlab\20241227-FeSeTe_paper-6x12-hextop-0Ep-dist-covStp5000-iniNei10\output.hdf5")
    # path = Path(r"G:\My Drive\_projects\ML-STS\Matlab\20241227-FeSeTe_paper-14x20-hextop-0Ep-dist-covStp5000-iniNei7\output.hdf5")
    # path = Path(r"G:\My Drive\_projects\ML-STS\Matlab\20241227-FeSeTe_paper-8x25-hextop-0Ep-dist-covStp5000-iniNei8\output.hdf5")
    dimensions = None
    # cmap='YlOrBr'
    # cmap='afmhot'
    # cmap='hot'
    # cmap='autumn'
    # cmap='hsv'
    # cmap='cool'
    # cmap='cividis'
    cmap='viridis'
    # cmap='inferno'
    # cmap='Greys'
    topology = 'hexagonal'
    hit_location = 'upper'
    
    som = SelfOrganizingMap.from_hdf5(path, topology=topology, dimensions=dimensions)
    matlab_output, _ = io.read_hdf5(path)
    sts_name = 'GridSpectroscopy_103_06222022001'
    hit_histogram = matlab_output[sts_name]['hitHistogram'].reshape(som.dimensions).astype(int)
    som.merge_nodes(threshold='loose')
    patches, ax = som.create_grid(patch_shape='hexagon', linewidth=0.5, alpha=0.5)
    som.color_patches(ax=ax, patches=patches, )
    som.overlay_hit_histogram(hit_histogram, location=hit_location, ax=ax)
    som.plot_weights_on_grid(ax = ax)
    plt.show()
    
    # som = SelfOrganizingMap.from_hdf5(path, topology=topology)
    # colors = np.zeros(som.dimensions + (3,))
    # colors[8, 5, :] = [1, 0, 0]
    # colors[6, 0, :] = [0, 1, 0]
    # colors[0, 3 ,:] = [0, 0, 1]

    # som.plot_distance_histogram()
    
    patches, ax = som.create_grid(patch_shape='hexagon', linewidth=0.5, alpha=0.5)
    color_matrix = som.assign_color('u-matrix')
    som.color_patches(ax=ax, patches=patches, color=color_matrix)
    som.overlay_hit_histogram(hit_histogram, location=hit_location, ax=ax)
    som.plot_weights_on_grid(ax = ax)
    plt.show()

    # som = SelfOrganizingMap.from_hdf5(path, topology=topology)
    # ax = som.create_grid(patch_shape='circle', linewidth=0.5, alpha=0.5, color='u_matrix', colormap=cmap)
    # som.overlay_hit_histogram(hit_histogram, location=hit_location, ax=ax)
    # som.plot_weights_on_grid(ax = ax)
    # plt.show()

import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from PIL import ImageColor
import random
import matplotlib.colors as mcolors
import matplotlib.pylab as plt
import numpy as np
from typing import Tuple
from skimage.feature import peak_local_max
import distinctipy
from scipy.spatial.distance import euclidean
from itertools import product
from matplotlib_scalebar.scalebar import ScaleBar
import numpy.typing as npt
from scipy.spatial.distance import pdist, squareform
from matplotlib.gridspec import GridSpec
from ..utils.arrays import adjust_dimensions2d, get_neighbors
from ..core.scan import STMScan

GRAY = (0.5, 0.5, 0.5)


def plot_hit_histogram(hit_histogram,
                  dimensions: Tuple = None,
                  plot_spectra: bool = True,
                  V=None,
                  som_weights=None,
                  offset=0.0,
                  savefig=None):
    hit_histogram = adjust_dimensions2d(hit_histogram, dimensions)
    dimensions = hit_histogram.shape
    if som_weights is not None and som_weights.ndim != 3:
        som_weights = som_weights.reshape(dimensions+(-1,))
    padded_hit_histogram = np.pad(
        hit_histogram, ((1, 1), (1, 1)), 'constant', constant_values=0)
    coordinates = peak_local_max(padded_hit_histogram, min_distance=1)
    coordinates = np.array([[i-1, j-1] for i, j in coordinates])
    colors = distinctipy.get_colors(len(coordinates))
    hex_separation = 1.8
    radius = hex_separation/(4*np.cos(np.pi/6))
    max_hit = hit_histogram.max()
    normalized_hist = hit_histogram / max_hit
    normalized_hist = np.clip(normalized_hist, 1e-10, 1)  # Avoid log(0) issues
    # Apply logarithmic scaling to the normalized histogram
    log_scaled_hist = np.log10(normalized_hist) + 1
    linewidth = 0.3
    patch_list = []
    text_list = []
    # radii_scaled = radius * log_scaled_hist
    radii_scaled = radius * normalized_hist
    # print(radii_scaled)
    txt_size = max(5, 20 - max(dimensions) // 2)
    BG_COLOR = GRAY
    FG_COLOR = distinctipy.BLACK
    counter = 0
    if plot_spectra:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(10, 10))

    for irow in range(dimensions[0]):
        for icolumn in range(dimensions[1]):
            radius_scaled = radii_scaled[irow, icolumn]

            x, y = irow, icolumn*0.9
            if icolumn % 2 != 0:
                x += 0.5
            # Background color
            hex = patches.RegularPolygon((x, y),
                                         numVertices=6,
                                         radius=radius,
                                         facecolor=BG_COLOR,
                                         alpha=0.7,
                                         edgecolor=distinctipy.BLACK)
            patch_list.append(hex)

            # Foreground color
            hex = patches.RegularPolygon((x, y),
                                         numVertices=6,
                                         radius=radius_scaled,
                                         facecolor=FG_COLOR,
                                         # alpha=1,
                                         # edgecolor=distinctipy.BLACK,
                                         )

            patch_list.append(hex)
            # Local maxima edge color
            if radius_scaled*100 >= txt_size:
                # print(radius_scaled, txt_size)
                color = BG_COLOR  # gray
            else:
                color = FG_COLOR  # distinctipy.BLACK
            if np.any(np.all(np.array([irow, icolumn]) == coordinates, axis=1)):
                hex = patches.RegularPolygon((x, y),
                                             numVertices=6,
                                             radius=radius_scaled,
                                             alpha=1.0,
                                             facecolor=colors[counter])
                color = colors[counter]
                counter += 1
                patch_list.append(hex)
            text_list.append([x, y, int(hit_histogram[irow, icolumn]), color])
    p = PatchCollection(patch_list, match_original=True)
    ax0.add_collection(p)

    for i, txt in enumerate(text_list):
        x, y, num, color = txt
        ax0.text(x, y, num, ha='center', va='center', size=txt_size,
                 color=distinctipy.get_text_color(color))

    ax0.set_xlim(-1, dimensions[0]+0.5)
    ax0.set_ylim(-1, dimensions[1]*0.9)
    ax0.set_aspect('equal')
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.grid(False)
    if plot_spectra:
        counter = 0
        for i, coord in enumerate(coordinates):
            ax1.plot(V, som_weights[coord[0], coord[1]]+i *
                     offset, linewidth=0.75, color=colors[counter])
            counter += 1
        ax1.set_ylim(0,)
        ax1.set_xlim(V[0], V[-1])
        ax1.set_xlabel("Bias (mV)")
        ax1.set_ylabel("dI/dV (a.u.)")

    # plt.savefig(base_dir / "Python" / 'SOM-peaks.pdf'
    #             )
    plt.show()
    if savefig is not None:
        plt.savefig(savefig)
        plt.clf()
    return

def som_didv_topo(hit_histogram,
                  som_weights,
                  cluster_index,
                  stm_scan: STMScan,
                  block_size=(1, -1),
                  dimensions: Tuple = None,
                  colormap: str = 'Grays',
                  offset=0.01,
                  savefig=None):
    origin = 'lower'
    hit_histogram = adjust_dimensions2d(hit_histogram, dimensions)
    dimensions = hit_histogram.shape
    n, m = dimensions
    if som_weights.ndim != 3:
        som_weights = som_weights.reshape(dimensions+(-1,))
    if cluster_index.ndim != 2:
        cluster_index = cluster_index.reshape(stm_scan.nx, stm_scan.ny)
    if block_size[1] == -1:
        block_size = (block_size[0], dimensions[1])
    colormap = plt.get_cmap(colormap)

    k, l = block_size
    for i in range(0, n, k):
        for j in range(0, m, l):
            coordinates = [(x, y) for x, y in product(
                range(i, min(i + k, dimensions[0])), range(j, min(j + l, dimensions[1])))]
            colors = distinctipy.get_colors(k*l)
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(20, 6))
            hex_separation = 1.8
            radius = hex_separation/(4*np.cos(np.pi/6))
            max_hit = hit_histogram.max()
            normalized_hist = hit_histogram / max_hit
            normalized_hist = np.clip(
                normalized_hist, 1e-10, 1)  # Avoid log(0) issues
            # Apply logarithmic scaling to the normalized histogram
            log_scaled_hist = np.log10(normalized_hist) + 1
            patch_list = []
            text_list = []
            radii_scaled = radius * log_scaled_hist
            # radii_scaled = radius * normalized_hist
            # print(radii_scaled)
            txt_size = max(5, 20 - max(dimensions) // 2)
            BG_COLOR = GRAY
            FG_COLOR = distinctipy.BLACK
            counter = 0
            for irow in range(dimensions[0]):
                for icolumn in range(dimensions[1]):
                    radius_scaled = radii_scaled[irow, icolumn]

                    x, y = irow, icolumn*0.9
                    if icolumn % 2 != 0:
                        x += 0.5
                    # Background color
                    hex = patches.RegularPolygon((x, y),
                                                 numVertices=6,
                                                 radius=radius,
                                                 facecolor=BG_COLOR,
                                                 alpha=0.7,
                                                 edgecolor=distinctipy.BLACK)
                    patch_list.append(hex)

                    # Foreground color
                    hex = patches.RegularPolygon((x, y),
                                                 numVertices=6,
                                                 radius=radius_scaled,
                                                 facecolor=FG_COLOR,
                                                 # alpha=1,
                                                 # edgecolor=distinctipy.BLACK,
                                                 )

                    patch_list.append(hex)
                    # Local maxima edge color
                    if radius_scaled*100 >= txt_size:
                        # print(radius_scaled, txt_size)
                        color = BG_COLOR  # gray
                    else:
                        color = FG_COLOR  # distinctipy.BLACK
                    if np.any(np.all(np.array([irow, icolumn]) == coordinates, axis=1)):
                        hex = patches.RegularPolygon((x, y),
                                                     numVertices=6,
                                                     radius=radius_scaled,
                                                     alpha=1.0,
                                                     facecolor=colors[counter])
                        color = colors[counter]
                        counter += 1
                        patch_list.append(hex)
                    text_list.append(
                        [x, y, int(hit_histogram[irow, icolumn]), color])

            p = PatchCollection(patch_list, match_original=True)
            ax0.add_collection(p)
            for txt in text_list:
                x, y, num, color = txt
                ax0.text(x, y, num, ha='center', va='center',
                         size=txt_size, color=distinctipy.get_text_color(color))

            ax0.set_xlim(-1, dimensions[0]+0.5)
            ax0.set_ylim(-1, dimensions[1]*0.9)
            ax0.set_aspect('equal')
            ax0.set_xticks([])
            ax0.set_yticks([])
            for counter, coord in enumerate(coordinates):
                # V = V*1e3 #TODO this might need to adjusted.
                ws = som_weights[coord[0], coord[1]]
                ax1.plot(stm_scan.V, ws+offset*counter,
                         linewidth=0.75, color=colors[counter])
            ax1.set_ylim(0,)
            ax1.set_xlim(stm_scan.V[0], stm_scan.V[-1])
            ax1.set_xlabel("Bias (mV)")
            ax1.set_ylabel("dI/dV (a.u.)")
            ax0.grid(False)
            norm = plt.Normalize(vmin=np.min(
                stm_scan.topography), vmax=np.max(stm_scan.topography))

            # [:, :, :3] to get rid of the 4th column (color alpha)
            masked = colormap(norm(stm_scan.topography))[:, :, :3]
            for counter, (irow, icolumn) in enumerate(coordinates):
                flat_index = np.ravel_multi_index([irow, icolumn], dimensions)
                idxs = np.where(cluster_index == flat_index)
                masked[idxs] = colors[counter]
            ax2.imshow(masked, origin=origin, cmap='YlOrBr', extent=[
                0, stm_scan.dimensions[1]/1e-9, 0, stm_scan.dimensions[0]/1e-9])
            ax3.imshow(stm_scan.topography, origin=origin, cmap='YlOrBr', extent=[
                0, stm_scan.dimensions[1]/1e-9, 0, stm_scan.dimensions[0]/1e-9])

            scalebar = ScaleBar(1, units='nm', dimension="si-length", length_fraction=0.4,
                                location='lower right', box_alpha=0, scale_loc='top')
            ax2.add_artist(scalebar)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.grid(False)
            if savefig is not None:
                name = savefig.stem + f"{i},{j}" + savefig.suffix
                savefig = savefig.with_name(name)
                plt.savefig(savefig)
                plt.clf()
            # plt.show()


def hist_som_distances(som_weights):
    # Reshape SOM weights to 2D (flatten grid while keeping weight vectors intact)
    flattened_weights = som_weights.reshape(-1, som_weights.shape[-1])
    
    # Compute pairwise distances between all weight vectors
    distances = pdist(flattened_weights, metric='euclidean')
    
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    # Define thresholds based on mean and std deviation
    aggressive_threshold = mean_distance - 0 * std_distance  # Aggressive
    medium_threshold = mean_distance - 0.5 * std_distance       # Medium
    loose_threshold = mean_distance - 1.0 * std_distance        # Loose
    thresholds = {'aggressive': aggressive_threshold,
                  'medium': medium_threshold,
                  'loose': loose_threshold}
    # Plot histogram
    plt.hist(distances, bins=50, alpha=0.75, label="Pairwise Distances")
    
    
    # Add vertical lines for thresholds
    aggressive_line =plt.axvline(aggressive_threshold, color='red', linestyle='--', linewidth=1, label='Aggressive')
    medium_line = plt.axvline(medium_threshold, color='orange', linestyle='--', linewidth=1, label='Medium')
    loose_line = plt.axvline(loose_threshold, color='green', linestyle='--', linewidth=1, label='Loose')
    
    # Add text for each threshold value
    ymin, ymax = plt.ylim()  # Get y-axis limits to position text
    plt.text(aggressive_threshold, ymax * 0.5, f'{aggressive_threshold:.2f}', color='red', fontsize=10, ha='right', rotation=90)
    plt.text(medium_threshold, ymax * 0.3, f'{medium_threshold:.2f}', color='orange', fontsize=10, ha='right', rotation=90)
    plt.text(loose_threshold, ymax * 0.1, f'{loose_threshold:.2f}', color='green', fontsize=10, ha='right', rotation=90)

    plt.xlabel("Pairwise Distance")
    plt.ylabel("Frequency")
    plt.title("Histogram of SOM Pairwise Distances")
    plt.legend()
    plt.show()
    
    # Convert to a square form distance matrix
    distance_matrix = squareform(distances)
    
    return distances, distance_matrix, thresholds

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

def som_merge(hit_histogram,
              som_weights,
              cluster_index,
              stm_scan: STMScan,
              grid_type='hex',
              dimensions: Tuple = None,
              colormap: str = 'Grays',
              offset=0.01,
              threshold=0.1,
              num_ax_per_row=4,
              num_spectra_per_ax=5,
              savefig=None):
    # Determine threshold based on string input
    if isinstance(threshold, str):
        _, _, thresholds = hist_som_distances(som_weights)
        threshold = thresholds[threshold]
    
    hit_histogram = adjust_dimensions2d(hit_histogram, dimensions)
    dimensions = hit_histogram.shape
    n, m = dimensions
    if som_weights.ndim != 3:
        som_weights = som_weights.reshape(dimensions+(-1,))
    if cluster_index.ndim != 2:
        cluster_index = cluster_index.reshape(stm_scan.nx, stm_scan.ny)
    colormap = plt.get_cmap(colormap)
    padded_hit_histogram = np.pad(
        hit_histogram, ((1, 1), (1, 1)), 'constant', constant_values=0)
    coordinates = peak_local_max(padded_hit_histogram, min_distance=1)
    coordinates = np.array([[i-1, j-1] for i, j in coordinates])
    
    visited = set()
    merged_nodes = set()
    merge_dict = {tuple((i, j)): [tuple((i, j))] for i in range(n) for j in range(m)}
    
    def find_parent(coord):
        """Find the parent node or cluster in merge_dict."""
        for parent, merged_nodes in merge_dict.items():
            if tuple(coord) in merged_nodes:
                return parent
        return tuple(coord)  # If not found, treat it as its own parent
    
    def weighted_average_weight(coord):
        """Compute the weighted average weight for a given coordinate."""
        parent = find_parent(coord)
        coords = merge_dict[parent]  # Use parent to get the cluster
        total_hits = sum(hit_histogram[c] for c in coords)
        return sum(som_weights[c] * hit_histogram[c] for c in coords) / total_hits
    
    def merge(coord):
        stack = [coord]
        while stack:
            current = stack.pop()
            visited.add(tuple(current))
            # current_hits = hit_histogram[tuple(current)]
            neighbors = get_neighbors(current, grid_type=grid_type, dimension=dimensions)
            
            neighbors = sorted(neighbors, key=lambda x: hit_histogram[tuple(x)], reverse=True)
            for neighbor in neighbors:
                if tuple(neighbor) in visited or hit_histogram[tuple(neighbor)] == 0:
                    continue
                
                avg_weight_current = weighted_average_weight(current)
                avg_weight_neighbor = weighted_average_weight(neighbor)
                
                dist = euclidean(avg_weight_current, avg_weight_neighbor)
                
                if dist < threshold:
                    # Merge clusters by extending merge_dict
                    parent_current = find_parent(current)
                    parent_neighbor = find_parent(neighbor)
                    if parent_neighbor != parent_current:  # Avoid self-merge
                        # Add newly merged nodes to merged_nodes
                        merged_nodes.update(merge_dict[parent_neighbor])
                        merged_nodes.update(merge_dict[parent_current])
                        
                        merge_dict[parent_current].extend(merge_dict.pop(parent_neighbor))

                    stack.append(neighbor)

    for coord in coordinates:
        if tuple(coord) not in visited:
            merge(coord)
    
    for i in range(n):
        for j in range(m):
            coord = (i, j)
            if coord not in visited and coord not in merged_nodes:
                merge([i, j])
    n_spectra = len(merge_dict)
    # Assign colors to merged groups
    colors = generate_colors_with_yellow_last(n_spectra)
    group_colors = {key: color for key, color in zip(merge_dict.keys(), colors)}
    # Plot merged nodes with consistent colors
    patch_list = []
    text_list = []
    radius = 1.8 / (4 * np.cos(np.pi / 6))
    normalized_hist = hit_histogram / hit_histogram.max()
    normalized_hist = np.ones_like(hit_histogram)
    radii_scaled = radius * normalized_hist
    txt_size = max(5, 20 - max(dimensions) // 2)
    fig, ax0 = plt.subplots(1, 1, figsize=(10, 10))
    for irow in range(dimensions[0]):
        for icolumn in range(dimensions[1]):
            radius_scaled = radii_scaled[irow, icolumn]
            x, y = irow, icolumn * 0.9
            if icolumn % 2 != 0:
                x += 0.5
            hex = patches.RegularPolygon((x, y),
                                         numVertices=6,
                                         radius=radius,
                                         facecolor=GRAY,
                                         alpha=0.7,
                                         edgecolor=distinctipy.BLACK)
            patch_list.append(hex)

            hex = patches.RegularPolygon((x, y),
                                         numVertices=6,
                                         radius=radius_scaled,
                                         facecolor=distinctipy.BLACK)

            patch_list.append(hex)

            for group, color in group_colors.items():
                if tuple((irow, icolumn)) in merge_dict[group]:
                    hex = patches.RegularPolygon((x, y),
                                                 numVertices=6,
                                                 radius=radius_scaled,
                                                 alpha=1.0,
                                                 facecolor=color)
                    patch_list.append(hex)
                    text_list.append([x, y, int(hit_histogram[irow, icolumn]), color])
                    break

    p = PatchCollection(patch_list, match_original=True)
    ax0.add_collection(p)

    for txt in text_list:
        x, y, num, color = txt
        ax0.text(x, y, num, ha='center', va='center', size=txt_size, color=distinctipy.get_text_color(color))

    ax0.set_xlim(-1, dimensions[0] + 0.5)
    ax0.set_ylim(-1, dimensions[1] * 0.9)
    ax0.set_aspect('equal')
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.grid(False)
    
    total_hits_all_nodes =  np.sum(hit_histogram)
    percentages = {}
    percentages = {}
    for parent, nodes in merge_dict.items():
        parent_hits = sum([hit_histogram[node] for node in nodes])
        percentages[parent] = 100 * parent_hits / total_hits_all_nodes

    num_spectra_per_row = num_spectra_per_ax*num_ax_per_row
    n_row = n_spectra//num_spectra_per_row+1
    n_col = num_ax_per_row
    fig, axes = plt.subplots(n_row, n_col , figsize=(10, 5*n_row), sharey=True, sharex=True)
    axes = axes.ravel()
    merge_dict = dict(sorted(merge_dict.items(), 
                            key=lambda item: sum(hit_histogram[node] for node in item[1]), 
                            reverse=True))
    num_spectra_per_ax = n_spectra//len(axes)
    V = stm_scan.V*1e3
    plots = [{} for x in range(len(axes))]
    for i, (parent, nodes) in enumerate(merge_dict.items()):
        # Use the weighted_average_weight function to compute the cluster's spectrum
        weighted_spectrum = weighted_average_weight(parent)
        # Plot the weighted spectrum using the parent cluster's color
        idx = i//num_spectra_per_ax
        percentage_label = f"{percentages[parent]:.1f}%"


        if idx >= n_row*n_col:
            idx=-1
        axes[idx].plot(V, 
                   weighted_spectrum + (i % num_spectra_per_ax) * offset, 
                   linewidth=0.75, 
                   color=group_colors[parent],
                   label=f'{percentage_label}')
        plots[idx][parent]=weighted_spectrum
        # midpoint = len(stm_scan.V) // 2

        # axes[idx].text(stm_scan.V[midpoint], 
        #            weighted_spectrum[midpoint] + (i % num_spectra_per_ax) * offset,
        #            f'{percentage_label}', fontsize=8, color=group_colors[parent])
    for ax in axes:
        ax.legend(fontsize=8, loc='upper center')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0)
        ax.set_xlim(V[0], V[-1])
    for ax in axes:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("dI/dV (a.u.)")
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Bias (mV)")
    plt.tight_layout()
    plt.show()
    if savefig is not None:
        plt.savefig(savefig)
        plt.clf()
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    # norm = plt.Normalize(vmin=np.min(
    #     stm_scan.topography), vmax=np.max(stm_scan.topography))
    # masked = colormap(norm(stm_scan.topography))[:, :, :3]
    masked = np.zeros(shape=(stm_scan.nx, stm_scan.ny, 3))
    for counter, parent in enumerate(merge_dict):
        irow, icolumn = parent
        flat_index = np.ravel_multi_index([irow, icolumn], dimensions)
        idxs = np.where(cluster_index == flat_index)
        masked[idxs] = group_colors[parent]
        for irow, icolumn in merge_dict[parent]:
            flat_index = np.ravel_multi_index([irow, icolumn], dimensions)
            idxs = np.where(cluster_index == flat_index)
            masked[idxs] = group_colors[parent]
    ax1.imshow(masked, cmap='YlOrBr', extent=[
        0, stm_scan.dimensions[1]/1e-9, 0, stm_scan.dimensions[0]/1e-9])

    # scalebar = ScaleBar(1, units='nm', dimension="si-length", length_fraction=0.4,
    #                     location='lower right', box_alpha=0, scale_loc='top')
    # ax1.add_artist(scalebar)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.grid(False)
    plt.show()


    for x in plots:    
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        norm = plt.Normalize(vmin=np.min(
            stm_scan.topography), vmax=np.max(stm_scan.topography))
        masked = colormap(norm(stm_scan.topography))[:, :, :3]
        for i, (parent, weighted_spectrum) in enumerate(x.items()):
            percentage_label = f"{percentages[parent]:.1f}%"
            axes[0].plot(V, 
                weighted_spectrum + i* offset, 
                linewidth=0.75, 
                color=group_colors[parent],
                label=f'{percentage_label}')
            
            axes[0].legend(fontsize=8, loc='upper center')
            axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1.0)
            axes[0].set_xlim(V[0], V[-1])
            axes[0].set_ylabel("dI/dV (a.u.)")
            axes[0].set_xlabel("Bias (mV)")

            irow, icolumn = parent
            flat_index = np.ravel_multi_index([irow, icolumn], dimensions)
            idxs = np.where(cluster_index == flat_index)
            masked[idxs] = group_colors[parent]
            for irow, icolumn in merge_dict[parent]:
                flat_index = np.ravel_multi_index([irow, icolumn], dimensions)
                idxs = np.where(cluster_index == flat_index)
                masked[idxs] = group_colors[parent]
            axes[1].imshow(masked, cmap='YlOrBr', extent=[
                0, stm_scan.dimensions[1]/1e-9, 0, stm_scan.dimensions[0]/1e-9])
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            axes[1].grid(False)
            

    plt.show()
            
        

    

    return merge_dict


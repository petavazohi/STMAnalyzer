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
from itertools import product
from matplotlib_scalebar.scalebar import ScaleBar
import numpy.typing as npt
from ..utils.arrays import adjust_dimensions2d
from ..core.scan import STMScan

GRAY = (0.5, 0.5, 0.5)

def hit_histogram(hit_histogram, dimensions: Tuple=None, plot_spectra: bool=True, V=None, som_weights=None, offset=0.0):

    hit_histogram = adjust_dimensions2d(hit_histogram, dimensions)
    dimensions = hit_histogram.shape
    if som_weights is not None and som_weights.ndim !=3:
        som_weights = som_weights.reshape(dimensions+(-1,))
    padded_hit_histogram = np.pad(hit_histogram, ((1, 1), (1, 1)), 'constant', constant_values=0)
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
    radii_scaled = radius * log_scaled_hist
    # radii_scaled = radius * normalized_hist
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
                color = BG_COLOR # gray
            else:
                color= FG_COLOR  # distinctipy.BLACK
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
        x, y, num, color= txt
        ax0.text(x, y, num, ha='center', va='center', size=txt_size, color=distinctipy.get_text_color(color))

    ax0.set_xlim(-1, dimensions[0]+0.5)
    ax0.set_ylim(-1, dimensions[1]*0.9)
    ax0.set_aspect('equal')
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.grid(False)
    if plot_spectra:
        counter = 0
        for i, coord in enumerate(coordinates):
            # V = V*1e3 #TODO this might need to adjusted.
            ax1.plot(V, som_weights[coord[0], coord[1]]+i*offset, linewidth=0.75, color=colors[counter])
            counter += 1
        ax1.set_ylim(0,)
        ax1.set_xlim(V[0], V[-1])
        ax1.set_xlabel("Bias (mV)")
        ax1.set_ylabel("dI/dV (a.u.)")

    # plt.savefig(base_dir / "Python" / 'SOM-peaks.pdf'
    #             )
    plt.show()
    return 

def som_didv_topo(hit_histogram,
                  som_weights,
                  cluster_index,
                  stm_scan: STMScan,
                  block_size=(1, -1),
                  dimensions: Tuple=None,
                  colormap: str='Grays',
                  offset=0.01):
    origin = 'lower'
    hit_histogram = adjust_dimensions2d(hit_histogram, dimensions)
    dimensions = hit_histogram.shape
    n, m = dimensions
    if som_weights.ndim !=3:
        som_weights = som_weights.reshape(dimensions+(-1,))
    if cluster_index.ndim !=2:
        cluster_index = cluster_index.reshape(stm_scan.nx, stm_scan.ny)
    if block_size[1] == -1:
        block_size = (block_size[0], dimensions[1])
    colormap = plt.get_cmap(colormap)

    k, l = block_size
    for i in range(0, n, k):
        for j in range(0, m, l):
            coordinates = [(x, y) for x, y in product(range(i, min(i + k, dimensions[0])), range(j, min(j + l, dimensions[1])))]
            colors = distinctipy.get_colors(k*l)
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(20, 6))
            hex_separation = 1.8
            radius = hex_separation/(4*np.cos(np.pi/6))
            max_hit = hit_histogram.max()
            normalized_hist = hit_histogram / max_hit
            normalized_hist = np.clip(normalized_hist, 1e-10, 1)  # Avoid log(0) issues
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
                        color = BG_COLOR # gray
                    else:
                        color= FG_COLOR  # distinctipy.BLACK
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
            for txt in text_list:
                x, y, num, color= txt
                ax0.text(x, y, num, ha='center', va='center', size=txt_size, color=distinctipy.get_text_color(color))

            ax0.set_xlim(-1, dimensions[0]+0.5)
            ax0.set_ylim(-1, dimensions[1]*0.9)
            ax0.set_aspect('equal')
            ax0.set_xticks([])
            ax0.set_yticks([])
            for counter, coord in enumerate(coordinates):
                # V = V*1e3 #TODO this might need to adjusted.
                ws = som_weights[coord[0], coord[1]]
                ax1.plot(stm_scan.V, ws+offset*counter, linewidth=0.75, color=colors[counter])
            ax1.set_ylim(0,)
            ax1.set_xlim(stm_scan.V[0], stm_scan.V[-1])
            ax1.set_xlabel("Bias (mV)")
            ax1.set_ylabel("dI/dV (a.u.)")
            ax0.grid(False)
            norm = plt.Normalize(vmin=np.min(stm_scan.topography), vmax=np.max(stm_scan.topography))
            
            masked = colormap(norm(stm_scan.topography))[:, :, :3] # [:, :, :3] to get rid of the 4th column (color alpha)
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
            plt.show()
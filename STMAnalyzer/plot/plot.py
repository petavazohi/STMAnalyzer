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
import numpy.typing as npt

GRAY = (0.5, 0.5, 0.5)
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)

def hit_histogram(hit_histogram, dimensions: Tuple=None, plot_spectra: bool=True, V=None, som_weights=None):
    if dimensions is None:
        print('Dimensions not provided. Guessing ...')
        n = np.sqrt(max(hit_histogram.shape))
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
    if som_weights is not None:
        som_weights = som_weights.T.reshape(dimensions+(-1,))
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
    FG_COLOR = BLACK
    counter = 0
    if plot_spectra: 
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
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
                                        edgecolor=BLACK)
            patch_list.append(hex)

            # Foreground color
            hex = patches.RegularPolygon((x, y),
                                        numVertices=6,
                                        radius=radius_scaled,
                                        facecolor=FG_COLOR,
                                        # alpha=1,
                                        # edgecolor=BLACK,
                                        )

            patch_list.append(hex)
            # Local maxima edge color
            if radius_scaled*100 >= txt_size:
                # print(radius_scaled, txt_size)
                color = BG_COLOR # gray
            else:
                color= FG_COLOR  # black
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
    ax.add_collection(p)


    for i, txt in enumerate(text_list):
        x, y, num, color= txt
        ax.text(x, y, num, ha='center', va='center', size=txt_size, color=distinctipy.get_text_color(color))

    ax.set_xlim(-1, dimensions[0]+0.5)
    ax.set_ylim(-1, dimensions[1]*0.9)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if plot_spectra:
        counter = 0
        for coord in coordinates:
            # V = V*1e3 #TODO this might need to adjusted.
            ax2.plot(V, som_weights[coord[0], coord[1]], linewidth=0.75, color=colors[counter])
            counter += 1
        ax2.set_ylim(0,)
        ax2.set_xlim(V[0], V[-1])
        ax2.set_xlabel("Bias (mV)")
        ax2.set_ylabel("dI/dV (a.u.)")
        ax.grid(False)
    # plt.savefig(base_dir / "Python" / 'SOM-peaks.pdf'
    #             )
    plt.show()
    return 

# def som_didv_topo(hit_histogram, som_weights, stm_scan, dimensions: Tuple=None):
#         if dimensions is None and :
#         print('Dimensions not provided. Guessing ...')
#         n = np.sqrt(max(hit_histogram.shape))
#         if n.is_integer():
#             n = int(n)
#             dimensions= (n, n)
#         else:
#             raise ValueError(
#                 f"Hit histogram is not squarable. Length: {len(hit_histogram)}. "
#                 f"Expected a perfect square."
#             )
#         print(f"SOM dimensions selected as {n}x{n}")
#     hit_histogram = np.reshape(hit_histogram, dimensions)
#     if som_weights is not None:
#         som_weights = som_weights.T.reshape(dimensions+(-1,))

# for flat_index in range(settings['dimension1']*settings['dimension2']):
#     irow, icolumn = np.unravel_index(flat_index, (settings['dimension1'], settings['dimension2']))
#     masked = np.where(matlab_output['clusterIndex'] != flat_index,  topography, np.nan)
#     avg = np.average(dIdV[matlab_output['clusterIndex'] == flat_index],axis=0)    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
#     ax2.imshow(masked, origin='lower', cmap='YlOrBr', extent=[0, 10, 0, 10])
#     ax2.grid(False)
#     scalebar = ScaleBar(1, "nm", dimension="si-length", length_fraction=0.25,
#                     location='lower right', box_alpha=0, scale_loc='top')
#     ax2.add_artist(scalebar)
#     # ax2.set_title(f'{int(hit_histogram[irow, icolumn])}- ({9-irow},{icolumn})')
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     for curve in dIdV[matlab_output['clusterIndex'] == flat_index]:
#         curve = (curve - curve.min()) / (curve.max() - curve.min())
#         line0, = ax1.plot(V*100, curve, color='C6', linewidth=0.4, label='Raw')
#     line1, = ax1.plot(V*-100, som_weights[flat_index, :],label='Learned')
#     avg =  (avg - avg.min()) / (avg.max() - avg.min())
#     line2, = ax1.plot(V*100, avg,color='C1',label='Average',linewidth=1)
#     print(f'{int(hit_histogram[irow, icolumn])}- ({irow},{icolumn})')
#     # ax1.set_title(f'{int(hit_histogram[irow, icolumn])}- ({irow},{icolumn})')
#     ax1.set_xlabel('Bias V (meV)')
#     ax1.set_ylabel('dI/dV (a.u.)')
#     ax1.set_xlim(-500, 500)
#     ax1.set_ylim(0,)
#     ax1.grid(False)
#     ax1.legend(handles=[line0, line1, line2], ncol=3)
#     plt.show()
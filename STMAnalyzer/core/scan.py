import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
import nanonispy.read as nap
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import exposure
import numpy.typing as npt
from re import findall
color_palette = [key for key in mcolors.BASE_COLORS if key != 'w']

class STMScan:
    def __init__(self,
                 topography: npt.ArrayLike,
                 dIdV: npt.ArrayLike,
                 I: npt.ArrayLike,
                 V: npt.ArrayLike,
                 dIdV_imaginary: npt.ArrayLike = None,
                 params: npt.ArrayLike = None,
                 dimensions: tuple = None,
                 metadata: dict = None,
                 ):
        self.topography = topography
        self.topography_unit8 = np.uint8(
            255 * (topography - np.min(topography)) / (np.max(topography) - np.min(topography)))
        self.dIdV = dIdV
        self.I = I
        self.V = V
        self.dIdV_imaginary = dIdV_imaginary
        self.params = params
        self.dimensions = dimensions
        self.nx, self.ny, self.nE = self.dIdV.shape
        self.metadata = metadata

    def histogram_equalization(self, clip_limit: float = 0.03) -> np.ndarray:
        self.topography_unit8 = 255 - exposure.equalize_adapthist(
            self.topography_unit8,
            clip_limit=clip_limit)
        topography_unit8 = self.topography_unit8
        self.topography = np.float_((topography_unit8 - np.min(topography_unit8)) / (
            np.max(topography_unit8) - np.min(topography_unit8)))
        return

    def flatten_topography(self, direction: str = 'x') -> np.ndarray:
        background = np.zeros(shape=(self.nx, self.ny))
        # should change to scan dirrection for generalization
        if direction == 'x':
            for row, line in enumerate(self.topography):
                pfit = np.polyfit(np.arange(self.ny), line, 2)
                background[row, :] = np.polyval(pfit, np.arange(self.ny))
        else:
            for column, line in enumerate(self.topography.T):
                pfit = np.polyfit(np.arange(self.nx), line, 2)
                background[:, row] = np.polyval(pfit, np.arange(self.nx))
        self.topography -= background
        return

    def normalize_to_setpoint(self):
        # self.dIdV = self.dIdV/np.array([self.dIdV[:, :, 0]] *
        #                       self.nE).reshape(self.nx, self.ny, self.nE)
        for ix in range(self.nx):
            for iy in range(self.ny):
                self.dIdV[ix, iy, :] = self.dIdV[ix, iy, :] / self.dIdV[ix, iy, 0]
        return

    @classmethod
    def from_file(cls, file_path: Path | str):
        # check extension if 3ds:
        file_path = Path(file_path)
        if file_path.suffix == '.3ds':
            print(f"Reading {file_path.name}")
            grid = nap.Grid(file_path.as_posix())
            try:
                a, b = findall(r',\s([0-9]*)/([0-9]*),',
                            grid.header['comment'])[0]
            except:
                raise ValueError(f"Could not find the required information in the file header: {grid.header['comment']}")
            divider = float(a)/float(b)
            if file_path.name == 'GridSpectroscopy-071-07182022001.3ds':
                divider = 0.1
            size = grid.header['size_xy']
            dIdV_key = [key for key in grid.signals.keys() if 'X' in key][0]
            dIdV = grid.signals[dIdV_key]
            V = grid.signals['sweep_signal']
            print(f'sweep signal range: {V[-1]}, {V[0]}')
            print(f'Divider {divider}')
            V *= divider
            # V = grid.signals['sweep_signal']
            topography = grid.signals['topo']
            current_key = [key for key in grid.signals.keys()
                           if "Current" in key][0]
            I = grid.signals[current_key]
            grid.header['file_path'] = file_path.as_posix()
        else:
            print("Functionality of this file has not been implemented")
        # add for other types of files
        return cls(
            topography=topography,
            dIdV=dIdV,
            I=I,
            V=V,
            dimensions=size,
            metadata=grid.header
        )

    def pixel_to_location(self, matrix: np.ndarray) -> np.ndarray:
        """
        Rescale elements of a matrix from one numeric range to another.

        Parameters:
        matrix (np.ndarray): The 2D matrix of numeric values to be rescaled.
        from_range (tuple): A tuple representing the original range of values, (min, max).
        to_range (tuple): A tuple representing the target range of values, (min, max).

        Returns:
        np.ndarray: A 2D matrix where each element has been rescaled to the target range.
        """
        from_range = [0, self.nx]
        to_range = [0, self.dimensions[0]/1e-9]
        return (matrix - from_range[0]) * (to_range[1] - to_range[0]) / (from_range[1] - from_range[0]) + to_range[0]

    def plot_topography(self, cmap='YlOrBr', ax=None):
        if not ax:
            _, ax = plt.subplots(1, 1)
        print(self.topography.shape)
        print(self.dimensions)
        ax.imshow(self.topography, cmap='YlOrBr'), extent=[
                  0, self.dimensions[1]/1e-9, 0, self.dimensions[0]/1e-9])
        scalebar = ScaleBar(1, units='nm', dimension="si-length", length_fraction=0.4,
                            location='lower right', box_alpha=0, scale_loc='top')
        ax.add_artist(scalebar)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def plot_dIdV(self, n_random=8, locations: list = None):
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax2 = self.plot_topography(ax=ax2)
        points = []
        V = self.V*1e3
        if locations is None:
            for x in range(n_random):
                index1 = np.random.randint(self.dIdV.shape[0])
                index2 = np.random.randint(self.dIdV.shape[1])
                points.append([index1, index2])
                ax1.plot(V, self.dIdV[index1, index2, :],
                         label=f'pixel location: ({index1}, {index2})', linewidth=1.0)
        else:
            for x in locations:
                index1 = x[0]
                index2 = x[1]
                points.append([index1, index2])
                ax1.plot(V, self.dIdV[index1, index2, :],
                         label=f'pixel location: ({index1}, {index2})', linewidth=1.0)
            n_random = len(locations)
        points = self.pixel_to_location(np.array(points))
        ax2.scatter(points[:, 0], points[:, 1], color=(
            color_palette*4)[:n_random], s=16.0)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.0)
        ax1.set_xlim(V[-1], V[0])
        ax1.set_xlabel("Bias (mV)")
        ax1.set_ylabel("dI/dV (a.u.)")


    def __str__(self):
        ret = ''
        ret += f'STS data grid: {self.nx}, {self.ny}\n'
        ret += f'Energy range: ({self.V[-1]*1e3} mV, {self.V[0]*1e3} mV) sampled with {self.nE} points'
        return ret
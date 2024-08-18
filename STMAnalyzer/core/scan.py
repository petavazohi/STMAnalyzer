import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nanonispy.read as nap
import numpy as np
import numpy.typing as npt
from pathlib import Path
from re import findall
from scipy.ndimage import zoom
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple, Union
from matplotlib_scalebar.scalebar import ScaleBar

color_palette = [key for key in mcolors.BASE_COLORS if key != 'w']


class STMScan:
    """
    Class for handling and analyzing STM scan data.
    """

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
        """
        Initialize an STMScan object.

        Parameters:
        topography (npt.ArrayLike): Topography data array.
        dIdV (npt.ArrayLike): dI/dV data array.
        I (npt.ArrayLike): Current data array.
        V (npt.ArrayLike): Voltage data array.
        dIdV_imaginary (npt.ArrayLike, optional): Imaginary part of dI/dV data. Default is None.
        params (npt.ArrayLike, optional): Additional parameters. Default is None.
        dimensions (tuple, optional): Spatial dimensions of the scan. Default is None.
        metadata (dict, optional): Metadata dictionary. Default is None.
        """
        self.topography = topography
        self.topography_unit8 = np.uint8(
            255 * (topography - np.min(topography)) / (np.max(topography) - np.min(topography)))
        self.dIdV = dIdV
        self.I = I
        self.V = V
        self.dIdV_imaginary = dIdV_imaginary
        self.params = params
        self.dimensions = dimensions
        self.metadata = metadata
        
    @property
    def nE(self):
        return self.dIdV.shape[2]
    
    @property    
    def ny(self):
        return self.dIdV.shape[1] 
    
    @property
    def nx(self):
        return self.dIdV.shape[0]

    def interpolate(self, zoom_factor: float):
        """
        Interpolates the data arrays by a given zoom factor.

        Parameters:
        zoom_factor (float): The factor by which to scale the spatial dimensions.
        """

        # Interpolate dIdV
        self.dIdV = zoom(self.dIdV, (1, 1, zoom_factor))
        nE = self.dIdV.shape[2]
        # Interpolate I
        self.I = zoom(self.I, (1, 1, zoom_factor))
        self.V = zoom(self.V, (zoom_factor,))
        # Optional: Interpolate the imaginary part of dIdV, if present
        if self.dIdV_imaginary is not None:
            self.dIdV_imaginary = zoom(
                self.dIdV_imaginary, (1, 1, zoom_factor))


    def crop_sts(self, V_limits: Tuple[float, float]):
        """
        Crops the data arrays based on a range of voltage limits.

        Parameters:
        V_limits (Tuple[float, float]): The lower and upper voltage limits.
        """
        idx1, idx2 = np.sort([np.argmin(np.abs(self.V - V_limits[0])),
                              np.argmin(np.abs(self.V - V_limits[1]))])

        self.V = self.V[idx1:idx2]
        self.I = self.I[idx1:idx2]
        self.dIdV = self.dIdV[:, :, idx1:idx2]

    def resample(self, nE: int, V_limits: Tuple[float, float]):
        """
        Resamples the data to a new number of energy points within a specific voltage range.

        Parameters:
        nE (int): The number of energy points to resample to.
        V_limits (Tuple[float, float]): The lower and upper voltage limits for cropping.
        """
        if min(V_limits) < min(self.V) or max(V_limits) > max(self.V):
            raise (ValueError(
                "Voltage limits are outside the range of the provided data."))
        zoom_factor = nE/self.nE
        print(f'old nE: {self.nE}, new nE: {nE}, zoom_factor: {zoom_factor}')
        
        self.crop_sts(V_limits=V_limits)
        self.interpolate(zoom_factor=zoom_factor)
        print(f'nE after resampling : {self.nE}')
        print('='*100)

    def histogram_equalization(self, clip_limit: float = 0.03):
        """
        Performs adaptive histogram equalization on the topography data for contrast enhancement.

        Parameters:
        clip_limit (float): Clipping limit for contrast enhancement. Default is 0.03.
        """
        self.topography_unit8 = 255 - exposure.equalize_adapthist(
            self.topography_unit8,
            clip_limit=clip_limit)
        topography_unit8 = self.topography_unit8
        self.topography = np.float_((topography_unit8 - np.min(topography_unit8)) / (
            np.max(topography_unit8) - np.min(topography_unit8)))
        return

    def flatten_topography(self, direction: str = 'x'):
        """
        Flattens the topography data by fitting a polynomial to the background.

        Parameters:
        direction (str): Direction along which to flatten the topography ('x' or 'y'). Default is 'x'.
        """
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
        """
        Normalizes the dI/dV data to the initial setpoint value at each pixel.
        """
        for ix in range(self.nx):
            for iy in range(self.ny):
                self.dIdV[ix, iy, :] = self.dIdV[ix,
                                                 iy, :] / self.dIdV[ix, iy, 0]
        return

    @classmethod
    def from_file(cls, file_path: Path | str):
        """
        Initializes an STMScan object from a file.

        Parameters:
        file_path (Union[Path, str]): The path to the file.

        Returns:
        STMScan: An STMScan object initialized from the file.
        """
        file_path = Path(file_path)

        if file_path.suffix == '.3ds':
            print(f"Reading {file_path.name}")
            grid = nap.Grid(file_path.as_posix())
            try:
                a, b = findall(r',\s([0-9]*)/([0-9]*),',
                               grid.header['comment'])[0]
            except:
                raise ValueError(f"Could not find the required information in the file header: {
                                 grid.header['comment']}")
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
        Converts pixel coordinates to real-space locations in nanometers.

        Parameters:
        matrix (np.ndarray): A matrix of pixel coordinates.

        Returns:
        np.ndarray: Matrix of locations in real-space (nm).
        """
        from_range = [0, self.nx]
        to_range = [0, self.dimensions[0]/1e-9]
        return (matrix - from_range[0]) * (to_range[1] - to_range[0]) / (from_range[1] - from_range[0]) + to_range[0]

    def plot_topography(self, cmap='YlOrBr', ax=None):
        """
        Plots the topography data.

        Parameters:
        cmap (str, optional): Colormap to use for the plot. Default is 'YlOrBr'.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Default is None.

        Returns:
        matplotlib.axes.Axes: The axes with the plotted topography.
        """
        if not ax:
            _, ax = plt.subplots(1, 1)
        print(self.topography.shape)
        print(self.dimensions)
        ax.imshow(self.topography, cmap='YlOrBr', extent=[
                  0, self.dimensions[1]/1e-9, 0, self.dimensions[0]/1e-9])
        scalebar = ScaleBar(1, units='nm', dimension="si-length", length_fraction=0.4,
                            location='lower right', box_alpha=0, scale_loc='top')
        ax.add_artist(scalebar)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def plot_dIdV(self, n_random=8, locations: list = []):
        """
        Plots random or specified pixel locations' dI/dV data and highlights their positions on the topography.

        Parameters:
        n_random (int, optional): Number of random pixel locations to plot. Default is 8.
        locations (list, optional): List of specific locations to plot. Default is empty.

        Returns:
        list: List of pixel locations plotted.
        """
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax2 = self.plot_topography(ax=ax2)
        points = []
        V = self.V*1e3
        if len(locations) == 0:
            for x in range(n_random):
                index1 = np.random.randint(self.dIdV.shape[0])
                index2 = np.random.randint(self.dIdV.shape[1])
                locations.append([index1, index2])
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
        points = self.pixel_to_location(np.array(locations))
        ax2.scatter(points[:, 0], points[:, 1], color=(
            color_palette*4)[:n_random], s=16.0)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.0)
        ax1.set_xlim(V[-1], V[0])
        ax1.set_xlabel("Bias (mV)")
        ax1.set_ylabel("dI/dV (a.u.)")
        return locations

    def __str__(self):
        """
        Returns a string representation of the STMScan object summarizing the grid and energy range.
        """
        ret = ''
        ret += f'STS data grid: {self.nx}, {self.ny}\n'
        ret += f'Energy range: ({self.V[-1]*1e3} mV, {self.V[0]
                                                      * 1e3} mV) sampled with {self.nE} points'
        return ret

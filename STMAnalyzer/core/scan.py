import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nanonispy.read as nap
import numpy as np
import numpy.typing as npt
from pathlib import Path
from re import findall
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
from skimage import exposure
from skimage.util import invert
from typing import List, Optional, Tuple, Union
from matplotlib_scalebar.scalebar import ScaleBar
import copy 
import distinctipy
import matplotlib.ticker as ticker

color_palette = [key for key in mcolors.BASE_COLORS if key != 'w']


class STMScan:
    """
    Class for handling and analyzing STM scan data.
    """

    def __init__(self,
                 dIdV_integral : npt.ArrayLike,
                 dIdV: npt.ArrayLike,
                 I: npt.ArrayLike,
                 V: npt.ArrayLike,
                 topography: npt.ArrayLike = None,
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
        self.dIdV_integral = dIdV_integral
        self.params = params
        self.dimensions = dimensions
        self.metadata = metadata
        self._fix_V_direction()
        
    def _fix_V_direction(self):
        if self.V[0] > self.V[-1]:
            self.V = np.flip(self.V)
            self.I = np.flip(self.I, axis=2)
            self.dIdV = np.flip(self.dIdV, axis=2)
            if self.dIdV_imaginary is not None:
                self.dIdV_imaginary = np.flip(self.dIdV_imaginary, axis=2)
            
    
    @property
    def v_range(self):
        return self.V.max() - self.V.min()
    
    @property 
    def dIdV_range(self):
        return self.dIdV.max() - self.dIdV.min()
    
    @property
    def nE(self):
        return self.dIdV.shape[2]
    
    @property    
    def ny(self):
        return self.dIdV.shape[1] 
    
    @property
    def nx(self):
        return self.dIdV.shape[0]

    def resample(self, nE: int,  V_limits: Tuple[float, float], kind: str='cubic'):
        """
        Interpolates the data arrays (dIdV, I, etc.) to match the target number of energy points.

        Parameters:
        nE (int): The target number of energy points after interpolation.
        kind (str): The type of interpolation to use. Default is 'linear'. 
                    Other options include 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.
        """
        # Ensure V_limits is a tuple for comparison

        # Only perform cropping if the input limits differ from the current range
        new_V = np.linspace(V_limits[0], V_limits[1], nE)
        # Initialize new arrays for the interpolated data
        new_dIdV = np.zeros(shape=(self.nx, self.ny, nE))
        new_I = np.zeros(shape=(self.nx, self.ny, nE))
        
        # Interpolate each (x, y) point separately
        for ix in range(self.nx):
            for iy in range(self.ny):
                # Interpolate dIdV
                f_dIdV = interp1d(self.V, self.dIdV[ix, iy, :], kind=kind)
                new_dIdV[ix, iy, :] = f_dIdV(new_V)
                
                # Interpolate I
                f_I = interp1d(self.V, self.I[ix, iy, :], kind=kind)
                new_I[ix, iy, :] = f_I(new_V)

        # Assign the new data to the instance
        self.dIdV = new_dIdV
        self.I = new_I
        self.V = new_V
        
        # Update nE to the new target number of energy points
        
        # Optional: Interpolate the imaginary part of dIdV, if present
        if self.dIdV_imaginary is not None:
            new_dIdV_imaginary = np.zeros((self.nx, self.ny, nE))
            for ix in range(self.nx):
                for iy in range(self.ny):
                    f_dIdV_imaginary = interp1d(self.V, self.dIdV_imaginary[ix, iy, :], kind=kind)
                    new_dIdV_imaginary[ix, iy, :] = f_dIdV_imaginary(new_V)
            self.dIdV_imaginary = new_dIdV_imaginary

    def interpolate_zoom(self, nE: int):
        """
        Interpolates the data arrays by a given zoom factor.

        Parameters:
        zoom_factor (float): The factor by which to scale the spatial dimensions.
        """
        if nE != self.nE:
            zoom_factor = nE/self.nE
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

    @property
    def V_limits(self) -> Tuple[float, float]:
        """
        Returns the voltage limits as a tuple.

        Returns:
        Tuple[float, float]: The minimum and maximum voltage values.

        Example:
        >>> stm_scan.V_limits
        (-0.1, 0.1)
        """
        return min(self.V), max(self.V)

    def crop_sts(self, V_limits: Union[Tuple[float, float], List[float]]):
        """
        Crops the data arrays based on a range of voltage limits.

        Parameters:
        V_limits (Tuple[float, float] or List[float]): The lower and upper voltage limits.
        """
        # Ensure V_limits is a tuple for comparison
        V_limits = tuple(V_limits)

        # Only perform cropping if the input limits differ from the current range
        if V_limits != self.V_limits:
            # Find the indices for the cropping
            idx1, idx2 = np.sort([np.argmin(np.abs(self.V - V_limits[0])),
                                np.argmin(np.abs(self.V - V_limits[1]))])
            
            # Perform the cropping
            self.V = self.V[idx1:idx2+1]
            self.I = self.I[idx1:idx2+1]
            self.dIdV = self.dIdV[:, :, idx1:idx2+1]

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
                                                 iy, :] / self.dIdV[ix, iy, -1]
        return

    @classmethod
    def from_file(cls, sts_path: Path | str, stm_path: Path | str = None):
        """
        Initializes an STMScan object from a file.

        Parameters:
        file_path (Union[Path, str]): The path to the file.

        Returns:
        STMScan: An STMScan object initialized from the file.
        """
        sts_path = Path(sts_path)

        if sts_path.suffix == '.3ds':
            print(f"Reading {sts_path.name}")
            grid = nap.Grid(sts_path.as_posix())
            try:
                a, b = findall(r',\s([0-9]*)/([0-9]*),',
                               grid.header['comment'])[0]
            except:
                raise ValueError(f"Could not find the required information in the file header: {grid.header['comment']}")
            divider = float(a)/float(b)
            size = grid.header['size_xy']
            dIdV_key = [key for key in grid.signals.keys() if 'X' in key][0]
            dIdV = grid.signals[dIdV_key]
            V = grid.signals['sweep_signal']
            print(f'sweep signal range: {V[-1]}, {V[0]}')
            print(f'Divider {divider}')
            V *= divider
            # V = grid.signals['sweep_signal']
            dIdV_integral = grid.signals['topo']
            current_key = [key for key in grid.signals.keys()
                           if "Current" in key][0]
            I = grid.signals[current_key]
            grid.header['file_path'] = sts_path.as_posix()
        else:
            print("Functionality of this file has not been implemented")
        # add for other types of files
        if stm_path is not None:
            stm_path = Path(stm_path)
            stm_scan = nap.Scan(stm_path)
            z_forward = stm_scan.signals['Z']['forward']
            topography = z_forward

            # z_forward = exposure.rescale_intensity(z_forward, in_range='image')
            # topography = exposure.equalize_hist(topography)
            # topography = invert(topography)
        else:
            topography = dIdV_integral #TODO this was added as test, switch it to None
        return cls(
            dIdV_integral=dIdV_integral,
            dIdV=dIdV,
            I=I,
            V=V,
            topography=topography,
            dimensions=size,
            metadata=grid.header
        )

    def _pixel_to_location(self, matrix: np.ndarray) -> np.ndarray:
        """
        Converts pixel coordinates to real-space locations in nanometers.

        Parameters:
        matrix (np.ndarray): A matrix of pixel coordinates, where each row represents [x, y].

        Returns:
        np.ndarray: Matrix of locations in real-space (nm).
        """
        assert matrix.shape[1] == 2, "Input matrix must have two columns representing [x, y] pixel coordinates."
        
        real_space = np.empty_like(matrix, dtype=float)
        
        for i, pixel_count in enumerate([self.nx, self.ny]):
            real_space[:, i] = (matrix[:, i] * self.dimensions[i]) / pixel_count
        return real_space


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
        
        
        ax.imshow(self.topography, cmap=cmap, extent=[
                  0, self.dimensions[1], 0, self.dimensions[0]])
        scalebar = ScaleBar(1, units='m', dimension="si-length", length_fraction=0.5, pad=0.3, sep=2,
                            location='lower left', box_alpha=0, scale_loc='top', color='indigo')#, font_properties={'size':'14'})
        ax.add_artist(scalebar)
        ax.set_aspect(1)
        # ax.set_xlabel("X Position (nm)")
        # ax.set_ylabel("Y Position (nm)")
        # xticks = np.linspace(0, self.dimensions[1] * 1e9, 7)  # 5 ticks for X
        # yticks = np.linspace(0, self.dimensions[0] * 1e9, 7)  # 5 ticks for Y
        # ax.set_xticks(xticks)
        # ax.set_yticks(yticks)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def plot_dIdV(self, n_random=8, locations: list = None, vertical_offset=0.0, cmap='YlOrBr'):
        """
        Plots random or specified pixel locations' dI/dV data and highlights their positions on the topography.

        Parameters:
        n_random (int, optional): Number of random pixel locations to plot. Default is 8.
        locations (list, optional): List of specific locations to plot. Default is empty.

        Returns:
        list: List of pixel locations plotted.
        """
        # fig_size = (self.nx/10*2, self.ny/10)
        fig_size = (self.nx/25*2, self.ny/25)
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
        aspect = self.dIdV_range/self.v_range*800
        # print(aspect, self.nx, self.ny)
        ax1.set_aspect(aspect)  # Adjust this value for different ratios; 1 means equal aspect ratio
        ax2 = self.plot_topography(ax=ax2, cmap=cmap, )
        points = []
        V = self.V*1e3
        color_palette = distinctipy.get_colors(n_random)
        if locations is None:
            locations = np.empty(shape=(n_random, 2), dtype=float)
            for i in range(n_random):
                index1 = np.random.randint(self.nx)
                index2 = np.random.randint(self.ny)
                locations[i, :] = [index1, index2]
                ax1.plot(V, self.dIdV[index1, index2, :]+i*vertical_offset,
                         label=f'pixel location: ({index1}, {index2})', linewidth=2.0, color=color_palette[i])
            points = self._pixel_to_location(locations)
        else:
            points = np.array(locations)
            for i, x in enumerate(locations):
                index1 = x[0]
                index2 = x[1]
                ax1.plot(V, self.dIdV[index1, index2, :]+i*vertical_offset,
                         label=f'pixel location: ({index1}, {index2})', linewidth=2.0, color=color_palette[i])
        ax2.scatter(points[:, 1], points[:, 0], color=
            color_palette, s=18.0)
        # ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.0)
        ax1.set_xlim(V[0], V[-1])
        ax1.set_ylim(0,)
        ax1.set_xlabel("Bias (mV)")
        ax1.set_ylabel("dI/dV (a.u.)")
        # Add ticks customization
        ax1.tick_params(axis='both', which='major', direction='inout')
        ax1.tick_params(axis='both', which='minor',direction='in')
        ax1.minorticks_on()  # Enable minor ticks
        plt.tight_layout()
        # Optionally, specify minor tick locations (if needed)
        # ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        # ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        return points

    def copy(self):
        """
        Creates and returns a deep copy of the STMScan object.

        Returns:
        STMScan: A deep copy of the current STMScan object.
        """
        return copy.deepcopy(self)

    def __str__(self):
        """
        Returns a string representation of the STMScan object summarizing the grid and energy range.
        """
        ret = ''
        ret += f'STS data grid: {self.nx}, {self.ny}\n'
        ret += f'Energy range: ({self.V[-1]*1e3} mV, {self.V[0] * 1e3} mV) sampled with {self.nE} points'
        return ret

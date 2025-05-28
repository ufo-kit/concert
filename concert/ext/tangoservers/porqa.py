"""
porqa.py
---------
Implements a device server to execute post-reconstruction quality assurance routines.
"""
import logging
from typing import AsyncIterator, List
import numpy as np
import matplotlib.pyplot as plt
try:
    import cupy as xp
    import cupy.fft as xft
except ModuleNotFoundError:
    print("cupy not available, defaulting to numpy")
    import numpy as xp
    import numpy.fft as xft
from tango import DebugIt, DevState
from tango.server import attribute, AttrWriteType, command
from tqdm import tqdm
from concert.ext.tangoservers.base import TangoRemoteProcessing
from concert.typing import ArrayLike


class TangoPostReconstructionQA(TangoRemoteProcessing):
    """Tango device for post-reconstruction quality assurance on a remote server"""

    freq_bins = attribute(
        label="Number of frequency bins",
        dtype=int,
        access=AttrWriteType.WRITE,
        fget="get_freq_bins",
        fset="set_freq_bins",
        doc="number of frequency bins for Fourier Shell Correlation (FSC) calculation"
    )

    plot = attribute(
        label="Plot",
        dtype=bool,
        access=AttrWriteType.WRITE,
        fget="get_plot",
        fset="set_plot",
        doc="Plot of the Fourier Shell Correlation (FSC) in different directions"
    )

    async def init_device(self) -> None:
        """
        Initializes the remote walker tango device. Sets the remote server's
        home directory as root as well as the current directory.
        """
        logging.getLogger('matplotlib.font_manager').disabled = True
        await super().init_device()
        self.set_state(DevState.STANDBY)
        self.info_stream(
            "%s in state: %s", self.__class__.__name__, self.get_state())
        
    def get_freq_bins(self) -> int:
        return self._freq_bins
    
    def set_freq_bins(self, fb: int) -> None:
        self._freq_bins = fb
        self.info_stream("%s: set frequency bins to %d", self.__class__.__name__, fb)

    def get_plot(self) -> bool:
        return self._plot
    
    def set_plot(self, plot: bool) -> None:
        self._plot = plot
        self.info_stream("%s: set plot to %s", self.__class__.__name__, plot)

    @DebugIt(show_args=True)
    @command()
    async def exec_qa(self) -> None:
        await self._process_stream(self._exec_fsc(self._receiver.subscribe()))

    def _compute_directional_fsc(self, fft1: ArrayLike, fft2: ArrayLike,
                                 direction: str = 'xy') -> ArrayLike:
        self.info_stream("%s: computing SFSC for direction %s", self.__class__.__name__, direction)
        z, y, x = np.indices(fft1.shape)
        center: ArrayLike = np.array(fft2.shape) // 2
        rz, ry, rx = z - center[0], y - center[1], x - center[2]
        # Directional distance
        if direction == 'xy':
            r: float = np.sqrt(rx**2 + ry**2)
        elif direction == 'yz':
            r: float = np.sqrt(ry**2 + rz**2)
        elif direction == 'zx':
            r: float = np.sqrt(rz**2 + rx**2)
        else:
            raise ValueError("Invalid direction")
        r = r.astype(int)
        fsc_num: ArrayLike = np.zeros(self._freq_bins)
        fsc_den1: ArrayLike = np.zeros(self._freq_bins)
        fsc_den2: ArrayLike = np.zeros(self._freq_bins)
        for i in tqdm(range(self._freq_bins), total=self._freq_bins):
            mask = (r == i)
            fft1_shell = fft1[mask]
            fft2_shell = fft2[mask]
            if len(fft1_shell) > 0:
                fsc_num[i] = np.sum(fft1_shell * np.conj(fft2_shell)).real
                fsc_den1[i] = np.sum(np.abs(fft1_shell)**2)
                fsc_den2[i] = np.sum(np.abs(fft2_shell)**2)
        fsc: ArrayLike = fsc_num / (np.sqrt(fsc_den1 * fsc_den2) + 1e-8)
        self.info_stream("%s: SFSC in %s direction : %s", self.__class__.__name__, direction, fsc)
        return fsc

    async def _exec_fsc(self, producer: AsyncIterator[ArrayLike]) -> None:
        """Computes Self Fourier Shell Correlation (FSC) for reconstructed volume"""
        slices: List[ArrayLike] = [slice async for slice in producer]
        volume: ArrayLike = np.stack(slices, axis=0)
        if volume.ndim != 3:
            self.error_stream(
                "%s: expected 3D volume, got %d", self.__class__.__name__, volume.ndim)
            return
        self.info_stream("%s: computing directional SFSC with %d frequency bins",
                         self.__class__.__name__, self._freq_bins)
        even_volume, odd_volume = volume[::2, :, :], volume[1::2, :, :]
        min_z = min(even_volume.shape[0], odd_volume.shape[0])
        even_volume, odd_volume = even_volume[:min_z], odd_volume[:min_z]
        self.info_stream("%s: dtsributed volumes into even and odd slices having shape: %s",
                         self.__class__.__name__, even_volume.shape)
        xp._default_memory_pool.free_all_blocks()
        # Compute Fourier transforms
        even_fft: ArrayLike = xp.asnumpy(
            xft.fftshift(xft.fftn(xp.asarray(even_volume.astype(np.float32)))))
        odd_fft: ArrayLike = xp.asnumpy(
            xft.fftshift(xft.fftn(xp.asarray(odd_volume.astype(np.float32)))))
        xp._default_memory_pool.free_all_blocks()
        fsc_xy: ArrayLike = self._compute_directional_fsc(even_fft, odd_fft, direction='xy')
        fsc_yz: ArrayLike = self._compute_directional_fsc(even_fft, odd_fft, direction='yz')
        fsc_zx: ArrayLike = self._compute_directional_fsc(even_fft, odd_fft, direction='zx')
        if self._plot:
            plt.figure()
            plt.plot(np.arange(10) + 1, fsc_xy, "o-", color="#E45756", label="SFSC-XY");
            plt.plot(np.arange(10) + 1, fsc_yz, "o-", color="#F58518", label="SFSC-YZ");
            plt.plot(np.arange(10) + 1, fsc_zx, "o-", color="#009392", label="SFSC-ZX");
            plt.legend()
            plt.show()


if __name__ == "__main__":
    pass

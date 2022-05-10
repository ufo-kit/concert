"""
Module to implement a monochromator with a second crystal/multilayer that can be scanned.
"""
import asyncio
import numpy as np
from concert.base import background, check
from concert.coroutines.base import broadcast
from concert.devices.monochromators.base import Monochromator as BaseMonochromator
from concert.processes.common import ascan
from concert.coroutines.sinks import Accumulate
from concert.quantities import q


class Monochromator(BaseMonochromator):
    """
    Base implementation of a monochromator with the ability to scan a second crystal/multilayer.
    """
    async def __ainit__(self, motor_2):
        """
        :param motor_2: Motor controlling the tilt of the second crystal or multilayer
        :type motor_2: concert.devices.motors.base.RotationMotor
        """
        self._last_scan_angles = None
        self._last_scan_counts = None
        self._motor_2 = motor_2
        await super().__ainit__()

    @background
    @check(source='standby', target='standby')
    async def scan_bragg_angle(self, diode, plot_callback=None, n_points=50,
                               tune_range=0.025 * q.deg, center_point=None):
        """
        Scans the second crystal or multilayer. After the scan, the motor is moved back to its
        initial position.

        A scan can be shown afterwards with :py:func:`.show_tune_scan`. To move the motor to the
            maximum call :py:func:`.select_maximum` and to go to the center of mass
            :py:func:`.select_center_of_mass`.

        :param diode: Diode to measure the intensity
        :type diode: concert.devices.photodiodes.base.Diode
        :param plot_callback: Function to plot the scanned intensity. Could be an instance of a
            PyplotViewer. If set to *None* the values of the scan are returned.
        :param n_points: Number of points, equally distributed between the
            current_angle - tune_range/2 and current_angle + tube_range/2
        :type n_points: int
        :param tune_range: Range to scan for maximum.
        :type tune_range: q.deg
        :param center_point: central point around which the scan is performed. If set to *None*
            the current position of the scanning motor is used.
        :type center_point: q.deg
        """

        async def _get_intensity():
            return await diode.get_intensity()

        current_bragg_angle = await self._motor_2.get_position()

        if center_point is None:
            center_point = current_bragg_angle

        scan_producer = ascan(
            self._motor_2['position'],
            center_point - tune_range / 2,
            center_point + tune_range / 2,
            tune_range / n_points,
            _get_intensity
        )

        acc = Accumulate()
        if plot_callback is not None:
            scan = broadcast(scan_producer, plot_callback, acc)
        else:
            scan = broadcast(scan_producer, acc)

        await asyncio.gather(*scan)
        self._last_scan_angles = np.zeros(n_points) * q.deg
        self._last_scan_counts = np.zeros(n_points) * diode['intensity'].unit

        for i in range(n_points):
            self._last_scan_angles[i] = acc.items[i][0]
            self._last_scan_counts[i] = acc.items[i][1]

        await self._motor_2.set_position(current_bragg_angle)

        if plot_callback is None:
            return [self._last_scan_angles, self._last_scan_counts]

    def get_last_tune_scan(self):
        """
        Shows a plot of the last tuning scan.
        """
        if self._last_scan_angles is None:
            raise Exception("No last scan data present")

        return self._last_scan_angles, self._last_scan_counts

    @background
    async def select_maximum(self):
        """
        Moves bragg2 to the maximum of the last tuning scan.
        """
        if self._last_scan_angles is None:
            raise Exception("No last scan data present.")
        new_bragg_angle = self._last_scan_angles[np.argmax(self._last_scan_counts)]
        await self._motor_2.set_position(new_bragg_angle)

    @background
    async def select_center_of_mass(self):
        """
        Moves bragg2 to the center of mass of the last tuning scan.
        """
        if self._last_scan_angles is None:
            raise Exception("No last scan data present.")
        new_bragg_angle = np.sum(self._last_scan_angles * self._last_scan_counts) / \
            np.sum(self._last_scan_counts)
        await self._motor_2.set_position(new_bragg_angle)

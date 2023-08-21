"""
power_supply.py
---------------

This module defines a Tango device subclass for testing only.
"""

from typing import Tuple, Dict, Union
import time
import tango
import numpy as np
from numpy import ndarray
from tango import AttrQuality, AttrWriteType, DispLevel, DevState, DebugIt
from tango.server import Device, attribute, command, pipe, device_property, DeviceMeta


class PowerSupply(Device, metaclass=DeviceMeta):

    voltage = attribute(
        label="Voltage",
        dtype=float,
        display_level=DispLevel.OPERATOR,
        access=AttrWriteType.READ,
        unit="V",
        format="8.4f",
        doc="the power supply voltage"
    )

    current = attribute(
        label="Current",
        dtype=float,
        display_level=DispLevel.EXPERT,
        access=AttrWriteType.READ_WRITE,
        unit="A",
        format="8.4f",
        min_value=0.0,
        max_value=8.5,
        min_alarm=0.1,
        max_alarm=8.4,
        min_warning=0.5,
        max_warning=8.0,
        fget="get_current",
        fset="set_current",
        doc="the power supply current"
    )

    noise = attribute(
        label="Noise",
        dtype=((int,),),
        max_dim_x=1024,
        max_dim_y=1024
    )

    info = pipe(label="Info")

    host = device_property(dtype=str)
    port = device_property(dtype=int, default_value=9788)

    def init_device(self) -> None:
        super().init_device()
        self.__current = 0.
        self.set_state(DevState.STANDBY)

    def read_voltage(self) -> Tuple[float, float, tango.AttrQuality]:
        self.info_stream("read_voltage(%s, %d)", self.host, self.port)
        return 9.99, time.time(), AttrQuality.ATTR_WARNING
    
    def get_current(self) -> float:
        return self.__current
    
    def set_current(self, curr: float) -> None:
        self.__current = curr

    def read_info(self) -> Tuple[str, Dict[str, Union[str, int]]]:
        return "Information", {
            "manufacturer": "Tango",
            "model": "PS2000",
            "version_number": 123
        }
    
    @DebugIt
    def read_noise(self) -> ndarray:
        return np.random.random_integers(1000, size=(100, 100))
    
    @command
    def TurnOn(self) -> None:
        self.set_state(DevState.ON)

    @command
    def TurnOff(self) -> None:
        self.set_state(DevState.OFF)

    @command(
        dtype_in=float,
        doc_in="Ramp target current",
        dtype_out=bool,
        doc_out="True if ramping went well, False otherwise"
    )
    def Ramp(self, target_curr: float) -> bool:
        return True


if __name__ == "__main__":
    PowerSupply.run_server()

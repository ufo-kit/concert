import unittest
import tango
from tango import DevState


class TestTango(unittest.TestCase):

    def test_abstract_power_supply(self) -> None:
        ps = tango.DeviceProxy("test/power_supply/1")
        self.assertTrue(ps.state() == DevState.STANDBY)
        ps.TurnOn()
        self.assertTrue(ps.state() == DevState.ON)
        self.assertTrue(ps.Ramp(2.1))
        ps.TurnOff()
        self.assertTrue(ps.state() == DevState.OFF)


if __name__ == "__main__":
    unittest.main()

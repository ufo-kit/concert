"""
dev_reg.py
----------

We attempt to register an abstract PowerSupply device with Tango database server
"""

import tango


if __name__ == "__main__":
    dev_info = tango.DbDevInfo()
    dev_info.server = "PowerSupply/test"
    dev_info._class = "PowerSupply"
    dev_info.name = "test/power_supply/1"
    db = tango.Database()
    db.add_device(dev_info)

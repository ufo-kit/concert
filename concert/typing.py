"""
typing.py
---------
Facilitates type annotations for concert
"""
from typing import Protocol, Any

#####################################################################
# Error Types
class StorageError(Exception):
    """
    Exceptions related to logical issues with storage.
    """
    pass
#####################################################################


#####################################################################
# Abstract Device Types

class AbstractTangoDevice(Protocol):
    """
    Abstract Tango device which let's users to write arbitrary attribute as
    key value pairs.
    """

    async def write_attribute(self, attr_name: str, value: Any) -> None:
        """Lets the caller write a device attribute

        :param attr_name: attribute name
        :type attr_name: str
        :param value: attribute value
        :type value: str
        """
#####################################################################


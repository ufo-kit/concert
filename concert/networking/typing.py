from typing import Union, Dict, Any, List, Tuple, Optional, Protocol
from numpy.typing import NDArray
from typing_extensions import TypeAlias

# Defines convenient type-aliases for ZMQ socket communication
Metadata_t: TypeAlias = Union[List[Any], str, int, float, Dict[Any, Any]]
Payload_t: TypeAlias = Union[Tuple[Metadata_t, Optional[NDArray]], Optional[NDArray]]
Subscription_t: TypeAlias = Union[Tuple[Metadata_t, NDArray], NDArray]


# Defines derived exception types
class BroadcastError(Exception):
    """
    BroadcastServer-related exceptions.
    """
    pass


class ConcertDeviceProxy(Protocol):
    """
    Abstract Tango device which lets users write arbitrary attribute as
    key value pairs.
    - *write_attribute(attr_name: str, value: Any)* writes a generic attribute as a key-value pair.
    """

    async def write_attribute(self, attr_name: str, value: Any) -> None:
        """
        Lets the caller write a device attribute

        :param attr_name: attribute name
        :type attr_name: str
        :param value: attribute value
        :type value: str
        """
        ...


if __name__ == "__main__":
    pass

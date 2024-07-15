"""
mocks.py
--------
Encapsulates mock devices which can be used to replace device servers for remote tests. The idea
is to approximately mock the async command invocations of the device servers, without actually
running them.
"""
import asyncio
import os
import unittest.mock as mock
from typing import Sequence, Any, Tuple


class MockWalkerDevice:
    """Attempts to mock the behavior of walker tango device server"""

    mock_device: mock.AsyncMock
    _logger_id: str

    def __init__(self, logger_id: str) -> None:
        self.mock_device = mock.AsyncMock()
        self._logger_id = logger_id

    async def write_attribute(self, attr_name: str, value: Any) -> None:
        await self.mock_device.write_attribute(attr_name=attr_name, value=value)

    def get_attribute_list(self) -> Sequence[str]:
        return []

    async def __getitem__(self, key: Any) -> Any:
        mock_value = mock.MagicMock()
        if key in ["current", "root"]:
            # For current and root we pass HOME to ensure that the check for an existing path
            # while running the experiment is satisfied. The actual path is irrelevant for
            # testing.
            mock_value.value = os.environ["HOME"]
        else:
            mock_value.value = "some_value"
        return mock_value

    async def descend(self, name: str) -> None:
        await self.mock_device.descend(name=name)
        
    async def ascend(self) -> None:
        await self.mock_device.ascend()

    async def exists(self, paths: str) -> bool:
        await self.mock_device.exists(paths=paths)

    async def write_sequence(self, name: str) -> None:
        await self.mock_device.write_sequence(name=name)
    
    async def register_logger(self, args: Tuple[str, str, str]) -> str:
        await self.mock_device.register_logger(args)
        return self._logger_id

    async def deregister_logger(self, logger_id: str) -> None:
        await self.mock_device.deregister_logger(logger_id)

    async def log(self, payload: Tuple[str, str, str]) -> None:
        print(f"Mock device got a log call")
        await self.mock_device.log(payload)

    async def log_to_json(self, payload: str) -> None:
        await self.mock_device.log_to_json(payload)


if __name__ == "__main__":
    pass


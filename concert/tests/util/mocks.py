"""
mocks.py
--------
Encapsulates mock devices which can be used to replace device servers for remote tests. The idea
is to approximately mock the async command invocations of the device servers, without actually
running them. We use `side_effect` to define the bare-minimum functionality for async method
invocations, which we need for testing.
"""
import os
import unittest.mock as mock
from typing import Sequence, Any, Tuple, List
import tango


class MockWalkerDevice:
    """Attempts to mock the behavior of walker tango device server"""

    mock_device: mock.AsyncMock
    _root: str
    _current: str
    _registered_log_paths: List[str]

    def _side_effect_descend(self, dir_name: str) -> None:
        self._current = os.path.join(self._current, dir_name)
        if not os.path.exists(self._current):
            self._create_dir(self._current)

    def _side_effect_ascend(self) -> None:
        self._current = os.path.dirname(self._current)

    def _side_effect_register_logger(self, args: Tuple[str, str, str]) -> str:
        return os.path.join(self._current, args[2])

    def _side_effect_deregister_logger(self, log_path: str) -> str:
        if os.path.dirname(log_path) != self._root:
            os.rmdir(os.path.dirname(log_path))

    def __init__(self) -> None:
        self.mock_device = mock.AsyncMock()
        self._root = os.environ["HOME"]
        self._current = self._root
        self._registered_log_paths = []
        self.mock_device.descend = mock.AsyncMock(side_effect=self._side_effect_descend)
        self.mock_device.ascend = mock.AsyncMock(side_effect=self._side_effect_ascend)
        self.mock_device.register_logger = mock.AsyncMock(
                side_effect=self._side_effect_register_logger)  # noqa E126
        self.mock_device.deregister_logger = mock.AsyncMock(
                side_effect=self._side_effect_deregister_logger)  # noqa E126

    async def write_attribute(self, attr: str, value: Any) -> None:
        await self.mock_device.write_attribute(attr, value)

    def get_attribute_list(self) -> Sequence[str]:
        return []

    async def __getitem__(self, key: Any) -> Any:
        mock_value = mock.MagicMock()
        # For current and root we pass HOME to ensure that the check for an existing path
        # while running the experiment is satisfied. The actual path is irrelevant for
        # testing.
        if key == "root":
            mock_value.value = self._root
        elif key == "current":
            mock_value.value = self._current
        else:
            mock_value.value = "some_value"
        return mock_value

    def lock(self, lock_validity: int = tango.constants.DEFAULT_LOCK_VALIDITY) -> None:
        # TODO: This should be an async method but its invocations in current implementation are
        # synchronous. When those are fixed it needs be fixed as well. When that is fixed we need
        # to register corresponding mock.AsyncMock in constructor for consistency.
        self.mock_device.lock(lock_validity=tango.constants.DEFAULT_LOCK_VALIDITY)

    async def descend(self, name: str) -> None:
        await self.mock_device.descend(name)

    async def ascend(self) -> None:
        await self.mock_device.ascend()

    @staticmethod
    def _create_dir(directory: str, mode: int = 0o0750) -> None:
        if not os.path.exists(directory):
            os.makedirs(name=directory, mode=mode)

    async def exists(self, paths: str) -> bool:
        await self.mock_device.exists(paths=paths)

    async def write_sequence(self, name: str) -> None:
        await self.mock_device.write_sequence(name=name)

    async def register_logger(self, args: Tuple[str, str, str]) -> str:
        return await self.mock_device.register_logger(args)

    async def deregister_logger(self, log_path: str) -> None:
        await self.mock_device.deregister_logger(log_path)

    async def log(self, payload: Tuple[str, str, str]) -> None:
        await self.mock_device.log(payload)

    async def log_to_json(self, payload: str) -> None:
        await self.mock_device.log_to_json(payload)

    def set_timeout_millis(self, timeout: int) -> None:
        pass


if __name__ == "__main__":
    pass

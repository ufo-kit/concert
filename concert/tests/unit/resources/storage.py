import os
from typing import Awaitable
from concert.storage import Walker


class TestableWalker(Walker):

    # Designates that this class is resource for testing and should not be collected by pytest
    # as a test case
    __test__ = False

    def __init__(self, root: str = "") -> None:
        super().__init__(root)
        self._paths = set([])

    @property
    def paths(self):
        return self._paths

    def exists(self, *paths):
        return os.path.join(*paths) in self._paths

    def _descend(self, name):
        self._current = os.path.join(self._current, name)
        self._paths.add(self._current)

    def _ascend(self):
        if self._current != self._root:
            self._current = os.path.dirname(self._current)

    def _create_writer(self, producer, dsetname=None) -> Awaitable:
        dsetname = dsetname or self.dsetname
        path = os.path.join(self._current, dsetname)

        async def _append_paths():
            i = 0
            async for item in producer:
                self._paths.add(os.path.join(path, str(i)))
                i += 1
        return _append_paths()


if __name__ == "__main__":
    pass

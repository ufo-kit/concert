"""
storage.py
------------
Defines high-level interfaces for storage utilities
"""

from __future__ import annotations
from typing import Protocol


class GenericWalker(Protocol):
    """Defines high-level requirements for a generic walker"""

    @property
    def current(self) -> str:
        """Returns current position of the walker

        :returns: current directory path of the walker
        :rtype: str
        """
        ...

    def exists(self, *paths: str) -> bool:
        """Returns True if path from current position specified by a list of *paths
        exists

        :param paths: an iterable collection of directory paths
        :type paths: Iterable[str]
        :returns: True if the path from the current position exists
        :rtype: bool
        """
        ...

    def descend(self, name: str) -> GenericWalker:
        """Descends to the path specified

        :param name: given directory path
        :type name: str
        :returns: self
        :rtype: Walker
        """
        ...

    def ascend(self) -> GenericWalker:
        """Ascend from current depth of the directory path

        :returns: self
        :rtype: Walker
        """
        ...









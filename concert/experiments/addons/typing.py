"""
typing.py
---------
Encapsulates type definitions for concert.experiments.addon.
"""
from typing import Protocol


class AbstractRAEDevice(Protocol):

    async def update_darks(self) -> None:
        """Accumulates the dark field projections"""
        ...

    async def update_flats(self) -> None:
        """Accumulates the flat field projections"""
        ...

    async def estimate_axis_of_rotation(self) -> None:
        """Executes axis of rotation estimation from radiogram projections"""
        ...


if __name__ == "__main__":
    pass

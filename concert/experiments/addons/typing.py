"""
typing.py
---------
Encapsulates type definitions for concert.experiments.addon.
"""
from typing import Protocol, Tuple


class AbstractQADevice(Protocol):

    async def update_darks(self) -> None:
        """Asynchronously accumulates the dark field projections and averages them"""
        ...

    async def update_flats(self) -> None:
        """Asynchronously accumulates the flat field projections and averages them"""
        ...

    async def estimate_center_of_rotation(self, args: Tuple[float]) -> None:
        """Executes center of rotation estimation from radio projections"""
        ...

if __name__ == "__main__":
    pass

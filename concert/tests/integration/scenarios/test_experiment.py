"""
test_experiment.py
------------------
Base class for experiment test cases with common functionality
"""
import json
import os
import shutil
import skimage.io as skio
from pathlib import Path
from typing import List, Any, Dict
from numpy import ndarray as ArrayLike
from concert.devices.cameras.uca import RemoteNetCamera
from concert.tests import TestCase
from abc import ABC, abstractmethod


class TestExperimentBase(TestCase, ABC):
    """Base class for experiment tests with common functionality."""

    @staticmethod
    def list_files(startpath: str) -> None:
        """List all files in a directory tree."""
        for root, dirs, files in os.walk(startpath):
            level: str = root.replace(startpath, '').count(os.sep)
            indent: str = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent: str = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))

    async def asyncSetUp(self) -> None:
        """Set up the experiment with common configuration."""
        await super().asyncSetUp()
        self._num_darks = 10
        self._num_flats = 10
        self._num_radios = 100
        self._camera = await RemoteNetCamera()
        if await self._camera.get_state() == 'recording':
            await self._camera.stop_recording()
        self._root = os.environ["DATA_ROOT"]
        
        # Setup experiment-specific components
        await self._setup_walker()
        await self._setup_experiment()
        
        # Clean up any existing scan data
        self._clean_scan_data()

    @abstractmethod
    async def _setup_walker(self):
        """Set up the directory walker - to be implemented by subclasses."""
        ...

    @abstractmethod
    async def _setup_experiment(self):
        """Set up the experiment - to be implemented by subclasses. Must create self._exp."""
        ...

    def _clean_scan_data(self):
        """Clean up any existing scan data."""
        base_path: Path = Path(self._root)
        items: List[str] = os.listdir(base_path)
        items = list(filter(lambda name: "scan" in name, items))
        if len(items) > 0:
            for item in items:
                abs_path: Path = base_path.joinpath(item)
                if os.path.isfile(abs_path):
                    os.remove(abs_path)
                else:
                    shutil.rmtree(abs_path)

    async def test_run(self) -> None:
        """Test running the experiment."""
        _ = await self._exp.run()
        base_path: Path = Path(self._root)
        items: List[str] = os.listdir(base_path)
        items = list(filter(lambda name: "scan" in name, items))
        self.assertTrue(len(items) > 0)
        self.list_files(base_path.__str__())
        for item in items:
            abs_path: Path = base_path.joinpath(item)
            self.assertTrue(os.path.exists(abs_path.joinpath("darks")))
            self.assertTrue(os.path.exists(abs_path.joinpath("flats")))
            self.assertTrue(os.path.exists(abs_path.joinpath("radios")))
            self.assertTrue(os.path.exists(abs_path.joinpath("experiment.log")))
            self.assertTrue(os.path.exists(abs_path.joinpath("experiment_start.json")))
            self.assertTrue(os.path.exists(abs_path.joinpath("experiment_finish.json")))
            darks: ArrayLike = skio.ImageCollection(
                abs_path.joinpath("darks/frame_000000.tif").__str__()
            )
            flats: ArrayLike = skio.ImageCollection(
                abs_path.joinpath("flats/frame_000000.tif").__str__()
            )
            radios: ArrayLike = skio.ImageCollection(
                abs_path.joinpath("radios/frame_000000.tif").__str__()
            )
            print(f"Num Darks: {len(darks)}")
            print(f"Num flats: {len(flats)}")
            print(f"Num radios: {len(radios)}")
            with open(abs_path.joinpath("experiment_finish.json")) as log:
                exp_log: Dict[str, Any] = json.load(log)["experiment"]
                self.assertTrue(len(darks) == int(exp_log["num_darks"]))
                self.assertTrue(len(flats) == int(exp_log["num_flats"]))
                self.assertTrue(len(radios) == int(exp_log["num_projections"]))

    async def asyncTearDown(self) -> None:
        """Clean up after the test."""
        await super().asyncTearDown()
        self._clean_scan_data()
        if hasattr(self, '_camera'):
            await self._camera.unregister_all()
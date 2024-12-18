import os
from pathlib import Path
from typing import List, Any, Dict
import json
from numpy import ndarray as ArrayLike
import skimage.io as skio


def list_files(startpath: str) -> None:
    for root, dirs, files in os.walk(startpath):
        level: str = root.replace(startpath, '').count(os.sep)
        indent: str = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent: str = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


if __name__ == "__main__":
    base_path: Path = Path(os.environ["TARGET_LOCATION"])
    items: List[str] = os.listdir(base_path)
    if len(items) > 0:
        for item in items:
            assert(item == "scan_0000")
            abs_path: Path = base_path.joinpath(item)
            assert(os.path.exists(abs_path.joinpath("darks")))
            assert(os.path.exists(abs_path.joinpath("flats")))
            assert(os.path.exists(abs_path.joinpath("radios")))
            assert(os.path.exists(abs_path.joinpath("experiment.log")))
            assert(os.path.exists(abs_path.joinpath("experiment.json")))
            darks: ArrayLike = skio.ImageCollection(abs_path.joinpath("darks/frame_000000.tif").__str__())
            flats: ArrayLike = skio.ImageCollection(abs_path.joinpath("flats/frame_000000.tif").__str__())
            radios: ArrayLike = skio.ImageCollection(abs_path.joinpath("radios/frame_000000.tif").__str__())
            print(f"Num Darks: {len(darks)}")
            print(f"Num flats: {len(flats)}")
            print(f"Num radios: {len(radios)}")
            with open(abs_path.joinpath("experiment.json")) as log:
                exp_log: Dict[str, Any] = json.load(log)["experiment"]
                assert(len(darks) == int(exp_log["num_darks"]))
                assert(len(flats) == int(exp_log["num_flats"]))
                assert(len(radios) == int(exp_log["num_projections"]))
            list_files(abs_path.__str__())
        print("Assertions succeeded, cleaning up")
import os
from pathlib import Path
import shutil
from typing import List

if __name__ == "__main__":
    base_path: Path = Path(os.environ["TARGET_LOCATION"])
    items: List[str] = os.listdir(base_path)
    if len(items) > 0:
        for item in items:
            abs_path: Path = base_path.joinpath(item)
            if os.path.isfile(abs_path):
                os.remove(abs_path)
            else:
                shutil.rmtree(abs_path)
from pathlib import Path

import pytest

from bip.config import Lake
from bip.lake import write
from bip.sources.synthetic import generate


@pytest.fixture(scope="session")
def synthetic_cells():
    return generate(seed=0)


@pytest.fixture(scope="session")
def lake(tmp_path_factory, synthetic_cells) -> Lake:
    lake = Lake(Path(tmp_path_factory.mktemp("data")))
    write(synthetic_cells, lake)
    return lake

import copy

import pytest

from carbonexus.config import load_config
from carbonexus.devtools import synthetic
from carbonexus import pipeline


def small_cfg(tmp_path):
    cfg = load_config(data_dir=tmp_path)
    cfg.raw = copy.deepcopy(cfg.raw)
    # coarser grid (0.2 deg) so the full 72-month analysis runs in a few seconds
    cfg.raw["study"].update(res=0.2, height=48, width=60)
    return cfg


@pytest.fixture(scope="session")
def analysis(tmp_path_factory):
    cfg = small_cfg(tmp_path_factory.mktemp("data"))
    raw, fires = synthetic.make_raw(cfg)
    res = pipeline.run_analysis(raw, fires, cfg, provenance=synthetic.PROVENANCE)
    pipeline.save(res, cfg)
    return cfg, raw, fires, res

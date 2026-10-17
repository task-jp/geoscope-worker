"""worker._job_zs: ジョブがスキャンするズームの組 (標準 / 広域 / 標準+広域)。"""
import os
import tempfile

import pytest

os.environ.setdefault("MODELS_DIR", tempfile.mkdtemp())
os.environ.setdefault("DATASETS_DIR", tempfile.mkdtemp())

import worker  # noqa: E402


def test_job_zs():
    assert worker._job_zs({}) == (16,)
    assert worker._job_zs({"tile_z": 15}) == (15,)
    assert worker._job_zs({"tile_z": 16, "tile_zs": [15, 16]}) == (16, 15)   # z16 を先に
    assert worker._job_zs(None) == (16,)
    with pytest.raises(ValueError):
        worker._job_zs({"tile_zs": [14]})

"""z15 DEM 合成 (dem_deps / load_dem_z) のテスト。torch 不要。"""

import numpy as np
import pytest

from app.core.dem import TILE_PX, dem_deps, load_dem_raw, load_dem_z


def _write_dem_webp(path, elev):
    """標高配列 (m) を本番と同じ符号付き 24bit WebP (可逆) で書く。"""
    import cv2
    x = np.round(elev * 100).astype(np.int64)
    x = np.where(x < 0, x + 2**24, x)
    img = np.zeros((*elev.shape, 3), np.uint8)
    img[:, :, 2] = (x >> 16) & 0xFF  # R
    img[:, :, 1] = (x >> 8) & 0xFF   # G
    img[:, :, 0] = x & 0xFF          # B
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 101])
    assert ok
    path.write_bytes(buf.tobytes())


def test_dem_deps():
    assert dem_deps(16, 100, 200) == [(100, 200)]
    assert set(dem_deps(15, 100, 200)) == {(200, 400), (201, 400), (200, 401), (201, 401)}
    with pytest.raises(ValueError):
        dem_deps(14, 0, 0)


def test_load_dem_z15_block_mean(tmp_path):
    # 4 子タイルに定数標高を入れ、z15 合成が各象限でその値になることを確認
    tx15, ty15 = 100, 200
    values = {(200, 400): 10.0, (201, 400): 20.0, (200, 401): -3.0, (201, 401): 40.0}
    for (cx, cy), v in values.items():
        _write_dem_webp(tmp_path / "16" / str(cx) / f"{cy}.webp",
                        np.full((TILE_PX, TILE_PX), v))
    load_dem_raw.cache_clear()
    dem = load_dem_z(str(tmp_path), 15, tx15, ty15)
    assert dem is not None and dem.shape == (TILE_PX, TILE_PX)
    h = TILE_PX // 2
    assert np.allclose(dem[:h, :h], 10.0)   # 左上 = 子 (200,400)
    assert np.allclose(dem[:h, h:], 20.0)   # 右上 = 子 (201,400)
    assert np.allclose(dem[h:, :h], -3.0)   # 左下 = 子 (200,401) 負標高も保持
    assert np.allclose(dem[h:, h:], 40.0)   # 右下 = 子 (201,401)


def test_load_dem_z15_averages_gradient(tmp_path):
    # 勾配のある子タイル 1 枚: 2×2 ブロック平均になっていること
    tx15, ty15 = 10, 20
    child = np.arange(TILE_PX * TILE_PX, dtype=np.float64).reshape(TILE_PX, TILE_PX) * 0.01
    _write_dem_webp(tmp_path / "16" / "20" / "40.webp", child)
    load_dem_raw.cache_clear()
    dem = load_dem_z(str(tmp_path), 15, tx15, ty15)
    assert dem is not None
    h = TILE_PX // 2
    expect = child.reshape(h, 2, h, 2).swapaxes(1, 2).reshape(h, h, 4).mean(axis=2)
    assert np.allclose(dem[:h, :h], expect, atol=1e-9)
    # 子が無い象限は NaN のまま
    assert np.isnan(dem[h:, h:]).all()


def test_load_dem_z15_missing_all_children(tmp_path):
    load_dem_raw.cache_clear()
    assert load_dem_z(str(tmp_path), 15, 1, 1) is None

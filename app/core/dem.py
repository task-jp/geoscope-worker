"""DEM decoding and coordinate utilities.

Ported from detect_kofun_v4.py and prepare_yolo_dataset.py.
"""

import functools
import io
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TILE_PX = 512


@functools.lru_cache(maxsize=128)
def load_dem_raw(tile_path: str) -> np.ndarray | None:
    """Load DEM tile as elevation array without 3ch conversion.

    LRU cache: 各 forkserver worker process 内で memoize する。
    `_gen_3ch_extended` は隣接タイル (右 / 下 / 右下) を併せて 4 枚読むため、
    proximity-sort で並んだスキャン中は同じ DEM タイルが直近で再 decode
    される。cache hit で WebP decode + float64 cast の重複を消す。
    呼び出し側は配列を mutate していないので cache 共有 (= 同一 ndarray
    返却) で安全。
    """
    try:
        data = Path(tile_path).read_bytes()
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        r = arr[:, :, 2].astype(np.float64)
        g = arr[:, :, 1].astype(np.float64)
        b = arr[:, :, 0].astype(np.float64)
        x = r * 65536 + g * 256 + b
        # 符号付き 24bit。x は 24bit なので `x > 2**24` は絶対に成立せず、
        # 以前はここが 2**24 だったため負の標高が 167,772m の突起になっていた。
        # 曲率の正規化は全画面の std/mean を使うので、1 画素の突起で 768x768 の
        # 全画素が狂う。decode_dem と同じ式にすること。
        elev = np.where(x == 2**23, np.nan, np.where(x > 2**23, (x - 2**24) * 0.01, x * 0.01))
        return elev
    except Exception:
        return None


def dem_deps(z: int, tx: int, ty: int) -> list[tuple[int, int]]:
    """z レベルのタイル 1 枚を作るのに必要な z16 タイル座標。

    DEM の実体は z16 のみ (サーバー / R2 とも)。z=15 はワーカーが z16 の
    子タイル 2×2 枚からその場で合成する (新規ストレージ不要で、ローカル /
    クラウドどちらのワーカーでも同じ動作)。z16 学習済みモデルを z15 に
    そのまま当てると実寸 2 倍 (200〜470m 級) の対象の検出器になる:
    3ch は物理量 (傾斜は対象サイズに不変) なので、巨大古墳が z15 では
    「普通の古墳の見た目」になる。
    """
    if z == 16:
        return [(tx, ty)]
    if z == 15:
        return [(2 * tx + i, 2 * ty + j) for j in (0, 1) for i in (0, 1)]
    raise ValueError(f"unsupported zoom: {z}")


def load_dem_z(tiles_dir: str, z: int, tx: int, ty: int) -> np.ndarray | None:
    """z レベルの 512×512 DEM 標高配列を返す (z=15 は z16 子タイルから合成)。

    ダウンサンプルは 2×2 ブロックの nanmean。標高の平均は物理的に正しい
    縮約で、cell_size_m(lat, 15) と組で傾斜・曲率も正しくなる。
    """
    if z == 16:
        return load_dem_raw(str(Path(tiles_dir) / "16" / str(tx) / f"{ty}.webp"))
    canvas = np.full((TILE_PX * 2, TILE_PX * 2), np.nan)
    found = False
    for cx, cy in dem_deps(z, tx, ty):
        child = load_dem_raw(str(Path(tiles_dir) / "16" / str(cx) / f"{cy}.webp"))
        if child is None:
            continue
        found = True
        ox, oy = (cx - 2 * tx) * TILE_PX, (cy - 2 * ty) * TILE_PX
        canvas[oy:oy + TILE_PX, ox:ox + TILE_PX] = child
    if not found:
        return None
    blocks = canvas.reshape(TILE_PX, 2, TILE_PX, 2).swapaxes(1, 2).reshape(TILE_PX, TILE_PX, 4)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=2)


def decode_dem(data: bytes) -> np.ndarray:
    """WebP DEM tile → float64 elevation array.

    Encoding: h = (R*65536 + G*256 + B) * 0.01 [m]
    Values >= 2^23 are treated as signed (subtract 2^24).
    Value == 2^23 is invalid (NaN).
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img, dtype=np.float64)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    x = r * 65536 + g * 256 + b
    return np.where(x == 2**23, np.nan, np.where(x > 2**23, (x - 2**24) * 0.01, x * 0.01))


def hillshade(elev: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Compute hillshade (azimuth=315, altitude=45). Returns uint8 [0,255]."""
    filled = elev.copy()
    mean_val = np.nanmean(elev) if not np.isnan(elev).all() else 0
    filled[np.isnan(filled)] = mean_val
    dy, dx = np.gradient(filled, cell_size)
    az, alt = math.radians(315), math.radians(45)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    shade = np.clip(
        math.sin(alt) * np.cos(slope) + math.cos(alt) * np.sin(slope) * np.cos(az - aspect),
        0,
        1,
    )
    return (shade * 255).astype(np.uint8)


def pixel_to_latlon(z: int, tile_x: int, tile_y: int, px: float, py: float) -> tuple[float, float]:
    """Convert tile pixel coordinates to lat/lon (EPSG:4326)."""
    n = 2**z
    lon = (tile_x + px / TILE_PX) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + py / TILE_PX) / n))))
    return lat, lon


def latlon_to_tile_px(lat: float, lon: float, z: int) -> tuple[int, int, float, float]:
    """Convert lat/lon to tile coordinates and pixel offset.

    Returns (tile_x, tile_y, pixel_x, pixel_y).
    """
    n = 2**z
    tx = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    ty = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    px = ((lon + 180) / 360 * n - tx) * TILE_PX
    py = ((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n - ty) * TILE_PX
    return tx, ty, px, py

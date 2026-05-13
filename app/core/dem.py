"""DEM decoding and coordinate utilities.

Ported from detect_kofun_v4.py and prepare_yolo_dataset.py.
"""

import io
import math

import numpy as np
from PIL import Image

TILE_PX = 512


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

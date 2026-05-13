"""Compute dome-ness metrics for detection results using DEM data.

Given a lat/lon + tile coordinates, loads the DEM tile and computes:
- dome_h: height above surrounding terrain [m]
- roundness: circularity of radial profile [0-1]
- isolation: dome_h / surrounding_std
- max_min_ratio: elongation of shape
"""

import math
from pathlib import Path

import cv2
import numpy as np

from app.core.dem import TILE_PX, decode_dem


def compute_domeness(
    tiles_dir: str, tile_z: int, tile_x: int, tile_y: int,
    bbox_px_cx: float, bbox_px_cy: float,
) -> dict | None:
    """Compute dome metrics at the given pixel position within a tile.

    bbox_px_cx/cy are normalized (0-1). Returns dict with metrics or None if tile missing.
    """
    tile_path = Path(tiles_dir) / f"{tile_z}/{tile_x}/{tile_y}.webp"
    if not tile_path.exists():
        return None

    try:
        elev = decode_dem(tile_path.read_bytes())
    except Exception:
        return None

    valid = elev[~np.isnan(elev)]
    if len(valid) < TILE_PX * TILE_PX * 0.3:
        return None

    filled = elev.copy()
    filled[np.isnan(filled)] = np.nanmean(elev)

    # Pixel position
    px = int(bbox_px_cx * TILE_PX) if bbox_px_cx <= 1 else int(bbox_px_cx)
    py = int(bbox_px_cy * TILE_PX) if bbox_px_cy <= 1 else int(bbox_px_cy)
    px = max(20, min(TILE_PX - 20, px))
    py = max(20, min(TILE_PX - 20, py))

    # Find local peak near the given position (search within 30px)
    search_r = 30
    y1 = max(0, py - search_r)
    y2 = min(TILE_PX, py + search_r)
    x1 = max(0, px - search_r)
    x2 = min(TILE_PX, px + search_r)
    region = filled[y1:y2, x1:x2]
    smoothed_region = cv2.GaussianBlur(region, (5, 5), 0)
    local_max_idx = np.unravel_index(np.argmax(smoothed_region), smoothed_region.shape)
    py = y1 + local_max_idx[0]
    px = x1 + local_max_idx[1]

    peak_h = filled[py, px]

    # Surrounding ring statistics (30-150px away)
    y1r = max(0, py - 200)
    y2r = min(TILE_PX, py + 200)
    x1r = max(0, px - 200)
    x2r = min(TILE_PX, px + 200)
    ring_region = filled[y1r:y2r, x1r:x2r]
    ry, rx = py - y1r, px - x1r
    yy, xx = np.ogrid[:ring_region.shape[0], :ring_region.shape[1]]
    dist = np.sqrt((xx - rx) ** 2 + (yy - ry) ** 2)

    ring_mask = (dist > 30) & (dist < 150)
    if ring_mask.sum() == 0:
        return {"dome_h": 0, "roundness": 0, "isolation": 0, "max_min_ratio": 0}

    surr_mean = float(np.mean(ring_region[ring_mask]))
    surr_std = float(np.std(ring_region[ring_mask]))

    dome_h = float(peak_h - surr_mean)
    isolation = dome_h / max(surr_std, 0.5)

    # Radial profile for roundness
    threshold = peak_h - dome_h * 0.5
    radii = []
    for angle_deg in range(0, 360, 12):  # 30方向（24-33が最適）
        rad = math.radians(angle_deg)
        r = 80
        for d in range(1, 80):
            sx = int(px + d * math.cos(rad))
            sy = int(py + d * math.sin(rad))
            if 0 <= sx < TILE_PX and 0 <= sy < TILE_PX:
                if filled[sy, sx] < threshold:
                    r = d
                    break
            else:
                break
        radii.append(r)

    radii = np.array(radii, dtype=float)
    mean_r = float(np.mean(radii))
    roundness = float(1.0 - np.std(radii) / mean_r) if mean_r > 0 else 0
    max_min_ratio = float(np.max(radii) / max(np.min(radii), 1.0))

    return {
        "dome_h": round(dome_h, 2),
        "roundness": round(roundness, 3),
        "isolation": round(isolation, 2),
        "max_min_ratio": round(max_min_ratio, 2),
        "mean_radius_px": round(mean_r, 1),
    }

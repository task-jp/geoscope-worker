"""Visualization functions for DEM data.

Red Relief Image Map (RRIM / 赤色立体地図) and support functions.
"""

import math

import cv2
import numpy as np


def _notch_moire(elev: np.ndarray) -> np.ndarray:
    """Remove DEM1A resampling moiré by notch-filtering the 1/5 frequency in Y."""
    from numpy.fft import fft2, ifft2, fftfreq
    h, w = elev.shape
    fy = fftfreq(h)
    mask = np.ones((h, w), dtype=np.float64)
    for harmonic in (1, 2):
        freq = 0.2 * harmonic
        for i in range(h):
            a = np.exp(-((abs(fy[i]) - freq) ** 2) / (2 * 0.015 ** 2))
            mask[i, :] *= 1 - a
    return np.real(ifft2(fft2(elev) * mask))


def _compute_openness(filled: np.ndarray, radius: int = 50, directions: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Compute positive and negative topographic openness (optimized).

    Uses sparse sampling along radial lines for speed.
    Returns (positive_openness, negative_openness) as float64 arrays in [0, 1].
    """
    h, w = filled.shape
    pos_open = np.full((h, w), math.pi / 2, dtype=np.float64)
    neg_open = np.full((h, w), math.pi / 2, dtype=np.float64)

    # Sparse distances: check fewer points along each direction
    distances = [1, 2, 3, 5, 8, 12, 18, 25, 35, 50]
    distances = [d for d in distances if d <= radius]

    for i in range(directions):
        angle = 2 * math.pi * i / directions
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)

        max_elev = np.zeros((h, w), dtype=np.float64)
        max_depr = np.zeros((h, w), dtype=np.float64)

        for dist in distances:
            iy = int(round(sin_a * dist))
            ix = int(round(cos_a * dist))
            if iy == 0 and ix == 0:
                continue

            # Compute slice bounds
            sy0, sy1 = max(0, -iy), min(h, h - iy)
            sx0, sx1 = max(0, -ix), min(w, w - ix)
            dy0, dy1 = sy0 + iy, sy1 + iy
            dx0, dx1 = sx0 + ix, sx1 + ix

            diff = filled[dy0:dy1, dx0:dx1] - filled[sy0:sy1, sx0:sx1]
            ang = np.arctan2(diff, dist)

            np.maximum(max_elev[sy0:sy1, sx0:sx1], ang, out=max_elev[sy0:sy1, sx0:sx1])
            np.maximum(max_depr[sy0:sy1, sx0:sx1], -ang, out=max_depr[sy0:sy1, sx0:sx1])

        pos_open = np.minimum(pos_open, math.pi / 2 - max_elev)
        neg_open = np.minimum(neg_open, math.pi / 2 - max_depr)

    pos_norm = np.clip(pos_open / (math.pi / 2), 0, 1)
    neg_norm = np.clip(neg_open / (math.pi / 2), 0, 1)
    return pos_norm, neg_norm


def cs_map(elev: np.ndarray) -> np.ndarray:
    """Generate Red Relief Image Map (RRIM / 赤色立体地図) with alpha channel.

    Uses slope + topographic openness instead of gradient-based curvature.
    Openness uses long baselines (50px) so it's robust against DEM grid artifacts.

    Returns BGRA uint8 image suitable for PNG encoding.
    """
    nan_mask = np.isnan(elev)
    filled = elev.copy()
    if nan_mask.any() and not nan_mask.all():
        from scipy.ndimage import distance_transform_edt
        _, indices = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
        filled[nan_mask] = filled[tuple(indices[:, nan_mask])]
    elif nan_mask.all():
        filled[:] = 0

    # DEM1Aリサンプリングモアレ除去（Y方向1/5周波数をノッチフィルタで除去）
    filled = _notch_moire(filled)

    # Slope: Sobel ksize=7 で広域勾配（グリッドノイズを回避）
    dx = cv2.Sobel(filled, cv2.CV_64F, 1, 0, ksize=7)
    dy = cv2.Sobel(filled, cv2.CV_64F, 0, 1, ksize=7)
    slope = np.sqrt(dx**2 + dy**2)
    slope_norm = np.clip(slope / 1800, 0, 1)  # p95≈1800で正規化

    # Topographic openness (positive = ridges bright, negative = valleys bright)
    pos_open, neg_open = _compute_openness(filled, radius=50, directions=8)

    # RRIM配色: 白背景 + 赤(傾斜) + 暗(谷)
    # brightness: 尾根=明るい、谷=暗い
    brightness = pos_open
    s = slope_norm

    # 赤色立体地図: R=255, G/B=明るさ-傾斜で減らす
    r = np.clip((brightness * 0.6 + 0.4) * 255, 0, 255).astype(np.uint8)
    g = np.clip(brightness * (1 - s) * 255, 0, 255).astype(np.uint8)
    b = np.clip(brightness * (1 - s) * 245, 0, 255).astype(np.uint8)

    # Alpha
    feature_strength = np.clip(s * 0.5 + (1 - brightness) * 0.4 + 0.1, 0, 1)
    a = np.clip(feature_strength * 220 + 35, 0, 255).astype(np.uint8)
    a[nan_mask] = 0

    return cv2.merge([b, g, r, a])


def multi_hillshade(filled: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Average hillshade from 4 directions (0, 90, 180, 270 degrees).

    Returns float64 array [0, 1].
    """
    dy, dx = np.gradient(filled, cell_size)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    alt = math.radians(45)

    shades = []
    for az_deg in [0, 90, 180, 270]:
        az = math.radians(az_deg)
        shade = np.clip(
            math.sin(alt) * np.cos(slope)
            + math.cos(alt) * np.sin(slope) * np.cos(az - aspect),
            0,
            1,
        )
        shades.append(shade)
    return np.mean(shades, axis=0)


def compute_slope(filled: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Compute slope angle normalized to [0, 1] (1.0 at 45 degrees)."""
    dy, dx = np.gradient(filled, cell_size)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.clip(slope / (math.pi / 4), 0, 1)


def compute_curvature(filled: np.ndarray, ksize: int = 31) -> np.ndarray:
    """Compute Laplacian curvature. Convex surfaces (domes) are bright.

    Returns float64 array [0, 1].
    """
    smoothed = cv2.GaussianBlur(filled, (ksize, ksize), 0)
    lap = cv2.Laplacian(smoothed, cv2.CV_64F, ksize=5)
    neg_lap = -lap
    std = max(np.std(neg_lap), 0.01)
    normalized = (neg_lap - np.mean(neg_lap)) / (3 * std) * 0.5 + 0.5
    return np.clip(normalized, 0, 1)


def _slope_aspect(filled: np.ndarray, cell_size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """共有計算: np.gradient → slope, aspect を 1 度だけ作る。
    dem_to_3ch から呼んで multi_hillshade / compute_slope 相当を再利用する。
    値は標準関数 (multi_hillshade / compute_slope) と bit-同値。"""
    dy, dx = np.gradient(filled, cell_size)
    slope = np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dy, dx)
    return slope, aspect


def _hillshade_from(slope: np.ndarray, aspect: np.ndarray) -> np.ndarray:
    """`multi_hillshade` と同じ値を、与えた slope/aspect で計算する。"""
    alt = math.radians(45)
    shades = []
    for az_deg in [0, 90, 180, 270]:
        az = math.radians(az_deg)
        shade = np.clip(
            math.sin(alt) * np.cos(slope)
            + math.cos(alt) * np.sin(slope) * np.cos(az - aspect),
            0,
            1,
        )
        shades.append(shade)
    return np.mean(shades, axis=0)


# 曲率の平滑化を「何メートルの範囲で行うか」。以前は ksize=31 画素固定で、
# z=16 では約 31m だったが、他のズームでは物理的な意味が変わっていた。
CURVATURE_SMOOTH_M = 31.0


def cell_size_m(lat: float, z: int, tile_px: int = 512) -> float:
    """1 画素が表す実距離 [m]。Web Mercator なので緯度に依存する。

    以前は `cell_size=1.0` 固定だった。これは z=16 の大阪付近 (0.983 m/px) で
    だけ正しく、実測で以下のずれがあった:
        沖縄 1.072 m/px (-6.7%) / 大阪 0.983 (+1.7%) / 札幌 0.872 (+14.7%)
        稚内 0.839 m/px (+19.2%) / z=15 では 1.966 m/px (-49.1%)
    傾斜は `np.gradient(filled, cell_size)` で求めるので、この値が違うと
    同じ地形が緯度・ズームによって別の見た目になる。傾斜チャンネルは
    45° で飽和するため、z=15 では飽和画素が 1% → 17% に増えることを実測した。
    """
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** z) / (tile_px / 256)


def dem_to_3ch(elev: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """DEM elevation → 3-channel uint8 image (H, W, 3).

    ch0: multi-direction hillshade
    ch1: slope
    ch2: curvature (Laplacian, convex=bright)

    cell_size は 1 画素の実距離 [m]。`cell_size_m(lat, z)` で求めて渡すこと。
    既定の 1.0 は z=16 の大阪付近の近似値で、他の緯度・ズームでは傾斜が
    誤って計算される (関数の docstring 参照)。

    実装メモ: 標準関数 (multi_hillshade / compute_slope) は各々 np.gradient を
    重複計算していた。768x768 では 1 度の gradient が ~10ms 効くので、
    共有ヘルパー経由で 1 回に統合する。dtype・出力値は標準関数経由と完全同値。
    """
    filled = elev.copy()
    mean_val = np.nanmean(elev) if not np.isnan(elev).all() else 0
    filled[np.isnan(filled)] = mean_val

    slope, aspect = _slope_aspect(filled, cell_size)
    ch0 = _hillshade_from(slope, aspect)
    ch1 = np.clip(slope / (math.pi / 4), 0, 1)   # = compute_slope(filled)
    # 平滑化は物理スケールで揃える。ksize は奇数でなければならない。
    ksize = max(3, int(round(CURVATURE_SMOOTH_M / cell_size)) | 1)
    ch2 = compute_curvature(filled, ksize=ksize)

    img = np.stack([ch0, ch1, ch2], axis=2)
    return (img * 255).astype(np.uint8)

"""resolve_detection のテスト。

torch 不要 (app/services/detections.py は scanning.py から分離されている)。
実行: リポジトリルートで `python3 -m pytest tests/ -q`
"""

import math

import pytest

from app.core.dem import TILE_PX, pixel_to_latlon
from app.services.detections import EDGE_MARGIN_PX, EXTENDED_PX, resolve_detection

TX, TY = 57168, 26022  # 実在のタイル座標 (草ケ部の誤検出があったタイル)


def _det(cx, cy, w, h, conf=0.5, cls=0):
    return {"cx": cx, "cy": cy, "w": w, "h": h, "conf": conf, "cls": cls}


def test_interior_detection_passes_unchanged():
    rec = resolve_detection(_det(300.0, 200.0, 100.0, 80.0), TX, TY)
    assert rec is not None
    assert (rec["tile_x"], rec["tile_y"]) == (TX, TY)
    assert (rec["bbox_cx"], rec["bbox_cy"]) == (300.0, 200.0)
    lat, lon = pixel_to_latlon(16, TX, TY, 300.0, 200.0)
    assert (rec["lat"], rec["lon"]) == (lat, lon)


def test_west_edge_truncated_view_is_dropped():
    # 草ケ部の実例: 左端から 1.28px の bbox (旧閾値 1px は素通りしていた)
    rec = resolve_detection(_det(30.7, 319.5, 58.9, 198.2, conf=0.461), TX, TY)
    assert rec is None


def test_bbox_within_margin_of_any_edge_is_dropped():
    m = EDGE_MARGIN_PX
    # 左・上・右・下それぞれの縁に余白未満で迫る bbox
    assert resolve_detection(_det(m + 20, 400, 2 * (20 + m) + 2, 50), TX, TY) is None
    assert resolve_detection(_det(400, m + 20, 50, 2 * (20 + m) + 2), TX, TY) is None
    assert resolve_detection(
        _det(EXTENDED_PX - m - 20, 400, 2 * (20 + m) + 2, 50), TX, TY) is None
    assert resolve_detection(
        _det(400, EXTENDED_PX - m - 20, 50, 2 * (20 + m) + 2), TX, TY) is None


def test_extension_center_reassigned_to_east_neighbor():
    # 中心が拡張領域 (x>512): 東隣タイルに付け替え、bbox 西端は負になってよい
    rec = resolve_detection(_det(540.0, 200.0, 120.0, 100.0), TX, TY)
    assert rec is not None
    assert (rec["tile_x"], rec["tile_y"]) == (TX + 1, TY)
    assert rec["bbox_cx"] == pytest.approx(540.0 - TILE_PX)  # = 28.0
    assert rec["bbox_cx"] - rec["bbox_w"] / 2 < 0  # 西へのはみ出し


def test_corner_extension_reassigned_diagonally():
    rec = resolve_detection(_det(600.0, 650.0, 80.0, 80.0), TX, TY)
    assert rec is not None
    assert (rec["tile_x"], rec["tile_y"]) == (TX + 1, TY + 1)
    assert rec["bbox_cx"] == pytest.approx(600.0 - TILE_PX)
    assert rec["bbox_cy"] == pytest.approx(650.0 - TILE_PX)


def test_same_object_resolves_identically_from_both_views():
    # 境界すぐ東の対象: 西タイルの視野では拡張領域、自タイルの視野では base。
    # どちらの視野から解決しても同じレコード (同じタイル・座標・実座標) になる。
    w, h = 100.0, 100.0
    from_west_view = resolve_detection(_det(TILE_PX + 60.0, 300.0, w, h), TX - 1, TY)
    from_own_view = resolve_detection(_det(60.0, 300.0, w, h), TX, TY)
    assert from_west_view is not None and from_own_view is not None
    for key in ("tile_x", "tile_y", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h"):
        assert from_west_view[key] == pytest.approx(from_own_view[key])
    assert math.isclose(from_west_view["lat"], from_own_view["lat"])
    assert math.isclose(from_west_view["lon"], from_own_view["lon"])


def test_resolve_detection_z15_latlon_matches_z16_children():
    # z15 タイル (tx,ty) の中心画素は、z16 子タイル (2tx,2ty) の右下端に相当する。
    # 同一地点を z15 と z16 の座標系で解決して lat/lon が一致することを確認。
    rec15 = resolve_detection(_det(256.0, 256.0, 50.0, 50.0), TX, TY, z=15)
    rec16 = resolve_detection(_det(512.0 - 0.0, 512.0, 50.0, 50.0), 2 * TX, 2 * TY, z=16)
    assert rec15 is not None and rec16 is not None
    assert math.isclose(rec15["lat"], rec16["lat"], abs_tol=1e-12)
    assert math.isclose(rec15["lon"], rec16["lon"], abs_tol=1e-12)

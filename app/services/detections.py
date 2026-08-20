"""検出結果の後処理: 拡張画像上の検出を実タイル座標のレコードに解決する。

scanning.py から分離しているのは、torch を積まない環境でも単体テストできる
ようにするため (scanning.py は import 時に torch を要求する)。
"""

from app.core.dem import TILE_PX, pixel_to_latlon

EXTENDED_PX = TILE_PX + TILE_PX // 2  # 768 = 512 + 256

# 拡張画像の縁からこの距離 (px) 未満に bbox がかかる検出は「画像端で切断された
# 部分ビュー」として捨てる。YOLO は端で切れた物体にも端から 1〜2px まで迫った
# box を出す (実例: 左端から 1.28px, conf 0.46 の誤検出が旧閾値 1px を素通り)
# ので、余白は広めに取る。
EDGE_MARGIN_PX = 8


def resolve_detection(d: dict, tx: int, ty: int, z: int = 16) -> dict | None:
    """拡張画像 (768×768) 上の検出 1 件を、実タイル座標のレコードに解決する。

    タイル境界付近の対象は複数タイルの視野に写る (自タイルの base 領域と、
    左/上隣タイルの拡張領域)。bbox が視野に完全に収まる検出は拡張領域中心の
    ものも全て通し、どの視野の検出を採用するかはサーバー側 dedup
    (IoU>0.3 で高スコアを残す) に委ねる。

    以前は「中心が base 512×512 内」の検出だけを残していたため、タイル境界の
    すぐ右/下にある対象は左/上端で切断されたビュー 1 つでしか判定されず、
    全体が見える左/上隣タイルの検出は比較される前に捨てられていた。

    bbox が視野の縁 EDGE_MARGIN_PX 未満に迫る検出は、切断された部分ビューと
    みなして捨てる。タイル原点は 512px 間隔、視野は 768px 幅なので、幅/高さ
    256-2*EDGE_MARGIN_PX (=240) px 以下の対象は必ずどこかの視野に余白付きで
    完全に収まり、本物なら別視野の検出が生き残る。

    tile_x/tile_y は中心を含むタイル。中心が拡張領域にある検出は隣タイルに
    付け替える。中心がタイル西/北端近くで bbox が隣へはみ出す場合、正規化後の
    bbox 西/北端は負になり得る (東/南の >1.0 と同じ扱いで、実座標の計算・
    サーバーの bbox_geom・ギャラリー描画のいずれも線形なので問題ない)。
    """
    gcx, gcy = d["cx"], d["cy"]
    w, h = d["w"], d["h"]
    if (gcx - w / 2 < EDGE_MARGIN_PX or gcy - h / 2 < EDGE_MARGIN_PX
            or gcx + w / 2 > EXTENDED_PX - EDGE_MARGIN_PX
            or gcy + h / 2 > EXTENDED_PX - EDGE_MARGIN_PX):
        return None
    real_tx = tx + int(gcx) // TILE_PX
    real_ty = ty + int(gcy) // TILE_PX
    local_cx = gcx - (real_tx - tx) * TILE_PX
    local_cy = gcy - (real_ty - ty) * TILE_PX
    lat, lon = pixel_to_latlon(z, real_tx, real_ty, local_cx, local_cy)
    return {
        "lat": lat, "lon": lon, "conf": d["conf"],
        "bbox_cx": local_cx, "bbox_cy": local_cy,
        "bbox_w": w, "bbox_h": h,
        "tile_x": real_tx, "tile_y": real_ty,
        "cls": d.get("cls", 0),
    }

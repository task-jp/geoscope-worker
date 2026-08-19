"""Tile scanning: run trained model inference on DEM tiles.

Called by the GPU worker. Mirrors the pipeline structure of scan_all_yolo.py:
multiprocess 3ch generation + GPU batch inference.
"""

import functools
import math
import os
import time
import multiprocessing as _mp
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from app.core.dem import TILE_PX, decode_dem, pixel_to_latlon
from app.core.visualization import cell_size_m, dem_to_3ch
from app.services.detections import EXTENDED_PX, resolve_detection


# ---------------------------------------------------------------------------
# DEM decode (runs in worker processes, lightweight)
# ---------------------------------------------------------------------------

def _load_dem(args: tuple) -> tuple | None:
    """Decode DEM tile (OpenCV, GIL-free) and fill NaN. Thread-safe."""
    tile_path_str, tx, ty = args
    try:
        data = Path(tile_path_str).read_bytes()
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        r = arr[:, :, 2].astype(np.float64)
        g = arr[:, :, 1].astype(np.float64)
        b = arr[:, :, 0].astype(np.float64)
        x = r * 65536 + g * 256 + b
        elev = np.where(x == 2**23, np.nan, np.where(x > 2**23, (x - 2**24) * 0.01, x * 0.01))
        valid = elev[~np.isnan(elev)]
        if len(valid) < TILE_PX * TILE_PX * 0.3:
            return None
        elev[np.isnan(elev)] = np.nanmean(elev) if len(valid) > 0 else 0
        return (elev, tx, ty)
    except Exception:
        return None


def _gen_3ch(args: tuple) -> tuple | None:
    """Convert a DEM tile to 3ch image in a subprocess. Uses cv2 for fast WebP decode."""
    tile_path_str, tx, ty = args
    try:
        data = Path(tile_path_str).read_bytes()
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        r = arr[:, :, 2].astype(np.float64)
        g = arr[:, :, 1].astype(np.float64)
        b = arr[:, :, 0].astype(np.float64)
        x = r * 65536 + g * 256 + b
        elev = np.where(x == 2**23, np.nan, np.where(x > 2**23, (x - 2**24) * 0.01, x * 0.01))
        valid = elev[~np.isnan(elev)]
        if len(valid) < TILE_PX * TILE_PX * 0.3:
            return None
        lat, _ = pixel_to_latlon(16, tx, ty, TILE_PX / 2, TILE_PX / 2)
        img = dem_to_3ch(elev, cell_size_m(lat, 16))
        if min(img[:, :, c].std() for c in range(3)) < 3:
            return None
        return (img, tx, ty)
    except Exception:
        return None


@functools.lru_cache(maxsize=128)
def _load_dem_raw(tile_path: str) -> np.ndarray | None:
    """Load DEM tile as elevation array without 3ch conversion.

    LRU cache: 各 forkserver worker process 内で memoize する。
    `_gen_3ch_extended` は隣接タイル (右 / 下 / 右下) を併せて 4 枚読むため、
    proximity-sort で並んだスキャン中は同じ DEM タイルが直近で再 decode
    される。cache hit で WebP decode + float64 cast の重複を消す。
    呼び出し側 (`_gen_3ch_extended` 内 `canvas[...] = main` 等) は配列を
    mutate していないので cache 共有 (= 同一 ndarray 返却) で安全。
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
        # 全画素が狂う。app/core/dem.py:decode_dem と同じ式にすること。
        elev = np.where(x == 2**23, np.nan, np.where(x > 2**23, (x - 2**24) * 0.01, x * 0.01))
        return elev
    except Exception:
        return None


# EXTENDED_PX (768 = 512 + 256) は app/services/detections.py で定義


CACHE_3CH_DIR = os.environ.get("CACHE_3CH_DIR", "")

# 3ch の生成方法を変えたら必ず上げる。上げると旧世代は参照されなくなる。
#
# 無効化を元 DEM の mtime 比較だけに頼ると、コードを変えても DEM は変わらないので
# 古い画像を黙って再利用してしまう。実際に v1 でそれを踏んだ:
# v1 は隣接タイルをスキャン対象集合で絞っており、都道府県スキャンの縁で
# 学習側と違う画像を作っていた (一致率 40%)。
#   v1: 隣接を tile_coord_set で絞る (破棄)
#   v2: 隣接はファイル存在で判定。ただし DEM デコードの符号バグを抱えたまま (破棄)
#   v3: _load_dem_raw の符号付き 24bit を修正 (学習側 decode_dem と一致)
#   v4: cell_size を緯度から計算するようにした (以前は 1.0 固定で、大阪付近でのみ
#       正しく、札幌で +14.7% / 稚内で +19.2% 傾斜を過小評価していた)
_CACHE_3CH_VERSION = "v4"


def _cache_base(tx: int) -> Path:
    return Path(CACHE_3CH_DIR) / _CACHE_3CH_VERSION / "16" / str(tx)


def _cache_3ch_read(tx: int, ty: int, src_paths: list[str]) -> tuple | None | str:
    """3ch キャッシュを引く。戻り値は 3 通り:
      ndarray      : キャッシュヒット (そのまま使える)
      "skip"       : 生成対象外と判定済み (前回 None を返したタイル)
      None         : キャッシュなし / 無効 → 生成する

    無効化は mtime 比較で行う。DEM は再変換で中身が変わることがあり
    (docs/DEM_PIPELINE.md の fill / refresh)、古い 3ch を返すと
    「更新したのに結果が変わらない」という最悪の不具合になる。
    stat 4 回は生成 308ms に対して無視できるコスト。
    """
    base = _cache_base(tx)
    img_p, skip_p = base / f"{ty}.webp", base / f"{ty}.skip"
    try:
        newest_src = max(os.stat(p).st_mtime for p in src_paths if os.path.exists(p))
    except ValueError:
        return None
    for p, kind in ((img_p, "img"), (skip_p, "skip")):
        try:
            if os.stat(p).st_mtime < newest_src:
                continue  # DEM の方が新しい → 作り直す
        except OSError:
            continue
        if kind == "skip":
            return "skip"
        arr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if arr is not None and arr.shape == (EXTENDED_PX, EXTENDED_PX, 3):
            return arr
    return None


_ENC_POOL = None
_ENC_PENDING: list = []


def _enc_submit(fn, *a) -> None:
    """符号化を背景スレッドに投げる。

    可逆 WebP の符号化は 254ms かかり、3ch 生成 (308ms) と直列にすると
    初回スキャンが約 1.8 倍に伸びる。プールワーカーは 32 個で 64 コアの半分しか
    使っていないため、符号化を別スレッドに出せば空きコアで処理され、初回も
    現状と同じ速度で終わる (OpenCV は GIL を解放するので Python でも並列に効く)。

    未完了が溜まると 1 件 1.7MB の画像を抱えたままメモリが膨らむので、
    2 件を超えたら最古の完了を待つ (背圧)。
    """
    global _ENC_POOL
    if _ENC_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        import atexit
        _ENC_POOL = ThreadPoolExecutor(max_workers=1)
        # maxtasksperchild でワーカーが作り直される際に書きかけを捨てない
        atexit.register(lambda: _ENC_POOL.shutdown(wait=True))
    _ENC_PENDING[:] = [f for f in _ENC_PENDING if not f.done()]
    while len(_ENC_PENDING) >= 2:
        _ENC_PENDING.pop(0).result()
        _ENC_PENDING[:] = [f for f in _ENC_PENDING if not f.done()]
    _ENC_PENDING.append(_ENC_POOL.submit(fn, *a))


def _cache_3ch_write_sync(tx: int, ty: int, img: np.ndarray | None) -> None:
    """3ch キャッシュを実際に書く (背景スレッドから呼ばれる)。
    img=None なら "生成対象外" マーカーを置く。

    可逆 WebP (quality 101) を使う。非可逆にすると学習時と推論時でピクセル値が
    変わり、モデルにとって別画像になる。実測で q80 は正例の再現率を
    52.8% → 49.5% に落とした (docs/CLAIMS.md 参照)。
    可逆の往復がビット完全一致することは実測で確認済み。

    32 プロセスが並行して書くので temp + os.replace で原子的に置換する。
    中断で切れたファイルが残ると、次回それを decode して壊れた画像で推論してしまう。
    """
    base = _cache_base(tx)
    try:
        base.mkdir(parents=True, exist_ok=True)
        final = base / (f"{ty}.skip" if img is None else f"{ty}.webp")
        tmp = base / f"{ty}.{os.getpid()}.tmp"
        if img is None:
            tmp.touch()
        else:
            ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 101])
            if not ok:
                return
            tmp.write_bytes(buf.tobytes())
        os.replace(tmp, final)
    except Exception:
        pass  # キャッシュは最適化なので、書けなくてもスキャンは続ける


def _cache_3ch_write(tx: int, ty: int, img: np.ndarray | None) -> None:
    """3ch キャッシュの書き込みを背景スレッドに委ねる。"""
    try:
        _enc_submit(_cache_3ch_write_sync, tx, ty, img)
    except Exception:
        _cache_3ch_write_sync(tx, ty, img)


def _gen_3ch_extended(args: tuple) -> tuple | None:
    """Generate 150% extended 3ch image (768×768) from tile (tx,ty).
    Covers full tile + 256px right + 256px down + 256×256 diagonal.
    args: (tiles_dir, tx, ty).

    CACHE_3CH_DIR が設定されていれば生成結果を可逆 WebP で保存し、
    次回以降は decode (1.7ms) だけで済ませる。3ch は静的な DEM の純粋関数なので
    スキャンごとに作り直す必要がない。実測で生成 308ms → 復号 1.7ms、
    全国スキャンの律速が CPU から GPU に移り 100 → 243 tile/s になる。
    既定は無効 (120 万枚で約 790GB 使うため、明示的に有効化させる)。
    """
    if not CACHE_3CH_DIR:
        return _gen_3ch_uncached(args)

    tiles_dir, tx, ty = args
    src = [str(Path(tiles_dir) / "16" / str(x) / f"{y}.webp")
           for x, y in ((tx, ty), (tx + 1, ty), (tx, ty + 1), (tx + 1, ty + 1))]

    cached = _cache_3ch_read(tx, ty, src)
    if isinstance(cached, str):          # "skip" = 前回 None だったタイル
        return None
    if cached is not None:
        return (cached, tx, ty)

    out = _gen_3ch_uncached(args)
    # None も記録する。有効ピクセル 30% 未満や平坦なタイルは全体の一定割合を占め、
    # 記録しないとそのぶんは毎回 308ms を払い続けることになる。
    _cache_3ch_write(tx, ty, out[0] if out is not None else None)
    return out


def _gen_3ch_uncached(args: tuple) -> tuple | None:
    """3ch 拡張タイルを DEM から実際に生成する (キャッシュを見ない)。

    隣接タイルは「存在すれば貼る」。以前はスキャン対象集合 (tile_coord_set) に
    入っているかで判定していたため、都道府県スキャンの縁では隣接 DEM が手元に
    あっても貼られず、学習側 (dataset.py の _load_3ch_extended は存在判定) と
    違う画像で推論していた。NaN 埋めが nanmean、曲率の正規化が全画面の std/mean
    なので、貼るか貼らないかで 768x768 の全画素が変わる (実測 一致率 40%)。
    3ch は DEM の関数であるべきで、スキャン範囲で変わってはいけない。
    """
    tiles_dir, tx, ty = args
    half = TILE_PX // 2  # 256

    canvas = np.full((EXTENDED_PX, EXTENDED_PX), np.nan)

    # メインタイル (0:512, 0:512)
    path = str(Path(tiles_dir) / "16" / str(tx) / f"{ty}.webp")
    main = _load_dem_raw(path)
    if main is None:
        return None
    # メインタイル単体で有効ピクセル30%未満ならスキップ
    # （隣接タイルが有効でも、メインがほぼNaNだと誤検出の元）
    main_valid = main[~np.isnan(main)]
    if len(main_valid) < TILE_PX * TILE_PX * 0.3:
        return None
    canvas[:TILE_PX, :TILE_PX] = main

    # 右タイル (512:768, 0:512) — 左半分の256列
    right = _load_dem_raw(str(Path(tiles_dir) / "16" / str(tx + 1) / f"{ty}.webp"))
    if right is not None:
        canvas[:TILE_PX, TILE_PX:] = right[:, :half]

    # 下タイル (0:512, 512:768) — 上半分の256行
    below = _load_dem_raw(str(Path(tiles_dir) / "16" / str(tx) / f"{ty + 1}.webp"))
    if below is not None:
        canvas[TILE_PX:, :TILE_PX] = below[:half, :]

    # 右下タイル (512:768, 512:768) — 左上256×256
    diag = _load_dem_raw(str(Path(tiles_dir) / "16" / str(tx + 1) / f"{ty + 1}.webp"))
    if diag is not None:
        canvas[TILE_PX:, TILE_PX:] = diag[:half, :half]

    # NaN埋め
    valid = canvas[~np.isnan(canvas)]
    if len(valid) < TILE_PX * TILE_PX * 0.3:
        return None
    canvas[np.isnan(canvas)] = np.nanmean(canvas) if len(valid) > 0 else 0

    try:
        lat, _ = pixel_to_latlon(16, tx, ty, EXTENDED_PX / 2, EXTENDED_PX / 2)
        img = dem_to_3ch(canvas, cell_size_m(lat, 16))
        if min(img[:, :, c].std() for c in range(3)) < 3:
            return None
        return (img, tx, ty)
    except Exception:
        return None


def _nms_detections(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Non-Maximum Suppression on detections with global coordinates."""
    if not detections:
        return []
    # confでソート（降順）
    dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
    keep = []
    for d in dets:
        overlap = False
        for k in keep:
            # 同一タイル座標系でなくグローバル座標(lat,lon)でIoU計算
            # 簡易: bboxの中心距離が小さければ重複とみなす
            dlat = abs(d["lat"] - k["lat"])
            dlon = abs(d["lon"] - k["lon"])
            # 約10m以内なら同一検出
            if dlat < 0.0001 and dlon < 0.0001:
                overlap = True
                break
        if not overlap:
            keep.append(d)
    return keep


def _batch_dem_to_3ch_gpu(elevs: np.ndarray) -> np.ndarray:
    """GPU batch 3ch generation using CuPy. Input: (N, 512, 512) float64. Output: (N, 512, 512, 3) uint8."""
    import cupy as cp
    from cupyx.scipy.ndimage import uniform_filter, laplace
    import math

    batch = cp.asarray(elevs)
    dy = cp.diff(batch, axis=1, prepend=batch[:, :1, :])
    dx = cp.diff(batch, axis=2, prepend=batch[:, :, :1])
    slope_angle = cp.arctan(cp.sqrt(dx**2 + dy**2))
    aspect = cp.arctan2(-dy, dx)

    # Multi-direction hillshade
    alt = math.radians(45)
    shades = []
    for az_deg in [0, 90, 180, 270]:
        az = math.radians(az_deg)
        shade = cp.clip(
            math.sin(alt) * cp.cos(slope_angle) + math.cos(alt) * cp.sin(slope_angle) * cp.cos(az - aspect),
            0, 1)
        shades.append(shade)
    hillshade = cp.mean(cp.stack(shades), axis=0)

    # Slope
    slope_norm = cp.clip(slope_angle / (cp.pi / 4), 0, 1)

    # Curvature
    smoothed = uniform_filter(batch, size=(1, 31, 31))
    lap = laplace(smoothed)
    neg_lap = -lap
    std = cp.maximum(cp.std(neg_lap, axis=(1, 2), keepdims=True), 0.01)
    mean = cp.mean(neg_lap, axis=(1, 2), keepdims=True)
    curvature = cp.clip((neg_lap - mean) / (3 * std) * 0.5 + 0.5, 0, 1)

    result = cp.stack([
        (hillshade * 255).astype(cp.uint8),
        (slope_norm * 255).astype(cp.uint8),
        (curvature * 255).astype(cp.uint8),
    ], axis=3)
    return result.get()


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device | None = None):
    """Load detection model. Supports DINO (directory) and YOLO (.pt)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = Path(model_path)

    # RT-DETR / DINO: directory with config.json
    if p.is_dir() and (p / "config.json").exists():
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        model = RTDetrForObjectDetection.from_pretrained(str(p)).to(device)
        model.eval()
        processor = RTDetrImageProcessor.from_pretrained(str(p))
        return model, processor, "rtdetr"

    # YOLO: .pt file (fallback for existing models)
    if p.suffix == ".pt" and p.is_file():
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        return model, None, "yolo"

    raise ValueError(f"Cannot determine model type for: {model_path}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _infer_rtdetr(
    model, processor, images: list[np.ndarray], conf_threshold: float, device: torch.device,
) -> list[list[dict]]:
    """Run RT-DETR inference on a batch."""
    import cv2 as _cv2

    rgb_images = [_cv2.cvtColor(img, _cv2.COLOR_BGR2RGB) for img in images]
    inputs = processor(images=rgb_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if k == "pixel_values"}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[TILE_PX, TILE_PX]] * len(images), device=device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=conf_threshold,
    )

    all_dets = []
    for r in results:
        dets = []
        boxes = r["boxes"].cpu().numpy()
        scores = r["scores"].cpu().numpy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            dets.append({
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
                "conf": float(score),
            })
        all_dets.append(dets)
    return all_dets


def _infer_yolo(
    model, images: list[np.ndarray], conf_threshold: float, device: torch.device,
    imgsz: int = 512,
) -> list[list[dict]]:
    """Run YOLO inference on a batch.

    画像は GPU 上でテンソルに組んでから渡す。ultralytics に numpy のリストを渡すと
    前処理を CPU でやるため、実測で 1 枚 3.12ms かかり全体の 69% を占めていた
    (推論本体は 1.31ms、後処理は 0.09ms)。内訳は np.stack + BGR→RGB が 2.46ms、
    transpose + ascontiguousarray が 1.13ms。BGR→RGB を CPU でやると負のストライドの
    コピーが走るのが主因で、GPU 上で permute + インデックスすれば消える。
    LetterBox も入力が既に EXTENDED_PX 角なので何もしないまま 0.41ms 払っていた。

    1 枚ずつ GPU へ送って torch.stack する形が最速だった (np.stack してから 1 回で
    送るより速い)。実測 4.32 → 2.64 ms/枚 = 231 → 379 tile/s。
    正例タイル 256 枚・検出 330 件で、座標・スコアともに完全一致を確認済み。

    注意: tensor を渡すと ultralytics は /255 の正規化を行わない (呼び出し側の責任)。
    """
    if images and isinstance(images[0], np.ndarray):
        gpu = [torch.from_numpy(im).to(device, non_blocking=True) for im in images]
        batch = torch.stack(gpu).permute(0, 3, 1, 2)[:, [2, 1, 0]].float() / 255
    else:
        batch = images
    results = model.predict(batch, conf=conf_threshold, iou=0.5, device=device,
                            imgsz=imgsz, verbose=False)
    all_dets = []
    for r in results:
        dets = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0]) if box.cls is not None else 0
            dets.append({
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
                "conf": conf,
                "cls": cls_id,
            })
        all_dets.append(dets)
    return all_dets



# ---------------------------------------------------------------------------
# Tile enumeration
# ---------------------------------------------------------------------------

def _latlon_to_tile(lat: float, lon: float, z: int = 16) -> tuple[int, int]:
    """Convert lat/lon to tile x,y at zoom level z."""
    n = 2 ** z
    tx = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    ty = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return tx, ty


def _enumerate_tiles(tiles_dir: str, region: dict | None = None,
                     tile_fetcher: Callable | None = None) -> list[tuple[str, int, int]]:
    """List all z=16 tiles, optionally filtered to a bounding box.

    If tile_fetcher is provided and region is set, generates tile coordinates
    even if local files don't exist (tile_fetcher will download them on demand).
    """
    scan_dir = Path(tiles_dir) / "16"

    # Compute tile range if region is given
    tx_min = ty_min = 0
    tx_max = ty_max = 2 ** 16 - 1
    if region:
        tx_min, ty_max_r = _latlon_to_tile(region["south"], region["west"])
        tx_max, ty_min_r = _latlon_to_tile(region["north"], region["east"])
        ty_min = ty_min_r
        ty_max = ty_max_r

    # If tile_fetcher is available and region is specified, enumerate by coordinate range
    # (no need for local files to exist)
    if tile_fetcher and region:
        tiles = []
        for tx in range(tx_min, tx_max + 1):
            for ty in range(ty_min, ty_max + 1):
                local = Path(tiles_dir) / "16" / str(tx) / f"{ty}.webp"
                tiles.append((str(local), tx, ty))
        return tiles

    # Default: enumerate local files
    if not scan_dir.exists():
        return []

    tiles = []
    for x_dir in sorted(scan_dir.iterdir()):
        if not x_dir.is_dir():
            continue
        tx = int(x_dir.name)
        if tx < tx_min or tx > tx_max:
            continue
        for f in sorted(x_dir.glob("*.webp")):
            ty = int(f.stem)
            if ty < ty_min or ty > ty_max:
                continue
            tiles.append((str(f), tx, ty))

    return tiles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_tiles(
    model_path: str,
    tiles_dir: str,
    conf_threshold: float = 0.3,
    region: dict | None = None,
    batch_size: int = 128,
    num_workers: int = 16,
    progress_callback: Callable[[float, str], None] | None = None,
    detection_callback: Callable[[list[dict]], None] | None = None,
    tile_fetcher: Callable[[int, int, int], str | None] | None = None,
    tile_set: set[tuple[int, int]] | None = None,
    resume_from: int = 0,
    progress_save_callback: Callable[[int], None] | None = None,
    tile_list: list[tuple[str, int, int]] | None = None,
    cancel_event: "threading.Event | None" = None,
) -> list[dict]:
    """Run inference on all (or region-filtered) DEM tiles.

    Pipeline:
    1. Enumerate z=16 tiles (optionally within region bbox).
    2. Multiprocess: DEM -> 3ch image generation.
    3. GPU batch: model inference.
    4. Convert pixel detections to lat/lon.

    Args:
        model_path: Path to trained model (directory for RT-DETR, .pth for FasterRCNN).
        tiles_dir: Root DEM tile directory.
        conf_threshold: Minimum confidence for detections.
        region: Optional bounding box {west, south, east, north}.
        batch_size: GPU batch size.
        num_workers: Number of parallel 3ch generation workers.
        progress_callback: fn(progress: float 0-1, message: str).

    Returns:
        List of detection dicts: [{lat, lon, conf, bbox_cx, bbox_cy, bbox_w, bbox_h, tile_x, tile_y}].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if progress_callback:
        progress_callback(0.0, "Loading model...")

    model, processor, model_type = load_model(model_path, device)

    if progress_callback:
        progress_callback(0.0, "Enumerating tiles...")

    if tile_list is not None:
        tiles = tile_list
    else:
        tiles = _enumerate_tiles(tiles_dir, region, tile_fetcher)
        if tile_set:
            tiles = [(p, tx, ty) for p, tx, ty in tiles if (tx, ty) in tile_set]
    if not tiles:
        if progress_callback:
            progress_callback(1.0, "No tiles found")
        return []

    # 150%拡張タイル: 各タイルを768×768に拡張（右+256, 下+256, 右下+256×256）
    # 隣接タイルはスキャン対象集合に入っているかで絞らない。絞ると都道府県
    # スキャンの縁で学習側と違う画像になる (_gen_3ch_uncached の docstring 参照)。
    extended_tiles = [(tiles_dir, tx, ty) for _, tx, ty in tiles]

    total_tiles = len(extended_tiles)

    # レジューム: 処理済みタイルをスキップ
    if resume_from > 0 and resume_from < total_tiles:
        extended_tiles = extended_tiles[resume_from:]
        if progress_callback:
            progress_callback(resume_from / total_tiles,
                              f"Resuming from tile {resume_from:,}/{total_tiles:,}")
    all_detections: list[dict] = []
    total_detections = 0
    processed = 0
    t0 = time.monotonic()

    # Pipeline: DL threads (if remote) → multiprocessing Pool (3ch on CPU) → queue → GPU YOLO
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    img_queue: queue.Queue = queue.Queue(maxsize=batch_size * 4)
    sentinel = object()

    def _is_cancelled():
        return cancel_event is not None and cancel_event.is_set()

    def _dl_and_feed(pool):
        """Download tiles in parallel threads, then feed to 3ch pool."""
        if tile_fetcher:
            # Remote: download adjacent tiles too, then generate extended 3ch
            dl_queue: queue.Queue = queue.Queue(maxsize=64)
            dl_done = threading.Event()

            def _downloader():
                # 拡張タイルごとに必要な隣接タイルを束ね、依存が満たされた瞬間に
                # dl_queue へ stream する。以前は全 needed タイルの fetch 完走を
                # 同期で待ってから dl_queue に put していたため、cache hit の
                # 多い 全国 scan でも consumer が数分〜数十分 starve していた。
                pending: dict = {}        # et -> set of base coords still missing
                waiters: dict = {}        # base coord -> list of et waiting
                for et in extended_tiles:
                    _, tx, ty = et
                    # 隣接も必ず取る。取らないと 3ch が学習側と食い違う
                    # (_gen_3ch_uncached の docstring 参照)。存在しない座標は
                    # fetch_tile が .404 マーカーに記録するので 2 回目以降は無料。
                    et_deps = {(tx, ty), (tx + 1, ty), (tx, ty + 1), (tx + 1, ty + 1)}
                    pending[et] = et_deps
                    for d in et_deps:
                        waiters.setdefault(d, []).append(et)

                needed = set(waiters.keys())
                lock = threading.Lock()

                def _fetch_one(coord):
                    ttx, tty = coord
                    local = Path(tiles_dir) / "16" / str(ttx) / f"{tty}.webp"
                    if not local.exists():
                        tile_fetcher(16, ttx, tty)
                    # この coord に依存していた et のうち deps 全部揃ったものを取り出す
                    released = []
                    with lock:
                        for et in waiters.pop(coord, ()):
                            deps_remaining = pending.get(et)
                            if deps_remaining is None:
                                continue
                            deps_remaining.discard(coord)
                            if not deps_remaining:
                                released.append(et)
                                del pending[et]
                    # lock 外で put (queue 満杯時にロック保持で他スレッドを止めない)
                    for et in released:
                        dl_queue.put(et)

                with ThreadPoolExecutor(max_workers=32) as executor:
                    list(executor.map(_fetch_one, needed, chunksize=16))
                dl_done.set()

            dl_thread = threading.Thread(target=_downloader, daemon=True)
            dl_thread.start()

            # dl_queue から取れた拡張タイルを 1 件ずつ pool に流す。
            # 旧コードは 16 件貯めてから chunksize=8 で imap_unordered していたため、
            # 1 バッチで使われるワーカーは 2 個だけ、かつ 16 件全部の結果が戻るまで
            # 次の dl_queue.get に進めず pool が休む時間が支配的になり、GPU 推論側
            # (img_queue.get) が starve していた。連続供給に切り替えて 16 worker 全部を
            # 常時飽和させる。
            def _yield_extended():
                while True:
                    if _is_cancelled():
                        return
                    try:
                        item = dl_queue.get(timeout=0.5)
                    except queue.Empty:
                        if dl_done.is_set() and dl_queue.empty():
                            return
                        continue
                    yield item

            for r in pool.imap_unordered(_gen_3ch_extended, _yield_extended(), chunksize=1):
                if _is_cancelled():
                    break
                img_queue.put(r)
            dl_thread.join()
        else:
            # Local tiles: 150%拡張タイル生成
            CHUNK = 1000
            for ci in range(0, len(extended_tiles), CHUNK):
                if _is_cancelled():
                    break
                for r in pool.imap_unordered(_gen_3ch_extended, extended_tiles[ci:ci+CHUNK], chunksize=16):
                    if _is_cancelled():
                        break
                    img_queue.put(r)
        img_queue.put(sentinel)

    # forkserver: workers don't inherit YOLO model / CUDA context → prevents OOM
    ctx = _mp.get_context("forkserver")
    with ctx.Pool(num_workers, maxtasksperchild=500) as pool:
        producer = threading.Thread(target=_dl_and_feed, args=(pool,), daemon=True)
        producer.start()

        imgs_batch: list[np.ndarray] = []
        metas_batch: list[tuple[int, int]] = []  # (tx, ty)

        def _process_batch():
            nonlocal all_detections, total_detections
            if model_type == "rtdetr":
                batch_dets = _infer_rtdetr(model, processor, imgs_batch, conf_threshold, device)
            else:
                batch_dets = _infer_yolo(model, imgs_batch, conf_threshold, device,
                                         imgsz=EXTENDED_PX)
            for dets, (tx, ty) in zip(batch_dets, metas_batch):
                for d in dets:
                    # 境界の対象は複数タイルの視野で検出され、サーバー側 dedup が
                    # 最良の 1 件を残す (resolve_detection の docstring 参照)
                    rec = resolve_detection(d, tx, ty)
                    if rec is not None:
                        all_detections.append(rec)

        while True:
            if _is_cancelled():
                pool.terminate()
                break
            item = img_queue.get()
            if item is sentinel:
                break
            processed += 1
            if item is not None:
                imgs_batch.append(item[0])
                metas_batch.append((item[1], item[2]))

            if len(imgs_batch) >= batch_size:
                _process_batch()
                imgs_batch.clear()
                metas_batch.clear()

                if detection_callback and all_detections:
                    detection_callback(all_detections[:])
                    total_detections += len(all_detections)
                    all_detections.clear()

            if progress_callback and processed % 1000 < 1:
                elapsed = time.monotonic() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                abs_processed = resume_from + processed
                det_count = total_detections + len(all_detections)
                progress_callback(
                    abs_processed / total_tiles,
                    f"{abs_processed:,}/{total_tiles:,} tiles, "
                    f"{det_count:,} detections ({rate:.0f} t/s)",
                )
                if progress_save_callback and processed % 10000 < 1:
                    progress_save_callback(abs_processed)

        producer.join()

        # 残り
        if imgs_batch:
            _process_batch()
            imgs_batch.clear()
            metas_batch.clear()

        if detection_callback and all_detections:
            detection_callback(all_detections[:])
            total_detections += len(all_detections)
            all_detections.clear()

    final_count = total_detections + len(all_detections)
    if progress_callback:
        progress_callback(1.0, f"Scan complete: {final_count:,} detections")

    # GPU メモリ解放
    del model
    if processor is not None:
        del processor
    torch.cuda.empty_cache()

    return all_detections

"""Tile scanning: run trained model inference on DEM tiles.

Called by the GPU worker. Mirrors the pipeline structure of scan_all_yolo.py:
multiprocess 3ch generation + GPU batch inference.
"""

import math
import time
import multiprocessing as _mp
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from app.core.dem import TILE_PX, decode_dem, pixel_to_latlon
from app.core.visualization import dem_to_3ch


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
        img = dem_to_3ch(elev)
        if min(img[:, :, c].std() for c in range(3)) < 3:
            return None
        return (img, tx, ty)
    except Exception:
        return None


def _load_dem_raw(tile_path: str) -> np.ndarray | None:
    """Load DEM tile as elevation array without 3ch conversion."""
    try:
        data = Path(tile_path).read_bytes()
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        r = arr[:, :, 2].astype(np.float64)
        g = arr[:, :, 1].astype(np.float64)
        b = arr[:, :, 0].astype(np.float64)
        x = r * 65536 + g * 256 + b
        elev = np.where(x == 2**23, np.nan, np.where(x > 2**24, (x - 2**24) * 0.01, x * 0.01))
        return elev
    except Exception:
        return None


EXTENDED_PX = TILE_PX + TILE_PX // 2  # 768 = 512 + 256


def _gen_3ch_extended(args: tuple) -> tuple | None:
    """Generate 150% extended 3ch image (768×768) from tile (tx,ty).
    Covers full tile + 256px right + 256px down + 256×256 diagonal.
    args: (tiles_dir, tx, ty, has_right, has_below, has_diag)."""
    tiles_dir, tx, ty, has_right, has_below, has_diag = args
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
    if has_right:
        path_r = str(Path(tiles_dir) / "16" / str(tx + 1) / f"{ty}.webp")
        right = _load_dem_raw(path_r)
        if right is not None:
            canvas[:TILE_PX, TILE_PX:] = right[:, :half]

    # 下タイル (0:512, 512:768) — 上半分の256行
    if has_below:
        path_b = str(Path(tiles_dir) / "16" / str(tx) / f"{ty + 1}.webp")
        below = _load_dem_raw(path_b)
        if below is not None:
            canvas[TILE_PX:, :TILE_PX] = below[:half, :]

    # 右下タイル (512:768, 512:768) — 左上256×256
    if has_diag:
        path_d = str(Path(tiles_dir) / "16" / str(tx + 1) / f"{ty + 1}.webp")
        diag = _load_dem_raw(path_d)
        if diag is not None:
            canvas[TILE_PX:, TILE_PX:] = diag[:half, :half]

    # NaN埋め
    valid = canvas[~np.isnan(canvas)]
    if len(valid) < TILE_PX * TILE_PX * 0.3:
        return None
    canvas[np.isnan(canvas)] = np.nanmean(canvas) if len(valid) > 0 else 0

    try:
        img = dem_to_3ch(canvas)
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
    """Run YOLO inference on a batch."""
    results = model.predict(images, conf=conf_threshold, iou=0.5, device=device,
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
    tile_coord_set = {(tx, ty) for _, tx, ty in tiles}
    extended_tiles = []
    for _, tx, ty in tiles:
        has_right = (tx + 1, ty) in tile_coord_set
        has_below = (tx, ty + 1) in tile_coord_set
        has_diag = (tx + 1, ty + 1) in tile_coord_set
        extended_tiles.append((tiles_dir, tx, ty, has_right, has_below, has_diag))

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
                # 拡張タイルに必要な隣接タイルも事前DL
                needed = set()
                for _, tx, ty, has_r, has_b, has_d in extended_tiles:
                    needed.add((tx, ty))
                    if has_r: needed.add((tx + 1, ty))
                    if has_b: needed.add((tx, ty + 1))
                    if has_d: needed.add((tx + 1, ty + 1))
                with ThreadPoolExecutor(max_workers=8) as executor:
                    def _fetch_one(coord):
                        ttx, tty = coord
                        local = Path(tiles_dir) / "16" / str(ttx) / f"{tty}.webp"
                        if not local.exists():
                            tile_fetcher(16, ttx, tty)
                    list(executor.map(_fetch_one, needed, chunksize=16))
                # DL完了後、拡張タイルをキューに投入
                for et in extended_tiles:
                    dl_queue.put(et)
                dl_done.set()

            dl_thread = threading.Thread(target=_downloader, daemon=True)
            dl_thread.start()

            batch = []
            while True:
                try:
                    item = dl_queue.get(timeout=0.5)
                    batch.append(item)
                    if len(batch) >= 16:
                        for r in pool.imap_unordered(_gen_3ch_extended, batch, chunksize=8):
                            img_queue.put(r)
                        batch.clear()
                except queue.Empty:
                    if dl_done.is_set() and dl_queue.empty():
                        break
            if batch:
                for r in pool.imap_unordered(_gen_3ch_extended, batch, chunksize=8):
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
                    gcx, gcy = d["cx"], d["cy"]
                    w, h = d["w"], d["h"]
                    real_tx = tx + int(gcx) // TILE_PX
                    real_ty = ty + int(gcy) // TILE_PX
                    local_cx = gcx - (real_tx - tx) * TILE_PX
                    local_cy = gcy - (real_ty - ty) * TILE_PX

                    # 中心がbase 512×512内の検出のみ残す
                    # 拡張領域(512-768)はコンテキスト用。そこに中心がある検出は
                    # 隣のタイルのbase領域で検出される
                    if gcx >= TILE_PX or gcy >= TILE_PX:
                        continue
                    # 画像左端/上端にかかるbboxはスキップ（部分的にしか見えない）
                    if gcx - w/2 < 1 or gcy - h/2 < 1:
                        continue

                    lat, lon = pixel_to_latlon(16, real_tx, real_ty, local_cx, local_cy)
                    all_detections.append({
                        "lat": lat, "lon": lon, "conf": d["conf"],
                        "bbox_cx": local_cx, "bbox_cy": local_cy,
                        "bbox_w": w, "bbox_h": h,
                        "tile_x": real_tx, "tile_y": real_ty,
                        "cls": d.get("cls", 0),
                    })

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

"""GPU 生成版のスキャン (scan_tiles の z=16 / CUDA / YOLO end2end 向け実装)。

データの流れ:
  復号プロセス (N 個): DEM WebP 4 枚 → 768x768x3 の生バイトを共有メモリ (/dev/shm) のスロットに直接書く
  主スレッド: スロット → GPU へ DMA → CUDA Graph で 3ch 生成 (別ストリーム、推論と並行)
             → forward + end2end 後処理 (バッチ) → resolve_detection

scan_tiles (CPU 生成 + 3ch キャッシュ) との違い:
  - 3ch キャッシュを使わない。生成が推論より速いので要らない (実測: 本番設定 111 → 413 tile/s)。
  - プロセス間で画像を pickle しない (1 枚 1.77MB × 数百/秒 が主スレッドの CPU を食っていた)。
  - 復号プロセスには「同じ列の連続した行」をまとめて渡す。隣接タイル (右・下・右下) を LRU で
    再利用でき、DEM の読みが 1 枚あたり ~1.3 ファイルで済む (NAS や R2 から読むときに効く)。
  - 推論は ultralytics の predict を通さない (gpu3ch.infer_yolo_direct)。

保つもの: progress_callback / detection_callback / progress_save_callback / cancel_event / resume_from /
tile_fetcher (リモート DEM 取得) の意味。進捗の保存はウィンドウ (SCAN_WINDOW 枚) 境界で行うので、
レジュームは保存位置より前のタイルを二度処理しない。
"""
from __future__ import annotations

import functools
import os
import queue
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from app.core.dem import TILE_PX, dem_deps, pixel_to_latlon
from app.core.gpu3ch import EXTENDED_PX, NAN_BGR, SLOT_BYTES, GraphGen, infer_yolo_direct, ksize_for
from app.core.visualization import cell_size_m
from app.services.detections import resolve_detection

HALF = TILE_PX // 2
WINDOW = int(os.environ.get("SCAN_WINDOW", "1024"))        # 進捗保存とタイル並べ替えの単位
GEN_BATCH = int(os.environ.get("GPU_3CH_BATCH", "32"))      # CUDA Graph のバッチ (静的)
LRU_TILES = int(os.environ.get("DEM_LRU_TILES", "1024"))    # 復号プロセスごとの DEM キャッシュ (約 2 列分)


def gpu_scan_available(model_type: str, z: int) -> bool:
    """この実装を使える条件。外れたら scan_tiles が従来経路で処理する。"""
    if os.environ.get("GPU_3CH", "1").lower() in ("0", "false", "no"):
        return False
    return model_type == "yolo" and z == 16 and torch.cuda.is_available()


def prepare_direct_model(model, conf_threshold: float, device):
    """predictor を初期化して end2end (NMS-free) の生モデルを返す。end2end でなければ None。
    判定をここで済ませ、scan_tiles_gpu の中では例外を握りつぶさない (途中の失敗を CPU 経路で
    黙ってやり直すと、遅くなった理由が見えなくなる)。"""
    dummy = torch.zeros((1, 3, EXTENDED_PX, EXTENDED_PX), dtype=torch.float32, device=device)
    model.predict(dummy, conf=conf_threshold, iou=0.5, device=device, imgsz=EXTENDED_PX, verbose=False)
    net = model.predictor.model
    with torch.no_grad():
        preds = net(dummy)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    if preds.ndim != 3 or preds.shape[-1] != 6:
        return None
    return net


# ---------------------------------------------------------------- 復号プロセス側
def _elev(bgr: np.ndarray) -> np.ndarray:
    r = bgr[:, :, 2].astype(np.float64); g = bgr[:, :, 1].astype(np.float64); b = bgr[:, :, 0].astype(np.float64)
    x = r * 65536 + g * 256 + b
    return np.where(x == 2**23, np.nan, np.where(x > 2**23, (x - 2**24) * 0.01, x * 0.01))


def _make_reader(maxsize: int):
    @functools.lru_cache(maxsize=maxsize)
    def read_bgr(path: str):
        try:
            data = Path(path).read_bytes()
        except OSError:
            return None
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return read_bgr


def build_canvas(read_bgr, tiles_dir: str, tx: int, ty: int, bgr: np.ndarray):
    """DEM 4 枚から 768x768x3 生バイトを bgr に書く。戻り値 (mean, cs, ksize) か None。
    判定は scanning._gen_3ch_uncached と同じ: 本体の有効画素 30% 未満 / canvas の有効画素 30% 未満 → None。"""
    main = read_bgr(str(Path(tiles_dir) / "16" / str(tx) / f"{ty}.webp"))
    if main is None:
        return None
    main_e = _elev(main)
    if np.count_nonzero(~np.isnan(main_e)) < TILE_PX * TILE_PX * 0.3:
        return None
    bgr[:] = NAN_BGR
    canvas = np.full((EXTENDED_PX, EXTENDED_PX), np.nan)
    bgr[:TILE_PX, :TILE_PX] = main
    canvas[:TILE_PX, :TILE_PX] = main_e
    for (dx, dy), (sl_y, sl_x), (src_y, src_x) in (
        ((1, 0), (slice(0, TILE_PX), slice(TILE_PX, None)), (slice(None), slice(0, HALF))),
        ((0, 1), (slice(TILE_PX, None), slice(0, TILE_PX)), (slice(0, HALF), slice(None))),
        ((1, 1), (slice(TILE_PX, None), slice(TILE_PX, None)), (slice(0, HALF), slice(0, HALF))),
    ):
        nb = read_bgr(str(Path(tiles_dir) / "16" / str(tx + dx) / f"{ty + dy}.webp"))
        if nb is not None:
            part = nb[src_y, src_x]
            bgr[sl_y, sl_x] = part
            canvas[sl_y, sl_x] = _elev(part)
    valid = np.count_nonzero(~np.isnan(canvas))
    if valid < TILE_PX * TILE_PX * 0.3:
        return None
    mean = float(np.nanmean(canvas)) if valid > 0 else 0.0
    lat, _ = pixel_to_latlon(16, tx, ty, EXTENDED_PX / 2, EXTENDED_PX / 2)
    cs = cell_size_m(lat, 16)
    return mean, cs, ksize_for(cs)


def _attach_shm(name: str, slots: int) -> np.ndarray:
    """/dev/shm/<name> を直接 mmap する。SharedMemory(name=) だと resource_tracker に登録され、
    終了時に二重 unregister で KeyError を吐くので使わない。"""
    import mmap
    fd = os.open(f"/dev/shm/{name}", os.O_RDWR)
    try:
        m = mmap.mmap(fd, slots * SLOT_BYTES, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    return np.ndarray((slots, EXTENDED_PX, EXTENDED_PX, 3), np.uint8, buffer=m)


def decode_worker(tiles_dir: str, shm_name: str, slots: int, task_q, free_q, res_q, lru: int) -> None:
    """task_q: (window_idx, task_id, [(tx, [ty, ...]), ...]) を順に処理し、res_q にタイルごとの結果を流す。
    結果: (slot, mean, cs, ksize, tx, ty, window_idx, task_id) か (None, tx, ty, window_idx, task_id)。None で終了。
    1 タスクは隣接する数列。同じプロセスが隣の列を続けて処理するので、右隣の列の DEM が LRU に残る。"""
    view = _attach_shm(shm_name, slots)
    read_bgr = _make_reader(lru)
    while True:
        task = task_q.get()
        if task is None:
            break
        widx, tid, cols = task
        for tx, tys in cols:
            for ty in tys:
                main = read_bgr(str(Path(tiles_dir) / "16" / str(tx) / f"{ty}.webp"))
                if main is None or np.count_nonzero(~np.isnan(_elev(main))) < TILE_PX * TILE_PX * 0.3:
                    res_q.put((None, tx, ty, widx, tid))
                    continue
                slot = free_q.get()
                r = build_canvas(read_bgr, tiles_dir, tx, ty, view[slot])
                if r is None:
                    free_q.put(slot)
                    res_q.put((None, tx, ty, widx, tid))
                else:
                    res_q.put((slot, r[0], r[1], r[2], tx, ty, widx, tid))
    res_q.put(("done",))


def _group_by_column(tiles: list[tuple[int, int]]) -> list[tuple[int, list[int]]]:
    by: dict[int, list[int]] = {}
    for tx, ty in tiles:
        by.setdefault(tx, []).append(ty)
    return [(tx, sorted(by[tx])) for tx in sorted(by)]


def _slot_count(requested: int) -> int:
    """/dev/shm の空きに収まるスロット数。--shm-size=2g なら約 1,000 枚分。"""
    try:
        st = os.statvfs("/dev/shm")
        avail = st.f_bavail * st.f_frsize
        return max(2 * GEN_BATCH, min(requested, int(avail * 0.8 // SLOT_BYTES)))
    except OSError:
        return requested


# ---------------------------------------------------------------- 本体
def scan_tiles_gpu(
    net,
    tiles_dir: str,
    tiles: list[tuple[str, int, int]],
    conf_threshold: float,
    batch_size: int,
    num_workers: int,
    progress_callback: Callable[[float, str], None] | None,
    detection_callback: Callable[[list[dict]], None] | None,
    tile_fetcher: Callable[[int, int, int], str | None] | None,
    resume_from: int,
    progress_save_callback: Callable[[int], None] | None,
    cancel_event,
    z: int = 16,
) -> list[dict]:
    import multiprocessing as mp
    device = torch.device("cuda")
    total_tiles = len(tiles)
    todo = [(tx, ty) for _, tx, ty in tiles[resume_from:]]
    if not todo:
        return []
    windows = [todo[i:i + WINDOW] for i in range(0, len(todo), WINDOW)]

    def _is_cancelled():
        return cancel_event is not None and cancel_event.is_set()

    ctx = mp.get_context("forkserver")
    slots = _slot_count(int(os.environ.get("GPU_3CH_SLOTS", "1024")))
    shm = shared_memory.SharedMemory(create=True, size=slots * SLOT_BYTES)
    procs: list = []
    feeder: threading.Thread | None = None
    stop = threading.Event()
    all_detections: list[dict] = []
    total_detections = 0
    try:
        view = np.ndarray((slots, EXTENDED_PX, EXTENDED_PX, 3), np.uint8, buffer=shm.buf)
        shm_t = torch.from_numpy(view)
        registered = int(torch.cuda.cudart().cudaHostRegister(shm_t.data_ptr(), shm_t.numel(), 0)) == 0
        gen = GraphGen(device, GEN_BATCH, shm_t, compiled=os.environ.get("GPU_3CH_COMPILE", "1") != "0")
        ksizes = {ksize_for(cell_size_m(pixel_to_latlon(16, tx, ty, EXTENDED_PX / 2, EXTENDED_PX / 2)[0], 16))
                  for tx, ty in todo}
        gen.prebuild(ksizes)
        if progress_callback:
            progress_callback(resume_from / total_tiles,
                              f"GPU 3ch ready (compiled={gen.compiled}, slots={slots}, decoders={num_workers})")

        task_q, free_q = ctx.Queue(), ctx.Queue()
        res_q = ctx.Queue(maxsize=slots)
        for i in range(slots):
            free_q.put(i)
        for _ in range(num_workers):
            p = ctx.Process(target=decode_worker, args=(tiles_dir, shm.name, slots, task_q, free_q, res_q, LRU_TILES),
                            daemon=True)
            p.start()
            procs.append(p)

        # ウィンドウごとに (リモートなら依存 DEM を揃えてから) 復号タスクを投入する。
        # タスクは隣接する最大 COLS_PER_TASK 列。同時に投入できるタスク数を抑えて、
        # 復号が主スレッドより先に走りすぎないようにする (スロットの枯渇とレジュームの粒度のため)。
        COLS_PER_TASK = 4
        inflight = threading.Semaphore(4 * max(1, num_workers))
        task_left: dict[int, int] = {}
        task_lock = threading.Lock()

        def _feed():
            from concurrent.futures import ThreadPoolExecutor
            tid = 0
            for widx, win in enumerate(windows):
                if stop.is_set():
                    return
                if tile_fetcher:
                    need = {c for tx, ty in win
                            for nx, ny in ((tx, ty), (tx + 1, ty), (tx, ty + 1), (tx + 1, ty + 1))
                            for c in dem_deps(z, nx, ny)}
                    missing = [c for c in need if not (Path(tiles_dir) / "16" / str(c[0]) / f"{c[1]}.webp").exists()]
                    with ThreadPoolExecutor(max_workers=32) as ex:
                        list(ex.map(lambda c: tile_fetcher(16, c[0], c[1]), missing, chunksize=16))
                cols = _group_by_column(win)
                for i in range(0, len(cols), COLS_PER_TASK):
                    part = cols[i:i + COLS_PER_TASK]
                    while not inflight.acquire(timeout=0.5):
                        if stop.is_set():
                            return
                    with task_lock:
                        task_left[tid] = sum(len(tys) for _, tys in part)
                    task_q.put((widx, tid, part))
                    tid += 1
            for _ in procs:
                task_q.put(None)

        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()

        buckets: dict[int, list] = {}
        ready: list[tuple] = []
        n_ready = 0
        pending_release: list[tuple] = []
        win_left = [len(w) for w in windows]
        done_prefix = 0          # 先頭から何ウィンドウ完了したか
        saved_at = resume_from
        processed = 0
        workers_done = 0
        t0 = time.monotonic()
        prof = {"wait": 0.0, "issue": 0.0, "gather": 0.0, "infer": 0.0, "n_gen": 0, "n_inf": 0}
        profiling = os.environ.get("GPU_3CH_PROFILE", "0") == "1"

        def release_slots():
            nonlocal pending_release
            keep_ = []
            for ev, sl in pending_release:
                if ev.query():
                    for s_ in sl:
                        free_q.put(s_)
                else:
                    keep_.append((ev, sl))
            pending_release = keep_

        def flush_bucket(k):
            nonlocal n_ready
            items = buckets.pop(k, [])
            if not items:
                return
            tq = time.perf_counter()
            imgs, keep, ev, h2d = gen(items)
            prof["issue"] += time.perf_counter() - tq; prof["n_gen"] += 1
            ready.append((imgs, keep, [(it[4], it[5]) for it in items], ev))
            n_ready += len(items)
            pending_release.append((h2d, [it[0] for it in items]))

        def flush_inf():
            nonlocal n_ready, total_detections
            if not ready:
                return
            tq = time.perf_counter()
            take, cnt = [], 0
            while ready and cnt < batch_size:
                take.append(ready.pop(0))
                cnt += len(take[-1][2])
            n_ready -= cnt
            cur = torch.cuda.current_stream()
            for r in take:
                cur.wait_event(r[3])
            keep = torch.cat([r[1] for r in take])
            imgs = torch.cat([r[0] for r in take])[keep]
            kl = keep.tolist()
            metas = [m for r in take for m in r[2]]
            metas = [m for m, kp in zip(metas, kl) if kp]
            prof["gather"] += time.perf_counter() - tq
            if not metas:
                return
            tq = time.perf_counter()
            batch_dets = infer_yolo_direct(net, imgs, conf_threshold)
            prof["infer"] += time.perf_counter() - tq; prof["n_inf"] += 1
            for dets, (tx, ty) in zip(batch_dets, metas):
                for d in dets:
                    rec = resolve_detection(d, tx, ty, z)
                    if rec is not None:
                        all_detections.append(rec)
            if detection_callback and all_detections:
                detection_callback(all_detections[:])
                total_detections += len(all_detections)
                all_detections.clear()

        def account(widx, tid):
            """タスク完了で投入枠を返し、ウィンドウ完了の前進に合わせて進捗を保存する (レジュームの正しさはここで決まる)。"""
            nonlocal done_prefix, saved_at
            with task_lock:
                task_left[tid] -= 1
                if task_left[tid] == 0:
                    del task_left[tid]
                    inflight.release()
            win_left[widx] -= 1
            if widx == done_prefix:
                while done_prefix < len(windows) and win_left[done_prefix] == 0:
                    done_prefix += 1
                abs_done = resume_from + sum(len(w) for w in windows[:done_prefix])
                if progress_save_callback and abs_done - saved_at >= 10000:
                    progress_save_callback(abs_done)
                    saved_at = abs_done

        while workers_done < len(procs):
            if _is_cancelled():
                break
            tq = time.perf_counter()
            try:
                r = res_q.get(timeout=0.5)
            except queue.Empty:
                prof["wait"] += time.perf_counter() - tq
                release_slots()
                continue
            prof["wait"] += time.perf_counter() - tq
            if r[0] == "done":
                workers_done += 1
                continue
            processed += 1
            widx, tid = r[-2], r[-1]
            if r[0] is None:
                account(widx, tid)
            else:
                buckets.setdefault(r[3], []).append(r)
                if len(buckets[r[3]]) >= GEN_BATCH:
                    flush_bucket(r[3])
            if n_ready >= batch_size + GEN_BATCH:
                flush_inf()
            release_slots()
            # 完了の計上: 生成に回したタイルは推論後ではなく受領時点で数える (旧実装と同じ粒度)
            if r[0] is not None:
                account(widx, tid)
            if progress_callback and processed % 1000 == 0:
                elapsed = time.monotonic() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                abs_processed = resume_from + processed
                det_count = total_detections + len(all_detections)
                progress_callback(abs_processed / total_tiles,
                                  f"{abs_processed:,}/{total_tiles:,} tiles, {det_count:,} detections ({rate:.0f} t/s)")
        if not _is_cancelled():
            for k in list(buckets):
                flush_bucket(k)
            while ready:
                flush_inf()
            torch.cuda.synchronize()
            if progress_save_callback:
                progress_save_callback(resume_from + processed)
            if profiling and progress_callback and processed:
                per = 1000 / processed
                progress_callback(1.0, "profile ms/tile: wait-for-decoders %.2f | gen-issue %.2f | gather %.2f | infer %.2f | wall %.2f (gen calls %d, infer calls %d)" % (
                    prof["wait"] * per, prof["issue"] * per, prof["gather"] * per, prof["infer"] * per,
                    (time.monotonic() - t0) * per, prof["n_gen"], prof["n_inf"]))
    finally:
        stop.set()
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        if feeder is not None:
            feeder.join(timeout=5)
        try:
            if registered:
                torch.cuda.cudart().cudaHostUnregister(shm_t.data_ptr())
        except Exception:
            pass
        shm.close()
        shm.unlink()
    return all_detections

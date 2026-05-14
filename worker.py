"""GPU Worker — polls the GeoScope server API for train/scan jobs and executes them.

Runs on a local GPU machine or in a cloud Pod (Docker). Communicates with the
server exclusively via HTTP API endpoints protected by WORKER_API_KEY.

Requirements:
  - Access to DEM tiles (TILES_DIR) or REMOTE_TILES=true
  - PyTorch with CUDA
  - ultralytics (YOLO)
"""

import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

import requests

# Add the backend directory so we can import app.services.*
_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.services.dataset import generate_dataset
from app.services.scanning import scan_tiles

import json
import shutil

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_URL = os.environ.get("GEOSCOPE_SERVER", "https://geoscope.jp")
WORKER_KEY = os.environ.get("WORKER_API_KEY", "")
TILES_DIR = os.environ.get("TILES_DIR", "/home/tasuku/jp/qchizu/gml_tiles")
MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
DATASETS_DIR = os.environ.get("DATASETS_DIR", "./datasets")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))

REMOTE_TILES = os.environ.get("REMOTE_TILES", "false").lower() in ("1", "true", "yes")
DISABLE_PID_LOCK = os.environ.get("DISABLE_PID_LOCK", "false").lower() in ("1", "true", "yes")
# DEM タイル配布元 (Cloudflare R2 等)。
# 例: "https://pub-abc.r2.dev"。空なら GEOSCOPE_SERVER/tiles/dem を使う。
DEM_TILE_BASE_URL = os.environ.get("DEM_TILE_BASE_URL", "").rstrip("/")
PREFETCH_PARALLEL = int(os.environ.get("PREFETCH_PARALLEL", "256"))

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    return {"Authorization": f"Bearer {WORKER_KEY}"}


# IP直接接続時のSSL証明書不一致を許容
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_VERIFY_SSL = not SERVER_URL.replace("https://", "").replace("http://", "").replace("/", "").replace(".", "").replace(":", "").isdigit()
# IPアドレスの場合はSSL検証を無効化
if any(c.isdigit() for c in SERVER_URL.split("//")[-1].split(":")[0].split("/")[0].split(".")[:1]):
    import re
    _VERIFY_SSL = not bool(re.match(r'https?://\d+\.\d+\.\d+\.\d+', SERVER_URL))


def api_get(path: str) -> dict | list | None:
    return _api_call("GET", path)


def _api_call(method: str, path: str, body: dict | None = None, timeout: int = 120) -> dict:
    """HTTP request with retry on transient errors (5xx, connection)."""
    import time as _time
    for attempt in range(4):
        try:
            fn = requests.put if method == "PUT" else requests.post if method == "POST" else requests.get
            resp = fn(f"{SERVER_URL}{path}", headers=_headers(), json=body, timeout=timeout, verify=_VERIFY_SSL)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < 3:
                print(f"API {method} {path} failed (attempt {attempt+1}): {e}, retrying...")
                _time.sleep(3 * (attempt + 1))
            else:
                raise
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code >= 500 and attempt < 3:
                print(f"API {method} {path} got {e.response.status_code} (attempt {attempt+1}), retrying...")
                _time.sleep(3 * (attempt + 1))
            else:
                raise


def api_put(path: str, body: dict | None = None) -> dict:
    return _api_call("PUT", path, body or {})


def api_post(path: str, body: dict) -> dict:
    return _api_call("POST", path, body, timeout=60)


_DEM_ARCHIVE_NAME = "dem-z16.tar"
_DEM_ARCHIVE_MARKER = ".dem_z16_extracted"


def _init_dem_from_archive() -> None:
    """初回 Pod 起動時に R2 から DEM tar アーカイブを stream DL + 展開する.

    既に展開済み (marker file 存在 or タイル数が一定以上) ならスキップ。
    REMOTE_TILES=true かつ DEM_TILE_BASE_URL 設定時のみ動作。
    SKIP_DEM_EXTRACT=true (永続 volume 無し設計) なら何もしない (per-tile fetch で動く)。
    """
    if os.environ.get("SKIP_DEM_EXTRACT", "").lower() in ("1", "true", "yes"):
        print("SKIP_DEM_EXTRACT set; DEM tiles will be fetched per-tile from R2")
        return
    if not (REMOTE_TILES and DEM_TILE_BASE_URL):
        return
    tiles_root = Path(TILES_DIR)
    marker = tiles_root / _DEM_ARCHIVE_MARKER
    if marker.exists():
        print(f"DEM archive already extracted (marker: {marker})")
        return

    # 念のため、既に大量のタイルが展開済みなら何もしない
    z16_dir = tiles_root / "16"
    if z16_dir.exists():
        try:
            existing = sum(1 for _ in z16_dir.iterdir())
            if existing >= 1000:  # サブディレクトリ (x 座標) が 1000+ あれば展開済み相当
                print(f"DEM tiles already present ({existing} x-dirs), skipping archive download")
                marker.write_text("ok")
                return
        except Exception:
            pass

    archive_url = f"{DEM_TILE_BASE_URL}/{_DEM_ARCHIVE_NAME}"
    tiles_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DEM archive from {archive_url} -> {tiles_root} (stream extract)...")
    t0 = time.time()
    import subprocess
    # curl で stream DL → tar に直接 pipe で展開 (一時ファイル不要)
    proc = subprocess.run(
        ["bash", "-c", f"curl -fsSL --retry 3 '{archive_url}' | tar -xf - -C '{tiles_root}'"],
        check=False,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"DEM archive extraction failed (rc={proc.returncode}, {elapsed:.0f}s). "
              f"Will fall back to per-tile fetch.")
        return
    try:
        marker.write_text("ok")
    except Exception:
        pass
    try:
        n = sum(1 for _ in z16_dir.iterdir()) if z16_dir.exists() else 0
        print(f"DEM archive extracted: {n} x-dirs ({elapsed:.0f}s)")
    except Exception:
        print(f"DEM archive extracted ({elapsed:.0f}s)")


def _tile_urls(z: int, x: int, y: int) -> list[str]:
    """DEM タイル取得URL候補. R2 優先、なければ GeoScope サーバー fallback."""
    urls = []
    if DEM_TILE_BASE_URL:
        urls.append(f"{DEM_TILE_BASE_URL}/{z}/{x}/{y}.webp")
    urls.append(f"{SERVER_URL}/tiles/dem/{z}/{x}/{y}.webp")
    return urls


def fetch_tile(z: int, x: int, y: int, session: requests.Session | None = None) -> str | None:
    """Ensure a DEM tile exists locally. Downloads from R2 or server if REMOTE_TILES is enabled.
    Returns the local path, or None if unavailable.

    session 指定時はそれを再利用 (HTTP keep-alive で並列 prefetch 高速化)。
    """
    local = Path(TILES_DIR) / str(z) / str(x) / f"{y}.webp"
    if local.exists():
        return str(local)
    if not REMOTE_TILES:
        return None
    s = session or requests
    for url in _tile_urls(z, x, y):
        try:
            resp = s.get(url, timeout=15, verify=_VERIFY_SSL)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(resp.content)
            return str(local)
        except Exception:
            continue
    return None


def _prefetch_tiles(tiles_needed: set[tuple[int, int, int]], label: str = "tiles",
                    max_workers: int | None = None) -> tuple[int, int]:
    """指定 (z, x, y) のセットを並列ダウンロード. 戻り値 (fetched, failed).

    R2 等 high-bandwidth origin 想定で並列度デフォルト PREFETCH_PARALLEL (=256)。
    HTTP keep-alive 用に session を ThreadLocal で再利用。
    """
    if not REMOTE_TILES or not tiles_needed:
        return 0, 0
    import concurrent.futures as _cf
    import threading as _th
    n = max_workers or PREFETCH_PARALLEL
    total = len(tiles_needed)
    print(f"  Pre-fetching {total:,} {label} ({n} parallel)...")
    _local = _th.local()
    def _ensure_session():
        s = getattr(_local, "s", None)
        if s is None:
            s = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_maxsize=n, pool_connections=n)
            s.mount("http://", adapter)
            s.mount("https://", adapter)
            _local.s = s
        return s
    fetched = failed = 0
    t0 = time.time()
    last_report = t0
    def _job(t):
        return fetch_tile(t[0], t[1], t[2], session=_ensure_session())
    with _cf.ThreadPoolExecutor(max_workers=n) as ex:
        for i, ok in enumerate(ex.map(_job, tiles_needed), 1):
            if ok:
                fetched += 1
            else:
                failed += 1
            now = time.time()
            if now - last_report >= 5.0:
                rate = i / (now - t0) if now > t0 else 0
                mbps = rate * 40 * 8 / 1000  # 40KB/tile 想定
                print(f"    progress: {i:,}/{total:,} ({rate:.0f} tile/s ≈ {mbps:.0f} Mbps)")
                last_report = now
    elapsed = time.time() - t0
    rate = total / max(elapsed, 0.001)
    print(f"  Pre-fetch {label}: {fetched:,} ok, {failed:,} missing ({elapsed:.1f}s, {rate:.0f} tile/s)")
    return fetched, failed


def _prefetch_annotation_tiles(annotations: list[dict]) -> None:
    """アノテーション (+ 拡張タイル用隣接) を prefetch."""
    if not REMOTE_TILES:
        return
    tiles_needed: set[tuple[int, int, int]] = set()
    for a in annotations:
        z = a.get("tile_z") or 16
        tx = a.get("tile_x")
        ty = a.get("tile_y")
        if tx is None or ty is None:
            continue
        # 150% extended tile uses neighbors (right, below, diagonal)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            tiles_needed.add((z, tx + dx, ty + dy))
    _prefetch_tiles(tiles_needed, label="annotation DEM tiles")


def upload_model(project_id: str, model_path: str):
    """Upload trained model to server."""
    import time as _time
    for attempt in range(4):
        try:
            with open(model_path, "rb") as f:
                resp = requests.post(
                    f"{SERVER_URL}/api/worker/models/{project_id}/upload",
                    headers=_headers(), files={"file": ("best.pt", f)},
                    timeout=120, verify=_VERIFY_SSL,
                )
            resp.raise_for_status()
            print(f"  Model uploaded ({resp.json().get('size', 0)} bytes)")
            return resp.json()
        except Exception as e:
            if attempt < 3:
                print(f"  Upload failed (attempt {attempt+1}): {e}, retrying...")
                _time.sleep(5)
            else:
                raise


def download_model(project_id: str, dest_path: str) -> bool:
    """Download model from server. Returns True if downloaded."""
    import time as _time
    for attempt in range(4):
        try:
            resp = requests.get(
                f"{SERVER_URL}/api/worker/models/{project_id}/download",
                headers=_headers(), timeout=120, verify=_VERIFY_SSL,
            )
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
            Path(dest_path).write_bytes(resp.content)
            print(f"  Model downloaded ({len(resp.content)} bytes)")
            return True
        except Exception as e:
            if attempt < 3:
                print(f"  Download failed (attempt {attempt+1}): {e}, retrying...")
                _time.sleep(5)
            else:
                raise


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

def poll_job() -> dict | None:
    """Fetch the oldest pending job. Returns None if no jobs available."""
    data = api_get("/api/worker/jobs/pending")
    if not data or "id" not in data:
        return None
    return data


def start_job(job_id: str, config_update: dict | None = None):
    api_put(f"/api/worker/jobs/{job_id}/start", {"config_update": config_update} if config_update else {})


class JobCancelled(Exception):
    pass

class JobAlreadyDone(Exception):
    """パラレルスキャン: 他ワーカーが先に完了済み"""
    pass


def update_progress(job_id: str, progress: float, message: str):
    resp = api_put(f"/api/worker/jobs/{job_id}/progress", {
        "progress": progress,
        "message": message,
    })
    if resp and resp.get("cancelled"):
        reason = resp.get("reason", "cancelled")
        raise JobCancelled(f"Job {job_id} {reason}")
    if resp and resp.get("done"):
        raise JobAlreadyDone(f"Job {job_id} already completed by another worker")


def complete_job(job_id: str, result: dict, model_path: str | None = None):
    api_put(f"/api/worker/jobs/{job_id}/complete", {
        "result": result,
        "model_path": model_path,
    })


def fail_job(job_id: str, error: str):
    api_put(f"/api/worker/jobs/{job_id}/fail", {"error": error[:2000]})


def post_detections(project_id: str, model_id: str | None, detections: list[dict]):
    """Upload detections to server in chunks."""
    CHUNK = 5000
    for i in range(0, len(detections), CHUNK):
        chunk = detections[i : i + CHUNK]
        api_post("/api/worker/detections/bulk", {
            "project_id": project_id,
            "model_id": model_id,
            "detections": chunk,
        })


# ---------------------------------------------------------------------------
# Job handlers
# ---------------------------------------------------------------------------

def _coco_to_yolo(coco_dir: str, yolo_dir: str):
    """Convert COCO dataset to YOLO format."""
    coco_path = Path(coco_dir)
    yolo_path = Path(yolo_dir)
    if yolo_path.exists():
        shutil.rmtree(yolo_path)

    for split in ["train", "val"]:
        coco_json = coco_path / f"{split}.json"
        if not coco_json.exists():
            continue
        with open(coco_json) as f:
            data = json.load(f)

        img_out = yolo_path / "images" / split
        lbl_out = yolo_path / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        # Group annotations by image_id
        img_annots = {}
        for ann in data.get("annotations", []):
            img_annots.setdefault(ann["image_id"], []).append(ann)

        for img_info in data["images"]:
            src = coco_path / "images" / img_info["file_name"]
            if not src.exists():
                continue
            # Symlink image
            dst = img_out / img_info["file_name"]
            if not dst.exists():
                dst.symlink_to(src.resolve())
            # Write YOLO label: class cx cy w h (normalized 0-1)
            iw, ih = img_info["width"], img_info["height"]
            annots = img_annots.get(img_info["id"], [])
            label_file = lbl_out / (Path(img_info["file_name"]).stem + ".txt")
            with open(label_file, "w") as lf:
                for ann in annots:
                    x, y, w, h = ann["bbox"]
                    cx = (x + w / 2) / iw
                    cy = (y + h / 2) / ih
                    nw = w / iw
                    nh = h / ih
                    cls_idx = ann.get("category_id", 1) - 1  # COCO 1-based → YOLO 0-based
                    lf.write(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Write data.yaml (read categories from COCO JSON)
    categories = data.get("categories", [{"id": 1, "name": "target"}])
    nc = len(categories)
    names = [c["name"] for c in sorted(categories, key=lambda c: c["id"])]
    yaml_path = yolo_path / "data.yaml"
    yaml_path.write_text(
        f"path: {yolo_path.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )


def train_yolo(yolo_dir: str, model_dir: str, epochs: int = 100,
               batch_size: int = 16, progress_callback=None) -> dict:
    """Train YOLO model using ultralytics."""
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    data_yaml = str(Path(yolo_dir) / "data.yaml")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=512,
        batch=batch_size,
        device=0,
        patience=20,
        augment=True,
        degrees=360,
        flipud=0.5,
        fliplr=0.5,
        scale=0.3,
        mosaic=0.5,
        exist_ok=True,
        verbose=False,
    )

    save_dir = Path(results.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.exists():
        best_path = save_dir / "weights" / "last.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No model found in {save_dir}")

    out_path = Path(model_dir) / "best.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, out_path)

    return {
        "model_type": "yolo",
        "epochs": epochs,
        "model_path": str(out_path),
    }


def train_rtdetr(coco_dir: str, model_dir: str, epochs: int = 30,
                 batch_size: int = 4, progress_callback=None) -> dict:
    """Train RT-DETR using HuggingFace Trainer. Apache 2.0 license."""
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        RTDetrForObjectDetection, RTDetrImageProcessor,
        TrainingArguments, Trainer,
    )

    MODEL_NAME = "PekingU/rtdetr_r18vd"
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = str(out_dir / "trainer_output")

    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    model = RTDetrForObjectDetection.from_pretrained(
        MODEL_NAME, num_labels=1, ignore_mismatched_sizes=True,
    )

    class CocoDataset(Dataset):
        def __init__(self, coco_json, images_dir):
            with open(coco_json) as f:
                data = json.load(f)
            self.images_dir = Path(images_dir)
            img_annots = {}
            for ann in data.get("annotations", []):
                img_annots.setdefault(ann["image_id"], []).append(ann)
            self.images = [img for img in data["images"] if img["id"] in img_annots]
            self.img_annots = img_annots

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            import cv2
            img_info = self.images[idx]
            img = cv2.imread(str(self.images_dir / img_info["file_name"]))
            if img is None:
                img = np.zeros((512, 512, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            annots = self.img_annots[img_info["id"]]
            iw, ih = img_info["width"], img_info["height"]
            coco_annots = []
            for ann in annots:
                x, y, w, h = ann["bbox"]
                if w < 1 or h < 1:
                    continue
                coco_annots.append({
                    "bbox": [x, y, w, h],
                    "category_id": 0,
                    "area": w * h,
                    "iscrowd": 0,
                })

            if not coco_annots:
                coco_annots = [{"bbox": [iw*0.4, ih*0.4, iw*0.01, ih*0.01], "category_id": 0, "area": 1, "iscrowd": 0}]

            target = {"image_id": img_info["id"], "annotations": coco_annots}
            encoding = processor(images=img, annotations=[target], return_tensors="pt")
            # Squeeze batch dim
            return {k: v.squeeze(0) if hasattr(v, 'squeeze') else v[0] for k, v in encoding.items()}

    coco_path = Path(coco_dir)
    train_ds = CocoDataset(coco_path / "train.json", coco_path / "images")
    val_ds = CocoDataset(coco_path / "val.json", coco_path / "images")

    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        labels = [b["labels"] for b in batch]
        return {"pixel_values": pixel_values, "labels": labels}

    args = TrainingArguments(
        output_dir=train_out,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1e-4,
        weight_decay=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.1,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    class ProgressTrainer(Trainer):
        def on_epoch_end(self, args, state, control, **kwargs):
            if progress_callback and state.epoch:
                train_loss = state.log_history[-1].get("loss", 0) if state.log_history else 0
                progress_callback(
                    state.epoch / args.num_train_epochs,
                    f"Epoch {int(state.epoch)}/{int(args.num_train_epochs)} — loss={train_loss:.4f}"
                )

    trainer = ProgressTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    trainer.train()

    # Save best model
    best_dir = str(out_dir / "best")
    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)

    # Get best metrics
    metrics = trainer.evaluate()
    best_val_loss = metrics.get("eval_loss", 0)

    return {
        "model_type": "rtdetr",
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "model_path": best_dir,
    }


def handle_train(job: dict):
    """Execute a training job."""
    job_id = job["id"]
    project_id = job["project_id"]
    config = job.get("config") or {}

    # 1. Fetch annotations from the server
    update_progress(job_id, 0.0, "Fetching annotations...")
    annotations = api_get(f"/api/worker/projects/{project_id}/annotations")
    if not annotations:
        fail_job(job_id, "No annotations found for this project")
        return

    print(f"  Fetched {len(annotations)} annotations")

    # 2. Generate COCO dataset
    train_label = config.get("train_label")
    update_progress(job_id, 0.05, f"Generating dataset ({train_label})...")
    _prefetch_annotation_tiles(annotations)
    dataset_dir = str(Path(DATASETS_DIR) / project_id)
    ds_info = generate_dataset(
        annotations,
        TILES_DIR,
        dataset_dir,
        train_label=train_label,
    )
    pos = ds_info.get("positive", 0)
    neg = ds_info.get("negative", 0)
    print(f"  Dataset: {ds_info}")
    update_progress(job_id, 0.08, f"Dataset: ⭕{pos} ❌{neg}")

    if pos == 0:
        fail_job(job_id, "No valid annotations for dataset generation")
        return

    # 3. Train with DINO (Deformable DETR, Apache 2.0)
    model_dir = str(Path(MODELS_DIR) / project_id)

    def on_progress(p: float, msg: str):
        scaled = 0.10 + p * 0.85
        update_progress(job_id, scaled, msg)

    # COCO → YOLO形式変換
    yolo_dir = str(Path(dataset_dir) / "yolo")
    _coco_to_yolo(dataset_dir, yolo_dir)

    result = train_yolo(
        yolo_dir,
        model_dir,
        epochs=config.get("epochs", 100),
        batch_size=config.get("batch_size", 16),
        progress_callback=on_progress,
    )
    print(f"  Training result: {result}")

    # 4. Upload model to server (for multi-worker support)
    if result.get("model_path") and Path(result["model_path"]).exists():
        update_progress(job_id, 0.97, "Uploading model...")
        upload_model(project_id, result["model_path"])

    # 5. Report model to server
    result["dataset"] = ds_info
    complete_job(job_id, result, model_path=result.get("model_path"))


def _post_annotations(project_id: str, annotations: list[dict], prefecture: str | None = None, scan_label: str | None = None):
    """Upload annotations to server in chunks."""
    CHUNK = 500
    for i in range(0, len(annotations), CHUNK):
        payload = {
            "project_id": project_id,
            "prefecture": prefecture,
            "annotations": annotations[i : i + CHUNK],
        }
        if scan_label:
            payload["scan_label"] = scan_label
        api_post("/api/worker/annotations/bulk", payload)


# バックグラウンドアップロード用スレッドプール
from concurrent.futures import ThreadPoolExecutor
_upload_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="upload")
_upload_futures = []

_MAX_PENDING_UPLOADS = 4  # これ以上溜まったら推論を待たせる

def _post_annotations_async(project_id: str, annotations: list[dict], prefecture: str | None = None, scan_label: str | None = None):
    """Upload annotations in background thread. 溜まりすぎたらブロックして待つ。"""
    global _upload_futures
    # 完了済みを刈り取り
    done = [f for f in _upload_futures if f.done()]
    for f in done:
        try:
            f.result()
        except Exception as e:
            print(f"[WARN] Background upload failed: {e}")
    _upload_futures = [f for f in _upload_futures if not f.done()]
    # 溜まりすぎたら最も古いのを待つ（バックプレッシャー）
    while len(_upload_futures) >= _MAX_PENDING_UPLOADS:
        _upload_futures[0].result()
        _upload_futures = [f for f in _upload_futures if not f.done()]
    # 投入
    fut = _upload_pool.submit(_post_annotations, project_id, list(annotations), prefecture, scan_label)
    _upload_futures.append(fut)

def _flush_uploads():
    """Wait for all pending background uploads to complete."""
    global _upload_futures
    for f in _upload_futures:
        try:
            f.result()
        except Exception as e:
            print(f"[WARN] Background upload failed: {e}")
    _upload_futures = []


def _annotations_hash(annotations: list[dict]) -> str:
    """Hash voted annotations to detect training data changes."""
    import hashlib
    voted = sorted(
        [(a["id"], a.get("annotation_vote", "")) for a in annotations if a.get("annotation_vote")],
        key=lambda x: x[0],
    )
    return hashlib.sha256(str(voted).encode()).hexdigest()[:16]


def _train_if_needed(job_id: str, project_id: str, config: dict) -> str | None:
    """Train model if annotations changed. Returns model path or None."""
    update_progress(job_id, 0.0, "Checking annotations...")
    annotations = _api_call("GET", f"/api/worker/projects/{project_id}/annotations", timeout=120)
    if not annotations:
        return None

    current_hash = _annotations_hash(annotations)
    model_dir = str(Path(MODELS_DIR) / project_id)
    model_path = str(Path(model_dir) / "best.pt")
    hash_file = Path(model_dir) / ".train_hash"

    # Check if model exists and hash matches
    if Path(model_path).exists() and hash_file.exists():
        prev_hash = hash_file.read_text().strip()
        if prev_hash == current_hash:
            print(f"  Annotations unchanged ({current_hash}), skipping training")
            update_progress(job_id, 0.0, "モデル再利用（アノテーション変更なし）")
            return model_path
    # Also check server model
    if not Path(model_path).exists():
        if download_model(project_id, model_path) and hash_file.exists():
            prev_hash = hash_file.read_text().strip()
            if prev_hash == current_hash:
                print(f"  Annotations unchanged ({current_hash}), using server model")
                update_progress(job_id, 0.0, "モデル再利用（アノテーション変更なし）")
                return model_path

    # Need training
    yes_count = sum(1 for a in annotations if a.get("annotation_vote") == "yes")
    no_count = sum(1 for a in annotations if a.get("annotation_vote") == "no")
    if yes_count < 3:
        return None

    update_progress(job_id, 0.02, f"学習開始... ⭕{yes_count} ❌{no_count}")

    # Generate dataset
    _prefetch_annotation_tiles(annotations)
    from app.services.dataset import generate_dataset
    dataset_dir = str(Path(DATASETS_DIR) / project_id)
    ds_info = generate_dataset(annotations, TILES_DIR, dataset_dir)
    pos = ds_info.get("positive", 0)
    if pos == 0:
        return None
    update_progress(job_id, 0.05, f"Dataset: ⭕{pos} ❌{ds_info.get('negative', 0)}")

    # COCO -> YOLO
    yolo_dir = str(Path(dataset_dir) / "yolo")
    _coco_to_yolo(dataset_dir, yolo_dir)

    # Train
    from ultralytics import YOLO
    model = YOLO("yolo26n.pt")
    def _abort_on_shutdown(trainer):
        if _shutdown_requested:
            trainer.stop = True
    model.add_callback("on_train_epoch_end", _abort_on_shutdown)
    results = model.train(
        data=str(Path(yolo_dir) / "data.yaml"),
        epochs=config.get("epochs", 100),
        imgsz=512, batch=config.get("batch_size", 16), device=0,
        patience=20, augment=True, degrees=360, flipud=0.5, fliplr=0.5,
        scale=0.3, mosaic=0.5, exist_ok=True, verbose=False,
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    if not best.exists():
        return None

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, model_path)
    hash_file.write_text(current_hash)

    # Upload to server
    update_progress(job_id, 0.20, "Uploading model...")
    upload_model(project_id, model_path)

    print(f"  Training complete ({current_hash})")
    return model_path


def handle_scan(job: dict):
    """Execute a scanning job. 学習が必要なら先に実行。"""
    job_id = job["id"]
    project_id = job["project_id"]
    config = job.get("config") or {}

    # 学習 (アノテーション変更時のみ)
    model_path = _train_if_needed(job_id, project_id, config)
    if not model_path:
        fail_job(job_id, "No trained model (need 3+ ⭕ annotations)")
        return

    region = config.get("region")
    prefecture = config.get("prefecture")
    conf_threshold = config.get("conf_threshold", 0.3)

    # 都道府県指定: サーバーから実際に県境と交差するタイルリストを取得
    pref_tiles = None
    if prefecture:
        update_progress(job_id, 0.0, f"{prefecture}のタイル一覧を取得中...")
        try:
            tiles_data = api_get(f"/api/prefectures/{requests.utils.quote(prefecture)}/tiles?z=16")
            if tiles_data:
                pref_tiles = {(t["x"], t["y"]) for t in tiles_data}
                print(f"  Prefecture {prefecture}: {len(pref_tiles)} tiles")
        except Exception as e:
            print(f"  Failed to get prefecture tiles: {e}")

    # スキャンごとのラベル名
    from datetime import datetime
    scan_label = config.get("scan_label", f"#{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # configにscan_labelを書き込み + labelsテーブルにsystemラベルを登録
    start_job(job_id, config_update={"scan_label": scan_label})
    try:
        api_post(f"/api/worker/projects/{project_id}/labels", {"name": scan_label, "emoji": "🔍"})
    except Exception:
        pass  # 既に存在する場合は無視

    # リトライ時: 同じscan_labelの既存アノテーションを削除
    try:
        resp = api_post(f"/api/worker/projects/{project_id}/annotations/delete-by-label",
                        {"label_name": scan_label})
        if resp and resp.get("deleted", 0) > 0:
            print(f"  Cleared {resp['deleted']} previous annotations for '{scan_label}'")
    except Exception:
        pass

    update_progress(job_id, 0.0, "Starting scan...")

    total_uploaded = 0

    def det_to_annotation(d: dict) -> dict:
        """検出結果をアノテーション形式に変換（ドーム度付き）"""
        from app.services.domeness import compute_domeness
        dome = compute_domeness(
            TILES_DIR, 16, d["tile_x"], d["tile_y"],
            d["bbox_cx"] / 512, d["bbox_cy"] / 512,
        )
        score = d["conf"]
        if dome:
            comment += f" dh={dome['dome_h']}m rnd={dome['roundness']} iso={dome['isolation']}"
            # スコア = conf * (1 + roundness) で丸い形状を優先
            score = d["conf"] * (1 + dome.get("roundness", 0))
        # bbox の実際の緯度経度を計算
        import math as _math
        z = 16
        n = 2 ** z
        tx, ty = d["tile_x"], d["tile_y"]
        cx, cy = d["bbox_cx"] / 512, d["bbox_cy"] / 512
        w, h = d["bbox_w"] / 512, d["bbox_h"] / 512
        lon_west = (tx + cx - w/2) / n * 360 - 180
        lon_east = (tx + cx + w/2) / n * 360 - 180
        lat_north = _math.degrees(_math.atan(_math.sinh(_math.pi * (1 - 2 * (ty + cy - h/2) / n))))
        lat_south = _math.degrees(_math.atan(_math.sinh(_math.pi * (1 - 2 * (ty + cy + h/2) / n))))
        return {
            "lat": d["lat"],
            "lon": d["lon"],
            "bbox_px_cx": cx, "bbox_px_cy": cy,
            "bbox_px_w": w, "bbox_px_h": h,
            "tile_x": tx, "tile_y": ty, "tile_z": z,
            "bbox_west": lon_west, "bbox_south": lat_south,
            "bbox_east": lon_east, "bbox_north": lat_north,
            "title": None,
            "labels": [{"name": scan_label, "emoji": "🔍", "vote": "yes", "system": "scan"}],
            "comment": comment,
            "score": score,
        }

    import threading as _threading
    _cancel_event = _threading.Event()
    _register_cancel_event(_cancel_event)

    def on_progress(p: float, msg: str):
        try:
            update_progress(job_id, 0.20 + p * 0.80, msg)
        except JobCancelled:
            _cancel_event.set()
            raise

    def on_detections(dets: list[dict]):
        nonlocal total_uploaded
        annots = [det_to_annotation(d) for d in dets]
        _post_annotations_async(project_id, annots, prefecture, scan_label)
        total_uploaded += len(annots)

    try:
        detections = scan_tiles(
            model_path=model_path,
            tiles_dir=TILES_DIR,
            conf_threshold=conf_threshold,
            region=region,
            batch_size=config.get("batch_size", 128),
            num_workers=config.get("num_workers", 8),
            progress_callback=on_progress,
            detection_callback=on_detections,
            tile_fetcher=fetch_tile if REMOTE_TILES else None,
            tile_set=pref_tiles,
            cancel_event=_cancel_event,
        )
    finally:
        _unregister_cancel_event(_cancel_event)

    # 残りがあればアップロード
    if detections:
        annots = [det_to_annotation(d) for d in detections]
        _post_annotations_async(project_id, annots, prefecture, scan_label)
        total_uploaded += len(annots)

    # バックグラウンドアップロード完了待ち
    _flush_uploads()

    complete_job(job_id, {
        "detection_count": total_uploaded,
        "conf_threshold": conf_threshold,
        "scan_label": scan_label,
        "region": region,
    })


# ---------------------------------------------------------------------------
# Multi-class batch scanning
# ---------------------------------------------------------------------------

def _requeue_job(job_id: str):
    """Return a claimed job back to queued status."""
    api_put(f"/api/worker/jobs/{job_id}/requeue", {})


def _drain_queued_scans() -> tuple[list[dict], list[dict]]:
    """Claim all queued scan/rescore jobs atomically. Returns (scan_jobs, rescore_jobs).
    同じプロジェクトの重複scanはrequeue。"""
    # バッチclaim: 全queuedジョブを一括でrunningに変更
    jobs = api_post("/api/worker/jobs/claim-all", {})
    if not jobs:
        return [], []

    all_jobs = []
    rescore_jobs = []
    for job in jobs:
        if job["job_type"] == "scan":
            all_jobs.append(job)
        elif job["job_type"] == "rescore":
            rescore_jobs.append(job)
        elif job["job_type"] == "train":
            try:
                handle_train(job)
            except Exception as e:
                _report_failure(job["id"], str(e)[-2000:])

    # プロジェクト重複排除: 最初の1つだけ採用、残りはqueuedに戻す
    seen = set()
    unique = []
    for job in all_jobs:
        pid = job["project_id"]
        if pid in seen:
            print(f"  Duplicate project {pid}, requeueing job {job['id']}")
            _requeue_job(job["id"])
        else:
            seen.add(pid)
            unique.append(job)
    return unique, rescore_jobs


def _peek_queued_scan() -> bool:
    """Check if any queued scan job exists (without claiming)."""
    try:
        peeked = api_get("/api/worker/jobs/pending?peek=true")
        return bool(peeked and "id" in peeked)
    except Exception:
        return False


def _prepare_scan_batch(scan_jobs: list[dict]) -> tuple[dict, list, list, list]:
    """Prepare class_map, project_annotations, hashes, and valid job_ids.
    同じプロジェクトの重複ジョブはマージ。失敗したジョブはスキップ。"""
    from datetime import datetime as dt

    # プロジェクトごとにジョブをグループ化（重複マージ）
    project_jobs: dict[str, list[dict]] = {}
    for job in scan_jobs:
        pid = job["project_id"]
        project_jobs.setdefault(pid, []).append(job)

    class_map = {}
    project_annotations = []
    per_project_hashes = []

    for cls_idx, (project_id, jobs) in enumerate(project_jobs.items()):
        primary_job = jobs[0]
        config = primary_job.get("config") or {}

        # 全ジョブのステータスを更新
        for j in jobs:
            try:
                update_progress(j["id"], 0.0, "Fetching annotations...")
            except Exception:
                pass

        annotations = _api_call("GET", f"/api/worker/projects/{project_id}/annotations", timeout=120)
        if not annotations:
            for j in jobs:
                fail_job(j["id"], "No annotations found")
            continue

        yes_count = sum(1 for a in annotations if a.get("annotation_vote") == "yes")
        if yes_count < 3:
            for j in jobs:
                fail_job(j["id"], f"Need 3+ ⭕ annotations (got {yes_count})")
            continue

        annot_hash = _annotations_hash(annotations)
        per_project_hashes.append((project_id, annot_hash))
        project_annotations.append((project_id, annotations))

        scan_label = config.get("scan_label", f"#{dt.now().strftime('%Y%m%d_%H%M%S')}")
        is_resume = config.get("scanned_tiles", 0) > 0
        for j in jobs:
            start_job(j["id"], config_update={"scan_label": scan_label})
        try:
            api_post(f"/api/worker/projects/{project_id}/labels", {"name": scan_label, "emoji": "🔍"})
        except Exception:
            pass
        if not is_resume:
            try:
                resp = api_post(f"/api/worker/projects/{project_id}/annotations/delete-by-label",
                                {"label_name": scan_label})
                if resp and resp.get("deleted", 0) > 0:
                    print(f"  Cleared {resp['deleted']} previous annotations for '{scan_label}'")
            except Exception:
                pass

        class_map[len(class_map)] = {
            "project_id": project_id,
            "scan_label": scan_label,
            "prefecture": config.get("prefecture"),
            "conf_threshold": config.get("conf_threshold", 0.3),
            "job_ids": [j["id"] for j in jobs],
            "scanned_tiles": config.get("scanned_tiles", 0),
        }

    all_job_ids = []
    for info in class_map.values():
        all_job_ids.extend(info["job_ids"])

    return class_map, project_annotations, per_project_hashes, all_job_ids


def _find_superset_model(needed: dict[str, str], models_dir: str):
    """Find a cached model trained with a superset of the needed projects (same hashes).
    Returns (model_path, cached_class_map, cls_remap) or None.
    cls_remap: {project_id: original_cls_idx} for class index remapping."""
    for d in Path(models_dir).glob("multi_*/.project_hashes.json"):
        model_path = d.parent / "best.pt"
        if not model_path.exists():
            continue
        try:
            cached = json.loads(d.read_text())
        except Exception:
            continue
        # cached: {"0": {"project_id": "...", "hash": "..."}, "1": ...}
        cached_by_pid = {v["project_id"]: (int(k), v["hash"]) for k, v in cached.items()}
        # 現在の全プロジェクトが含まれ、ハッシュが一致するか
        remap = {}
        for pid, h in needed.items():
            if pid not in cached_by_pid or cached_by_pid[pid][1] != h:
                break
            remap[pid] = cached_by_pid[pid][0]  # project_id → original cls_idx
        else:
            # 全プロジェクトがマッチ
            return str(model_path), cached, remap
    return None


def _train_multi(class_map, project_annotations, per_project_hashes, job_ids) -> str | None:
    """Train multi-class model. Returns model_path or None.
    学習中にYOLOコールバックでpeekし、新ジョブ検知時は学習を早期終了する。
    新ジョブが見つかった場合は 'new_job_found' を返す。"""
    import hashlib

    def update_all(progress, message):
        for jid in job_ids:
            try:
                update_progress(jid, progress, message)
            except (JobCancelled, JobAlreadyDone):
                pass

    combined_hash = hashlib.sha256(str(sorted(per_project_hashes)).encode()).hexdigest()[:16]
    model_dir = str(Path(MODELS_DIR) / f"multi_{combined_hash}")
    model_path = str(Path(model_dir) / "best.pt")
    class_map_file = Path(model_dir) / ".class_map.json"

    # キャッシュチェック (exact match)
    if Path(model_path).exists() and class_map_file.exists():
        print(f"  Model cache hit ({combined_hash}), reusing")
        update_all(0.0, "モデル再利用（アノテーション変更なし）")
        return model_path

    # スーパーセットモデル検索: 現在の全プロジェクトを含む既存モデルを探す
    needed = {pid: h for pid, h in per_project_hashes}  # {project_id: hash}
    superset_model = _find_superset_model(needed, MODELS_DIR)
    if superset_model:
        sup_path, sup_class_map, cls_remap = superset_model
        print(f"  Superset model found: {sup_path}, remap={cls_remap}")
        update_all(0.0, "モデル再利用（スーパ��セットモデル）")
        # class_mapのcls_idを元モデルのcls_idにリマップ
        new_class_map = {}
        for new_idx, info in class_map.items():
            old_idx = cls_remap.get(info["project_id"])
            if old_idx is not None:
                new_class_map[old_idx] = info
        class_map.clear()
        class_map.update(new_class_map)
        return sup_path

    nc = len(project_annotations)
    update_all(0.02, f"Dataset生成中... ({nc}クラス)")
    # 全プロジェクトのアノテーションのタイルを事前 fetch (REMOTE_TILES=true 時のみ)
    for _pid, annots in project_annotations:
        _prefetch_annotation_tiles(annots)
    from app.services.dataset import generate_multi_dataset
    dataset_dir = str(Path(DATASETS_DIR) / f"multi_{combined_hash}")
    ds_info = generate_multi_dataset(project_annotations, TILES_DIR, dataset_dir)
    total_pos = sum(ds_info["positive"].values())
    if total_pos == 0:
        return None
    update_all(0.05, f"Dataset: {total_pos} positives, {nc} classes")

    yolo_dir = str(Path(dataset_dir) / "yolo")
    _coco_to_yolo(dataset_dir, yolo_dir)

    update_all(0.06, f"学習開始... ({nc}クラス)")
    from ultralytics import YOLO
    _new_job_found = False

    def _on_epoch_end(trainer):
        nonlocal _new_job_found
        # SIGTERM/SIGINT 受信時は即座に学習中断 (進捗保存はジョブ requeue で復帰)
        if _shutdown_requested:
            trainer.stop = True
            print(f"  Shutdown requested — stopping training early")
            return

        # ハートビート: 学習中にprogress更新を送り、stale job auto-failを防ぐ
        epoch = trainer.epoch + 1
        total = trainer.epochs
        # 学習フェーズは progress 0.06〜0.50 の範囲
        pct = 0.06 + (epoch / total) * 0.44
        update_all(pct, f"学習中... (epoch {epoch}/{total})")

        if _new_job_found:
            return
        if _peek_queued_scan():
            _new_job_found = True
            trainer.stop = True
            print(f"  New scan job detected during training — stopping early")

    model = YOLO("yolo26n.pt")
    model.add_callback("on_train_epoch_end", _on_epoch_end)

    results = model.train(
        data=str(Path(yolo_dir) / "data.yaml"),
        epochs=100, imgsz=512, batch=-1, device=0, cache=True,
        patience=20, augment=True, degrees=360, flipud=0.5, fliplr=0.5,
        scale=0.3, mosaic=0.5, exist_ok=True, verbose=False,
    )

    if _new_job_found:
        return "new_job_found"

    save_dir = Path(results.save_dir)

    # 学習品質チェック: best mAP50 が閾値未満ならスキャン中止
    min_map50 = float(os.environ.get("MIN_TRAINING_MAP50", "0.2"))
    best_map50 = 0.0
    csv_path = save_dir / "results.csv"
    if csv_path.exists():
        import csv
        with csv_path.open() as f:
            best_map50 = max(
                (float(r.get("metrics/mAP50(B)", 0) or 0) for r in csv.DictReader(f)),
                default=0.0,
            )
    if best_map50 < min_map50:
        msg = f"学習品質不足 (best mAP50={best_map50:.3f} < {min_map50}) — 探索を中止"
        print(f"  {msg}")
        update_all(0.5, msg)
        return "low_quality"

    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    if not best.exists():
        return None

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, model_path)
    class_map_file.write_text(json.dumps({str(k): v for k, v in class_map.items()}))
    # プロジェクトごとのハッシュを保存（スーパーセット検索用）
    hash_file = Path(model_dir) / ".project_hashes.json"
    hash_data = {}
    for cls_idx, (pid, _) in enumerate(project_annotations):
        for p_id, p_hash in per_project_hashes:
            if p_id == pid:
                hash_data[str(cls_idx)] = {"project_id": pid, "hash": p_hash}
                break
    hash_file.write_text(json.dumps(hash_data))
    print(f"  Multi-class training complete ({combined_hash}, {nc} classes)")
    return model_path


def _scan_multi(model_path, class_map, job_ids, project_annotations=None):
    """Run multi-class scan and route results to projects."""
    from collections import defaultdict
    import threading as _threading
    _cancel_event = _threading.Event()
    _register_cancel_event(_cancel_event)

    def update_all(progress, message):
        for jid in job_ids:
            try:
                update_progress(jid, progress, message)
            except JobCancelled:
                _cancel_event.set()
            except JobAlreadyDone:
                pass

    # タイルセット: 全プロジェクトの和集合
    update_all(0.30, "スキャン準備中...")
    combined_tile_set = None
    for info in class_map.values():
        pref = info["prefecture"]
        if not pref:
            combined_tile_set = None
            break
        try:
            tiles_data = api_get(f"/api/prefectures/{requests.utils.quote(pref)}/tiles?z=16")
            if tiles_data:
                pref_tiles = {(t["x"], t["y"]) for t in tiles_data}
                combined_tile_set = pref_tiles if combined_tile_set is None else combined_tile_set | pref_tiles
        except Exception as e:
            print(f"  Failed to get tiles for {pref}: {e}")

    min_conf = min(info["conf_threshold"] for info in class_map.values())
    upload_counts = defaultdict(int)

    def det_to_annotation(d: dict, scan_label: str) -> dict:
        score = d["conf"]
        import math as _math
        z, tx, ty = 16, d["tile_x"], d["tile_y"]
        cx, cy, w, h = d["bbox_cx"], d["bbox_cy"], d["bbox_w"], d["bbox_h"]
        n = 2 ** z
        lon_west = (tx + (cx - w/2) / 512) / n * 360 - 180
        lon_east = (tx + (cx + w/2) / 512) / n * 360 - 180
        merc_s = (ty + (cy + h/2) / 512) / n
        merc_n = (ty + (cy - h/2) / 512) / n
        lat_south = _math.degrees(_math.atan(_math.sinh(_math.pi * (1 - 2 * merc_s))))
        lat_north = _math.degrees(_math.atan(_math.sinh(_math.pi * (1 - 2 * merc_n))))
        return {
            "lat": d["lat"], "lon": d["lon"],
            "bbox_px_cx": cx / 512, "bbox_px_cy": cy / 512,
            "bbox_px_w": w / 512, "bbox_px_h": h / 512,
            "tile_x": tx, "tile_y": ty, "tile_z": z,
            "bbox_west": lon_west, "bbox_south": lat_south,
            "bbox_east": lon_east, "bbox_north": lat_north,
            "title": None,
            "labels": [{"name": scan_label, "emoji": "🔍", "vote": "yes", "system": "scan"}],
            "comment": None, "score": score,
        }

    _chunk_start = [0]
    _chunk_count = [0]

    def on_progress(p: float, msg: str):
        # チャンク内進捗を全体進捗に変換
        global_done = _chunk_start[0] + int(p * _chunk_count[0])
        global_p = global_done / total_tiles if total_tiles > 0 else p
        progress = 0.20 + global_p * 0.80
        rate_part = ""
        if "t/s)" in msg:
            rate_part = " (" + msg.split("(")[-1]
        for info in class_map.values():
            pid = info["project_id"]
            det_msg = f"{global_done:,}/{total_tiles:,} tiles, {upload_counts[pid]:,} detections{rate_part}"
            for jid in info["job_ids"]:
                try:
                    update_progress(jid, progress, det_msg)
                except (JobCancelled, JobAlreadyDone):
                    pass

    def on_detections(dets: list[dict]):
        by_class = defaultdict(list)
        for d in dets:
            cls_id = d.get("cls", 0)
            if cls_id in class_map:
                info = class_map[cls_id]
                if d["conf"] >= info["conf_threshold"]:
                    by_class[cls_id].append(d)
        for cls_id, cls_dets in by_class.items():
            info = class_map[cls_id]
            annots = [det_to_annotation(d, info["scan_label"]) for d in cls_dets]
            _post_annotations_async(info["project_id"], annots, info["prefecture"], info["scan_label"])
            upload_counts[info["project_id"]] += len(annots)

    # レジューム: 全ジョブの scanned_tiles の最小値から再開
    resume_from = min(
        (info.get("scanned_tiles", 0) for info in class_map.values()),
        default=0,
    )
    if resume_from > 0:
        print(f"  Resuming scan from tile {resume_from:,}")

    def _save_progress(tiles_done: int):
        for jid in job_ids:
            try:
                api_put(f"/api/worker/jobs/{jid}/start",
                        {"config_update": {"scanned_tiles": tiles_done}})
            except Exception:
                pass

    # タイル列挙してparallel configを設定（スタンドアロンワーカーが参加可能にする）
    from app.services.scanning import _enumerate_tiles
    all_tiles = _enumerate_tiles(TILES_DIR, None, fetch_tile if REMOTE_TILES else None)
    if combined_tile_set:
        all_tiles = [(p, tx, ty) for p, tx, ty in all_tiles if (tx, ty) in combined_tile_set]
    total_tiles = len(all_tiles)

    # スキャン範囲のタイル全部 + 拡張タイル用隣接を prefetch (R2 並列で高速取得)
    if REMOTE_TILES and total_tiles > 0:
        update_all(0.18, f"スキャン範囲のDEMタイル取得中 ({total_tiles:,}枚)...")
        z_scan = 16  # 探索は z=16 固定
        scan_tiles_needed: set[tuple[int, int, int]] = set()
        for _path, tx, ty in all_tiles:
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                scan_tiles_needed.add((z_scan, tx + dx, ty + dy))
        _prefetch_tiles(scan_tiles_needed, label="scan-range DEM tiles")

    # ⭕タイルに近い順にソート（有望エリアを先にスキャン）
    if project_annotations:
        pos_tiles = set()
        for _pid, annots in project_annotations:
            for a in annots:
                if a.get("annotation_vote") == "yes" and a.get("tile_x") is not None:
                    pos_tiles.add((a["tile_x"], a["tile_y"]))
        if pos_tiles:
            import numpy as np
            pos_arr = np.array(list(pos_tiles), dtype=np.int32)
            tile_coords = np.array([(tx, ty) for _, tx, ty in all_tiles], dtype=np.int32)
            # 各タイルから最近接⭕タイルまでの距離²（チャンク処理でメモリ節約）
            chunk_size = max(10000, 200_000_000 // max(len(pos_arr), 1))  # ~200MB上限
            min_dists = np.empty(len(tile_coords), dtype=np.float32)
            for i in range(0, len(tile_coords), chunk_size):
                chunk = tile_coords[i:i+chunk_size]
                dx = chunk[:, 0:1] - pos_arr[:, 0]  # (chunk, pos)
                dy = chunk[:, 1:2] - pos_arr[:, 1]
                min_dists[i:i+len(chunk)] = (dx * dx + dy * dy).min(axis=1)
            order = np.argsort(min_dists)
            all_tiles = [all_tiles[j] for j in order]
            print(f"  Total tiles: {total_tiles:,} (sorted by proximity to {len(pos_tiles)} ⭕ tiles)")
        else:
            print(f"  Total tiles: {total_tiles:,}")
    else:
        print(f"  Total tiles: {total_tiles:,}")

    # モデルをサーバーにアップロード（スタンドアロンワーカーがダウンロードできるように）
    for info in class_map.values():
        try:
            upload_model(info["project_id"], model_path)
        except Exception as e:
            print(f"  Model upload failed for {info['project_id']}: {e}")

    for jid in job_ids:
        try:
            api_put(f"/api/worker/jobs/{jid}/start", {"config_update": {
                "parallel": True,
                "total_tiles": total_tiles,
                "tile_cursor": 0,
                "model_uploaded": True,
            }})
        except Exception:
            pass

    # claim-tilesでタイルを分担スキャン（スタンドアロンワーカーと並列動作）
    CLAIM_SIZE = 50000
    while True:
        # 全ジョブのcursorを同時に進める（どのジョブに参加しても同じ範囲が返る）
        chunks = []
        for jid in job_ids:
            try:
                c = _api_call("POST", f"/api/worker/jobs/{jid}/claim-tiles",
                              body={"count": CLAIM_SIZE}, timeout=30)
                chunks.append(c)
            except Exception:
                pass
        if not chunks or chunks[0].get("count", 0) == 0:
            break

        start_idx = chunks[0]["start"]
        count = chunks[0]["count"]
        _chunk_start[0] = start_idx
        _chunk_count[0] = count
        my_tiles = all_tiles[start_idx:start_idx + count]

        if not my_tiles:
            print(f"  Warning: empty tile slice [{start_idx}:{start_idx+count}] (all_tiles={len(all_tiles)}), skipping chunk")
            continue

        scan_tiles(
            model_path=model_path,
            tiles_dir=TILES_DIR,
            conf_threshold=min_conf,
            batch_size=128,
            num_workers=8,
            progress_callback=on_progress,
            detection_callback=on_detections,
            tile_fetcher=fetch_tile if REMOTE_TILES else None,
            tile_list=my_tiles,
            cancel_event=_cancel_event,
        )

    _flush_uploads()

    # オーバーラップ検出の重複除去（NMS）
    for info in class_map.values():
        try:
            resp = api_post(f"/api/worker/projects/{info['project_id']}/annotations/dedup",
                            {"scan_label": info["scan_label"]})
            if resp:
                print(f"  NMS dedup: {resp.get('deleted', 0)} duplicates removed for {info['project_id']}")
        except Exception as e:
            print(f"  NMS dedup failed: {e}")

    # 全ジョブ完了
    for cls_idx, info in class_map.items():
        for jid in info["job_ids"]:
            complete_job(jid, {
                "detection_count": upload_counts[info["project_id"]],
                "conf_threshold": info["conf_threshold"],
                "scan_label": info["scan_label"],
                "multi_class": True,
                "num_classes": len(class_map),
            })

    _unregister_cancel_event(_cancel_event)


def handle_scan_jobs(initial_jobs: list[dict]):
    """Scan pipeline: 学習→新ジョブチェック→再学習...→スキャン（中断なし）。
    レジューム対応: config.model_path があれば学習スキップ。"""
    scan_jobs = list(initial_jobs)

    # レジューム判定: 全ジョブに model_path と scanned_tiles があれば学習スキップ
    resume_model = None
    configs = [j.get("config") or {} for j in scan_jobs]
    if all(c.get("model_path") and c.get("scanned_tiles", 0) > 0 for c in configs):
        resume_model = configs[0]["model_path"]
        if Path(resume_model).exists():
            print(f"  Resuming with existing model: {resume_model}")

    while True:
        print(f"  Preparing batch ({len(scan_jobs)} jobs, "
              f"{len(set(j['project_id'] for j in scan_jobs))} projects)")

        class_map, project_annotations, hashes, job_ids = _prepare_scan_batch(scan_jobs)
        if not project_annotations:
            return  # 全ジョブが失敗

        if resume_model and Path(resume_model).exists():
            model_path = resume_model
            resume_model = None  # 1回だけ
        else:
            # 学習
            model_path = _train_multi(class_map, project_annotations, hashes, job_ids)

            if model_path == "new_job_found":
                new_scans, new_rescores = _drain_queued_scans()
                if new_scans:
                    print(f"  +{len(new_scans)} new scan job(s) — restarting training")
                    scan_jobs.extend(new_scans)
                    continue
                for jid in job_ids:
                    fail_job(jid, "Training interrupted but no new jobs found")
                return

            if model_path == "low_quality":
                for jid in job_ids:
                    fail_job(jid, "学習品質が閾値未満のため探索を中止しました（教師データ不足の可能性）")
                return

            if model_path is None:
                for jid in job_ids:
                    fail_job(jid, "Training failed: no model produced")
                return

            # 学習完了 → スキャン前に最終チェック
            if _peek_queued_scan():
                new_scans, new_rescores = _drain_queued_scans()
                if new_scans:
                    print(f"  +{len(new_scans)} new scan job(s) before scan — restarting training")
                    scan_jobs.extend(new_scans)
                    continue

        # model_path をジョブに保存（次回レジューム用）
        for jid in job_ids:
            try:
                start_job(jid, config_update={"model_path": model_path})
            except Exception:
                pass

        # スキャン実行
        _scan_multi(model_path, class_map, job_ids, project_annotations)
        return


def handle_rescore(job: dict):
    """再学習 + フィルタ結果タイルのみ再推論してスコア更新。"""
    job_id = job["id"]
    project_id = job["project_id"]
    config = job.get("config") or {}
    tiles_config = config.get("tiles", [])

    start_job(job_id)

    # 1. 再学習（_train_if_neededと同じロジック）
    update_progress(job_id, 0.0, "学習データ確認中...")
    annotations = _api_call("GET", f"/api/worker/projects/{project_id}/annotations", timeout=120)
    if not annotations:
        fail_job(job_id, "No annotations found")
        return

    yes_count = sum(1 for a in annotations if a.get("annotation_vote") == "yes")
    if yes_count < 3:
        fail_job(job_id, f"Need 3+ ⭕ annotations (got {yes_count})")
        return

    current_hash = _annotations_hash(annotations)
    model_dir = str(Path(MODELS_DIR) / project_id)
    model_path = str(Path(model_dir) / "best.pt")
    hash_file = Path(model_dir) / ".train_hash"

    need_train = True
    if Path(model_path).exists() and hash_file.exists():
        if hash_file.read_text().strip() == current_hash:
            need_train = False
            update_progress(job_id, 0.1, "モデル再利用")

    if need_train:
        update_progress(job_id, 0.02, f"再学習中... ⭕{yes_count}")
        _prefetch_annotation_tiles(annotations)
        from app.services.dataset import generate_dataset
        dataset_dir = str(Path(DATASETS_DIR) / project_id)
        ds_info = generate_dataset(annotations, TILES_DIR, dataset_dir)
        if ds_info.get("positive", 0) == 0:
            fail_job(job_id, "No positive annotations")
            return
        yolo_dir = str(Path(dataset_dir) / "yolo")
        _coco_to_yolo(dataset_dir, yolo_dir)

        from ultralytics import YOLO
        model = YOLO("yolo26n.pt")
        def _abort_on_shutdown(trainer):
            if _shutdown_requested:
                trainer.stop = True
        model.add_callback("on_train_epoch_end", _abort_on_shutdown)
        results = model.train(
            data=str(Path(yolo_dir) / "data.yaml"),
            epochs=100, imgsz=512, batch=-1, device=0, cache=True,
            patience=20, augment=True, degrees=360, flipud=0.5, fliplr=0.5,
            scale=0.3, mosaic=0.5, exist_ok=True, verbose=False,
        )
        save_dir = Path(results.save_dir)
        best = save_dir / "weights" / "best.pt"
        if not best.exists():
            best = save_dir / "weights" / "last.pt"
        if not best.exists():
            fail_job(job_id, "Training failed")
            return
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, model_path)
        hash_file.write_text(current_hash)
        update_progress(job_id, 0.3, "学習完了")

    # 2. 対象タイルのみ再推論
    tile_set = {(t[1], t[2]) for t in tiles_config} if tiles_config else None
    if not tile_set:
        fail_job(job_id, "No target tiles")
        return

    update_progress(job_id, 0.3, f"{len(tile_set)}タイルを再推論中...")

    import threading as _threading
    _cancel_event = _threading.Event()
    _register_cancel_event(_cancel_event)

    def _rescore_progress(p, msg):
        try:
            update_progress(job_id, 0.3 + p * 0.6, msg)
        except JobCancelled:
            _cancel_event.set()
            raise

    from app.services.scanning import scan_tiles
    try:
        detections = scan_tiles(
            model_path=model_path,
            tiles_dir=TILES_DIR,
            conf_threshold=0.01,  # 低閾値で全検出
            batch_size=128,
            num_workers=8,
            tile_set=tile_set,
            progress_callback=_rescore_progress,
            cancel_event=_cancel_event,
        )
    finally:
        _unregister_cancel_event(_cancel_event)

    # 3. 既存アノテーションとIoUマッチング → スコア更新
    update_progress(job_id, 0.9, "スコア更新中...")

    # フィルタ結果のアノテーション取得（フィルタ指定時はフィルタ結果、なければ投票済み全件）
    filter_data = config.get("filter", {})
    if filter_data:
        target_annotations = api_post(
            f"/api/worker/projects/{project_id}/annotations/by-filter",
            {"filter": filter_data},
        )
    else:
        target_annotations = _api_call("GET", f"/api/worker/projects/{project_id}/annotations", timeout=120)

    # タイル別にグループ化
    from collections import defaultdict
    det_by_tile = defaultdict(list)
    for d in detections:
        det_by_tile[(d["tile_x"], d["tile_y"])].append(d)

    score_updates = []
    for a in target_annotations:
        tx, ty = a.get("tile_x"), a.get("tile_y")
        if (tx, ty) not in det_by_tile:
            score_updates.append({"annotation_id": a["id"], "score": 0})
            continue

        # IoUマッチング
        a_cx = a.get("bbox_px_cx", 0)
        a_cy = a.get("bbox_px_cy", 0)
        a_w = a.get("bbox_px_w", 0)
        a_h = a.get("bbox_px_h", 0)
        # 正規化値をピクセルに
        if a_cx <= 1:
            a_cx *= 512
            a_cy *= 512
            a_w *= 512
            a_h *= 512

        best_iou = 0
        best_conf = 0
        for d in det_by_tile[(tx, ty)]:
            d_cx, d_cy, d_w, d_h = d["bbox_cx"], d["bbox_cy"], d["bbox_w"], d["bbox_h"]
            # IoU計算
            a_x1, a_y1 = a_cx - a_w/2, a_cy - a_h/2
            a_x2, a_y2 = a_cx + a_w/2, a_cy + a_h/2
            d_x1, d_y1 = d_cx - d_w/2, d_cy - d_h/2
            d_x2, d_y2 = d_cx + d_w/2, d_cy + d_h/2
            inter_x1 = max(a_x1, d_x1)
            inter_y1 = max(a_y1, d_y1)
            inter_x2 = min(a_x2, d_x2)
            inter_y2 = min(a_y2, d_y2)
            inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union = a_w * a_h + d_w * d_h - inter
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_conf = d["conf"]

        score = best_conf if best_iou > 0.3 else 0
        score_updates.append({"annotation_id": a["id"], "score": score})

    # バルク更新（500件ずつ）
    updated = 0
    for i in range(0, len(score_updates), 500):
        chunk = score_updates[i:i+500]
        resp = api_put("/api/worker/annotations/bulk-score", {"updates": chunk})
        updated += resp.get("updated", 0) if resp else 0

    update_progress(job_id, 1.0, f"スコア更新完了: {updated}件更新, {len(detections)}件検出")
    complete_job(job_id, {"updated": updated, "detections": len(detections)})


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_current_job_ids: list[str] = []
_shutdown_requested = False
_active_cancel_events: list = []


def _register_cancel_event(ev):
    """Register a cancel event so SIGTERM can trigger it."""
    _active_cancel_events.append(ev)
    if _shutdown_requested:
        ev.set()


def _unregister_cancel_event(ev):
    try:
        _active_cancel_events.remove(ev)
    except ValueError:
        pass


def _request_shutdown(*_):
    """SIGTERM/SIGINT handler: signal graceful shutdown to all active scans."""
    global _shutdown_requested
    if _shutdown_requested:
        return
    _shutdown_requested = True
    print("Shutdown requested, draining in-flight jobs (saved progress will resume next run)...")
    for ev in list(_active_cancel_events):
        try:
            ev.set()
        except Exception:
            pass


def _shutdown_cleanup():
    """ワーカー終了時: 処理中のジョブを再キュー (graceful) または fail (crash) する。"""
    for jid in _current_job_ids:
        try:
            if _shutdown_requested:
                print(f"Shutdown: requeueing job {jid}")
                _requeue_job(jid)
            else:
                print(f"Shutdown: failing job {jid}")
                fail_job(jid, "Worker shutdown")
        except Exception:
            pass


_PIDFILE = Path(__file__).parent / ".worker.pid"


def _acquire_pidlock():
    """PIDファイルで排他制御。既に動作中なら即終了。"""
    if _PIDFILE.exists():
        try:
            old_pid = int(_PIDFILE.read_text().strip())
            # プロセスが生きているか確認
            os.kill(old_pid, 0)
            print(f"ERROR: Worker already running (PID {old_pid}). Exiting.")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass  # 古いPIDファイル — 上書きOK
        except PermissionError:
            print(f"ERROR: Worker already running (PID {old_pid}). Exiting.")
            sys.exit(1)
    _PIDFILE.write_text(str(os.getpid()))


def _release_pidlock():
    try:
        if _PIDFILE.exists() and _PIDFILE.read_text().strip() == str(os.getpid()):
            _PIDFILE.unlink()
    except Exception:
        pass


def main():
    if not WORKER_KEY:
        print("ERROR: WORKER_API_KEY environment variable not set")
        sys.exit(1)

    if not DISABLE_PID_LOCK:
        _acquire_pidlock()

    import atexit
    if not DISABLE_PID_LOCK:
        atexit.register(_release_pidlock)
    atexit.register(_shutdown_cleanup)
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    print(f"Worker started")
    print(f"  Server:    {SERVER_URL}")
    print(f"  Tiles:     {TILES_DIR}")
    print(f"  Models:    {MODELS_DIR}")
    print(f"  Datasets:  {DATASETS_DIR}")
    print(f"  Poll:      {POLL_INTERVAL}s")
    print(f"  R2 base:   {DEM_TILE_BASE_URL or '(none, GeoScope server only)'}")
    print()

    # 初回起動: DEM tar archive を R2 から stream 展開 (REMOTE_TILES + DEM_TILE_BASE_URL のとき)
    try:
        _init_dem_from_archive()
    except Exception:
        print("DEM archive init failed (will fall back to per-tile fetch):")
        traceback.print_exc()

    poll_failures = 0
    while not _shutdown_requested:
        # 1. queued scan ジョブを全て取得（train jobは即処理）
        try:
            scan_jobs, rescore_jobs = _drain_queued_scans()
            poll_failures = 0
        except Exception as e:
            poll_failures += 1
            wait = min(POLL_INTERVAL * poll_failures, 60)
            print(f"Poll error ({poll_failures}): {e}, retry in {wait}s")
            time.sleep(wait)
            continue

        if not scan_jobs and not rescore_jobs:
            time.sleep(POLL_INTERVAL)
            continue

        # Rescore jobs: handle individually
        for job in rescore_jobs:
            print(f"=== Rescore {job['id']} ===")
            try:
                handle_rescore(job)
            except JobAlreadyDone as e:
                print(f"  {e}")
            except JobCancelled as e:
                _report_failure(job["id"], str(e))
            except Exception:
                tb = traceback.format_exc()
                print(f"Rescore failed:\n{tb}")
                _report_failure(job["id"], tb[-2000:])

        if not scan_jobs:
            import gc; gc.collect()
            print()
            continue

        # 2. Scan pipeline: 学習→チェック→再学習...→スキャン
        print(f"=== Scan batch ({len(scan_jobs)} jobs) ===")
        _current_job_ids[:] = [j["id"] for j in scan_jobs]
        try:
            handle_scan_jobs(scan_jobs)
        except JobAlreadyDone as e:
            print(f"  {e}")
        except JobCancelled as e:
            print(f"  {e}")
            pass
        except Exception:
            tb = traceback.format_exc()
            print(f"Scan batch failed:\n{tb}")
            for j in scan_jobs:
                _report_failure(j["id"], tb[-2000:])
        finally:
            _current_job_ids.clear()

        # メモリクリーンアップ
        import gc; gc.collect()
        print()


def _report_failure(job_id: str, error: str):
    """Report job failure with persistent retry (up to 5 min)."""
    for attempt in range(20):
        try:
            fail_job(job_id, error)
            print(f"Reported failure for {job_id}")
            return
        except Exception as e:
            wait = min(15 * (attempt + 1), 60)
            print(f"Failed to report failure (attempt {attempt+1}): {e}, retrying in {wait}s...")
            time.sleep(wait)


if __name__ == "__main__":
    main()

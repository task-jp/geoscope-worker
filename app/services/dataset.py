"""Dataset generation: annotations -> COCO format for YOLO training.

Called by the GPU worker (not by the FastAPI server).
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from app.core.dem import TILE_PX, decode_dem
from app.core.visualization import dem_to_3ch


def _tile_path(tiles_dir: str, z: int, tx: int, ty: int) -> Path:
    return Path(tiles_dir) / f"{z}/{tx}/{ty}.webp"


def _load_3ch(tiles_dir: str, z: int, tx: int, ty: int) -> np.ndarray | None:
    """Load a DEM tile and convert to 3-channel image (512x512x3 uint8)."""
    path = _tile_path(tiles_dir, z, tx, ty)
    if not path.exists():
        return None
    try:
        elev = decode_dem(path.read_bytes())
        valid = elev[~np.isnan(elev)]
        if len(valid) < TILE_PX * TILE_PX * 0.3:
            return None
        return dem_to_3ch(elev)
    except Exception:
        return None


EXTENDED_PX = TILE_PX + TILE_PX // 2  # 768


def _load_3ch_extended(tiles_dir: str, z: int, tx: int, ty: int) -> np.ndarray | None:
    """Load 150% extended 3ch image (768×768) by stitching tile + right/below/diagonal neighbors."""
    half = TILE_PX // 2
    canvas = np.full((EXTENDED_PX, EXTENDED_PX), np.nan)

    # メインタイル (0:512, 0:512)
    path = _tile_path(tiles_dir, z, tx, ty)
    if not path.exists():
        return None
    try:
        main = decode_dem(path.read_bytes())
    except Exception:
        return None
    canvas[:TILE_PX, :TILE_PX] = main

    # 右タイル → 左256列
    p_r = _tile_path(tiles_dir, z, tx + 1, ty)
    if p_r.exists():
        try:
            canvas[:TILE_PX, TILE_PX:] = decode_dem(p_r.read_bytes())[:, :half]
        except Exception:
            pass

    # 下タイル → 上256行
    p_b = _tile_path(tiles_dir, z, tx, ty + 1)
    if p_b.exists():
        try:
            canvas[TILE_PX:, :TILE_PX] = decode_dem(p_b.read_bytes())[:half, :]
        except Exception:
            pass

    # 右下タイル → 左上256×256
    p_d = _tile_path(tiles_dir, z, tx + 1, ty + 1)
    if p_d.exists():
        try:
            canvas[TILE_PX:, TILE_PX:] = decode_dem(p_d.read_bytes())[:half, :half]
        except Exception:
            pass

    valid = canvas[~np.isnan(canvas)]
    if len(valid) < TILE_PX * TILE_PX * 0.3:
        return None
    canvas[np.isnan(canvas)] = np.nanmean(canvas) if len(valid) > 0 else 0
    try:
        return dem_to_3ch(canvas)
    except Exception:
        return None


HALF = TILE_PX // 2  # 256


def _crop_512(img: np.ndarray, ox: int, oy: int) -> np.ndarray:
    """Crop a 512×512 region from a >=512 image at offset (ox, oy)."""
    return img[oy:oy + TILE_PX, ox:ox + TILE_PX].copy()


def _needs_shift(annots: list[dict]) -> tuple[bool, bool]:
    """Check if any annotation crosses the right or bottom tile boundary."""
    shift_right = shift_down = False
    for a in annots:
        cx, cy = a["bbox_px_cx"], a["bbox_px_cy"]
        w, h = a["bbox_px_w"], a["bbox_px_h"]
        if cx + w / 2 > 1.0:
            shift_right = True
        if cy + h / 2 > 1.0:
            shift_down = True
    return shift_right, shift_down


def _bbox_px_to_coco(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Convert normalized bbox to COCO format [x, y, w, h] in 512px space."""
    px_cx = cx * TILE_PX
    px_cy = cy * TILE_PX
    px_w = w * TILE_PX
    px_h = h * TILE_PX
    x = max(0, px_cx - px_w / 2)
    y = max(0, px_cy - px_h / 2)
    return (x, y, min(px_w, TILE_PX - x), min(px_h, TILE_PX - y))


def _make_crop_entry(img, annots, ox, oy, images_dir, z, tx, ty, suffix,
                     image_id, annot_id, coco_images, coco_annotations,
                     cat_id=1, cls_annots_map=None):
    """Generate one 512×512 crop and its COCO annotations.

    If cls_annots_map is provided (multi-class), annots is ignored and
    cls_annots_map maps cls_idx -> list of annotations.
    """
    crop = _crop_512(img, ox, oy)
    filename = f"{z}_{tx}_{ty}{suffix}.png"
    cv2.imwrite(str(images_dir / filename), crop)
    coco_images.append({"id": image_id, "file_name": filename, "width": TILE_PX, "height": TILE_PX})

    norm_ox = ox / TILE_PX
    norm_oy = oy / TILE_PX

    items = []
    if cls_annots_map:
        for cls_idx, anns in cls_annots_map.items():
            items.extend((cls_idx + 1, a) for a in anns)
    else:
        items = [(cat_id, a) for a in annots]

    for cid, a in items:
        cx = a["bbox_px_cx"] - norm_ox
        cy = a["bbox_px_cy"] - norm_oy
        w, h = a["bbox_px_w"], a["bbox_px_h"]
        if cx < 0 or cx > 1.0 or cy < 0 or cy > 1.0:
            continue
        bbox = _bbox_px_to_coco(cx, cy, w, h)
        area = bbox[2] * bbox[3]
        if area < 1:
            continue
        coco_annotations.append({"id": annot_id, "image_id": image_id, "category_id": cid,
                                 "bbox": list(bbox), "area": area, "iscrowd": 0})
        annot_id += 1

    return image_id + 1, annot_id


def generate_dataset(
    annotations: list[dict],
    tiles_dir: str,
    output_dir: str,
    train_label: str | None = None,
    val_ratio: float = 0.2,
) -> dict:
    """Generate a COCO-format dataset from annotations using label votes.

    Uses the vote status of ``train_label`` to split annotations:
    - ⭕ (yes) → positive examples (tile image + bbox annotation)
    - ❌ (no) → negative examples (tile image only, no bbox)
    - ❓ (unvoted / missing) → excluded

    Args:
        annotations: List of annotation dicts (from API response).
            Each must have: tile_x, tile_y, tile_z, bbox_px_cx/cy/w/h, labels.
        tiles_dir: Path to DEM tile directory.
        output_dir: Where to write images/ and annotations JSON.
        train_label: Label name to use for yes/no vote splitting.
        val_ratio: Fraction of images for validation split.

    Returns:
        dict with train_images, val_images, positive/negative annotation counts.
    """
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)

    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Split annotations by vote ---
    positive_annots = []  # ⭕ yes → bbox付き正例
    negative_annots = []  # ❌ no → タイルのみ負例
    for a in annotations:
        # annotation_vote フィールドを優先、なければラベルのvoteを使う
        vote = a.get("annotation_vote")
        if not vote and train_label:
            labels = a.get("labels") or []
            entry = next((l for l in labels if l.get("name") == train_label), None)
            if entry:
                vote = entry.get("vote")
        if not vote:
            continue
        if vote == "yes":
            positive_annots.append(a)
        elif vote == "no":
            negative_annots.append(a)
        # unvoted → skip

    if not positive_annots:
        return {"train_images": 0, "val_images": 0, "positive": 0, "negative": 0}

    # --- 2. Group positive annotations by tile ---
    tile_annots: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for a in positive_annots:
        key = (a["tile_z"], a["tile_x"], a["tile_y"])
        tile_annots[key].append(a)

    # --- 3. Generate positive images (with bbox annotations) ---
    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    image_id = 0
    annot_id = 0
    used_tiles: set[tuple[int, int, int]] = set()

    for (z, tx, ty), annots in tile_annots.items():
        img = _load_3ch_extended(tiles_dir, z, tx, ty)
        if img is None:
            img = _load_3ch(tiles_dir, z, tx, ty)
            if img is None:
                continue
        used_tiles.add((z, tx, ty))

        # 512 crops with boundary shift
        is_extended = img.shape[0] == EXTENDED_PX
        image_id, annot_id = _make_crop_entry(
            img, annots, 0, 0, images_dir, z, tx, ty, "",
            image_id, annot_id, coco_images, coco_annotations)

        if is_extended:
            shift_right, shift_down = _needs_shift(annots)
            if shift_right:
                image_id, annot_id = _make_crop_entry(
                    img, annots, HALF, 0, images_dir, z, tx, ty, "_r",
                    image_id, annot_id, coco_images, coco_annotations)
            if shift_down:
                image_id, annot_id = _make_crop_entry(
                    img, annots, 0, HALF, images_dir, z, tx, ty, "_d",
                    image_id, annot_id, coco_images, coco_annotations)
            if shift_right and shift_down:
                image_id, annot_id = _make_crop_entry(
                    img, annots, HALF, HALF, images_dir, z, tx, ty, "_rd",
                    image_id, annot_id, coco_images, coco_annotations)

    # --- 4. Add negative samples from ❌ annotations ---
    neg_tiles: dict[tuple[int, int, int], bool] = {}
    for a in negative_annots:
        key = (a["tile_z"], a["tile_x"], a["tile_y"])
        if key not in used_tiles:
            neg_tiles[key] = True

    for z, tx, ty in neg_tiles:
        img = _load_3ch_extended(tiles_dir, z, tx, ty)
        if img is None:
            img = _load_3ch(tiles_dir, z, tx, ty)
            if img is None:
                continue
        crop = _crop_512(img, 0, 0)
        filename = f"{z}_{tx}_{ty}.png"
        cv2.imwrite(str(images_dir / filename), crop)
        coco_images.append({"id": image_id, "file_name": filename, "width": TILE_PX, "height": TILE_PX})
        used_tiles.add((z, tx, ty))
        image_id += 1

    # --- 5. Train / val split ---
    all_ids = list(range(len(coco_images)))
    random.shuffle(all_ids)
    split_idx = max(1, int(len(all_ids) * (1 - val_ratio)))
    train_ids = set(all_ids[:split_idx])
    val_ids = set(all_ids[split_idx:])

    categories = [{"id": 1, "name": train_label, "supercategory": "none"}]

    def _make_coco(image_ids: set[int]) -> dict:
        images = [img for img in coco_images if img["id"] in image_ids]
        annots = [a for a in coco_annotations if a["image_id"] in image_ids]
        return {"images": images, "annotations": annots, "categories": categories}

    train_coco = _make_coco(train_ids)
    val_coco = _make_coco(val_ids)

    # --- 6. Write JSON ---
    with open(output / "train.json", "w") as f:
        json.dump(train_coco, f)
    with open(output / "val.json", "w") as f:
        json.dump(val_coco, f)

    return {
        "train_images": len(train_coco["images"]),
        "val_images": len(val_coco["images"]),
        "positive": len(positive_annots),
        "negative": len(neg_tiles),
    }


def generate_multi_dataset(
    project_annotations: list[tuple[str, list[dict]]],
    tiles_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
) -> dict:
    """Generate a multi-class COCO dataset from multiple projects.

    Each project becomes a separate YOLO class.
    Returns dict with class_names, train/val counts, per-class positive counts.
    """
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Assign class per project (COCO category_id is 1-based)
    class_names = []
    categories = []
    # tile -> {class_idx: [annotations]}
    tile_pos: dict[tuple[int, int, int], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    neg_tiles: set[tuple[int, int, int]] = set()
    pos_counts = {}

    for cls_idx, (project_id, annotations) in enumerate(project_annotations):
        cat_id = cls_idx + 1
        class_names.append(project_id)
        categories.append({"id": cat_id, "name": project_id, "supercategory": "none"})

        pos = 0
        for a in annotations:
            vote = a.get("annotation_vote")
            if not vote:
                continue
            key = (a["tile_z"], a["tile_x"], a["tile_y"])
            if vote == "yes":
                tile_pos[key][cls_idx].append(a)
                pos += 1
            elif vote == "no":
                neg_tiles.add(key)
        pos_counts[project_id] = pos

    # Generate images (one per tile, multiple classes' bboxes)
    coco_images = []
    coco_annotations = []
    image_id = 0
    annot_id = 0
    used_tiles = set()

    for (z, tx, ty), cls_annots in tile_pos.items():
        img = _load_3ch_extended(tiles_dir, z, tx, ty)
        if img is None:
            img = _load_3ch(tiles_dir, z, tx, ty)
            if img is None:
                continue
        used_tiles.add((z, tx, ty))
        is_extended = img.shape[0] == EXTENDED_PX

        # Base crop
        image_id, annot_id = _make_crop_entry(
            img, [], 0, 0, images_dir, z, tx, ty, "",
            image_id, annot_id, coco_images, coco_annotations,
            cls_annots_map=cls_annots)

        # Shifted crops for boundary-crossing annotations
        if is_extended:
            all_annots = [a for anns in cls_annots.values() for a in anns]
            shift_right, shift_down = _needs_shift(all_annots)
            if shift_right:
                image_id, annot_id = _make_crop_entry(
                    img, [], HALF, 0, images_dir, z, tx, ty, "_r",
                    image_id, annot_id, coco_images, coco_annotations,
                    cls_annots_map=cls_annots)
            if shift_down:
                image_id, annot_id = _make_crop_entry(
                    img, [], 0, HALF, images_dir, z, tx, ty, "_d",
                    image_id, annot_id, coco_images, coco_annotations,
                    cls_annots_map=cls_annots)
            if shift_right and shift_down:
                image_id, annot_id = _make_crop_entry(
                    img, [], HALF, HALF, images_dir, z, tx, ty, "_rd",
                    image_id, annot_id, coco_images, coco_annotations,
                    cls_annots_map=cls_annots)

    # Negative samples
    for z, tx, ty in neg_tiles:
        if (z, tx, ty) in used_tiles:
            continue
        img = _load_3ch_extended(tiles_dir, z, tx, ty)
        if img is None:
            img = _load_3ch(tiles_dir, z, tx, ty)
            if img is None:
                continue
        crop = _crop_512(img, 0, 0)
        filename = f"{z}_{tx}_{ty}.png"
        cv2.imwrite(str(images_dir / filename), crop)
        coco_images.append({"id": image_id, "file_name": filename, "width": TILE_PX, "height": TILE_PX})
        used_tiles.add((z, tx, ty))
        image_id += 1

    # Train/val split
    all_ids = list(range(len(coco_images)))
    random.shuffle(all_ids)
    split_idx = max(1, int(len(all_ids) * (1 - val_ratio)))
    train_ids = set(all_ids[:split_idx])
    val_ids = set(all_ids[split_idx:])

    def _make_coco(image_ids: set[int]) -> dict:
        images = [img for img in coco_images if img["id"] in image_ids]
        annots = [a for a in coco_annotations if a["image_id"] in image_ids]
        return {"images": images, "annotations": annots, "categories": categories}

    train_coco = _make_coco(train_ids)
    val_coco = _make_coco(val_ids)

    with open(output / "train.json", "w") as f:
        json.dump(train_coco, f)
    with open(output / "val.json", "w") as f:
        json.dump(val_coco, f)

    return {
        "class_names": class_names,
        "train_images": len(train_coco["images"]),
        "val_images": len(val_coco["images"]),
        "positive": pos_counts,
        "negative": len(neg_tiles - set(tile_pos.keys())),
    }

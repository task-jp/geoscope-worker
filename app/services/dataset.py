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

from app.core.dem import TILE_PX, decode_dem, pixel_to_latlon
from app.core.visualization import cell_size_m, dem_to_3ch


def dataset_code_hash() -> str:
    """このファイル自身の内容ハッシュ (8 hex 文字)。

    モデルキャッシュのキー (worker.py:_annotations_hash) に混ぜて、教師データ
    生成コードの変更時にローカル/サーバ/superset の全キャッシュを無効化する。
    境界クリップ修正のように「アノテーションは同じでも教師データが変わる」
    変更は、これ無しでは古いコードで学習したモデルが再利用され続ける
    (実例: 2026-08-20 の千葉スキャンが修正前学習のモデルを 74 秒で再利用)。
    """
    import hashlib
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:8]


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
        lat, _ = pixel_to_latlon(z, tx, ty, TILE_PX / 2, TILE_PX / 2)
        return dem_to_3ch(elev, cell_size_m(lat, z))
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
        # 学習と推論で cell_size を揃えないと 3ch が食い違う (scanning.py と同じ式)
        lat, _ = pixel_to_latlon(z, tx, ty, EXTENDED_PX / 2, EXTENDED_PX / 2)
        return dem_to_3ch(canvas, cell_size_m(lat, z))
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
                     cat_id=1, cls_annots_map=None, emitted=None):
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
        # bbox 全体がこの切り出しの内側にあるときだけ書き出す。
        # 中心だけで判定していた頃は、端にかかる対象が `_bbox_px_to_coco` の
        # max(0,...) で黙ってクリップされ、半分の形のまま「これが古墳だ」と
        # 学習されていた。モデルはその切れた形を覚え、完全な形には反応しなくなる
        # (実測: 指摘された 8 件のうち 4 件は隣タイルで conf 0.05 でも検出されず)。
        # 拡張タイルと `_needs_shift` のシフト切り出しは、境界をまたぐ対象が
        # どれかの切り出しに完全に収まるようにするための仕組みなので、
        # 「完全に収まる切り出しにだけ書く」が本来の意図。
        if cx - w / 2 < 0 or cx + w / 2 > 1.0 or cy - h / 2 < 0 or cy + h / 2 > 1.0:
            continue
        if emitted is not None:
            emitted.add(id(a))
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

    # ⭕ のクラスを先に全部並べ、その後ろに ❌ の除外クラスを並べる。
    # ❌ を「bbox のない背景画像」として与えていた頃は、48px の対象を含む 512px の
    # タイル 1 枚が背景になるだけで、対象は画像面積の 0.9%。勾配が全アンカーに薄く
    # 分散し、⭕ が持つ「ここにこれがある」という鋭さが ❌ 側に無かった。
    # 実測で ❌ 2,431 件のうち 154 件 (6.3%) が同一対象として再検出されていた
    # (20m でも 50m でも件数が変わらない = 近傍の別物ではない)。
    # 除外クラスにすると ❌ にも bbox 付きの教師データが与えられ、⭕ と同じ強さで
    # 学習される。推論側は class_map に無いクラスを捨てるので (worker.py の
    # on_detections)、除外クラスの検出はそのまま落ちる。
    n_proj = len(project_annotations)
    for cls_idx, (project_id, annotations) in enumerate(project_annotations):
        class_names.append(project_id)
        categories.append({"id": cls_idx + 1, "name": project_id, "supercategory": "none"})
    for cls_idx, (project_id, _) in enumerate(project_annotations):
        neg_idx = n_proj + cls_idx
        class_names.append(f"{project_id}__negative")
        categories.append({"id": neg_idx + 1, "name": f"{project_id}__negative",
                           "supercategory": "none"})

    def _place(a: dict, tiles_dir: str) -> tuple[tuple[int, int, int], dict] | None:
        """アノテーションを「完全な形で写るタイル」に割り当てる。

        拡張タイルは右下にしか伸びないので、bbox が左/上にはみ出す対象は自タイルの
        画像では切れている。`_bbox_px_to_coco` が max(0,...) で黙ってクリップするため、
        半分の形のまま「これが古墳だ」と学習されていた (実測 ⭕ 631 件中 77 件 = 12.2%)。
        モデルはその切れた形を覚え、完全な形には反応しなくなる — 実際に、指摘された
        8 件のうち 4 件は隣タイルで conf を 0.05 まで下げても検出されなかった。

        対象は左/上隣タイルの拡張画像には完全に写っているので、そちらの座標系
        (cx += 1.0) で登録する。すると `_needs_shift` が右/下シフトを発火させ、
        ox=256 の切り出しに bbox 全体が収まる。既存の仕組みで完結する。

        隣タイルの DEM が無ければ完全な形を作れないので None を返す (学習に入れない)。
        """
        cx, cy = a["bbox_px_cx"], a["bbox_px_cy"]
        w, h = a["bbox_px_w"], a["bbox_px_h"]
        z, tx, ty = a["tile_z"], a["tile_x"], a["tile_y"]
        dx = 1 if cx - w / 2 < 0 else 0
        dy = 1 if cy - h / 2 < 0 else 0
        if not dx and not dy:
            return (z, tx, ty), a
        ntx, nty = tx - dx, ty - dy
        if not _tile_path(tiles_dir, z, ntx, nty).exists():
            return None
        shifted = dict(a)
        shifted["bbox_px_cx"] = cx + dx
        shifted["bbox_px_cy"] = cy + dy
        shifted["tile_x"], shifted["tile_y"] = ntx, nty
        return (z, ntx, nty), shifted

    neg_counts = {}
    moved = dropped = 0
    for cls_idx, (project_id, annotations) in enumerate(project_annotations):
        neg_idx = n_proj + cls_idx
        pos = neg = 0
        for a in annotations:
            vote = a.get("annotation_vote")
            if not vote:
                continue
            if vote == "no" and not (a.get("bbox_px_w") and a.get("bbox_px_h")):
                # bbox を持たない ❌ (古い import 等) は従来どおり背景タイル扱い
                neg_tiles.add((a["tile_z"], a["tile_x"], a["tile_y"]))
                continue
            placed = _place(a, tiles_dir)
            if placed is None:
                dropped += 1
                continue
            key, ann = placed
            if (key[1], key[2]) != (a["tile_x"], a["tile_y"]):
                moved += 1
            if vote == "yes":
                tile_pos[key][cls_idx].append(ann)
                pos += 1
            elif vote == "no":
                tile_pos[key][neg_idx].append(ann)
                neg += 1
        pos_counts[project_id] = pos
        neg_counts[project_id] = neg

    emitted: set = set()
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
            cls_annots_map=cls_annots, emitted=emitted)

        # Shifted crops for boundary-crossing annotations
        if is_extended:
            all_annots = [a for anns in cls_annots.values() for a in anns]
            shift_right, shift_down = _needs_shift(all_annots)
            if shift_right:
                image_id, annot_id = _make_crop_entry(
                    img, [], HALF, 0, images_dir, z, tx, ty, "_r",
                    image_id, annot_id, coco_images, coco_annotations,
                    cls_annots_map=cls_annots, emitted=emitted)
            if shift_down:
                image_id, annot_id = _make_crop_entry(
                    img, [], 0, HALF, images_dir, z, tx, ty, "_d",
                    image_id, annot_id, coco_images, coco_annotations,
                    cls_annots_map=cls_annots, emitted=emitted)
            if shift_right and shift_down:
                image_id, annot_id = _make_crop_entry(
                    img, [], HALF, HALF, images_dir, z, tx, ty, "_rd",
                    image_id, annot_id, coco_images, coco_annotations,
                    cls_annots_map=cls_annots, emitted=emitted)

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
        # negative: bbox 付きで除外クラスの教師データにした件数 (プロジェクト別)
        # negative_bg: bbox が無く従来どおり背景タイルにした枚数
        "negative": neg_counts,
        "negative_bg": len(neg_tiles - set(tile_pos.keys())),
        # moved: 左/上にはみ出すため隣タイルの座標系に移して完全な形で学習させた件数
        # dropped: 隣タイルの DEM が無く完全な形を作れなかったため学習に入れなかった件数
        "moved_to_neighbor": moved,
        "dropped_no_neighbor": dropped,
        # どの切り出しにも完全には収まらず、学習に入らなかった件数。
        # 512 の切り出しに対して大きすぎる対象 (幅 400px 超など) が該当する。
        "not_emitted": sum(len(v) for d in tile_pos.values() for v in d.values())
                       - len(emitted),
    }

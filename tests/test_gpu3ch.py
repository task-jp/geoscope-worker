"""GPU 3ch 生成の同一性テスト。

CPU 版 (visualization.dem_to_3ch + scanning._gen_3ch_uncached の判定) と GPU 版が同じ画素を出すこと、
直接推論の後処理が ultralytics の end2end 後処理と同じ箱を返すことを確かめる。
GPU が無い環境では GPU のテストはスキップし、純粋な Python/CPU のテストだけ走る。
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from app.core import gpu3ch
from app.core.visualization import cell_size_m, dem_to_3ch
from app.services import scanning_gpu

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA が必要")


def _synthetic_dem(seed: int, size: int = 768) -> np.ndarray:
    """起伏 + ノイズ + 一部 NaN の標高 (float64)。"""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    elev = 100 + 30 * np.sin(x / 40) * np.cos(y / 55) + rng.normal(0, 0.5, (size, size))
    elev = np.round(elev, 2)                 # DEM は 0.01m 刻み
    elev[:40, :40] = np.nan                  # 一部欠損
    return elev


def _encode_bgr(elev: np.ndarray) -> np.ndarray:
    """標高 → DEM タイルの 24bit 符号付きエンコード (cv2 の BGR 順)。NaN は x=2^23。"""
    x = np.where(np.isnan(elev), 2**23, np.round(elev * 100).astype(np.int64))
    x = np.where(x < 0, x + 2**24, x)
    bgr = np.empty(elev.shape + (3,), np.uint8)
    bgr[..., 2] = (x >> 16) & 255; bgr[..., 1] = (x >> 8) & 255; bgr[..., 0] = x & 255
    return bgr


def test_ksize_matches_cpu_formula():
    for cs in (0.84, 0.983, 1.0, 1.072):
        k = gpu3ch.ksize_for(cs)
        assert k % 2 == 1 and k == max(3, int(round(31.0 / cs)) | 1)


def test_group_by_column_sorts_rows():
    g = scanning_gpu._group_by_column([(5, 3), (4, 9), (5, 1), (4, 2)])
    assert g == [(4, [2, 9]), (5, [1, 3])]


@cuda
def test_gpu_matches_cpu_pixels():
    dev = torch.device("cuda")
    elevs = [_synthetic_dem(s) for s in range(3)]
    css = [cell_size_m(lat, 16) for lat in (35.0, 43.0, 26.5)]
    bgr = torch.from_numpy(np.stack([_encode_bgr(e) for e in elevs])).to(dev)
    cs = torch.tensor(css, dtype=gpu3ch.DT)
    for i, (e, c) in enumerate(zip(elevs, css)):
        filled = e.copy(); filled[np.isnan(filled)] = np.nanmean(e)
        ref = dem_to_3ch(filled, c)
        k = gpu3ch.ksize_for(c)
        curv = gpu3ch.CurvFFT(k, 768, 768, dev)
        img, keep = gpu3ch.dem_to_3ch_gpu(bgr[i:i + 1], cs[i:i + 1].to(dev), curv)
        got = img[0].cpu().numpy()
        diff = np.abs(got.astype(np.int16) - ref.astype(np.int16))
        # 丸め順序の違いで 1 階調の差がごく稀に出る (実測 51 億値に 1 値)。2 以上の差は不正。
        assert diff.max() <= 1
        assert np.count_nonzero(diff) <= 3
        assert bool(keep[0]) == (min(ref[:, :, ch].std() for ch in range(3)) >= 3)


@cuda
def test_graph_matches_eager(monkeypatch):
    monkeypatch.setattr(gpu3ch, "PRECISION", "fp64")   # 基準の eager 版は fp64 なので Graph も fp64 で作る
    dev = torch.device("cuda")
    B = 4
    shm = torch.zeros((8, 768, 768, 3), dtype=torch.uint8).pin_memory()
    elevs = [_synthetic_dem(10 + s) for s in range(B)]
    for i, e in enumerate(elevs):
        shm[i] = torch.from_numpy(_encode_bgr(e))
    cs = cell_size_m(36.0, 16)
    items = [(i, cs, gpu3ch.ksize_for(cs)) for i, e in enumerate(elevs)]
    gen = gpu3ch.GraphGen(dev, B, shm, compiled=False)
    imgs, keep, ev, h2d = gen(items)
    torch.cuda.synchronize()
    curv = gpu3ch.CurvFFT(items[0][2], 768, 768, dev)
    ref, rkeep = gpu3ch.dem_to_3ch_gpu(shm[:B].to(dev), torch.tensor([cs] * B, dtype=gpu3ch.DT, device=dev), curv)
    assert torch.equal(imgs, ref) and torch.equal(keep, rkeep)
    # 2 回目 (同じ Graph の再生) も同じ
    imgs2, keep2, _, _ = gen(items)
    torch.cuda.synchronize()
    assert torch.equal(imgs2, ref)


def test_direct_postprocess_matches_ultralytics():
    """end2end 出力 (B, 300, 6) に対する後処理が、ultralytics の nms + scale_boxes(clip) と一致する。"""
    from ultralytics.utils import nms, ops
    torch.manual_seed(0)
    preds = torch.rand(3, 300, 6)
    preds[..., :4] = preds[..., :4] * 900 - 60          # 一部は画像外にはみ出す
    preds[..., 5] = torch.randint(0, 3, (3, 300)).float()

    class Net:
        def __call__(self, batch):
            return preds
    imgs = torch.zeros((3, 768, 768, 3), dtype=torch.uint8)
    got = gpu3ch.infer_yolo_direct(Net(), imgs, conf=0.3)
    ref_out = nms.non_max_suppression(preds.clone(), 0.3, 0.5, end2end=True, max_det=300)
    for b in range(3):
        pred = ref_out[b].clone()
        pred[:, :4] = ops.scale_boxes((768, 768), pred[:, :4], (768, 768, 3))
        exp = [{"cx": float((x1 + x2) / 2), "cy": float((y1 + y2) / 2), "w": float(x2 - x1), "h": float(y2 - y1),
                "conf": float(cf), "cls": int(cl)} for x1, y1, x2, y2, cf, cl in pred.numpy()]
        assert got[b] == exp


@cuda
def test_fp32_path_close_to_fp64():
    dev = torch.device("cuda")
    e = _synthetic_dem(21); c = cell_size_m(35.0, 16); k = gpu3ch.ksize_for(c)
    bgr = torch.from_numpy(_encode_bgr(e)[None]).to(dev); cs = torch.tensor([c], dtype=gpu3ch.DT, device=dev)
    img64, keep64 = gpu3ch.dem_to_3ch_gpu(bgr, cs, gpu3ch.CurvFFT(k, 768, 768, dev), fp32=False)
    img32, keep32 = gpu3ch.dem_to_3ch_gpu(bgr, cs, gpu3ch.CurvFFT32(k, 768, 768, dev), fp32=True)
    d = (img32.to(torch.int16) - img64.to(torch.int16)).abs()
    assert int(d.max()) <= 1 and float((d > 0).float().mean()) < 1e-3
    assert bool(keep32[0]) == bool(keep64[0])


@cuda
def test_tensorrt_matches_eager(tmp_path):
    """自前ビルドの TensorRT engine (FP16) が eager fp16 と同じ検出を返す (tensorrt が無ければスキップ)。"""
    import shutil
    pytest.importorskip("tensorrt")
    from app.services import scanning_gpu
    from app.services.scanning import load_model
    dev = torch.device("cuda")
    pt = tmp_path / "yolo26n.pt"
    shutil.copy2(Path(__file__).resolve().parents[1] / "yolo26n.pt", pt)
    model, _, _ = load_model(str(pt), dev)
    eager = scanning_gpu.prepare_direct_model(model, 0.3, dev)
    trt = scanning_gpu._try_tensorrt(model, dev)
    assert trt is not None
    rng = np.random.default_rng(0)
    imgs = torch.from_numpy(rng.integers(0, 255, (8, 768, 768, 3), dtype=np.uint8)).to(dev)
    a = gpu3ch.infer_yolo_direct(eager, imgs, 0.05)
    b = gpu3ch.infer_yolo_direct(trt, imgs, 0.05)
    assert len(a) == len(b) == 8
    # 乱数画像でも検出 (低 conf) が出る。箱の数とおおよその conf が一致すること
    na, nb = sum(len(x) for x in a), sum(len(x) for x in b)
    assert abs(na - nb) <= max(2, na // 10)

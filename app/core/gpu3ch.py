"""3ch 生成 (dem_to_3ch) の GPU 版。CPU 版 (visualization.dem_to_3ch) と同じ画素を出す。

なぜ GPU か:
  CPU 版は 1 枚 308ms かかり、GPU の推論 (2.7ms) に追いつかないので 3ch キャッシュ (768px、
  全国で 782GB) が必要だった。GPU で生成すれば 1 枚 0.9ms (融合時) で、キャッシュ自体が要らなくなる。
  ワーカーはどの構成でも GPU を持つ (YOLO を動かすため) ので、CPU 数に依存しない。

同一性の担保:
  - 倍精度 (float64) で、CPU 版と同じ演算を同じ順序で行う。勾配は np.gradient と同じ差分、
    ガウス核とラプラシアン核は cv2.getGaussianKernel / cv2.getDerivKernels から取り、境界は
    reflect-101 (OpenCV の既定) = torch の 'reflect'。
  - 曲率は「ガウス平滑化 → ラプラシアン」を 1 つの合成核にまとめて FFT で畳み込む。両方の核が
    対称で境界が reflect-101 なので、reflect(17px) → 合成核 1 回と同値。
  - 陰影の三角関数は代数に置換 (cos(aspect)=dx/r 等)。数学的に同値で、丸めが変わるだけ。
  - 実測: 既存キャッシュ 2,911 枚 (51 億値) と比べて 1 値だけ 1 階調差。採用基準は
    「YOLO のスキャンに影響のない範囲の誤差は許容」(2026-10-12 決定)。

速度:
  要素演算は torch.compile (inductor → Triton) で融合する。コンパイラが無い環境では eager に落ちる
  (1.6ms/枚、画素は同じ)。生成全体を ksize ごとに CUDA Graph に捕捉し、専用ストリームで再生する。
  復号プロセスが CPU を使っていても、Graph の再生時間は変わらない (launch が 1 回だから)。

入力の約束:
  復号側は DEM WebP の生バイト (768x768x3 uint8、cv2 の BGR 順) と、現行コードと同じ np.nanmean で
  求めた NaN 埋め値を渡す。標高への復号 (x*0.01 など) はここで倍精度で行う。IEEE の乗算なので
  CPU と同じ値になる。隣接タイルが無い領域は (B,G,R)=(128,0,0)... ではなく R=128,G=0,B=0
  (x = 2^23 = 無効) で埋める (NAN_BGR)。
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.core.dem import TILE_PX
from app.core.visualization import CURVATURE_SMOOTH_M

EXTENDED_PX = TILE_PX + TILE_PX // 2   # 768 (detections.py と同じ定義)
DT = torch.float64
NAN_BGR = np.array([0, 0, 128], np.uint8)   # cv2 BGR 順。R=128,G=0,B=0 → x = 2^23 → NaN
SLOT_BYTES = EXTENDED_PX * EXTENDED_PX * 3


def slot_px(z: int) -> int:
    """復号プロセスが共有メモリに置く生バイト canvas の一辺。z16 は DEM 4 枚から 768。
    z15 は z16 の 3×3 枚 (1536) をそのまま置き、GPU で 2×2 平均して 768 にする (dem.load_dem_z と同じ縮約)。"""
    if z == 16:
        return EXTENDED_PX
    if z == 15:
        return 2 * EXTENDED_PX
    raise ValueError(f"unsupported zoom: {z}")


def ksize_for(cell_size: float) -> int:
    """visualization.dem_to_3ch と同じ ksize (曲率の平滑化幅、奇数)。"""
    return max(3, int(round(CURVATURE_SMOOTH_M / cell_size)) | 1)


# ---------------------------------------------------------------- 復号
MIN_VALID = TILE_PX * TILE_PX * 0.3   # _gen_3ch_uncached と同じ「有効画素 30% 未満は生成しない」(768 canvas に対しても 512² 基準)


def _pool2_nanmean(elev: torch.Tensor, invalid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """z16 の標高 (B,2H,2W) を 2×2 ブロックの nanmean で (B,H,W) にする。dem.load_dem_z と同じ縮約:
    np.nanmean(blocks, axis=2) は NaN を 0 にした和 ((a00+a01)+a10)+a11 を有効数で割る。同じ順で足す。
    戻り値 (標高, 無効マスク)。4 画素とも無効なら無効。"""
    B, H2, W2 = elev.shape
    z = torch.where(invalid, torch.zeros((), dtype=elev.dtype, device=elev.device), elev)
    z = z.view(B, H2 // 2, 2, W2 // 2, 2)
    v = (~invalid).view(B, H2 // 2, 2, W2 // 2, 2)
    s = ((z[:, :, 0, :, 0] + z[:, :, 0, :, 1]) + z[:, :, 1, :, 0]) + z[:, :, 1, :, 1]
    n = ((v[:, :, 0, :, 0].to(torch.int32) + v[:, :, 0, :, 1]) + v[:, :, 1, :, 0]) + v[:, :, 1, :, 1]
    inv = n == 0
    return s / n.clamp_min(1).to(elev.dtype), inv


def gpu_decode_fill(bgr_u8: torch.Tensor, dtype=DT) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,H,W,3) uint8 BGR → (B,H,W) 標高 (dtype) と有効画素数 (B,)。dem.load_dem_raw と同じ式。
    無効画素 (x = 2^23) は有効画素の平均で埋める (_gen_3ch_uncached の canvas[np.isnan] = np.nanmean と同じ)。
    平均は倍精度で合計して求める。以前は復号プロセスが倍精度の標高配列を作って np.nanmean していたが、
    その変換 (タイル 1 枚あたり数十 MB のメモリ往復) が 32 プロセスでメモリ帯域を食い潰し復号が律速になった。
    入力が 1536 角 (z15) なら、復号した z16 の標高を倍精度で 2×2 平均してから同じ処理をする。"""
    pooled = bgr_u8.shape[1] == 2 * EXTENDED_PX
    c = bgr_u8.to(DT if pooled else dtype)
    x = c[..., 2] * 65536 + c[..., 1] * 256 + c[..., 0]      # < 2^24 なので float32 でも厳密
    invalid = x == 2**23
    elev = torch.where(x > 2**23, (x - 2**24) * 0.01, x * 0.01)
    if pooled:   # CPU 版は倍精度で nanmean するので、平均までは倍精度で行う
        elev, invalid = _pool2_nanmean(elev, invalid)
        elev = elev.to(dtype)
    cnt = (~invalid).sum(dim=(1, 2))
    mean = torch.where(invalid, torch.zeros((), dtype=DT, device=elev.device), elev.to(DT)).sum(dim=(1, 2)) / cnt.clamp_min(1).to(DT)
    elev = torch.where(invalid, mean.to(dtype).view(-1, 1, 1), elev)
    return elev, cnt


# ---------------------------------------------------------------- 勾配・陰影・傾斜
def _gradient(f: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """np.gradient(f, h) と同じ: 内側は中心差分、端は片側差分。f: (B,1,H,W)、h: (B,1,1,1)。戻り値 (dy, dx)。"""
    dy = torch.empty_like(f)
    dy[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * h)
    dy[:, :, :1] = (f[:, :, 1:2] - f[:, :, :1]) / h
    dy[:, :, -1:] = (f[:, :, -1:] - f[:, :, -2:-1]) / h
    dx = torch.empty_like(f)
    dx[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2 * h)
    dx[..., :1] = (f[..., 1:2] - f[..., :1]) / h
    dx[..., -1:] = (f[..., -1:] - f[..., -2:-1]) / h
    return dy, dx


def _shade_slope(dy: torch.Tensor, dx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """multi_hillshade (4 方位の平均) と compute_slope を、三角関数を代数に置き換えて計算する。
    cos(az-aspect) を az∈{0,90,180,270} で展開すると cos a, sin a, -cos a, -sin a。
    cos(aspect)=dx/r, sin(aspect)=-dy/r, cos(slope)=1/sqrt(1+r²), sin(slope)=r/sqrt(1+r²)。"""
    alt = math.radians(45)
    r2 = dx * dx + dy * dy
    r = torch.sqrt(r2)
    slope = torch.atan(r)
    inv = torch.rsqrt(1 + r2)
    cos_s, sin_s = inv, r * inv
    safe = torch.where(r > 0, r, torch.ones_like(r))
    cos_a = torch.where(r > 0, dx / safe, torch.ones_like(r))   # atan2(0,0)=0 → cos=1, sin=0
    sin_a = torch.where(r > 0, -dy / safe, torch.zeros_like(r))
    base = math.sin(alt) * cos_s
    k = math.cos(alt) * sin_s
    shade = (torch.clamp(base + k * cos_a, 0, 1) + torch.clamp(base + k * sin_a, 0, 1)
             + torch.clamp(base - k * cos_a, 0, 1) + torch.clamp(base - k * sin_a, 0, 1))
    return shade / 4, torch.clamp(slope / (math.pi / 4), 0, 1)


# ---------------------------------------------------------------- 曲率 (FFT 合成核)
def _fft_size(n: int) -> int:
    """n 以上で 2,3,5 だけを素因数に持つ長さ。実測では 900 (768+34 に対して) が最速だった。"""
    m = max(n, 900)
    while True:
        x = m
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        if x == 1:
            return m
        m += 1


class CurvFFT:
    """compute_curvature の「GaussianBlur(k) → Laplacian(5)」を合成核 1 回の FFT 畳み込みで行う。
    出力は -lap (neg_lap) 相当。正規化 (mean/std) は呼び出し側。"""

    def __init__(self, k: int, H: int, W: int, device: torch.device):
        g = cv2.getGaussianKernel(k, 0)                                   # (k,1) float64
        kd, ks = cv2.getDerivKernels(2, 0, 5, normalize=False, ktype=cv2.CV_64F)
        G2 = g @ g.T
        lap = ks @ kd.T + kd @ ks.T                                       # d2x + d2y
        K = -cv2.filter2D(np.pad(G2, 2), -1, lap, borderType=cv2.BORDER_CONSTANT)
        self.pad = K.shape[0] // 2
        self.H, self.W = H, W
        self.N1, self.N2 = _fft_size(H + 2 * self.pad), _fft_size(W + 2 * self.pad)
        kern = np.zeros((self.N1, self.N2))
        p = self.pad
        for i in range(-p, p + 1):
            for j in range(-p, p + 1):
                kern[i % self.N1, j % self.N2] = K[i + p, j + p]
        self.Kf = torch.fft.rfft2(torch.from_numpy(kern).to(device, DT))

    def __call__(self, f: torch.Tensor) -> torch.Tensor:
        p = self.pad
        mu = f.mean(dim=(2, 3), keepdim=True)                              # 平均を引いて丸め誤差を抑える
        fp = F.pad(f - mu, (p, p, p, p), mode="reflect")
        fp = F.pad(fp, (0, self.N2 - fp.shape[-1], 0, self.N1 - fp.shape[-2]))
        out = torch.fft.irfft2(torch.fft.rfft2(fp) * self.Kf, s=(self.N1, self.N2))
        return out[..., p:p + self.H, p:p + self.W]


def _curvature(curv: CurvFFT, f: torch.Tensor) -> torch.Tensor:
    neg = curv(f)
    mean = neg.mean(dim=(1, 2, 3), keepdim=True)
    std = neg.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(0.01)
    return torch.clamp((neg - mean) / (3 * std) * 0.5 + 0.5, 0, 1)


# ---------------------------------------------------------------- 単精度版 (GPU_3CH_PRECISION=fp32)
# 実測 (2026-10-12): fp64 0.84ms → fp32 0.29ms/枚。uint8 出力の差は最大 1 階調で、値の 0.02〜0.07%。
# 前方後円墳モデル・964 枚で fp64 経路と検出 777/777 一致 (conf 差 最大 0.004)、⭕ 再現 527/632 で同数 → 既定にした。
PRECISION = os.environ.get("GPU_3CH_PRECISION", "fp32")   # 既定 fp32。fp64 は検証・比較用


class CurvFFT32(CurvFFT):
    """CurvFFT の単精度版。平均を引いてから fp32 にするので、標高の絶対値 (〜3000m) の丸めを持ち込まない。"""

    def __init__(self, k: int, H: int, W: int, device: torch.device):
        super().__init__(k, H, W, device)
        self.Kf32 = self.Kf.to(torch.complex64)

    def __call__(self, f: torch.Tensor) -> torch.Tensor:
        p = self.pad
        mu = f.mean(dim=(2, 3), keepdim=True)
        fp = F.pad((f - mu).float(), (p, p, p, p), mode="reflect")
        fp = F.pad(fp, (0, self.N2 - fp.shape[-1], 0, self.N1 - fp.shape[-2]))
        out = torch.fft.irfft2(torch.fft.rfft2(fp) * self.Kf32, s=(self.N1, self.N2))
        return out[..., p:p + self.H, p:p + self.W].double()


def front32(bgr_u8: torch.Tensor, cs: torch.Tensor):
    """front の単精度版。戻り値 (f (B,1,H,W) fp32, ch0, ch1, 有効画素数)。"""
    elev, cnt = gpu_decode_fill(bgr_u8, torch.float32)
    f = elev.unsqueeze(1)
    dy, dx = _gradient(f, cs.view(-1, 1, 1, 1).float())
    ch0, ch1 = _shade_slope(dy, dx)
    return f, ch0, ch1, cnt


# ---------------------------------------------------------------- 前段 / 後段 (compile の単位)
def front(bgr_u8: torch.Tensor, cs: torch.Tensor):
    """復号 + 勾配 + 陰影/傾斜。戻り値 (f (B,1,H,W) fp64, ch0, ch1, 有効画素数)。"""
    elev, cnt = gpu_decode_fill(bgr_u8, DT)
    f = elev.unsqueeze(1)
    dy, dx = _gradient(f, cs.view(-1, 1, 1, 1))
    ch0, ch1 = _shade_slope(dy, dx)
    return f, ch0, ch1, cnt


def back(ch0: torch.Tensor, ch1: torch.Tensor, ch2: torch.Tensor, cnt: torch.Tensor):
    """uint8 画像と、_gen_3ch_uncached の生成対象外判定 (canvas の有効画素 30% 未満、または
    min(img[:,:,c].std()) < 3 → None)。uint8 なので和・二乗和は int64 で厳密に取れる。
    戻り値 (img (B,H,W,3) uint8, keep (B,) bool)。"""
    img = torch.cat([ch0, ch1, ch2], dim=1).permute(0, 2, 3, 1)
    img = (img * 255).to(torch.uint8)
    x = img.to(torch.int64)
    n = img.shape[1] * img.shape[2]
    s1 = x.sum(dim=(1, 2)).to(DT)
    s2 = (x * x).sum(dim=(1, 2)).to(DT)
    std = torch.sqrt(torch.clamp(s2 / n - (s1 / n) ** 2, min=0))
    return img, (std.min(dim=1).values >= 3) & (cnt >= MIN_VALID)


def back_infer(ch0: torch.Tensor, ch1: torch.Tensor, ch2: torch.Tensor, cnt: torch.Tensor):
    """back と同じ判定をしつつ、uint8 画像ではなく推論入力 (B,3,H,W) fp16・RGB・/255 を返す。
    uint8 に量子化してから /255 するので、uint8 画像を経由した場合とモデル入力は同じ値になる。
    (uint8 → permute → half → /255 のコピー 3 回と、集約の torch.cat を無くすため)"""
    img = torch.cat([ch0, ch1, ch2], dim=1)                 # (B,3,H,W) チャンネル順は BGR (ch0 が B)
    q = (img * 255).to(torch.uint8)
    x = q.to(torch.int64)
    n = q.shape[2] * q.shape[3]
    s1 = x.sum(dim=(2, 3)).to(DT)
    s2 = (x * x).sum(dim=(2, 3)).to(DT)
    std = torch.sqrt(torch.clamp(s2 / n - (s1 / n) ** 2, min=0))
    keep = (std.min(dim=1).values >= 3) & (cnt >= MIN_VALID)
    # チャンネル順 BGR → RGB。[2,1,0] の添字指定は CPU 側の index テンソルを同期コピーするため CUDA Graph の
    # 捕捉中に使えない (eager で捕捉する環境で落ちた)。flip は GPU だけで完結する
    return q.flip(1).half() / 255, keep


def dem_to_3ch_gpu(bgr_u8: torch.Tensor, cs: torch.Tensor, curv: CurvFFT, fp32: bool = False):
    """eager 版 (テストと compile 不可環境用)。"""
    f, ch0, ch1, cnt = (front32 if fp32 else front)(bgr_u8, cs)
    return back(ch0.double(), ch1.double(), _curvature(curv, f.double()), cnt)


def try_compile(fn):
    """torch.compile を試し、コンパイラ (gcc/Triton) が無ければ None。"""
    try:
        return torch.compile(fn, dynamic=False)
    except Exception:
        return None


# ---------------------------------------------------------------- CUDA Graph 生成器
class GraphGen:
    """ksize ごとに CUDA Graph を持ち、専用ストリームで非同期に再生する。

    入力は共有メモリ上の (slots, H, W, 3) uint8 テンソル (cudaHostRegister 済み) のスロット番号。
    __call__ は GPU に仕事を投入してすぐ戻る。戻り値の画像は ready_event の後に完成している。
    捕捉は常に 1 本の cap_stream で行う (別ストリームだと共有プールのブロックが再利用されず
    5 本で 10GB に膨らんで OOM になった)。
    """

    def __init__(self, device: torch.device, B: int, shm_tensor: torch.Tensor, compiled: bool = True,
                 output: str = "uint8"):
        """output="uint8": (n,H,W,3) uint8 画像を返す (検証・テスト用)。
        output="infer": 推論入力 (n,3,H,W) fp16 を __call__ の dst に直接書く (本番)。
        shm_tensor の一辺が 1536 なら z15 (z16 の 3×3 枚を GPU で 2×2 平均する)。出力は常に 768 角。"""
        self.device, self.B, self.shm = device, B, shm_tensor
        self.in_px = shm_tensor.shape[1]
        self.output = output
        self.stream = torch.cuda.Stream()
        self.cap_stream = torch.cuda.Stream()
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[int, tuple] = {}
        self.fp32 = PRECISION == "fp32"
        self.c_front = try_compile(front32 if self.fp32 else front) if compiled else None
        self.c_back = try_compile(back_infer if output == "infer" else back) if compiled else None
        self.compiled = self.c_front is not None and self.c_back is not None
        self.host_cs = [torch.empty(B, dtype=DT).pin_memory() for _ in range(2)]
        self.host_ev = [torch.cuda.Event(), torch.cuda.Event()]
        self.hi = 0

    def _build(self, k: int) -> None:
        curv = (CurvFFT32 if self.fp32 else CurvFFT)(k, EXTENDED_PX, EXTENDED_PX, self.device)
        fr = front32 if self.fp32 else front
        if self.compiled:
            c_curv = try_compile(lambda f: _curvature(curv, f.double() if self.fp32 else f))
            if c_curv is None:
                self.compiled = False
        if self.compiled:
            def gen_fn(b, c):
                f, ch0, ch1, cnt = self.c_front(b, c)
                return self.c_back(ch0.double(), ch1.double(), c_curv(f), cnt)
        else:
            bk = back_infer if self.output == "infer" else back
            def gen_fn(b, c):
                f, ch0, ch1, cnt = fr(b, c)
                return bk(ch0.double(), ch1.double(), _curvature(curv, f.double() if self.fp32 else f), cnt)
        s_bgr = torch.zeros((self.B, self.in_px, self.in_px, 3), dtype=torch.uint8, device=self.device)
        s_bgr[..., 2] = 1   # 有効な適当値 (x = 65536) でウォームアップ
        s_cs = torch.full((self.B,), 1.0, dtype=DT, device=self.device)
        torch.cuda.synchronize()
        try:
            with torch.cuda.stream(self.cap_stream):
                for _ in range(2):
                    gen_fn(s_bgr, s_cs)
            torch.cuda.synchronize()
        except Exception as e:
            if not self.compiled:
                raise
            # コンパイル済み関数の実行に失敗 (Triton のビルド失敗など) → eager で作り直す。理由は必ず残す
            # (eager は 1.6 ms/枚で、黙って落ちると「なぜか遅い」になる)
            print(f"[gpu3ch] torch.compile path failed, falling back to eager: {type(e).__name__}: {str(e)[:300]}", flush=True)
            self.compiled = False
            self._build(k)
            return
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool, stream=self.cap_stream):
            o_img, o_keep = gen_fn(s_bgr, s_cs)
        torch.cuda.synchronize()
        # curv (FFT 核) と gen_fn を必ず保持する。解放されると Graph が再利用済みメモリを読み、曲率が壊れる。
        self.graphs[k] = (g, s_bgr, s_cs, o_img, o_keep, torch.cuda.Event(), (curv, gen_fn))

    def prebuild(self, ksizes) -> None:
        for k in sorted(set(ksizes)):
            if k not in self.graphs:
                self._build(k)

    @torch.no_grad()
    def __call__(self, items: list[tuple], dst: torch.Tensor | None = None, dst_free: torch.cuda.Event | None = None):
        """items: 同じ ksize の (slot, cs, ksize, ...) (len ≤ B)。
        戻り値 (out, keep (n,) bool cuda, ready_event, h2d_event)。out は output="uint8" なら (n,H,W,3) uint8 の
        clone、output="infer" なら dst (推論入力 (n,3,H,W) fp16 を書き込んだテンソル)。
        h2d_event の後は共有メモリのスロットを再利用してよい。
        dst_free: dst を前に読んでいた処理 (推論) の完了イベント。dst への書き込みはその後に行う。"""
        k = items[0][2]
        if k not in self.graphs:
            self._build(k)
        g, s_bgr, s_cs, o_img, o_keep, done_ev, _ = self.graphs[k]
        n = len(items)
        hi = self.hi
        self.hi ^= 1
        self.host_ev[hi].synchronize()
        cs = self.host_cs[hi]
        cs.copy_(torch.tensor([it[1] for it in items] + [items[0][1]] * (self.B - n), dtype=DT))
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(done_ev)            # 同じ Graph の前回再生 (静的バッファ使用中) を待つ
            start_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record(self.stream)
            for i, it in enumerate(items):
                s_bgr[i].copy_(self.shm[it[0]], non_blocking=True)
            if n < self.B:
                s_bgr[n:].copy_(s_bgr[0].expand(self.B - n, -1, -1, -1))
            s_cs.copy_(cs, non_blocking=True)
            self.host_ev[hi].record(self.stream)
            h2d_ev = torch.cuda.Event()
            h2d_ev.record(self.stream)
            g.replay()
            if dst is not None:
                if dst_free is not None:
                    self.stream.wait_event(dst_free)   # リングの同じ面をまだ推論が読んでいるかもしれない
                dst.copy_(o_img[:n], non_blocking=True)
                imgs = dst
            else:
                imgs = o_img[:n].clone()
            keep = o_keep[:n].clone()
            done_ev.record(self.stream)
            ready_ev = torch.cuda.Event(enable_timing=True)
            ready_ev.record(self.stream)
        self.last_start_ev = start_ev   # profile 用: ready_ev.elapsed_time(start_ev) で GPU 上の所要時間
        return imgs, keep, ready_ev, h2d_ev


# ---------------------------------------------------------------- 直接推論
def _is_half(net) -> bool:
    """重みの dtype で判定する。AutoBackend が来た場合は内部モデル (net.model) を見る。"""
    inner = getattr(net, "model", net) if net.__class__.__name__ == "AutoBackend" else net
    try:
        return next(inner.parameters()).dtype == torch.float16
    except (StopIteration, AttributeError, TypeError):
        return False


@torch.no_grad()
def infer_yolo_direct(net, imgs_u8: torch.Tensor, conf: float, max_det: int = 300) -> list[list[dict]]:
    """ultralytics の model.predict と同じ結果を、ラッパーを通さずに得る。

    predict は入力バッチ全体を CPU に戻して numpy 化し (convert_torch2numpy_batch)、画像ごとに Results を
    作る。実測で 1 枚 0.9ms (CPU が混むと 3ms) かかり、forward (1.0ms) と同じかそれ以上だった。
    end2end モデルの後処理は `pred[pred[:,4] > conf][:max_det]` と、同サイズなので gain=1/pad=0 の
    scale_boxes = clip だけ。ここではそれをバッチで行い、CPU への転送を 1 回にする。
    辞書化は scanning._infer_yolo と同じ float32 演算。実測で predict と 442/442 件ビット一致。
    """
    return infer_yolo_collect(infer_yolo_launch(net, imgs_u8, conf, max_det))


@torch.no_grad()
def infer_yolo_launch(net, imgs_u8: torch.Tensor, conf: float, max_det: int = 300):
    """infer_yolo_direct の前半。forward と後処理を現在のストリームに投入し、結果の CPU への転送も
    非同期に始めてすぐ戻る。戻り値を infer_yolo_collect に渡すと検出の辞書が得られる。

    スキャンでは、結果を待つ間に主スレッドが次の生成を GPU に投入できるよう、この 2 つを分けて呼ぶ。
    同期 (.cpu()) で待つと、その間 GPU の生成ストリームが空になり、GPU の使用率が半分に落ちた (実測)。"""
    half = _is_half(net)
    if imgs_u8.dtype == torch.uint8:
        batch = imgs_u8.permute(0, 3, 1, 2)[:, [2, 1, 0]]
        batch = (batch.half() if half else batch.float()) / 255
    else:   # back_infer が作った前処理済み入力 (fp16, RGB, /255)
        batch = imgs_u8 if (imgs_u8.dtype == torch.float16) == half else (imgs_u8.half() if half else imgs_u8.float())
    preds = net(batch)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    if preds.ndim != 3 or preds.shape[-1] != 6:
        raise ValueError(f"end2end (NMS-free) の出力を想定: {tuple(preds.shape)}")
    preds = preds.float()[:, :max_det]
    mask = preds[..., 4] > conf
    p = preds.clone()   # TensorRT の出力は使い回しのバッファなので、次の推論に上書きされる前に複製する
    w = h = float(imgs_u8.shape[2])
    p[..., 0].clamp_(0, w); p[..., 1].clamp_(0, h); p[..., 2].clamp_(0, w); p[..., 3].clamp_(0, h)
    if not p.is_cuda:
        return None, mask, p
    mask_h = torch.empty(mask.shape, dtype=mask.dtype, pin_memory=True)
    p_h = torch.empty(p.shape, dtype=p.dtype, pin_memory=True)
    mask_h.copy_(mask, non_blocking=True)
    p_h.copy_(p, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    return ev, mask_h, p_h


def infer_yolo_collect(pending) -> list[list[dict]]:
    """infer_yolo_launch の結果を待って (転送完了まで) 検出の辞書にする。"""
    ev, mask_h, p_h = pending
    if ev is not None:
        ev.synchronize()
    mask_c = mask_h.numpy()
    p_c = p_h.numpy()
    out = []
    for b in range(p_c.shape[0]):
        dets = []
        for x1, y1, x2, y2, cf, cl in p_c[b][mask_c[b]]:
            dets.append({"cx": float((x1 + x2) / 2), "cy": float((y1 + y2) / 2), "w": float(x2 - x1),
                         "h": float(y2 - y1), "conf": float(cf), "cls": int(cl)})
        out.append(dets)
    return out

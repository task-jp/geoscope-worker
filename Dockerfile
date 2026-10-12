# GeoScope クラウドワーカー Docker image (AGPL-3.0)
# Built from this repository's source, distributed at:
#   ghcr.io/task-jp/geoscope-worker:latest
#
# Source code: https://github.com/task-jp/geoscope-worker

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
# gcc は torch.compile (Triton) の補助モジュールのビルドに必要。無いと 3ch 生成が eager に落ちる (1.6ms/枚、画素は同じ)。
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ca-certificates curl tar gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # TensorRT の builder リソースは GPU 世代別に 250〜650MB ずつ入っている。Windows 用 (_win_) と
    # 使わない世代 (sm90=H100, sm100=B200) を外して約 3.6GB 減らす。残すのは sm80 (A100)、sm89 (4090)、
    # sm120 (RTX 50 系)、ptx (それ以外の世代の汎用)。別の GPU を使うときはここを見直すこと
    && cd /opt/conda/lib/python3.11/site-packages/tensorrt_libs \
    && rm -f libnvinfer_builder_resource_win_*.so.* libnvinfer_builder_resource_sm90.so.* libnvinfer_builder_resource_sm100.so.* \
    && ls -la libnvinfer_builder_resource_* | awk '{print $5, $9}'

COPY worker.py .
COPY app ./app
COPY yolo26n.pt yolo11n.pt ./

# クラウド実行用デフォルト環境変数
ENV REMOTE_TILES=true \
    DISABLE_PID_LOCK=true \
    GEOSCOPE_SERVER=https://geoscope.jp \
    POLL_INTERVAL=10 \
    PYTHONUNBUFFERED=1 \
    TILES_DIR=/workspace/tiles \
    MODELS_DIR=/workspace/models \
    DATASETS_DIR=/workspace/datasets

RUN mkdir -p /workspace/tiles /workspace/models /workspace/datasets

ENTRYPOINT ["python3", "-u", "worker.py"]

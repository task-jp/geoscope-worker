# GeoScope クラウドワーカー Docker image (AGPL-3.0)
# Built from this repository's source, distributed at:
#   docker.io/tasukusuzukisignalslot/geoscope-worker:latest
#
# Source code: https://github.com/task-jp/geoscope-worker

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ca-certificates curl tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

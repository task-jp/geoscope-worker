# geoscope-worker

[GeoScope](https://geoscope.jp) のクラウド GPU ワーカー。BYO (Bring Your Own) クラウドGPU 構成で、各ユーザーが自分の [RunPod](https://runpod.io?ref=ok9s0q0q) アカウントで Pod を起動して DEM 標高データから地形特徴を検出する。

## ライセンス

このリポジトリのコードは **GNU Affero General Public License v3.0 (AGPL-3.0)** で配布されています。([LICENSE](./LICENSE))

依存ライブラリ:
- [ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- PyTorch (BSD)
- OpenCV (Apache 2.0)

## 関連リポジトリ

- **GeoScope backend / frontend**: 別リポジトリで運用 (このリポジトリの worker と HTTP API 経由で通信)
- **公開 Docker image**: `docker.io/tasukusuzukisignalslot/geoscope-worker:latest`

## アーキテクチャ

```
[ユーザーの RunPod Pod]
   ├ worker.py が GeoScope サーバー API を polling
   ├ DEM タイルを Cloudflare R2 から stream 取得 (初回のみ)
   ├ ultralytics (YOLO) で学習・推論
   └ 結果を GeoScope サーバーに POST
```

Pod の `/workspace` (volumeInGb) に DEM キャッシュ・モデルを永続化し、Pod stop/start で再利用する。

## ローカル実行

```bash
# 環境変数
export WORKER_API_KEY=<自分のGeoScope API key>
export GEOSCOPE_SERVER=https://geoscope.jp
export REMOTE_TILES=true
# 任意: Cloudflare R2 (DEM tar 配布)
export DEM_TILE_BASE_URL=https://pub-xxxxx.r2.dev

pip install -r requirements.txt
python3 -u worker.py
```

## Docker

```bash
docker build -t geoscope-worker .
docker run --gpus all \
  -e WORKER_API_KEY=... \
  -e GEOSCOPE_SERVER=https://geoscope.jp \
  -v /path/to/workspace:/workspace \
  geoscope-worker
```

## 環境変数

| 変数 | 説明 | デフォルト |
|------|------|----------|
| `WORKER_API_KEY` | GeoScope API 認証キー (必須) | - |
| `GEOSCOPE_SERVER` | GeoScope サーバー URL | `https://geoscope.jp` |
| `REMOTE_TILES` | DEM タイルをサーバー/R2 から取得 | `true` |
| `DEM_TILE_BASE_URL` | DEM タイル配布元 (Cloudflare R2 等) | `(なし)` |
| `PREFETCH_PARALLEL` | DEM プリフェッチ並列度 | `256` |
| `DISABLE_PID_LOCK` | PID ロックを無効化 (複数 Pod 並列起動用) | `false` |
| `TILES_DIR`, `MODELS_DIR`, `DATASETS_DIR` | 各データ保存先 | (Dockerfile デフォルト) |
| `MIN_TRAINING_MAP50` | 学習品質ゲート (mAP50 閾値) | `0.2` |

## ソース要件 (AGPL-3.0)

このコードを改変して配布する場合、AGPL-3.0 のもとでソースコードの公開が必要です。サービスとして提供する場合 (= Pod を他人に使わせる場合) も同様です。

## 出処

GeoScope 本体の `webapp/backend/worker.py` + `app/services/{dataset,scanning,domeness}.py` + `app/core/{dem,visualization}.py` を AGPL-3.0 として切り出したものです。

# Bear Sighting Risk Prediction API - FastAPI + Folium

東北3県（秋田・岩手・福島）のクマ出没予測システム用FastAPI + Folium地図UIです。

## ファイル構成

```
api/
├── __init__.py
├── main.py                    # FastAPIエントリポイント
├── routes/
│   ├── __init__.py
│   ├── risk.py               # リスク予測エンドポイント
│   └── sightings.py          # 目撃情報エンドポイント
└── templates/
    └── map.html              # Folium地図テンプレート
```

## セットアップ

### 1. 依存パッケージのインストール

```bash
cd /sessions/clever-brave-brahmagupta/mnt/CU_JAXA
source .venv/bin/activate
pip install -r requirements.txt
```

必須パッケージ:
- `fastapi>=0.110`
- `uvicorn>=0.27`
- `folium>=0.17`
- `pandas>=2.0`
- `numpy>=1.24`
- `lightgbm>=4.0` (オプション)
- `jinja2>=3.1`

### 2. サーバー起動

```bash
cd /sessions/clever-brave-brahmagupta/mnt/CU_JAXA
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

サーバーが起動すると:
- Web UI: http://localhost:8000/
- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## APIエンドポイント

### GET / (ルート)

Folium地図UIをHTMLで返します。

```bash
curl http://localhost:8000/
```

**機能:**
- OpenStreetMapベースの地図表示
- リスクヒートマップ（メッシュ塗り分け）
- 目撃情報マーカー表示
- 月選択セレクタ
- リスクレベル凡例

---

### GET /api/risk

指定月のリスクスコアをメッシュ単位で返します。

```bash
curl "http://localhost:8000/api/risk?date=2025-02"
```

**パラメータ:**
- `date` (string, required): 対象月（YYYY-MM形式）
  - デフォルト: 現在月

**レスポンス例:**
```json
{
  "date": "2025-02",
  "risk_scores": {
    "mesh_0_0": 45.3,
    "mesh_0_1": 62.1,
    "mesh_0_2": 38.5,
    ...
  },
  "statistics": {
    "mean_risk": 48.5,
    "median_risk": 50.2,
    "max_risk": 98.2,
    "min_risk": 5.3,
    "std_risk": 25.1,
    "high_risk_count": 123,      // 80以上
    "medium_risk_count": 456,    // 40-79
    "low_risk_count": 789        // 0-39
  },
  "mesh_count": 4425,
  "bbox": [139.2, 36.8, 142.1, 40.5]
}
```

**注意:**
- リスクスコアは0-100でスケーリング
- 特徴量テーブル(`data/processed/features.parquet`)がない場合、ダミーデータを返す
- モデル(`models/outputs/model_full.lgb`)がない場合、簡易スコアリングを使用

---

### GET /api/sightings

指定期間の目撃情報を返します。

```bash
curl "http://localhost:8000/api/sightings?from=2024-01-01&to=2025-02-13"
```

**パラメータ:**
- `from` (string): YYYY-MM-DD開始日（デフォルト: 今月1日）
- `to` (string): YYYY-MM-DD終了日（デフォルト: 今日）
- `prefecture` (string): 県名フィルタ（秋田/岩手/福島）
- `limit` (int): 取得件数上限（デフォルト: 1000, 最大: 10000）

**レスポンス例:**
```json
{
  "from": "2024-01-01",
  "to": "2025-02-13",
  "prefecture": null,
  "total_count": 456,
  "returned_count": 100,
  "sightings": [
    {
      "id": 1,
      "date": "2025-02-10",
      "latitude": 39.5123,
      "longitude": 140.3456,
      "prefecture": "Akita",
      "location": "Kakunodate City",
      "details": "Sighting near mountain area..."
    },
    ...
  ]
}
```

---

### GET /api/sightings/statistics

目撃情報の統計情報を返します。

```bash
curl "http://localhost:8000/api/sightings/statistics?from=2024-01-01"
```

**パラメータ:**
- `from` (string): YYYY-MM-DD開始日
- `to` (string): YYYY-MM-DD終了日

**レスポンス例:**
```json
{
  "from": "2024-01-01",
  "to": "2025-02-13",
  "total_sightings": 456,
  "by_prefecture": {
    "Akita": 150,
    "Iwate": 200,
    "Fukushima": 106
  },
  "by_month": {
    "2024-01": 40,
    "2024-02": 35,
    "2024-03": 42
  }
}
```

---

### GET /api/feature-importance

予測モデルの特徴量重要度を返します。

```bash
curl http://localhost:8000/api/feature-importance
```

**レスポンス例:**
```json
{
  "feature_importance": {
    "NDVI": 0.25,
    "elevation": 0.20,
    "forest_cover": 0.18,
    "precipitation": 0.15,
    "temperature": 0.12,
    "distance_to_road": 0.10
  },
  "note": "ダミーデータ（特徴量テーブルが見つかりません）"
}
```

**注意:**
- モデルが利用できない場合、ダミー値が返される
- 重要度の合計は1.0に正規化される

---

### GET /api/health

ヘルスチェック・リソース確認

```bash
curl http://localhost:8000/api/health
```

**レスポンス例:**
```json
{
  "status": "ok",
  "features_available": false,
  "sightings_available": false,
  "model_available": false
}
```

---

## 設定ファイル

### config.py

プロジェクトルートの`config.py`で以下が定義されています:

```python
# 対象地域: 東北3県
BBOX = [139.2, 36.8, 142.1, 40.5]  # [west, south, east, north]

# ディレクトリパス
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SIGHTINGS_DIR = PROJECT_ROOT / "data" / "raw" / "sightings"

# ファイルパス
SIGHTINGS_CSV = SIGHTINGS_DIR / "tohoku_sightings.csv"  # 目撃データ
```

## フォールバック処理

データやモデルが見つからない場合でもサーバーが起動できるように設計されています:

| リソース | 見つからない場合 |
|---------|----------------|
| 特徴量テーブル | ダミーのグリッドベースリスクスコア (0-100) |
| 目撃データCSV | 空の配列を返す |
| LightGBMモデル | 特徴量の平均値ベースの簡易スコアリング |

## データファイル期待形式

### 目撃データ (`data/raw/sightings/tohoku_sightings.csv`)

必須カラム:
- `date` or `日付`: 目撃日（YYYY-MM-DD形式推奨）
- `latitude` or `緯度`: 緯度
- `longitude` or `経度`: 経度
- `prefecture` or `都道府県` or `県`: 県名
- `location` or `市町村`: 位置情報（オプション）
- `details` or `備考`: 詳細情報（オプション）

### 特徴量テーブル (`data/processed/features.parquet`)

期待形式:
- Parquetファイル
- `mesh_id`: メッシュID
- `year_month` or `date`: 時間情報
- `NDVI`, `elevation`, `forest_cover` 等の数値特徴量

### LightGBMモデル (`models/outputs/model_full.lgb`)

- LightGBM Boosterオブジェクト
- `lgb.save_model()`で保存されたバイナリファイル

## 開発・テスト

### APIテスト

```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# ルートエンドポイント
response = client.get("/")
assert response.status_code == 200

# リスクAPI
response = client.get("/api/risk?date=2025-02")
assert response.status_code == 200
data = response.json()
assert "risk_scores" in data
assert "statistics" in data

# 目撃情報API
response = client.get("/api/sightings?from=2024-01-01&to=2025-02-13")
assert response.status_code == 200
data = response.json()
assert "sightings" in data
assert "total_count" in data
```

### ローカル開発サーバー起動

```bash
uvicorn api.main:app --reload --port 8000
```

`--reload` フラグで自動リロード有効

## トラブルシューティング

### ModuleNotFoundError: No module named 'fastapi'

```bash
source .venv/bin/activate
pip install fastapi uvicorn folium httpx
```

### ポート8000が既に使用されている

別のポートで起動:
```bash
uvicorn api.main:app --port 8001
```

### Folium地図が表示されない

- ブラウザのコンソールで JavaScript エラーを確認
- CDNからLeafletが正常に読み込まれているか確認
- キャッシュをクリアしてリロード

### リスクスコアがすべてダミーデータ

- `data/processed/features.parquet` が存在するか確認
- ファイルが正しいParquet形式か確認
- ログを確認: 読み込みエラーが記録されている

## ポート設定

| ポート | 用途 |
|-------|------|
| 8000 | メイン (推奨) |
| 8001 | 代替（バックアップ） |
| 8080 | 本番環境 |

## パフォーマンスチューニング

### リスク計算のキャッシング

グローバル変数でデータをキャッシュ:
```python
_features_cache = None
_sightings_cache = None
_model_cache = None
```

初回起動時のみディスク読み込み → 2回目以降はメモリから取得

### 大規模データセット対応

目撃情報APIの `limit` パラメータで件数制限 (デフォルト: 1000)

```bash
# 最大10000件まで取得可能
curl "http://localhost:8000/api/sightings?from=2024-01-01&limit=10000"
```

## セキュリティ考慮事項

- CORS設定がまだない → 本番環境では追加推奨
- 入力値の検証は基本的なもののみ → 詳細な検証が必要な場合は追加
- APIキー認証がない → 本番環境では実装推奨

### CORS設定例

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では具体的なオリジンを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## ドキュメント

### 自動生成ドキュメント

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

これらは FastAPI により自動生成されます。

## ライセンス

プロジェクトのLICENSEファイルを参照してください。

## 問い合わせ

バグ報告や機能リクエストは GitHub Issues で。

"""
FastAPI エントリポイント
東北3県クマ出没予測 - リスク地図UI
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

from config import (
    BBOX,
    PROCESSED_DIR,
    SIGHTINGS_DIR,
    PROJECT_ROOT as CONFIG_ROOT,
)

# ===== ログ設定 =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== FastAPI初期化 =====
app = FastAPI(
    title="クマ出没予測API",
    description="東北3県（秋田・岩手・福島）のクマ出没リスク予測システム",
    version="1.0.0",
)

# ===== グローバル変数: キャッシュデータ =====
_features_cache = None
_sightings_cache = None
_model_cache = None
_feature_importance_cache = None


def load_features():
    """特徴量テーブルをキャッシュから読み込む"""
    global _features_cache
    if _features_cache is not None:
        return _features_cache

    features_path = PROCESSED_DIR / "features.parquet"
    if features_path.exists():
        try:
            _features_cache = pd.read_parquet(features_path)
            logger.info(f"特徴量テーブル読み込み成功: {features_path}")
            return _features_cache
        except Exception as e:
            logger.error(f"特徴量テーブル読み込み失敗: {e}")
            return None
    else:
        logger.warning(f"特徴量テーブルが見つかりません: {features_path}")
        return None


def load_sightings():
    """目撃データをキャッシュから読み込む"""
    global _sightings_cache
    if _sightings_cache is not None:
        return _sightings_cache

    sightings_path = SIGHTINGS_DIR / "tohoku_sightings.csv"
    if sightings_path.exists():
        try:
            df = pd.read_csv(sightings_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            _sightings_cache = df
            logger.info(f"目撃データ読み込み成功: {len(df)}件")
            return _sightings_cache
        except Exception as e:
            logger.error(f"目撃データ読み込み失敗: {e}")
            return None
    else:
        logger.warning(f"目撃データが見つかりません: {sightings_path}")
        return None


def load_model():
    """LightGBMモデルをキャッシュから読み込む"""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    try:
        import lightgbm as lgb
        model_path = CONFIG_ROOT / "models" / "outputs" / "model_full.lgb"
        if model_path.exists():
            _model_cache = lgb.Booster(model_file=str(model_path))
            logger.info(f"モデル読み込み成功: {model_path}")
            return _model_cache
        else:
            logger.warning(f"モデルが見つかりません: {model_path}")
            return None
    except ImportError:
        logger.warning("lightgbmがインストールされていません")
        return None
    except Exception as e:
        logger.error(f"モデル読み込み失敗: {e}")
        return None


def generate_risk_heatmap(features_df, target_date=None):
    """
    リスクスコアを生成（メッシュ単位）

    Args:
        features_df: 特徴量データフレーム
        target_date: 対象月（YYYY-MM形式）

    Returns:
        {mesh_id: risk_score} の辞書
    """
    if features_df is None:
        logger.warning("特徴量データがないため、ダミーのリスクスコアを返す")
        return _generate_dummy_risk_scores()

    model = load_model()
    if model is None:
        logger.info("モデルが利用できないため、簡易スコアリングを使用")
        return _generate_simple_risk_scores(features_df)

    try:
        # モデルで予測
        risk_scores = {}
        predictions = model.predict(features_df)

        # 0-100スケールに正規化
        if len(predictions) > 0:
            min_pred, max_pred = predictions.min(), predictions.max()
            if max_pred > min_pred:
                normalized = ((predictions - min_pred) / (max_pred - min_pred)) * 100
            else:
                normalized = np.full_like(predictions, 50.0)

            # メッシュIDとスコアを紐付け
            if "mesh_id" in features_df.columns:
                for idx, score in zip(features_df["mesh_id"], normalized):
                    risk_scores[str(idx)] = float(score)

        logger.info(f"リスクスコア計算完了: {len(risk_scores)}メッシュ")
        return risk_scores
    except Exception as e:
        logger.error(f"リスク予測エラー: {e}")
        return _generate_simple_risk_scores(features_df)


def _generate_simple_risk_scores(features_df):
    """簡易的なリスクスコア（特徴量の平均値ベース）"""
    risk_scores = {}
    try:
        # NDVI, 標高, 森林率などの特徴を基に簡易スコア化
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for idx, row in features_df.iterrows():
                # 特徴量を0-100にスケーリング
                values = row[numeric_cols].values
                score = np.nanmean(values) * 10  # 簡易的なスケーリング
                score = np.clip(score, 0, 100)

                mesh_id = row.get("mesh_id", idx)
                risk_scores[str(mesh_id)] = float(score)
    except Exception as e:
        logger.error(f"簡易スコアリング失敗: {e}")

    return risk_scores if risk_scores else _generate_dummy_risk_scores()


def _generate_dummy_risk_scores():
    """ダミーのリスクスコア（開発/テスト用）"""
    west, south, east, north = BBOX

    # グリッド生成（0.05度間隔 ≈ 5km）
    lat_points = np.arange(south, north, 0.05)
    lon_points = np.arange(west, east, 0.05)

    dummy_scores = {}
    for i, lat in enumerate(lat_points):
        for j, lon in enumerate(lon_points):
            mesh_id = f"mesh_{i}_{j}"
            # 緯度経度ベースのランダムスコア
            score = (np.sin(lat * 0.1) + np.cos(lon * 0.1)) * 25 + 50
            score = np.clip(score, 0, 100)
            dummy_scores[mesh_id] = float(score)

    logger.info(f"ダミーリスクスコア生成: {len(dummy_scores)}メッシュ")
    return dummy_scores


def get_feature_importance():
    """特徴量重要度を取得"""
    global _feature_importance_cache
    if _feature_importance_cache is not None:
        return _feature_importance_cache

    model = load_model()
    if model is None:
        logger.warning("モデルが利用できないため、ダミーの重要度を返す")
        _feature_importance_cache = {
            "NDVI": 0.25,
            "elevation": 0.20,
            "forest_cover": 0.18,
            "precipitation": 0.15,
            "temperature": 0.12,
            "distance_to_road": 0.10,
        }
        return _feature_importance_cache

    try:
        importance = model.feature_importance()
        feature_names = model.feature_name()

        # 正規化
        total = importance.sum()
        if total > 0:
            importance_dict = {
                name: float(imp / total)
                for name, imp in zip(feature_names, importance)
            }
        else:
            importance_dict = {name: 0.0 for name in feature_names}

        _feature_importance_cache = importance_dict
        return _feature_importance_cache
    except Exception as e:
        logger.error(f"特徴量重要度取得失敗: {e}")
        return {}


# ===== エンドポイント =====

@app.get("/", response_class=HTMLResponse)
async def index():
    """
    地図UI（Folium + Jinja2）
    """
    try:
        # リスクスコアを読み込み
        features = load_features()
        risk_scores = generate_risk_heatmap(features)

        # 目撃データを読み込み
        sightings = load_sightings()

        # 地図作成（中心: 東北3県中心）
        center_lat, center_lon = 38.5, 140.5
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles="OpenStreetMap"
        )

        # リスクヒートマップを追加
        if risk_scores:
            heat_data = []
            for mesh_id, score in list(risk_scores.items())[:500]:  # 最初の500メッシュ
                # メッシュIDから座標を復元（ダミー: ランダム）
                np.random.seed(hash(mesh_id) % 2**32)
                lat = center_lat + (np.random.random() - 0.5) * 3
                lon = center_lon + (np.random.random() - 0.5) * 3

                # スコアを熱量として追加
                heat_data.append([lat, lon, score])

            HeatMap(heat_data, radius=30, blur=20, max_zoom=1, name="リスクヒートマップ").add_to(m)

        # 目撃情報マーカーを追加
        if sightings is not None and len(sightings) > 0:
            marker_cluster = MarkerCluster(name="目撃情報").add_to(m)

            for idx, row in sightings.iterrows():
                if "latitude" in row and "longitude" in row:
                    try:
                        lat = float(row["latitude"])
                        lon = float(row["longitude"])

                        # ポップアップテキスト
                        popup_text = f"""
                        <b>目撃情報</b><br>
                        日付: {row.get('date', '不明')}<br>
                        位置: {lat:.3f}, {lon:.3f}<br>
                        """

                        folium.Marker(
                            location=[lat, lon],
                            popup=folium.Popup(popup_text, max_width=250),
                            icon=folium.Icon(color="red", icon="info-sign")
                        ).add_to(marker_cluster)
                    except (ValueError, TypeError):
                        continue

        # レイヤーコントロール
        folium.LayerControl().add_to(m)

        # 地図をHTML文字列で取得
        map_html = m._repr_html_()

        # Jinja2テンプレートにレンダリング
        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>クマ出没予測 - 東北3県リスク地図</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            font-size: 28px;
            margin-bottom: 5px;
        }}

        .header p {{
            font-size: 14px;
            opacity: 0.9;
        }}

        .controls {{
            background: white;
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .control-group label {{
            font-weight: 500;
            font-size: 14px;
        }}

        .control-group select {{
            padding: 8px 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
            background-color: white;
            cursor: pointer;
        }}

        .control-group select:hover {{
            border-color: #667eea;
        }}

        .legend {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 12px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            border: 1px solid #ddd;
        }}

        .map-container {{
            position: relative;
            height: calc(100vh - 200px);
            background: #eee;
        }}

        .map-container iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}

        .info {{
            background: white;
            padding: 15px 20px;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #666;
        }}

        .info a {{
            color: #667eea;
            text-decoration: none;
        }}

        .info a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Bear Sighting Risk Prediction System</h1>
        <p>Tohoku 3 Prefecture Risk Map (Akita, Iwate, Fukushima) | Machine Learning Based Prediction</p>
    </div>

    <div class="controls">
        <div class="control-group">
            <label for="month-select">表示月:</label>
            <select id="month-select" onchange="updateRisk()">
                <option value="2025-02">2025年2月</option>
                <option value="2025-01">2025年1月</option>
                <option value="2024-12">2024年12月</option>
                <option value="2024-11">2024年11月</option>
                <option value="2024-10">2024年10月</option>
            </select>
        </div>

        <div class="control-group legend">
            <div style="font-weight: 600; margin-bottom: 10px;">リスクレベル</div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ff0000;"></div>
                <span>高リスク（80-100）</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ffaa00;"></div>
                <span>中リスク（40-79）</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #00cc00;"></div>
                <span>低リスク（0-39）</span>
            </div>
        </div>
    </div>

    <div class="map-container">
        {map_html}
    </div>

    <div class="info">
        <p>
            Sightings: Red markers indicate bear sighting locations |
            Risk Map: Heatmap color intensity indicates risk level |
            Updates: Monthly (1st of each month) |
            <a href="/api/feature-importance">Feature Importance API</a>
        </p>
    </div>

    <script>
        function updateRisk() {{
            const month = document.getElementById("month-select").value;
            console.log("リスク月を更新:", month);
            // 実装: /api/risk?date=YYYY-MM へリクエスト
            fetch(`/api/risk?date=${{month}}`)
                .then(r => r.json())
                .then(data => {{
                    console.log("リスクスコア更新:", data);
                    // 地図を更新（実装予定）
                }})
                .catch(e => console.error("リスク更新エラー:", e));
        }}
    </script>
</body>
</html>
        """

        return html_content

    except Exception as e:
        logger.error(f"地図UI生成エラー: {e}")
        return f"""
<html>
<head><title>エラー</title></head>
<body>
<h1>地図UIの生成に失敗しました</h1>
<p>{str(e)}</p>
</body>
</html>
        """


@app.get("/api/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "features_available": load_features() is not None,
        "sightings_available": load_sightings() is not None,
        "model_available": load_model() is not None,
    }


@app.get("/api/feature-importance")
async def get_feature_importance_endpoint():
    """特徴量重要度を返す"""
    return {
        "feature_importance": get_feature_importance(),
        "note": "モデルが利用できない場合はダミー値が返されます"
    }


# ===== routes のインポート=====
from api.routes import risk_router, sightings_router

app.include_router(risk_router)
app.include_router(sightings_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

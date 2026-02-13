"""
リスク予測エンドポイント
GET /api/risk?date=YYYY-MM
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, HTTPException, Query
from config import PROCESSED_DIR, BBOX

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Risk Prediction"])


@router.get("/risk")
async def get_risk_scores(date: str = Query("2025-02", description="YYYY-MM形式の対象月")):
    """
    指定月のリスクスコアを返す（メッシュ単位）

    Args:
        date: YYYY-MM形式の対象月

    Returns:
        {
            "date": "2025-02",
            "risk_scores": {
                "mesh_001": 75.5,
                "mesh_002": 42.3,
                ...
            },
            "statistics": {
                "mean_risk": 48.5,
                "max_risk": 98.2,
                "min_risk": 5.3,
                "high_risk_count": 123
            },
            "bbox": [139.2, 36.8, 142.1, 40.5]
        }
    """
    try:
        # 日付形式の検証
        try:
            target_date = datetime.strptime(date, "%Y-%m")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="日付形式がYYYY-MMではありません"
            )

        # 特徴量テーブルを読み込み
        features_path = PROCESSED_DIR / "features.parquet"
        if not features_path.exists():
            logger.warning("特徴量テーブルが見つかりません")
            # ダミーデータを返す
            return _generate_dummy_risk_response(date)

        df_features = pd.read_parquet(features_path)

        # 日付フィルタリング
        if "year_month" in df_features.columns:
            df_filtered = df_features[df_features["year_month"] == date]
        elif "date" in df_features.columns:
            df_filtered = df_features[
                df_features["date"].dt.strftime("%Y-%m") == date
            ]
        else:
            df_filtered = df_features  # すべてのデータを使用

        if df_filtered.empty:
            logger.warning(f"対象月 {date} のデータが見つかりません")
            return _generate_dummy_risk_response(date)

        # リスクスコアを計算（簡易版: 特徴量の平均）
        risk_scores = {}
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns

        for idx, row in df_filtered.iterrows():
            mesh_id = row.get("mesh_id", f"mesh_{idx}")
            values = row[numeric_cols].values
            # 0-100スケールのリスクスコア
            score = np.nanmean(values) * 10
            score = np.clip(score, 0, 100)
            risk_scores[str(mesh_id)] = float(score)

        # 統計情報を計算
        scores_array = np.array(list(risk_scores.values()))
        stats = {
            "mean_risk": float(np.mean(scores_array)),
            "median_risk": float(np.median(scores_array)),
            "max_risk": float(np.max(scores_array)),
            "min_risk": float(np.min(scores_array)),
            "std_risk": float(np.std(scores_array)),
            "high_risk_count": int(np.sum(scores_array >= 80)),
            "medium_risk_count": int(np.sum((scores_array >= 40) & (scores_array < 80))),
            "low_risk_count": int(np.sum(scores_array < 40)),
        }

        return {
            "date": date,
            "risk_scores": risk_scores,
            "statistics": stats,
            "mesh_count": len(risk_scores),
            "bbox": BBOX,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"リスク予測エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")


def _generate_dummy_risk_response(date: str):
    """ダミーのリスクスコア応答（開発用）"""
    west, south, east, north = BBOX

    # グリッド生成（0.05度間隔）
    lat_points = np.arange(south, north, 0.05)
    lon_points = np.arange(west, east, 0.05)

    risk_scores = {}
    for i, lat in enumerate(lat_points):
        for j, lon in enumerate(lon_points):
            mesh_id = f"mesh_{i}_{j}"
            # 位置ベースのスコア（擬似乱数）
            score = (np.sin(lat * 0.1) + np.cos(lon * 0.1)) * 25 + 50
            score = np.clip(score, 0, 100)
            risk_scores[mesh_id] = float(score)

    scores_array = np.array(list(risk_scores.values()))
    stats = {
        "mean_risk": float(np.mean(scores_array)),
        "median_risk": float(np.median(scores_array)),
        "max_risk": float(np.max(scores_array)),
        "min_risk": float(np.min(scores_array)),
        "std_risk": float(np.std(scores_array)),
        "high_risk_count": int(np.sum(scores_array >= 80)),
        "medium_risk_count": int(np.sum((scores_array >= 40) & (scores_array < 80))),
        "low_risk_count": int(np.sum(scores_array < 40)),
    }

    return {
        "date": date,
        "risk_scores": risk_scores,
        "statistics": stats,
        "mesh_count": len(risk_scores),
        "bbox": BBOX,
        "note": "ダミーデータ（特徴量テーブルが見つかりません）",
    }

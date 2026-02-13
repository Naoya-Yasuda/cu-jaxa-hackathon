"""
目撃情報エンドポイント
GET /api/sightings?from=YYYY-MM-DD&to=YYYY-MM-DD
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, HTTPException, Query
from config import SIGHTINGS_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Sightings"])


@router.get("/sightings")
async def get_sightings(
    from_date: str = Query(None, alias="from", description="YYYY-MM-DD開始日"),
    to_date: str = Query(None, alias="to", description="YYYY-MM-DD終了日"),
    prefecture: str = Query(None, description="県名フィルタ（秋田/岩手/福島）"),
    limit: int = Query(1000, ge=1, le=10000, description="取得件数上限"),
):
    """
    指定期間の目撃情報を返す

    Args:
        from_date: YYYY-MM-DD形式の開始日（デフォルト: 今月1日）
        to_date: YYYY-MM-DD形式の終了日（デフォルト: 今日）
        prefecture: 県名フィルタ（秋田/岩手/福島）
        limit: 取得件数上限（デフォルト: 1000）

    Returns:
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
                    "latitude": 39.5,
                    "longitude": 140.3,
                    "prefecture": "秋田県",
                    "location": "鹿角市",
                    "details": "目撃情報..."
                },
                ...
            ]
        }
    """
    try:
        # デフォルト日付を設定
        if from_date is None:
            from_date = datetime.now().strftime("%Y-%m-01")
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")

        # 日付形式の検証
        try:
            datetime.strptime(from_date, "%Y-%m-%d")
            datetime.strptime(to_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="日付形式がYYYY-MM-DDではありません"
            )

        # 目撃データを読み込み
        sightings_path = SIGHTINGS_DIR / "tohoku_sightings.csv"
        if not sightings_path.exists():
            logger.warning("目撃データが見つかりません")
            return {
                "from": from_date,
                "to": to_date,
                "prefecture": prefecture,
                "total_count": 0,
                "returned_count": 0,
                "sightings": [],
                "note": "データが利用できません",
            }

        # CSVを読み込み
        df = pd.read_csv(sightings_path)

        # 日付列を確認・変換
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        elif "日付" in df.columns:
            df["date"] = pd.to_datetime(df["日付"], errors="coerce")
        else:
            logger.warning("日付列が見つかりません")
            df["date"] = pd.to_datetime("2025-01-01")

        # 日付範囲でフィルタリング
        df_filtered = df[
            (df["date"] >= from_date) & (df["date"] <= to_date)
        ].copy()

        # 県名フィルタリング
        if prefecture:
            pref_cols = ["prefecture", "都道府県", "県", "pref"]
            for col in pref_cols:
                if col in df_filtered.columns:
                    df_filtered = df_filtered[
                        df_filtered[col].str.contains(prefecture, na=False)
                    ]
                    break

        total_count = len(df_filtered)

        # 件数制限
        df_output = df_filtered.head(limit)
        returned_count = len(df_output)

        # JSONシリアライズ用の辞書リストに変換
        sightings_list = []
        for idx, row in df_output.iterrows():
            sighting = {
                "id": int(idx),
                "date": str(row["date"].date()) if pd.notna(row.get("date")) else None,
            }

            # 座標情報
            for lat_col in ["latitude", "lat", "緯度"]:
                if lat_col in row and pd.notna(row[lat_col]):
                    try:
                        sighting["latitude"] = float(row[lat_col])
                        break
                    except (ValueError, TypeError):
                        pass
            sighting.setdefault("latitude", None)

            for lon_col in ["longitude", "lon", "経度"]:
                if lon_col in row and pd.notna(row[lon_col]):
                    try:
                        sighting["longitude"] = float(row[lon_col])
                        break
                    except (ValueError, TypeError):
                        pass
            sighting.setdefault("longitude", None)

            # 位置情報
            for pref_col in ["prefecture", "都道府県", "県"]:
                if pref_col in row and pd.notna(row[pref_col]):
                    sighting["prefecture"] = str(row[pref_col])
                    break
            sighting.setdefault("prefecture", "不明")

            for loc_col in ["location", "市町村", "場所"]:
                if loc_col in row and pd.notna(row[loc_col]):
                    sighting["location"] = str(row[loc_col])
                    break
            sighting.setdefault("location", "")

            # 詳細情報
            for detail_col in ["details", "remarks", "備考", "説明"]:
                if detail_col in row and pd.notna(row[detail_col]):
                    sighting["details"] = str(row[detail_col])[:200]  # 最大200文字
                    break
            sighting.setdefault("details", "")

            sightings_list.append(sighting)

        return {
            "from": from_date,
            "to": to_date,
            "prefecture": prefecture,
            "total_count": total_count,
            "returned_count": returned_count,
            "sightings": sightings_list,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"目撃情報取得エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")


@router.get("/sightings/statistics")
async def get_sightings_statistics(
    from_date: str = Query(None, alias="from", description="YYYY-MM-DD開始日"),
    to_date: str = Query(None, alias="to", description="YYYY-MM-DD終了日"),
):
    """
    目撃情報の統計情報を返す

    Returns:
        {
            "from": "2024-01-01",
            "to": "2025-02-13",
            "total_sightings": 456,
            "by_prefecture": {
                "秋田県": 150,
                "岩手県": 200,
                "福島県": 106
            },
            "by_month": {
                "2024-01": 40,
                "2024-02": 35,
                ...
            }
        }
    """
    try:
        # デフォルト日付を設定
        if from_date is None:
            from_date = "2024-01-01"
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")

        # 目撃データを読み込み
        sightings_path = SIGHTINGS_DIR / "tohoku_sightings.csv"
        if not sightings_path.exists():
            return {
                "from": from_date,
                "to": to_date,
                "total_sightings": 0,
                "by_prefecture": {},
                "by_month": {},
            }

        df = pd.read_csv(sightings_path)

        # 日付処理
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["date"] = pd.to_datetime("2025-01-01")

        # 日付範囲でフィルタリング
        df_filtered = df[
            (df["date"] >= from_date) & (df["date"] <= to_date)
        ].copy()

        # 統計計算
        total = len(df_filtered)

        # 県別集計
        by_pref = {}
        for pref_col in ["prefecture", "都道府県", "県"]:
            if pref_col in df_filtered.columns:
                by_pref = df_filtered[pref_col].value_counts().to_dict()
                break

        # 月別集計
        by_month = {}
        if "date" in df_filtered.columns:
            df_filtered["year_month"] = df_filtered["date"].dt.strftime("%Y-%m")
            by_month = df_filtered["year_month"].value_counts().to_dict()
            by_month = dict(sorted(by_month.items()))

        return {
            "from": from_date,
            "to": to_date,
            "total_sightings": total,
            "by_prefecture": {str(k): int(v) for k, v in by_pref.items()},
            "by_month": {str(k): int(v) for k, v in by_month.items()},
        }

    except Exception as e:
        logger.error(f"統計情報取得エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")

"""プロジェクト共通設定"""
from pathlib import Path

# ===== ディレクトリ =====
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

JAXA_DIR = RAW_DIR / "jaxa"
SIGHTINGS_DIR = RAW_DIR / "sightings"
GEO_DIR = RAW_DIR / "geo"
WEATHER_DIR = RAW_DIR / "weather"

# ===== 対象地域（目撃データの範囲 + バッファ） =====
# 目撃データ範囲: lat 35.60~35.90, lon 138.95~139.30
# 周囲に0.05度（約5km）バッファを追加
BBOX = [138.90, 35.55, 139.35, 35.95]  # [west, south, east, north]

# ===== 対象期間 =====
DATE_START = "2023-10-01T00:00:00"
DATE_END = "2025-09-30T00:00:00"

# ===== JAXA Earth API コレクション（確認済みパラメータ） =====
JAXA_COLLECTIONS = {
    "ndvi_monthly": {
        "collection": "JAXA.JASMES_Terra.MODIS-Aqua.MODIS_ndvi.v811_global_monthly",
        "band": "ndvi",
        "ppu": 20,
        "description": "NDVI（植生指数）月次 ~5km",
        "time_series": True,
    },
    "dem": {
        "collection": "JAXA.EORC_ALOS.PRISM_AW3D30.v3.2_global",
        "band": "DSM",
        "ppu": 120,
        "description": "AW3D30 数値表層モデル（標高） ~1km",
        "time_series": False,
        "fixed_date": ["2021-02-01T00:00:00", "2021-02-28T00:00:00"],
    },
    "fnf": {
        "collection": "JAXA.EORC_ALOS-2.PALSAR-2_FNF.v2.1.0_global_yearly",
        "band": "FNF",
        "ppu": 120,
        "description": "森林・非森林マップ ~1km",
        "time_series": False,
        "fixed_date": ["2020-01-01T00:00:00", "2020-12-31T00:00:00"],
    },
    "gsmap_monthly": {
        "collection": "JAXA.EORC_GSMaP_standard.Gauge.00Z-23Z.v6_monthly",
        "band": "PRECIP",
        "ppu": 10,
        "description": "GSMaP 降水量 月次 ~10km",
        "time_series": True,
    },
    "lst_monthly": {
        "collection": "NASA.EOSDIS_Terra.MODIS_MOD11C3-LST.daytime.v061_global_monthly",
        "band": "LST",
        "ppu": 20,
        "description": "地表面温度（昼間）月次 ~5km",
        "time_series": True,
    },
    "landcover": {
        "collection": "Copernicus.C3S_PROBA-V_LCCS_global_yearly",
        "band": "LCCS",
        "ppu": 20,
        "description": "土地被覆分類 (Copernicus) ~5km",
        "time_series": False,
        "fixed_date": ["2019-01-01T00:00:00", "2019-12-31T00:00:00"],
    },
}

# ===== 目撃データ =====
SIGHTINGS_CSV = DATA_DIR / "tukinowaguma_source1120.csv"

"""プロジェクト共通設定（東北3県対応版）"""
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

# ===== 対象地域: 東北3県（秋田・岩手・福島） =====
# まずは秋田県を中心にプロトタイプ。データが揃い次第3県に拡大。
#
# 秋田県: lat 39.0~40.5, lon 139.7~140.6（内陸の山間部中心）
# 岩手県: lat 38.7~40.5, lon 140.6~142.1
# 福島県: lat 36.8~37.9, lon 139.2~140.5
#
# → 3県統合BBOX（山間部にフォーカス、海岸平野部は除外可）
BBOX = [139.2, 36.8, 142.1, 40.5]  # [west, south, east, north]

# 秋田県のみで先行する場合のBBOX
BBOX_AKITA = [139.7, 39.0, 140.6, 40.5]

# ===== 対象期間 =====
# クマダス: 2022年度〜, 福島データ: R4(2022)年度〜
DATE_START = "2022-04-01T00:00:00"
DATE_END = "2025-12-31T00:00:00"

# ===== JAXA Earth API コレクション（確認済みパラメータ） =====
JAXA_COLLECTIONS = {
    "ndvi_monthly": {
        "collection": "JAXA.JASMES_Terra.MODIS-Aqua.MODIS_ndvi.v811_global_monthly",
        "band": "ndvi",
        "ppu": 20,
        "description": "NDVI（植生指数）月次 ~5km → 時間弁別用（季節・植生状態）",
        "time_series": True,
    },
    "dem": {
        "collection": "JAXA.EORC_ALOS.PRISM_AW3D30.v3.2_global",
        "band": "DSM",
        "ppu": 120,
        "description": "AW3D30 数値表層モデル（標高） ~1km → 空間弁別用",
        "time_series": False,
        "fixed_date": ["2021-02-01T00:00:00", "2021-02-28T00:00:00"],
    },
    "fnf": {
        "collection": "JAXA.EORC_ALOS-2.PALSAR-2_FNF.v2.1.0_global_yearly",
        "band": "FNF",
        "ppu": 120,
        "description": "森林・非森林マップ ~1km → 空間弁別用",
        "time_series": False,
        "fixed_date": ["2020-01-01T00:00:00", "2020-12-31T00:00:00"],
    },
    "gsmap_monthly": {
        "collection": "JAXA.EORC_GSMaP_standard.Gauge.00Z-23Z.v6_monthly",
        "band": "PRECIP",
        "ppu": 10,
        "description": "GSMaP 降水量 月次 ~10km → 時間弁別用",
        "time_series": True,
    },
    "lst_monthly": {
        "collection": "NASA.EOSDIS_Terra.MODIS_MOD11C3-LST.daytime.v061_global_monthly",
        "band": "LST",
        "ppu": 20,
        "description": "地表面温度（昼間）月次 ~5km → 時間弁別用",
        "time_series": True,
    },
    "landcover": {
        "collection": "Copernicus.C3S_PROBA-V_LCCS_global_yearly",
        "band": "LCCS",
        "ppu": 20,
        "description": "土地被覆分類 (Copernicus) ~5km → 空間弁別補助",
        "time_series": False,
        "fixed_date": ["2019-01-01T00:00:00", "2019-12-31T00:00:00"],
    },
    "ndvi_normal": {
        "collection": "JAXA.JASMES_Terra.MODIS-Aqua.MODIS_ndvi.v811_global_monthly-normal",
        "band": "ndvi_2012_2021",
        "ppu": 20,
        "description": "NDVI月次平年値(2012-2021) ~5km → 堅果類豊凶の代理指標（偏差算出用）",
        "time_series": False,
        "fixed_date": ["2012-01-01T00:00:00", "2021-12-31T00:00:00"],
    },
}

# ===== 予測メッシュ設定 =====
MESH_SIZE_KM = 1  # 1km × 1km メッシュ（空間弁別はDEM/FNFが担う）

# ===== 目撃データ =====
# 東北3県の統合CSV（秋田クマダス + 岩手 + 福島）
SIGHTINGS_CSV = SIGHTINGS_DIR / "tohoku_sightings.csv"

# 旧・静岡データ（参考用に残す）
SIGHTINGS_CSV_SHIZUOKA = DATA_DIR / "tukinowaguma_source1120.csv"

# ===== ネガティブサンプリング設定 =====
NEG_SAMPLE_RATIO = 15  # 正例1件に対して負例15件を抽出

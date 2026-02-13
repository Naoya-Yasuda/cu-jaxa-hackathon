"""
東北3県のクマ目撃データを統一フォーマットに変換・統合するスクリプト

入力:
  data/raw/sightings/akita_kumadas.csv     (秋田県クマダス)
  data/raw/sightings/fukushima_*.xlsx      (福島県Excel)
  data/raw/sightings/iwate_*.csv           (岩手県、あれば)

出力:
  data/raw/sightings/tohoku_sightings.csv  (統一フォーマット)

Usage:
  python scripts/prepare_sightings.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SIGHTINGS_DIR, SIGHTINGS_CSV, BBOX

# ===== 統一スキーマ =====
UNIFIED_COLUMNS = [
    "sighting_id",  # 一意識別子 (source_連番)
    "date",         # 目撃日 (YYYY-MM-DD)
    "lat",          # 緯度 (WGS84)
    "lon",          # 経度 (WGS84)
    "type",         # 種別: 目撃 / 痕跡 / 被害 / 捕獲
    "source",       # データソース: akita_kumadas / fukushima / iwate
    "prefecture",   # 都道府県: 秋田 / 岩手 / 福島
    "municipality", # 市町村（あれば）
    "description",  # 備考
]


def load_akita_kumadas(filepath: Path) -> pd.DataFrame:
    """秋田県クマダスCSVを読み込み、統一フォーマットに変換"""
    print(f"\n[秋田県クマダス] {filepath}")

    # エンコーディングを試行
    for enc in ["utf-8", "utf-8-sig", "shift_jis", "cp932"]:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        print("  ERROR: CSVの読み込みに失敗しました")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    print(f"  読み込み: {len(df)}行, カラム: {list(df.columns)}")

    # カラム名を確認して対応
    # クマダスのカラム名はデータ公開時期により異なる可能性があるため、
    # 柔軟にマッピングする
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "緯度" in col or "lat" in col_lower:
            col_map["lat"] = col
        elif "経度" in col or "lon" in col_lower or "lng" in col_lower:
            col_map["lon"] = col
        elif "日時" in col or "日付" in col or "年月日" in col or "date" in col_lower:
            col_map["date"] = col
        elif "情報種別" in col or "区分" in col:
            col_map["type"] = col
        elif "獣種" in col or "動物" in col or "animal" in col_lower:
            col_map["animal"] = col
        elif "種別" in col or "type" in col_lower:
            col_map["type"] = col
        elif "市町村" in col or "municipality" in col_lower:
            col_map["municipality"] = col
        elif "状況" in col or "備考" in col or "内容" in col or "description" in col_lower:
            col_map["description"] = col

    print(f"  カラムマッピング: {col_map}")

    # ツキノワグマのみフィルタリング
    if "animal" in col_map:
        before = len(df)
        df = df[df[col_map["animal"]].astype(str).str.contains("クマ|熊|ツキノワ|bear", case=False, na=False)]
        print(f"  ツキノワグマフィルタ: {before} → {len(df)}件")

    # 統一フォーマットに変換
    result = pd.DataFrame()
    result["lat"] = pd.to_numeric(df.get(col_map.get("lat", ""), pd.Series(dtype=float)), errors="coerce")
    result["lon"] = pd.to_numeric(df.get(col_map.get("lon", ""), pd.Series(dtype=float)), errors="coerce")
    if "date" in col_map and col_map["date"] in df.columns:
        result["date"] = pd.to_datetime(df[col_map["date"]], errors="coerce")
    else:
        result["date"] = pd.NaT

    if "type" in col_map and col_map["type"] in df.columns:
        result["type"] = df[col_map["type"]].astype(str)
    else:
        result["type"] = "不明"

    result["source"] = "akita_kumadas"
    result["prefecture"] = "秋田"

    if "municipality" in col_map and col_map["municipality"] in df.columns:
        result["municipality"] = df[col_map["municipality"]].astype(str)
    else:
        result["municipality"] = ""

    if "description" in col_map and col_map["description"] in df.columns:
        result["description"] = df[col_map["description"]].astype(str)
    else:
        result["description"] = ""

    # 緯度経度が欠損の行を除去
    before = len(result)
    result = result.dropna(subset=["lat", "lon", "date"])
    print(f"  欠損除去: {before} → {len(result)}件")

    # BBOX内のみ保持
    before = len(result)
    result = result[
        (result["lon"] >= BBOX[0]) & (result["lon"] <= BBOX[2]) &
        (result["lat"] >= BBOX[1]) & (result["lat"] <= BBOX[3])
    ]
    print(f"  BBOX内: {before} → {len(result)}件")

    return result


def load_fukushima_excel(dirpath: Path) -> pd.DataFrame:
    """福島県Excelファイルを読み込み、統一フォーマットに変換"""
    xlsx_files = list(dirpath.glob("fukushima_*.xlsx")) + list(dirpath.glob("fukushima_*.xls"))
    if not xlsx_files:
        print("\n[福島県] ファイルが見つかりません。スキップします。")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    all_dfs = []
    for f in xlsx_files:
        print(f"\n[福島県] {f}")
        try:
            df = pd.read_excel(f)
            print(f"  読み込み: {len(df)}行, カラム: {list(df.columns)}")

            # 福島のExcelフォーマットは要確認
            # 緯度経度があるか、住所のみかで処理が変わる
            has_latlon = any("緯度" in str(c) or "lat" in str(c).lower() for c in df.columns)
            if has_latlon:
                print("  → 緯度経度カラムあり")
            else:
                print("  → 緯度経度なし。ジオコーディングが必要です。")
                print("    スキップします（手動で緯度経度を追加してください）。")
                continue

            all_dfs.append(df)
        except Exception as e:
            print(f"  ERROR: {e}")

    if not all_dfs:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    # TODO: 福島県固有のカラムマッピング
    print("  福島県データの統合は、カラム構造確認後に実装してください。")
    return pd.DataFrame(columns=UNIFIED_COLUMNS)


def load_iwate(dirpath: Path) -> pd.DataFrame:
    """岩手県データを読み込み（あれば）"""
    csv_files = list(dirpath.glob("iwate_*.csv"))
    if not csv_files:
        print("\n[岩手県] ファイルが見つかりません。スキップします。")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    # TODO: 岩手データのフォーマット確認後に実装
    print(f"\n[岩手県] {len(csv_files)}ファイル発見。フォーマット確認後に実装してください。")
    return pd.DataFrame(columns=UNIFIED_COLUMNS)


def main():
    SIGHTINGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("東北3県 クマ目撃データ 統合スクリプト")
    print(f"  BBOX: {BBOX}")
    print(f"  出力: {SIGHTINGS_CSV}")
    print("=" * 60)

    # 各県のデータを読み込み
    dfs = []

    # 秋田県
    akita_path = SIGHTINGS_DIR / "akita_kumadas.csv"
    if akita_path.exists():
        dfs.append(load_akita_kumadas(akita_path))
    else:
        print(f"\n[秋田県] {akita_path} が見つかりません。")
        print("  → docs/data_fetch_guide.md を参照してダウンロードしてください。")

    # 福島県
    dfs.append(load_fukushima_excel(SIGHTINGS_DIR))

    # 岩手県
    dfs.append(load_iwate(SIGHTINGS_DIR))

    # 統合
    combined = pd.concat([d for d in dfs if len(d) > 0], ignore_index=True)

    if len(combined) == 0:
        print("\n統合データが0件です。まずデータをダウンロードしてください。")
        return

    # sighting_id を付与
    combined["sighting_id"] = [
        f"{row['source']}_{i:05d}"
        for i, (_, row) in enumerate(combined.iterrows(), 1)
    ]

    # 日付でソート
    combined = combined.sort_values("date").reset_index(drop=True)

    # カラム順を揃えて保存
    combined = combined[UNIFIED_COLUMNS]
    combined.to_csv(SIGHTINGS_CSV, index=False, encoding="utf-8-sig")

    # サマリ表示
    print("\n" + "=" * 60)
    print("統合結果サマリ")
    print("=" * 60)
    print(f"  総件数: {len(combined)}")
    print(f"  期間: {combined['date'].min()} ~ {combined['date'].max()}")
    print(f"  緯度: {combined['lat'].min():.4f} ~ {combined['lat'].max():.4f}")
    print(f"  経度: {combined['lon'].min():.4f} ~ {combined['lon'].max():.4f}")
    print(f"\n  県別件数:")
    print(combined["prefecture"].value_counts().to_string())
    print(f"\n  種別件数:")
    print(combined["type"].value_counts().to_string())
    print(f"\n  年別件数:")
    combined["year"] = pd.to_datetime(combined["date"]).dt.year
    print(combined["year"].value_counts().sort_index().to_string())

    print(f"\n  保存先: {SIGHTINGS_CSV}")


if __name__ == "__main__":
    main()

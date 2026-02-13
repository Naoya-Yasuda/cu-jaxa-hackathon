"""
メッシュ分割 & 特徴量テーブル生成スクリプト

1kmメッシュを生成し、各メッシュに以下の特徴量を付与:
  [空間弁別: DEM/FNF/距離]  → どこが危ないか
  [時間弁別: NDVI/LST/GSMaP] → いつ危ないか

正例（目撃あり）+ ネガティブサンプリングした負例で学習テーブルを構成。

Usage:
  python scripts/build_features.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BBOX, JAXA_DIR, SIGHTINGS_CSV, PROCESSED_DIR,
    MESH_SIZE_KM, NEG_SAMPLE_RATIO, DATE_START, DATE_END,
)

# 1度 ≈ 111km（緯度方向）/ 1度 ≈ 91km（経度方向、北緯40度付近）
LAT_PER_KM = 1.0 / 111.0
LON_PER_KM = 1.0 / 91.0  # cos(40°) ≈ 0.766, 111*0.766 ≈ 85-91km


def create_mesh_grid(bbox, mesh_size_km):
    """1kmメッシュグリッドを生成"""
    west, south, east, north = bbox
    lat_step = mesh_size_km * LAT_PER_KM
    lon_step = mesh_size_km * LON_PER_KM

    lats = np.arange(south, north, lat_step)
    lons = np.arange(west, east, lon_step)

    mesh_records = []
    mesh_id = 0
    for lat in lats:
        for lon in lons:
            mesh_records.append({
                "mesh_id": mesh_id,
                "lat_center": lat + lat_step / 2,
                "lon_center": lon + lon_step / 2,
                "lat_south": lat,
                "lat_north": lat + lat_step,
                "lon_west": lon,
                "lon_east": lon + lon_step,
            })
            mesh_id += 1

    df = pd.DataFrame(mesh_records)
    print(f"メッシュ生成: {len(lats)} × {len(lons)} = {len(df)} メッシュ")
    print(f"  緯度方向: {len(lats)}段 (step={lat_step:.4f}°)")
    print(f"  経度方向: {len(lons)}列 (step={lon_step:.4f}°)")
    return df


def load_jaxa_data(name):
    """JAXAデータを読み込む"""
    data_dir = JAXA_DIR / name
    meta_path = data_dir / "metadata.json"

    if not meta_path.exists():
        print(f"  Warning: {meta_path} not found")
        return None, None

    with open(meta_path) as f:
        meta = json.load(f)

    images = []
    for i in range(meta["n_images"]):
        npy_path = data_dir / f"{name}_{i:03d}.npy"
        images.append(np.load(npy_path))

    return images, meta


def latlon_to_pixel(lat, lon, bbox, shape):
    """緯度経度 → ラスタピクセル座標"""
    west, south, east, north = bbox
    rows, cols = shape[0], shape[1]
    x_frac = (lon - west) / (east - west)
    y_frac = (north - lat) / (north - south)
    col = int(x_frac * cols)
    row = int(y_frac * rows)
    if 0 <= row < rows and 0 <= col < cols:
        return row, col
    return None


def get_raster_value(lat, lon, raster, bbox):
    """ラスタ画像から緯度経度に対応する値を取得"""
    if raster is None:
        return np.nan
    px = latlon_to_pixel(lat, lon, bbox, raster.shape)
    if px is None:
        return np.nan
    val = raster[px[0], px[1]]
    if hasattr(val, "__len__"):  # (row, col, band) の場合
        val = val[0]
    return float(val)


def assign_spatial_features(mesh_df, bbox):
    """空間弁別用の静的特徴量を付与（DEM, FNF, Landcover）"""
    print("\n[空間特徴量]")

    # DEM（標高）
    dem_imgs, dem_meta = load_jaxa_data("dem")
    if dem_imgs:
        dem = dem_imgs[0][:, :, 0] if dem_imgs[0].ndim == 3 else dem_imgs[0]
        mesh_df["elevation"] = mesh_df.apply(
            lambda r: get_raster_value(r["lat_center"], r["lon_center"], dem, bbox), axis=1
        )
        # 簡易的な傾斜角（周辺ピクセルとの標高差）
        print(f"  DEM: shape={dem.shape}, assigned to {mesh_df['elevation'].notna().sum()} meshes")
    else:
        mesh_df["elevation"] = np.nan
        print("  DEM: データなし")

    # FNF（森林/非森林）
    fnf_imgs, fnf_meta = load_jaxa_data("fnf")
    if fnf_imgs:
        fnf = fnf_imgs[0][:, :, 0] if fnf_imgs[0].ndim == 3 else fnf_imgs[0]
        mesh_df["fnf"] = mesh_df.apply(
            lambda r: get_raster_value(r["lat_center"], r["lon_center"], fnf, bbox), axis=1
        )
        mesh_df["is_forest"] = (mesh_df["fnf"] == 1).astype(int)
        print(f"  FNF: shape={fnf.shape}, 森林={mesh_df['is_forest'].sum()}")
    else:
        mesh_df["fnf"] = np.nan
        mesh_df["is_forest"] = np.nan
        print("  FNF: データなし")

    # Landcover（土地被覆）
    lc_imgs, lc_meta = load_jaxa_data("landcover")
    if lc_imgs:
        lc = lc_imgs[0][:, :, 0] if lc_imgs[0].ndim == 3 else lc_imgs[0]
        mesh_df["landcover"] = mesh_df.apply(
            lambda r: get_raster_value(r["lat_center"], r["lon_center"], lc, bbox), axis=1
        )
        print(f"  Landcover: shape={lc.shape}")
    else:
        mesh_df["landcover"] = np.nan
        print("  Landcover: データなし")

    return mesh_df


def get_monthly_periods(start, end):
    """開始〜終了の月リスト"""
    return pd.period_range(start=start[:7], end=end[:7], freq="M")


def build_training_data(mesh_df, sightings_df, bbox):
    """
    正例 + ネガティブサンプリングした負例で学習テーブルを生成

    正例: 目撃があったメッシュ×月
    負例: 同じ月の目撃メッシュ以外からランダム抽出（正例の NEG_SAMPLE_RATIO 倍）
    """
    print("\n[学習データ生成]")

    # 月の一覧
    months = get_monthly_periods(DATE_START, DATE_END)
    print(f"  対象月数: {len(months)} ({months[0]} ~ {months[-1]})")

    # 時系列衛星データ読み込み
    ndvi_imgs, ndvi_meta = load_jaxa_data("ndvi_monthly")
    gsmap_imgs, gsmap_meta = load_jaxa_data("gsmap_monthly")
    lst_imgs, lst_meta = load_jaxa_data("lst_monthly")

    # 目撃データに月カラム追加
    sightings_df = sightings_df.copy()
    sightings_df["date"] = pd.to_datetime(sightings_df["date"])
    sightings_df["year_month"] = sightings_df["date"].dt.to_period("M")

    # 各目撃を最寄りメッシュに割り当て
    def find_mesh(lat, lon, mesh_df):
        dists = ((mesh_df["lat_center"] - lat) ** 2 + (mesh_df["lon_center"] - lon) ** 2)
        return mesh_df.iloc[dists.idxmin()]["mesh_id"]

    sightings_df["mesh_id"] = sightings_df.apply(
        lambda r: find_mesh(r["lat"], r["lon"], mesh_df), axis=1
    )

    # 目撃エリアの範囲を計算（バッファ付き）→ ネガティブサンプリングを近傍に制限
    BUFFER_DEG = 0.3  # 約30km のバッファ
    sighting_lat_min = sightings_df["lat"].min() - BUFFER_DEG
    sighting_lat_max = sightings_df["lat"].max() + BUFFER_DEG
    sighting_lon_min = sightings_df["lon"].min() - BUFFER_DEG
    sighting_lon_max = sightings_df["lon"].max() + BUFFER_DEG

    # 目撃エリア内のメッシュのみを負例候補にする
    sighting_area_mask = (
        (mesh_df["lat_center"] >= sighting_lat_min) &
        (mesh_df["lat_center"] <= sighting_lat_max) &
        (mesh_df["lon_center"] >= sighting_lon_min) &
        (mesh_df["lon_center"] <= sighting_lon_max)
    )
    sighting_area_mesh_ids = set(mesh_df[sighting_area_mask]["mesh_id"].values)
    print(f"  目撃エリア（バッファ{BUFFER_DEG}°）内メッシュ: {len(sighting_area_mesh_ids)} / {len(mesh_df)}")
    print(f"    緯度: {sighting_lat_min:.2f} ~ {sighting_lat_max:.2f}")
    print(f"    経度: {sighting_lon_min:.2f} ~ {sighting_lon_max:.2f}")

    records = []
    np.random.seed(42)

    for month_idx, month in enumerate(months):
        # この月の目撃メッシュ
        month_sightings = sightings_df[sightings_df["year_month"] == month]
        positive_meshes = set(month_sightings["mesh_id"].unique())

        # 正例を追加
        for mesh_id in positive_meshes:
            records.append({
                "mesh_id": int(mesh_id),
                "year_month": str(month),
                "month": month.month,
                "target": 1,
                "n_sightings": int((month_sightings["mesh_id"] == mesh_id).sum()),
            })

        # 負例をサンプリング（目撃エリア近傍のみ）
        negative_candidates = np.array([
            mid for mid in sighting_area_mesh_ids if mid not in positive_meshes
        ])
        n_neg = min(len(positive_meshes) * NEG_SAMPLE_RATIO, len(negative_candidates))
        if n_neg > 0:
            neg_sampled = np.random.choice(negative_candidates, size=n_neg, replace=False)
            for mesh_id in neg_sampled:
                records.append({
                    "mesh_id": int(mesh_id),
                    "year_month": str(month),
                    "month": month.month,
                    "target": 0,
                    "n_sightings": 0,
                })

    train_df = pd.DataFrame(records)
    print(f"  学習データ: {len(train_df)}件 (正例={train_df['target'].sum()}, 負例={len(train_df) - train_df['target'].sum()})")
    print(f"  正例率: {train_df['target'].mean():.1%}")

    # メッシュの空間特徴量をjoin
    train_df = train_df.merge(mesh_df, on="mesh_id", how="left")

    # 時系列特徴量を付与
    if ndvi_imgs:
        ndvi_data = {str(m): ndvi_imgs[i] for i, m in enumerate(months) if i < len(ndvi_imgs)}
        train_df["ndvi"] = train_df.apply(
            lambda r: get_raster_value(
                r["lat_center"], r["lon_center"],
                ndvi_data.get(r["year_month"], np.array([[[np.nan]]]))[..., 0]
                if ndvi_data.get(r["year_month"]) is not None and ndvi_data.get(r["year_month"]).ndim == 3
                else ndvi_data.get(r["year_month"]),
                bbox
            ), axis=1
        )
        print(f"  NDVI付与: {train_df['ndvi'].notna().sum()}件")

        # NDVI前月差分（ndvi_diff）: 植生変化の急変を捉える
        # 同メッシュの前月NDVIを取得して差分を計算
        def get_prev_ndvi(row):
            ym = row["year_month"]
            try:
                current_period = pd.Period(ym, freq="M")
                prev_period = current_period - 1
                prev_key = str(prev_period)
            except Exception:
                return np.nan
            raster = ndvi_data.get(prev_key)
            if raster is None:
                return np.nan
            if raster.ndim == 3:
                raster = raster[..., 0]
            return get_raster_value(row["lat_center"], row["lon_center"], raster, bbox)

        train_df["ndvi_prev"] = train_df.apply(get_prev_ndvi, axis=1)
        train_df["ndvi_diff"] = train_df["ndvi"] - train_df["ndvi_prev"]
        train_df.drop(columns=["ndvi_prev"], inplace=True)
        print(f"  NDVI差分付与: {train_df['ndvi_diff'].notna().sum()}件")

        # NDVI偏差（ndvi_anomaly）: 月別平年値との差分 → 堅果類豊凶の代理指標
        # 2022-2025データから自前の月別クリマトロジーを計算し、偏差を算出
        from collections import defaultdict
        month_images = defaultdict(list)
        for period_key, raster in ndvi_data.items():
            m = int(period_key.split("-")[1])
            r = raster[..., 0] if raster.ndim == 3 else raster
            month_images[m].append(r)

        ndvi_normals = {}
        for m in sorted(month_images.keys()):
            stacked = np.stack(month_images[m], axis=0)
            ndvi_normals[m] = np.nanmean(stacked, axis=0)

        # JAXA公式の平年値データが data/raw/jaxa/ndvi_normal/ にあればそちらを優先
        normal_dir = JAXA_DIR / "ndvi_normal"
        if (normal_dir / "metadata.json").exists():
            print("  NDVI平年値: JAXA公式データを使用")
            with open(normal_dir / "metadata.json") as f_meta:
                normal_meta = json.load(f_meta)
            for i in range(normal_meta["n_images"]):
                npy_path = normal_dir / f"ndvi_normal_{i:03d}.npy"
                if npy_path.exists():
                    nr = np.load(npy_path)
                    if nr.ndim == 3:
                        nr = nr[..., 0]
                    ndvi_normals[i + 1] = nr  # 1-indexed month
        else:
            print("  NDVI平年値: 自前クリマトロジー（2022-2025平均）を使用")

        def get_ndvi_anomaly(row):
            ym = row["year_month"]
            try:
                m = int(ym.split("-")[1])
            except Exception:
                return np.nan
            normal_raster = ndvi_normals.get(m)
            if normal_raster is None:
                return np.nan
            normal_val = get_raster_value(
                row["lat_center"], row["lon_center"], normal_raster, bbox
            )
            ndvi_val = row.get("ndvi", np.nan)
            if np.isnan(ndvi_val) or np.isnan(normal_val):
                return np.nan
            return ndvi_val - normal_val

        train_df["ndvi_anomaly"] = train_df.apply(get_ndvi_anomaly, axis=1)
        print(f"  NDVI偏差付与: {train_df['ndvi_anomaly'].notna().sum()}件")

    if lst_imgs:
        lst_data = {str(m): lst_imgs[i] for i, m in enumerate(months) if i < len(lst_imgs)}
        train_df["lst"] = train_df.apply(
            lambda r: get_raster_value(
                r["lat_center"], r["lon_center"],
                lst_data.get(r["year_month"], np.array([[[np.nan]]]))[..., 0]
                if lst_data.get(r["year_month"]) is not None and lst_data.get(r["year_month"]).ndim == 3
                else lst_data.get(r["year_month"]),
                bbox
            ), axis=1
        )
        # ケルビン→摂氏
        train_df["lst_celsius"] = train_df["lst"] - 273.15
        print(f"  LST付与: {train_df['lst'].notna().sum()}件")

    if gsmap_imgs:
        gsmap_data = {str(m): gsmap_imgs[i] for i, m in enumerate(months) if i < len(gsmap_imgs)}
        train_df["precip"] = train_df.apply(
            lambda r: get_raster_value(
                r["lat_center"], r["lon_center"],
                gsmap_data.get(r["year_month"], np.array([[[np.nan]]]))[..., 0]
                if gsmap_data.get(r["year_month"]) is not None and gsmap_data.get(r["year_month"]).ndim == 3
                else gsmap_data.get(r["year_month"]),
                bbox
            ), axis=1
        )
        print(f"  GSMaP付与: {train_df['precip'].notna().sum()}件")

    return train_df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("特徴量テーブル生成")
    print(f"  BBOX: {BBOX}")
    print(f"  メッシュサイズ: {MESH_SIZE_KM}km")
    print(f"  ネガティブサンプリング比: 1:{NEG_SAMPLE_RATIO}")
    print("=" * 60)

    # 1. メッシュグリッド生成
    mesh_df = create_mesh_grid(BBOX, MESH_SIZE_KM)

    # 2. 空間特徴量付与
    mesh_df = assign_spatial_features(mesh_df, BBOX)

    # メッシュデータ保存
    mesh_path = PROCESSED_DIR / "mesh_grid.parquet"
    mesh_df.to_parquet(mesh_path, index=False)
    print(f"\nメッシュグリッド保存: {mesh_path}")

    # 3. 目撃データ読み込み
    if not SIGHTINGS_CSV.exists():
        print(f"\n目撃データが見つかりません: {SIGHTINGS_CSV}")
        print("先に python scripts/prepare_sightings.py を実行してください。")
        return

    sightings_df = pd.read_csv(SIGHTINGS_CSV)
    print(f"\n目撃データ: {len(sightings_df)}件")

    # 4. 学習データ生成（正例 + ネガティブサンプリング負例）
    train_df = build_training_data(mesh_df, sightings_df, BBOX)

    # 5. 特徴量カラムを整理
    feature_cols = [
        "mesh_id", "year_month", "month", "target", "n_sightings",
        "lat_center", "lon_center",
        "elevation", "is_forest", "landcover",
    ]
    # 時系列特徴量があれば追加
    for col in ["ndvi", "ndvi_diff", "ndvi_anomaly", "lst_celsius", "precip"]:
        if col in train_df.columns:
            feature_cols.append(col)

    train_df = train_df[[c for c in feature_cols if c in train_df.columns]]

    # 保存
    output_path = PROCESSED_DIR / "features.parquet"
    train_df.to_parquet(output_path, index=False)

    print("\n" + "=" * 60)
    print("特徴量テーブル生成完了")
    print("=" * 60)
    print(f"  出力: {output_path}")
    print(f"  レコード数: {len(train_df)}")
    print(f"  カラム: {list(train_df.columns)}")
    print(f"  正例: {train_df['target'].sum()}, 負例: {len(train_df) - train_df['target'].sum()}")
    print(f"  正例率: {train_df['target'].mean():.1%}")
    print(f"\n  特徴量統計:")
    print(train_df.describe().to_string())


if __name__ == "__main__":
    main()

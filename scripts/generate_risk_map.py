"""
リスクマップ生成スクリプト v3 — ハイブリッドアプローチ

設計思想:
  Baselineモデル(AUC 0.954) → 「どこが危ないか」（空間リスク）
  実績データの季節パターン → 「いつ危ないか」（季節係数）
  統合リスク = 空間リスク × 季節係数

  これにより、7月は山間部、10月は里山～平野部の谷筋、といった
  季節による出没パターンの違いが明確にマップに反映される。

Usage:
  python scripts/generate_risk_map.py --month 2025-07
  python scripts/generate_risk_map.py --month 2025-10
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BBOX, BBOX_AKITA, PROCESSED_DIR, SIGHTINGS_CSV

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm が必要です。pip install lightgbm")
    sys.exit(1)

try:
    import folium
    from folium.plugins import HeatMap
except ImportError:
    print("ERROR: folium が必要です。pip install folium")
    sys.exit(1)


MODEL_DIR = Path(__file__).parent.parent / "models" / "outputs"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# 1kmメッシュの解像度
LAT_STEP = 1.0 / 111.0
LON_STEP = 1.0 / 91.0


# ===== Step 1: 空間リスク（Baselineモデル） =====

def predict_spatial_risk(model, mesh_df, month_num):
    """Baselineモデルで空間リスクを予測（位置+標高+森林+月）"""
    predict_df = mesh_df.copy()
    predict_df["month"] = month_num

    model_features = model.feature_name()
    for f in model_features:
        if f not in predict_df.columns:
            predict_df[f] = np.nan

    X = predict_df[model_features]
    scores = model.predict(X)
    predict_df["spatial_risk"] = np.clip(scores * 100, 0, 100)

    return predict_df[["mesh_id", "lat_center", "lon_center", "spatial_risk"]]


# ===== Step 2: 季節係数（実測データ） =====

def compute_seasonal_factors(sightings_df, mesh_df, sigma_km=5.0):
    """
    目撃データから季節係数を計算（メッシュ単位、空間平滑化付き）

    各メッシュの「その月にどれだけ目撃が集中するか」を算出。
    年間平均を1.0として、10月は3.0（3倍）、2月は0.1（1/10）のように変動。

    空間平滑化: 個別メッシュでは目撃数が少ないため、
    周辺sigma_km内のメッシュの目撃パターンを加重平均する。
    """
    print("\n[季節係数の計算]")

    sdf = sightings_df.copy()
    sdf["date"] = pd.to_datetime(sdf["date"])
    sdf["month"] = sdf["date"].dt.month

    # 各目撃を最寄りメッシュに割り当て
    sdf["mesh_lat_idx"] = ((sdf["lat"] - BBOX[1]) / LAT_STEP).astype(int)
    sdf["mesh_lon_idx"] = ((sdf["lon"] - BBOX[0]) / LON_STEP).astype(int)
    sdf["mesh_key"] = sdf["mesh_lat_idx"].astype(str) + "_" + sdf["mesh_lon_idx"].astype(str)

    # メッシュ×月のカウント
    mesh_month_counts = sdf.groupby(["mesh_key", "month"]).size().reset_index(name="count")

    # 月別の全体件数比率（グローバル季節パターン）
    total_by_month = sdf.groupby("month").size()
    global_monthly_ratio = total_by_month / total_by_month.mean()

    print(f"  グローバル季節パターン:")
    for m in range(1, 13):
        r = global_monthly_ratio.get(m, 0)
        bar = "█" * int(r * 10)
        print(f"    {m:2d}月: ×{r:.2f} {bar}")

    # 秋田県内メッシュにフィルタ
    akita_mesh = mesh_df[
        (mesh_df["lat_center"] >= BBOX_AKITA[1]) &
        (mesh_df["lat_center"] <= BBOX_AKITA[3]) &
        (mesh_df["lon_center"] >= BBOX_AKITA[0]) &
        (mesh_df["lon_center"] <= BBOX_AKITA[2])
    ].copy()
    akita_mesh["mesh_lat_idx"] = ((akita_mesh["lat_center"] - BBOX[1]) / LAT_STEP).astype(int)
    akita_mesh["mesh_lon_idx"] = ((akita_mesh["lon_center"] - BBOX[0]) / LON_STEP).astype(int)
    akita_mesh["mesh_key"] = akita_mesh["mesh_lat_idx"].astype(str) + "_" + akita_mesh["mesh_lon_idx"].astype(str)

    # 目撃のあるメッシュの緯度経度
    sighting_meshes = sdf.groupby("mesh_key").agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    ).reset_index()

    # 各メッシュ×月のローカル季節係数を計算（周辺の目撃パターンで平滑化）
    sigma_deg = sigma_km / 111.0  # km → 度に変換

    # 全月の季節係数テーブル
    seasonal_factors = {}

    for month_num in range(1, 13):
        month_data = mesh_month_counts[mesh_month_counts["month"] == month_num]
        month_count_dict = dict(zip(month_data["mesh_key"], month_data["count"]))

        # 全メッシュの全月合計
        total_count_dict = mesh_month_counts.groupby("mesh_key")["count"].sum().to_dict()

        factors = []
        for _, row in akita_mesh.iterrows():
            lat_c = row["lat_center"]
            lon_c = row["lon_center"]

            # 周辺メッシュの目撃データを加重集計
            nearby_mask = (
                (sighting_meshes["lat"] >= lat_c - 3 * sigma_deg) &
                (sighting_meshes["lat"] <= lat_c + 3 * sigma_deg) &
                (sighting_meshes["lon"] >= lon_c - 3 * sigma_deg) &
                (sighting_meshes["lon"] <= lon_c + 3 * sigma_deg)
            )
            nearby = sighting_meshes[nearby_mask]

            if len(nearby) == 0:
                # 周辺に目撃データなし → グローバル係数を使用
                factors.append(global_monthly_ratio.get(month_num, 1.0))
                continue

            # ガウシアン重み
            dists_sq = ((nearby["lat"] - lat_c) ** 2 + (nearby["lon"] - lon_c) ** 2)
            weights = np.exp(-dists_sq / (2 * sigma_deg ** 2))

            # 加重した当月件数 / 加重した全月平均件数
            weighted_month_count = sum(
                month_count_dict.get(mk, 0) * w
                for mk, w in zip(nearby["mesh_key"], weights)
            )
            weighted_total_count = sum(
                total_count_dict.get(mk, 0) * w
                for mk, w in zip(nearby["mesh_key"], weights)
            )

            if weighted_total_count > 0:
                # ローカル月比率
                local_ratio = (weighted_month_count * 12) / weighted_total_count
                # グローバルとブレンド（データが少ない場所はグローバル寄り）
                data_weight = min(weighted_total_count / 50, 1.0)
                blended = data_weight * local_ratio + (1 - data_weight) * global_monthly_ratio.get(month_num, 1.0)
                factors.append(blended)
            else:
                factors.append(global_monthly_ratio.get(month_num, 1.0))

        seasonal_factors[month_num] = factors

    return akita_mesh, seasonal_factors, global_monthly_ratio


# ===== Step 3: 統合リスク =====

def compute_combined_risk(spatial_risk_df, akita_mesh, seasonal_factors, month_num):
    """空間リスク × 季節係数 = 統合リスク"""
    # akita_meshのmesh_idで空間リスクをjoin
    combined = akita_mesh[["mesh_id", "lat_center", "lon_center"]].copy()
    combined = combined.merge(
        spatial_risk_df[["mesh_id", "spatial_risk"]],
        on="mesh_id", how="left"
    )

    # 季節係数を付与
    combined["seasonal_factor"] = seasonal_factors[month_num]

    # 統合リスク = 空間リスク × 季節係数（上限100）
    combined["risk_score"] = np.clip(
        combined["spatial_risk"] * combined["seasonal_factor"], 0, 100
    )

    print(f"\n  統合リスク統計 ({month_num}月):")
    print(f"    空間リスク:   平均={combined['spatial_risk'].mean():.1f}%, 最大={combined['spatial_risk'].max():.1f}%")
    print(f"    季節係数:     平均={combined['seasonal_factor'].mean():.2f}, "
          f"最大={combined['seasonal_factor'].max():.2f}")
    print(f"    統合リスク:   平均={combined['risk_score'].mean():.1f}%, 最大={combined['risk_score'].max():.1f}%")
    print(f"    高リスク(>30%): {(combined['risk_score'] > 30).sum()}メッシュ")

    return combined


# ===== Step 4: Foliumマップ =====

def generate_folium_map(combined_df, sightings_df, target_month, seasonal_ratio, output_path=None):
    """Foliumリスクマップ生成（季節反映版）"""
    month_num = int(target_month.split("-")[1])

    center_lat = (BBOX_AKITA[1] + BBOX_AKITA[3]) / 2
    center_lon = (BBOX_AKITA[0] + BBOX_AKITA[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="OpenStreetMap",
    )

    # --- Layer 1: ヒートマップ ---
    threshold = 3
    visible = combined_df[combined_df["risk_score"] >= threshold]
    print(f"  ヒートマップ表示: {len(visible)} / {len(combined_df)} メッシュ")

    if len(visible) > 0:
        heat_data = visible[["lat_center", "lon_center", "risk_score"]].values.tolist()
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_zoom=13,
            radius=18,
            blur=25,
            gradient={
                "0.2": "#1a9850",
                "0.4": "#91cf60",
                "0.6": "#fee08b",
                "0.8": "#fc8d59",
                "1.0": "#d73027",
            },
            name="リスクヒートマップ",
        ).add_to(m)

    # --- Layer 2: 高リスクTOP10（デフォルト表示、ランキング付き） ---
    if len(combined_df) > 0:
        top_layer = folium.FeatureGroup(name="高リスク地点 TOP10", show=True)
        top_risk = combined_df.nlargest(10, "risk_score")

        for rank, (_, row) in enumerate(top_risk.iterrows(), 1):
            score = row["risk_score"]
            sf = row["seasonal_factor"]

            # ランキング番号付きのDivIcon
            folium.Marker(
                location=[row["lat_center"], row["lon_center"]],
                icon=folium.DivIcon(
                    html=f'<div style="background:{"#d73027" if score>=50 else "#fc8d59" if score>=30 else "#fee08b"};'
                         f'color:{"white" if score>=30 else "black"};'
                         f'border-radius:50%;width:28px;height:28px;text-align:center;'
                         f'line-height:28px;font-weight:bold;font-size:14px;'
                         f'border:2px solid white;box-shadow:1px 1px 3px rgba(0,0,0,0.5);">'
                         f'{rank}</div>',
                    icon_size=(28, 28),
                    icon_anchor=(14, 14),
                ),
                popup=folium.Popup(
                    f"<b>#{rank} リスク: {score:.1f}%</b><br>"
                    f"空間リスク: {row['spatial_risk']:.1f}%<br>"
                    f"季節係数: ×{sf:.2f}<br>"
                    f"({row['lat_center']:.3f}, {row['lon_center']:.3f})",
                    max_width=220,
                ),
            ).add_to(top_layer)

        top_layer.add_to(m)

    # --- Layer 3: 実際の目撃地点（当月分） ---
    if sightings_df is not None and len(sightings_df) > 0:
        sdf = sightings_df.copy()
        sdf["date"] = pd.to_datetime(sdf["date"])
        sdf["month"] = sdf["date"].dt.month

        # 同月の過去全年分（検証用）
        month_sightings = sdf[sdf["month"] == month_num]

        sighting_layer = folium.FeatureGroup(
            name=f"過去の{month_num}月の目撃地点 ({len(month_sightings)}件)",
            show=False,
        )

        display = month_sightings.sample(min(300, len(month_sightings)), random_state=42)
        for _, row in display.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=3,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.6,
                weight=1,
            ).add_to(sighting_layer)

        sighting_layer.add_to(m)
        print(f"  目撃地点: {len(month_sightings)}件 ({month_num}月全年分), 表示: {len(display)}件")

    # レイヤーコントロール
    folium.LayerControl(collapsed=False).add_to(m)

    # 月別の季節パターンバー（凡例内）
    month_bars = ""
    for mo in range(1, 13):
        r = seasonal_ratio.get(mo, 1.0)
        bar_h = max(2, min(40, int(r * 15)))
        color = "#d73027" if r > 2 else "#fc8d59" if r > 1.3 else "#91cf60" if r > 0.5 else "#1a9850"
        highlight = "border:2px solid #333;" if mo == month_num else ""
        month_bars += (
            f'<div style="display:inline-block;width:16px;margin:0 1px;text-align:center;">'
            f'<div style="background:{color};height:{bar_h}px;width:14px;{highlight}"></div>'
            f'<div style="font-size:9px;">{mo}</div></div>'
        )

    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 12px; border: 2px solid gray;
                border-radius: 5px; font-size: 13px; line-height: 1.6;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.2); max-width: 280px;">
        <b>クマ出没リスク予測</b> <span style="color:#d73027;">{target_month}</span><br>
        <small>秋田県 | JAXA衛星データ + LightGBM</small><br>
        <hr style="margin:4px 0;">
        <span style="background:#d73027;padding:2px 8px;color:white;">■</span> 高リスク<br>
        <span style="background:#fc8d59;padding:2px 8px;">■</span> 中高リスク<br>
        <span style="background:#fee08b;padding:2px 8px;">■</span> 中リスク<br>
        <span style="background:#1a9850;padding:2px 8px;color:white;">■</span> 低リスク<br>
        <hr style="margin:4px 0;">
        <b>月別出没パターン:</b><br>
        <div style="display:flex;align-items:flex-end;height:50px;padding-top:5px;">
            {month_bars}
        </div>
        <hr style="margin:4px 0;">
        <small>空間リスク(AW3D30+PALSAR-2) × 季節係数</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # タイトル
    sr = seasonal_ratio.get(month_num, 1.0)
    level = "危険" if sr > 2.5 else "警戒" if sr > 1.5 else "注意" if sr > 0.8 else "低め"
    level_color = "#d73027" if sr > 2.5 else "#fc8d59" if sr > 1.5 else "#fee08b" if sr > 0.8 else "#91cf60"

    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background-color: white; padding: 8px 16px;
                border: 2px solid #333; border-radius: 5px; font-size: 16px;
                font-weight: bold; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        秋田県 クマ出没リスクマップ
        <span style="color:{level_color}; margin-left:8px;">{target_month} [{level}]</span>
        <br><small style="font-weight:normal; font-size:11px;">
        JAXA: AW3D30標高 + PALSAR-2森林 + MODIS NDVI/LST/GSMaP | 季節係数: ×{sr:.1f}</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # 保存
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"risk_map_{target_month.replace('-', '')}.html"

    m.save(str(output_path))
    file_size_kb = output_path.stat().st_size / 1024
    print(f"\n  リスクマップ保存: {output_path} ({file_size_kb:.0f}KB)")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="クマ出没リスクマップ生成 v3")
    parser.add_argument("--month", type=str, default=None,
                        help="予測対象月 (YYYY-MM)")
    parser.add_argument("--output", type=str, default=None,
                        help="出力HTMLパス")
    args = parser.parse_args()

    if args.month is None:
        now = datetime.now()
        target_month = f"{now.year}-{now.month:02d}" if 4 <= now.month <= 11 else f"{now.year}-07"
        print(f"対象月未指定のため {target_month} を使用")
    else:
        target_month = args.month

    month_num = int(target_month.split("-")[1])

    print("=" * 60)
    print("クマ出没リスクマップ v3（空間リスク × 季節係数）")
    print(f"  対象月: {target_month}")
    print("=" * 60)

    # Baselineモデル読み込み（AUC 0.954）
    model_path = MODEL_DIR / "model_baseline.lgb"
    if not model_path.exists():
        # フォールバック: fullモデル
        model_path = MODEL_DIR / "model_full.lgb"
    model = lgb.Booster(model_file=str(model_path))
    print(f"モデル: {model_path.name} (特徴量: {model.feature_name()})")

    # メッシュグリッド
    mesh_df = pd.read_parquet(PROCESSED_DIR / "mesh_grid.parquet")
    print(f"メッシュ: {len(mesh_df)}個")

    # 目撃データ
    sightings_df = pd.read_csv(SIGHTINGS_CSV)
    print(f"目撃データ: {len(sightings_df)}件")

    # Step 1: 空間リスク予測
    print("\n[Step 1: 空間リスク予測（Baselineモデル）]")
    spatial_risk = predict_spatial_risk(model, mesh_df, month_num)

    # Step 2: 季節係数計算
    print("\n[Step 2: 季節係数計算（実測データ）]")
    akita_mesh, seasonal_factors, global_ratio = compute_seasonal_factors(
        sightings_df, mesh_df, sigma_km=5.0
    )

    # Step 3: 統合リスク
    print("\n[Step 3: 統合リスク計算]")
    combined = compute_combined_risk(spatial_risk, akita_mesh, seasonal_factors, month_num)

    # Step 4: マップ生成
    print("\n[Step 4: マップ生成]")
    output_path = generate_folium_map(
        combined, sightings_df, target_month, global_ratio,
        output_path=args.output,
    )

    print("\n完了！")


if __name__ == "__main__":
    main()

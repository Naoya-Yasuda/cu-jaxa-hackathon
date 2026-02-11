# 相関のありそうなパラメータ候補

- 市街地からの距離
- 川との距離
- どんぐりの豊作・凶作（植生の変化率）
- ソーラーパネル（or 人工改変地なのか自然地なのか）
- 標高

---

## JAXA Earth API 確定取得パラメータ

| # | データ | コレクションID | band | ppu | 結果ピクセル | 解像度 | 期間 | 対応する仮説 |
|---|---|---|---|---|---|---|---|---|
| 1 | **NDVI（植生指数）** | `JAXA.JASMES_Terra.MODIS-Aqua.MODIS_ndvi.v811_global_monthly` | `ndvi` | 20 | 8×9 | ~5km | 2023-10〜2025-09 (24ヶ月) | どんぐり豊凶・植生変化率 |
| 2 | **DEM（標高）** | `JAXA.EORC_ALOS.PRISM_AW3D30.v3.2_global` | `DSM` | 120 | 35×40 | ~1km | 2021-02 (固定) | 標高 |
| 3 | **FNF（森林・非森林）** | `JAXA.EORC_ALOS-2.PALSAR-2_FNF.v2.1.0_global_yearly` | `FNF` | 120 | 35×40 | ~1km | 2020 (固定) | 市街地距離 |
| 4 | **GSMaP（降水量）** | `JAXA.EORC_GSMaP_standard.Gauge.00Z-23Z.v6_monthly` | `PRECIP` | 10 | 3×4 | ~10km | 2023-10〜2025-09 | 川・降水量 |
| 5 | **LST（地表面温度）** | `NASA.EOSDIS_Terra.MODIS_MOD11C3-LST.daytime.v061_global_monthly` | `LST` | 20 | 8×9 | ~5km | 2023-10〜2025-09 | 気温（活動パターン） |
| 6 | **Landcover（土地被覆）** | `Copernicus.C3S_PROBA-V_LCCS_global_yearly` | `LCCS` | 20 | 8×9 | ~5km | 2019 (固定) | 人工改変地 vs 自然地 |
| 7 | **高解像度土地被覆図** | JAXA HRLULC（Earth API外・別途取得） | - | - | 10m | 10m | 2021〜 | ソーラーパネル検出 |

### 共通設定
- **BBOX**: `[138.90, 35.55, 139.35, 35.95]` （目撃データ範囲 + 約5kmバッファ）
- **日付フォーマット**: ISO 8601 (`YYYY-MM-DDT00:00:00`)
- **ライブラリバグ修正**: `jaxa-earth` の `dn2p.py` L40,53 で `!= None` → `is not None` に要修正

### ソーラーパネルについて
JAXA Earth API（90+データセット）にはソーラーパネル直接検出データはないが、
**JAXA高解像度土地利用土地被覆図（HRLULC）** v21.11以降で「ソーラーパネル」が分類カテゴリに追加済み（10m解像度、正答率100%）。
Earth API外のデータのため別途取得が必要（JAXA EORC または G空間情報センター）。

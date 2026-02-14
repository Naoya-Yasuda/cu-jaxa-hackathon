# セルフレビュー 注意点まとめ

## 修正済みバグ (2025-02-14)

### Bug1: TOP10ランキングが正規化後スコアで選択されていた
- **症状**: TOP10マーカーがヒートマップの赤い領域と一致しない
- **原因**: 99.5%ile正規化で69メッシュが100.0%に同率化 → `nlargest(10)`がインデックス順（南側優先）で恣意的に選択
- **修正**: `raw_risk`（正規化前の生スコア）でランキングするよう変更

### Bug2: タイトルに堅果類豊凶倍率が未表示
- **症状**: タイトルに「季節係数: ×1.1」のみ表示、堅果類の×1.50が不可視
- **修正**: タイトル・ポップアップに堅果類豊凶情報を追加

## 今後の注意点

### ① lst_celsius（地表面温度）NaN率 62.3%
- MODIS LST月次データの欠損率が非常に高い
- LightGBMはNaN処理可能だが、Fullモデルの特徴量としての信頼性が低い
- **対策候補**: LST欠損月を前後月の補間で埋める、またはLSTを除外してモデル再学習

### ② Fullモデル(AUC 0.9426) < Baseline(AUC 0.9543)
- 衛星データ（NDVI/LST/GSMaP）が5km解像度のため、1kmメッシュ予測にノイズ
- 同じ5km圏内の複数メッシュが同一NDVI値を持ち、細粒度差異を表現不可
- **対策**: GCOM-C APIも5km → APIでの解像度改善は不可。G-Portal 250mデータ手動DLか、メッシュサイズを5kmに拡大して衛星データを活かすアプローチを検討

### ③ 堅果類豊凶データの粒度
- 現在は秋田県全体で年次一律スコア
- 実際はブナ林の標高帯・地域ごとに豊凶が異なる
- **対策候補**: 東北森林管理局の山林署別データに細分化、NDVIと堅果類スコアの交差項追加

### ④ 季節係数の計算コスト
- `compute_seasonal_factors()` = O(メッシュ数 × 目撃メッシュ数) per month
- 13,694メッシュで実用的（数十秒）だが3県拡大時は遅い
- **対策候補**: scipy.spatial.KDTreeで空間検索を高速化

### ⑤ GCOM-C NDVI: APIでは250m取得不可（重要）
- **調査結果 (2025-02-14)**: JAXA Earth APIのGCOM-C NDVIは L3 1/24deg ≈ **約5km**（MODIS同等）
- 250mネイティブデータはG-Portal直接ダウンロードのみ（API非対応）
- API上の確認済みコレクション: `JAXA.JASMES_GCOM-C.SGLI_standard.L2-NDVI.daytime.v3_japan_8-day`（8日間隔、月次版なし）
- **config.py更新済み**: MODIS月次をメインに戻し、GCOM-C 8日版を参考情報として記載
- **要ローカル最終確認**:
  ```python
  from jaxa.earth import je
  # GCOM-C NDVI全コレクション検索
  cols, bands = je.ImageCollectionList(ssl_verify=True).filter_name(keywords=["ndvi","GCOM-C"])
  for i, c in enumerate(cols):
      print(f"{c} → bands: {bands[i]}")
  # SGLI NDVI検索
  cols2, bands2 = je.ImageCollectionList(ssl_verify=True).filter_name(keywords=["ndvi","SGLI"])
  for i, c in enumerate(cols2):
      print(f"{c} → bands: {bands2[i]}")
  ```
- **250mが本当に必要な場合**: G-Portalから手動DL → ローカルGeoTIFF → build_features.pyで直接読み込み

### ⑥ 交差検証なし
- 単一のtrain/test分割（2022-2024訓練 / 2025テスト）のみ
- 堅果類豊凶が年ごとに大きく異なるため、年ごとの安定性が不明
- **対策候補**: Leave-One-Year-Out交差検証

### ⑦ 堅果類乗数の線形仮定
- `multiplier = 1.5 - score × 0.8` は線形
- 大凶作と凶作の差が実際の出没増に比例するかは未検証
- **対策候補**: 過去の目撃数と豊凶スコアの回帰分析で非線形関係を推定

## モデル性能サマリー
| モデル | AUC | Recall | Precision | F1 |
|--------|-----|--------|-----------|-----|
| Baseline (位置+標高+森林+月) | 0.9543 | - | - | - |
| Full (衛星データ含む) | 0.9426 | - | - | - |

※ Baselineモデルを空間リスク予測に使用中（AUCが高いため）

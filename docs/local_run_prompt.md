# ローカル再実行手順（修正版 v2）

> CU_JAXAディレクトリに移動してから実行してください
>
> **修正内容**: NaN処理修正、scale_pos_weight修正、時系列分割導入、ndvi_diff追加、HTML軽量化

---

## Step 4のみ再実行: 特徴量テーブル再生成（ndvi_diff追加）

```bash
# features.parquetを再生成（ndvi_diff特徴量が追加される）
python scripts/build_features.py
```

出力:
- `data/processed/features.parquet`（ndvi_diff含む学習テーブル）

確認:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/features.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'ndvi_diff NaN: {df[\"ndvi_diff\"].isna().sum()} / {len(df)}')
"
```

---

## Step 5のみ再実行: モデル学習（修正版）

```bash
python models/train.py
```

修正点:
- **NaN処理**: dropna廃止、LightGBMネイティブNaN処理（データ量維持）
- **scale_pos_weight = 1.0**: ネガティブサンプリング済みなので二重補正しない
- **時系列分割**: ~2024年=Train, 2025年~=Test（未来予測の検証）
- **閾値最適化**: Recall≥0.8制約下でF1最大化
- **ndvi_diff**: NDVI前月差分を新特徴量として追加

出力:
- `models/outputs/model_full.lgb`
- `models/outputs/evaluation_results.json`
- `models/outputs/feature_importance.png`（Baseline/Full並列比較、衛星データ特徴量を赤色表示）

---

## Step 7のみ再実行: リスクマップ生成（軽量版）

```bash
python scripts/generate_risk_map.py --month 2025-07
python scripts/generate_risk_map.py --month 2025-10
```

修正点:
- **HTML軽量化**: Rectangle描画廃止、HeatMap + 上位50マーカーのみ
- **秋田県表示**: タイトル・表示範囲を実データ範囲（秋田県）に修正
- **ndvi_diff対応**: 予測時にNDVI差分も計算

---

## まとめて再実行する場合（Step 4→5→7）

```bash
# Step 4: 特徴量テーブル再生成
python scripts/build_features.py

# Step 5: モデル学習
python models/train.py

# Step 7: リスクマップ生成
python scripts/generate_risk_map.py --month 2025-07
python scripts/generate_risk_map.py --month 2025-10
```

---

## 初回セットアップ（Step 1〜3は済んでいる前提）

Step 1〜3 は前回の実行で完了済み。以下は参考用:

- Step 1: 秋田県クマダスCSV → `data/raw/sightings/akita_kumadas.csv`
- Step 2: `python scripts/prepare_sightings.py` → `tohoku_sightings.csv`
- Step 3: `python scripts/fetch_jaxa.py` → `data/raw/jaxa/`

---

## トラブルシューティング

### パッケージが足りない場合

```bash
pip install -r requirements.txt
```

### Step 4 で「ndvi_diff が生成されない」

→ NDVI時系列データが2ヶ月分以上必要（前月データがないとdiffが計算できない）

### Step 5 で「テストデータが不十分です」

→ 2025年のデータが少ない場合は自動でランダム分割にフォールバックします

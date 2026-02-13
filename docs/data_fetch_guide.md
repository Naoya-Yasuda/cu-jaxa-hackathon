# データ取得手順書（東北3県版）

> この環境から外部APIへのアクセスがブロックされているため、
> 以下の手順はローカル環境（自分のPC）で実行してください。

---

## 1. 秋田県クマダスCSV

```bash
# ダウンロードURL
curl -o data/raw/sightings/akita_kumadas.csv \
  "https://ckan.pref.akita.lg.jp/dataset/f801a10f-f076-47e4-b5a6-0bb5569639e0/resource/326bfe79-3f64-401b-9862-b37a477c7211/download/050008_kumadas.csv"
```

または直接ブラウザでアクセス:
https://ckan.pref.akita.lg.jp/dataset/050008_shizenhogoka_003

「ダウンロード」ボタンからCSVを取得 → `data/raw/sightings/akita_kumadas.csv` に保存

**注意**: ツキノワグマ以外（イノシシ・シカ）も含まれるのでフィルタリングが必要。
前処理スクリプト `scripts/prepare_sightings.py` で自動フィルタリングされます。

---

## 2. 福島県ツキノワグマ目撃情報

ブラウザで以下にアクセス:
https://www.pref.fukushima.lg.jp/sec/16035b/tukinowaguma-mokugeki.html

Excel (.xlsx) ファイルをダウンロード → `data/raw/sightings/fukushima_*.xlsx` に保存

---

## 3. 岩手県データ（確認が必要）

https://www.pref.iwate.jp/opendata/1000081/index.html

CSVが公開されているか確認。なければ秋田+福島の2県で先行。

---

## 4. JAXA衛星データ再取得

config.pyを東北用に更新済みなので、以下を実行:

```bash
cd CU_JAXA
python scripts/fetch_jaxa.py
```

3県統合BBOXは広いので時間がかかる場合があります。
秋田県のみで先行する場合は config.py の `BBOX` を `BBOX_AKITA` に一時変更してください。

---

## 5. 取得後の確認

```bash
# CSVデータの確認
python scripts/prepare_sightings.py

# JAXAデータの確認
python -c "
import json
from pathlib import Path
for d in sorted(Path('data/raw/jaxa').iterdir()):
    meta = d / 'metadata.json'
    if meta.exists():
        m = json.load(open(meta))
        print(f'{d.name}: {m[\"n_images\"]} images, shape={m[\"shape\"][0]}')
"
```

"""
JAXA Earth APIから衛星データを取得するスクリプト

取得データ（確認済みパラメータ: docs/old/param.md参照）:
  1. NDVI（植生指数）月次 MODIS ~5km     - 植生の活性度・どんぐり豊凶
  2. DEM（標高）AW3D30 ~1km              - 標高・傾斜
  3. FNF（森林・非森林マップ）~1km        - 市街地距離の算出
  4. GSMaP（降水量）月次 ~10km           - 降水と行動の相関
  5. LST（地表面温度）月次 ~5km          - 気温（活動パターン）
  6. Landcover（土地被覆）~5km           - 人工改変地 vs 自然地

Usage:
  python scripts/fetch_jaxa.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BBOX, DATE_START, DATE_END, JAXA_DIR, JAXA_COLLECTIONS

from jaxa.earth import je


def fetch_collection(name: str, spec: dict, output_dir: Path) -> bool:
    """1つのコレクションを取得して保存する"""
    print(f"\n{'='*60}")
    print(f"[{name}] {spec['description']}")
    print(f"  Collection: {spec['collection']}")
    print(f"  Band: {spec['band']}")
    print(f"  PPU: {spec['ppu']}")
    print(f"  BBOX: {BBOX}")

    # 日付範囲の決定
    if spec["time_series"]:
        date_range = [DATE_START, DATE_END]
        print(f"  Date: {DATE_START} ~ {DATE_END}")
    else:
        date_range = spec["fixed_date"]
        print(f"  Date: {date_range[0]} ~ {date_range[1]} (固定)")

    print(f"{'='*60}")

    save_dir = output_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        ic = je.ImageCollection(
            collection=spec["collection"],
            ssl_verify=False,
        )

        # 全データで filter_date が必要（静的データも含む）
        ic = ic.filter_date(dlim=date_range)
        ic = ic.filter_resolution(ppu=spec["ppu"])
        ic = ic.filter_bounds(bbox=BBOX)
        ic = ic.select(band=spec["band"])

        print("  Fetching images...")
        data = ic.get_images()

        # raster画像があれば numpy + メタデータとして保存
        if hasattr(data, "raster") and data.raster is not None:
            raster = data.raster
            if hasattr(raster, "img") and raster.img is not None:
                for i, img in enumerate(raster.img):
                    npy_path = save_dir / f"{name}_{i:03d}.npy"
                    np.save(npy_path, img)

                print(f"  Saved {len(raster.img)} raster image(s) as .npy")

                # メタデータ保存
                meta = {
                    "collection": spec["collection"],
                    "band": spec["band"],
                    "ppu": spec["ppu"],
                    "bbox": BBOX,
                    "date_range": date_range,
                    "time_series": spec["time_series"],
                    "n_images": len(raster.img),
                    "shape": [list(img.shape) for img in raster.img],
                    "fetched_at": datetime.now().isoformat(),
                }
                if hasattr(raster, "date") and raster.date:
                    meta["dates"] = [str(d) for d in raster.date]
                if hasattr(raster, "bounds"):
                    meta["bounds"] = str(raster.bounds)

                meta_path = save_dir / "metadata.json"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                print(f"  Saved metadata: {meta_path}")

                # データの概要表示
                for i, img in enumerate(raster.img):
                    valid = img[~np.isnan(img)] if np.issubdtype(img.dtype, np.floating) else img
                    print(f"    [{i:03d}] shape={img.shape}, "
                          f"min={np.nanmin(img):.4f}, max={np.nanmax(img):.4f}, "
                          f"mean={np.nanmean(img):.4f}")
            else:
                print("  Warning: raster.img is None")
                return False
        else:
            print("  Warning: No raster data in response")
            return False

        print(f"  [OK] {name} completed")
        return True

    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    JAXA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("JAXA Earth API データ取得")
    print(f"  対象範囲: {BBOX}")
    print(f"  対象期間: {DATE_START} ~ {DATE_END}")
    print(f"  保存先: {JAXA_DIR}")
    print(f"  コレクション数: {len(JAXA_COLLECTIONS)}")
    print("=" * 60)

    results = {}
    for name, spec in JAXA_COLLECTIONS.items():
        success = fetch_collection(name, spec, JAXA_DIR)
        results[name] = "OK" if success else "FAILED"

    # 結果サマリ
    print("\n" + "=" * 60)
    print("取得結果サマリ")
    print("=" * 60)
    for name, status in results.items():
        icon = "✓" if status == "OK" else "✗"
        desc = JAXA_COLLECTIONS[name]["description"]
        print(f"  {icon} {name}: {status}  ({desc})")

    failed = [n for n, s in results.items() if s != "OK"]
    if failed:
        print(f"\n{len(failed)}件の取得に失敗しました: {failed}")
    else:
        print(f"\n全{len(results)}件の取得が完了しました！")


if __name__ == "__main__":
    main()

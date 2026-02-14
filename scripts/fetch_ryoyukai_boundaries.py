"""
秋田県25市町村の境界ポリゴンを取得し、猟友会管轄メタデータを付与して GeoJSON 出力する。

データソース:
  - 境界ポリゴン: OpenStreetMap (Overpass API) admin_level=7
  - 猟友会メタデータ: docs/akita_ryoyukai_data.md から手動整理

Usage:
  python scripts/fetch_ryoyukai_boundaries.py
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# 秋田県25市町村の猟友会管轄メタデータ
# region: 地域振興局名, bureau_tel: 振興局TEL, dept: 市町村の担当部署, tel: 担当TEL
RYOYUKAI_META = {
    "鹿角市":     {"region": "鹿角地域",   "bureau_tel": "0186-23-2275", "dept": "農地林務課",       "tel": "0186-30-0264"},
    "小坂町":     {"region": "鹿角地域",   "bureau_tel": "0186-23-2275", "dept": "町役場（代表）",   "tel": "0186-29-3901"},
    "大館市":     {"region": "北秋田地域", "bureau_tel": "0186-62-1445", "dept": "産業部林務係",     "tel": "0186-43-7147"},
    "北秋田市":   {"region": "北秋田地域", "bureau_tel": "0186-62-1445", "dept": "農林課林業振興係", "tel": "0186-62-5517"},
    "上小阿仁村": {"region": "北秋田地域", "bureau_tel": "0186-62-1445", "dept": "村役場（代表）",   "tel": "0186-77-2221"},
    "能代市":     {"region": "山本地域",   "bureau_tel": "0185-52-2181", "dept": "農業振興課",       "tel": "0185-89-2183"},
    "藤里町":     {"region": "山本地域",   "bureau_tel": "0185-52-2181", "dept": "町役場（代表）",   "tel": "0185-79-2111"},
    "三種町":     {"region": "山本地域",   "bureau_tel": "0185-52-2181", "dept": "町役場（代表）",   "tel": "0185-85-2111"},
    "八峰町":     {"region": "山本地域",   "bureau_tel": "0185-52-2181", "dept": "町役場（代表）",   "tel": "0185-76-2111"},
    "秋田市":     {"region": "秋田地域",   "bureau_tel": "018-860-3381", "dept": "農地森林整備課",   "tel": "018-888-5740"},
    "男鹿市":     {"region": "秋田地域",   "bureau_tel": "018-860-3381", "dept": "農林水産課",       "tel": "0185-24-9139"},
    "潟上市":     {"region": "秋田地域",   "bureau_tel": "018-860-3381", "dept": "農林水産振興課",   "tel": "018-853-5336"},
    "五城目町":   {"region": "秋田地域",   "bureau_tel": "018-860-3381", "dept": "農林振興課",       "tel": "018-852-5215"},
    "八郎潟町":   {"region": "秋田地域",   "bureau_tel": "018-860-3381", "dept": "町役場（代表）",   "tel": "018-875-5800"},
    "井川町":     {"region": "秋田地域",   "bureau_tel": "018-860-3381", "dept": "町役場（代表）",   "tel": "018-874-2211"},
    "大潟村":     {"region": "秋田地域",   "bureau_tel": "018-860-3381", "dept": "村役場（代表）",   "tel": "0185-45-2111"},
    "由利本荘市": {"region": "由利地域",   "bureau_tel": "0184-22-8351", "dept": "農山漁村振興課",   "tel": "0184-24-6355"},
    "にかほ市":   {"region": "由利地域",   "bureau_tel": "0184-22-8351", "dept": "農林水産課",       "tel": "0184-38-4303"},
    "大仙市":     {"region": "仙北地域",   "bureau_tel": "0187-63-6113", "dept": "農林整備課",       "tel": "0187-63-1111"},
    "仙北市":     {"region": "仙北地域",   "bureau_tel": "0187-63-6113", "dept": "農林整備課",       "tel": "0187-43-2207"},
    "美郷町":     {"region": "仙北地域",   "bureau_tel": "0187-63-6113", "dept": "町役場（代表）",   "tel": "0187-84-1111"},
    "横手市":     {"region": "平鹿地域",   "bureau_tel": "0182-32-9505", "dept": "農林整備課",       "tel": "0182-32-2114"},
    "湯沢市":     {"region": "雄勝地域",   "bureau_tel": "0183-73-5112", "dept": "農林課林務班",     "tel": "0183-55-8569"},
    "羽後町":     {"region": "雄勝地域",   "bureau_tel": "0183-73-5112", "dept": "町役場（代表）",   "tel": "0183-62-2111"},
    "東成瀬村":   {"region": "雄勝地域",   "bureau_tel": "0183-73-5112", "dept": "村役場（代表）",   "tel": "0182-47-3401"},
}

KENBU_TEL = "018-883-1607"  # 秋田県猟友会 本部


def fetch_overpass(query: str) -> dict:
    """Overpass API にクエリを送信して結果を返す"""
    url = "https://overpass-api.de/api/interpreter"
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
    req.add_header("User-Agent", "CU_JAXA_RiskMap/1.0")

    print("  Overpass API にリクエスト中...")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def relation_to_polygons(elements: list) -> dict:
    """Overpass の relation 結果から name → Polygon/MultiPolygon の辞書を作る"""
    # node の座標辞書を作成
    nodes = {}
    for el in elements:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    # way の座標列辞書
    ways = {}
    for el in elements:
        if el["type"] == "way":
            coords = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
            if coords:
                ways[el["id"]] = coords

    # relation を処理
    results = {}
    for el in elements:
        if el["type"] != "relation":
            continue
        name = el.get("tags", {}).get("name", "")
        if not name:
            continue

        # outer メンバーの way を収集
        outer_ways = []
        for member in el.get("members", []):
            if member["type"] == "way" and member.get("role", "") in ("outer", ""):
                wid = member["ref"]
                if wid in ways:
                    outer_ways.append(ways[wid])

        if not outer_ways:
            continue

        # way をつなげてリングを作る
        rings = merge_ways_to_rings(outer_ways)
        if not rings:
            continue

        if len(rings) == 1:
            results[name] = {"type": "Polygon", "coordinates": rings}
        else:
            results[name] = {"type": "MultiPolygon", "coordinates": [[r] for r in rings]}

    return results


def merge_ways_to_rings(ways: list) -> list:
    """複数の way をつなげて閉じたリングのリストにする"""
    # 未使用 way のリストを作る
    remaining = [list(w) for w in ways]
    rings = []

    while remaining:
        ring = remaining.pop(0)

        changed = True
        while changed:
            changed = False
            for i, w in enumerate(remaining):
                # ring の末尾と w の先頭が一致
                if ring[-1] == w[0]:
                    ring.extend(w[1:])
                    remaining.pop(i)
                    changed = True
                    break
                # ring の末尾と w の末尾が一致（逆向き）
                elif ring[-1] == w[-1]:
                    ring.extend(reversed(w[:-1]))
                    remaining.pop(i)
                    changed = True
                    break
                # ring の先頭と w の末尾が一致
                elif ring[0] == w[-1]:
                    ring = w[:-1] + ring
                    remaining.pop(i)
                    changed = True
                    break
                # ring の先頭と w の先頭が一致（逆向き）
                elif ring[0] == w[0]:
                    ring = list(reversed(w[1:])) + ring
                    remaining.pop(i)
                    changed = True
                    break

        # 閉じる
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        if len(ring) >= 4:
            rings.append(ring)

    return rings


def build_geojson(polygons: dict) -> dict:
    """ポリゴン辞書 + 猟友会メタデータ → GeoJSON FeatureCollection"""
    features = []
    matched = set()

    for name, geometry in polygons.items():
        # OSM の name から市町村名を抽出（例: "秋田市" がそのまま入る）
        meta = None
        for muni_name, m in RYOYUKAI_META.items():
            if muni_name in name:
                meta = m
                matched.add(muni_name)
                break

        if meta is None:
            print(f"  警告: '{name}' に対応する猟友会データがありません。スキップ。")
            continue

        props = {
            "name": name,
            "region": meta["region"],
            "dept": meta["dept"],
            "tel": meta["tel"],
            "bureau_tel": meta["bureau_tel"],
            "kenbu_tel": KENBU_TEL,
        }

        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": geometry,
        })

    # マッチしなかった市町村を報告
    unmatched = set(RYOYUKAI_META.keys()) - matched
    if unmatched:
        print(f"  警告: OSMデータに見つからなかった市町村: {unmatched}")

    return {"type": "FeatureCollection", "features": features}


def simplify_coords(geojson: dict, precision: int = 4) -> dict:
    """座標の精度を下げてファイルサイズを削減 (小数点以下 precision 桁)"""
    def round_coords(obj):
        if isinstance(obj, list):
            if obj and isinstance(obj[0], (int, float)):
                return [round(v, precision) for v in obj]
            return [round_coords(item) for item in obj]
        return obj

    for feature in geojson.get("features", []):
        geom = feature.get("geometry", {})
        if "coordinates" in geom:
            geom["coordinates"] = round_coords(geom["coordinates"])

    return geojson


def main():
    print("=" * 60)
    print("秋田県 猟友会管轄境界データ生成")
    print("=" * 60)

    # Overpass API で秋田県内の市町村境界 (admin_level=7) を取得
    # area[name="秋田県"] で秋田県のエリア内に絞る
    query = """
    [out:json][timeout:120];
    area["name"="秋田県"]["admin_level"="4"]->.akita;
    (
      relation["admin_level"="7"]["boundary"="administrative"](area.akita);
    );
    out body;
    >;
    out skel qt;
    """

    print("\n[Step 1] Overpass API から秋田県市町村境界を取得")
    result = fetch_overpass(query)
    elements = result.get("elements", [])
    print(f"  取得要素数: {len(elements)}")

    print("\n[Step 2] relation → ポリゴン変換")
    polygons = relation_to_polygons(elements)
    print(f"  変換された市町村: {len(polygons)}")
    for name in sorted(polygons.keys()):
        geom_type = polygons[name]["type"]
        print(f"    {name} ({geom_type})")

    print("\n[Step 3] 猟友会メタデータをマージ")
    geojson = build_geojson(polygons)
    print(f"  GeoJSON features: {len(geojson['features'])}")

    print("\n[Step 4] 座標精度を最適化して出力")
    geojson = simplify_coords(geojson, precision=4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "ryoyukai_boundaries.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(",", ":"))

    file_size_kb = output_path.stat().st_size / 1024
    print(f"\n  出力: {output_path} ({file_size_kb:.0f}KB)")
    print(f"  市町村数: {len(geojson['features'])}")
    print("\n完了!")


if __name__ == "__main__":
    main()

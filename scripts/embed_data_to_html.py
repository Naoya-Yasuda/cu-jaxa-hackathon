"""
risk_data_2025.json と ryoyukai_boundaries.json を HTML に埋め込み、
file:// プロトコルでも完全に動作するスタンドアロン HTML を生成する。

Usage:
  python scripts/embed_data_to_html.py
"""

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

HTML_PATH = OUTPUT_DIR / "risk_app_2025hatsu.html"
RISK_JSON = OUTPUT_DIR / "risk_data_2025.json"
RYOYUKAI_JSON = OUTPUT_DIR / "ryoyukai_boundaries.json"


def main():
    print("=" * 60)
    print("HTML へデータ埋め込み")
    print("=" * 60)

    # 1. 読み込み
    html = HTML_PATH.read_text(encoding="utf-8")
    risk_data = RISK_JSON.read_text(encoding="utf-8").strip()
    ryoyukai_data = RYOYUKAI_JSON.read_text(encoding="utf-8").strip()

    print(f"  HTML: {len(html):,} bytes")
    print(f"  risk_data_2025.json: {len(risk_data):,} bytes")
    print(f"  ryoyukai_boundaries.json: {len(ryoyukai_data):,} bytes")

    # 2. risk_data を埋め込み
    #    - `let RISK_DATA = null;` → `let RISK_DATA = {...};`
    #    - `init()` 内の fetch 部分を削除
    old_risk_decl = "let RISK_DATA = null;"
    new_risk_decl = f"let RISK_DATA = {risk_data};"
    if old_risk_decl not in html:
        print("  ERROR: 'let RISK_DATA = null;' が見つかりません")
        return
    html = html.replace(old_risk_decl, new_risk_decl, 1)

    # fetch を使う init() を、埋め込みデータを使う形に書き換え
    old_init = '''async function init() {
  const resp = await fetch("./risk_data_2025.json");
  if (!resp.ok) throw new Error("risk_data_2025.json の読み込みに失敗しました");
  RISK_DATA = await resp.json();

  buildTicks();'''
    new_init = '''async function init() {
  buildTicks();'''
    if old_init not in html:
        print("  ERROR: init() 関数のfetch部分が見つかりません")
        return
    html = html.replace(old_init, new_init, 1)

    # エラーメッセージも更新
    old_err = '''`<b>読み込みエラー</b><br>${err.message}<br><br>` +
      `HTTPサーバー経由で開いてください。`;'''
    new_err = '''`<b>読み込みエラー</b><br>${err.message}`;'''
    html = html.replace(old_err, new_err, 1)

    print("  ✓ risk_data 埋め込み完了")

    # 3. ryoyukai データを埋め込み
    #    - Overpass API fetch → 埋め込み GeoJSON を使用
    old_ryoyukai_load = '''async function loadRyoyukaiData() {
  try {
    const q = `[out:json][timeout:60];area["name"="秋田県"]["admin_level"="4"]->.akita;relation["admin_level"="7"]["boundary"="administrative"](area.akita);out geom;`;
    const url = "https://overpass-api.de/api/interpreter";
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8" },
      body: "data=" + encodeURIComponent(q)
    });
    if (!res.ok) throw new Error("Overpass error: " + res.status);
    const data = await res.json();
    const geojson = overpassAdminToGeoJSON(data.elements || []);
    map.getSource("ryoyukai").setData(geojson);
    setupRyoyukaiHover();
    console.log("猟友会管轄: " + geojson.features.length + "市町村を取得");
  } catch (e) {
    console.warn("猟友会データの取得に失敗:", e);
  }
}'''
    new_ryoyukai_load = f'''const RYOYUKAI_GEOJSON = {ryoyukai_data};

async function loadRyoyukaiData() {{
  try {{
    map.getSource("ryoyukai").setData(RYOYUKAI_GEOJSON);
    setupRyoyukaiHover();
    console.log("猟友会管轄: " + RYOYUKAI_GEOJSON.features.length + "市町村を読み込み");
  }} catch (e) {{
    console.warn("猟友会データの設定に失敗:", e);
  }}
}}'''
    if old_ryoyukai_load not in html:
        print("  ERROR: loadRyoyukaiData() 関数が見つかりません")
        return
    html = html.replace(old_ryoyukai_load, new_ryoyukai_load, 1)

    print("  ✓ ryoyukai_boundaries 埋め込み完了")

    # 4. Overpass API での mergeWaysToRings / overpassAdminToGeoJSON は
    #    もう不要だが、残しても害はないのでそのまま

    # 5. 出力
    HTML_PATH.write_text(html, encoding="utf-8")
    file_size_mb = len(html.encode("utf-8")) / (1024 * 1024)
    print(f"\n  出力: {HTML_PATH}")
    print(f"  サイズ: {file_size_mb:.1f}MB")
    print("\n完了! file:// プロトコルでそのまま開けます。")


if __name__ == "__main__":
    main()

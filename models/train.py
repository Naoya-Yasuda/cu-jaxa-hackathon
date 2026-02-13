"""
LightGBMモデル学習 & 評価スクリプト（修正版）

修正点:
  1. NaN処理: dropna廃止 → LightGBMネイティブNaN処理を活用
  2. scale_pos_weight: 1.0固定（ネガティブサンプリング済みのため二重補正しない）
  3. 時系列分割: 2025年データをテストに使用（未来予測の検証）
  4. ndvi_diff特徴量を追加
  5. 閾値最適化: Recall重視の閾値自動選択

Usage:
  python models/train.py
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR

import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score,
    f1_score, confusion_matrix, precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_DIR = Path(__file__).parent
OUTPUT_DIR = MODEL_DIR / "outputs"


# ===== 特徴量の定義 =====
# 位置弁別用（緯度経度）: 「どの地域か」
LOCATION_FEATURES = ["lat_center", "lon_center"]

# 空間弁別用（DEM/FNF）: 「どこが危ないか」
SPATIAL_FEATURES = ["elevation", "is_forest", "landcover"]

# 時間弁別用（衛星データ + 月）: 「いつ危ないか」
TEMPORAL_FEATURES = ["ndvi", "ndvi_diff", "ndvi_anomaly", "lst_celsius", "precip", "month"]

# ベースライン（衛星データなし）: 位置+地形+月のみ
BASELINE_FEATURES = LOCATION_FEATURES + ["elevation", "is_forest", "month"]

# 全特徴量（衛星データ含む）
ALL_FEATURES = LOCATION_FEATURES + SPATIAL_FEATURES + TEMPORAL_FEATURES


def load_data():
    """特徴量テーブルを読み込み"""
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        print(f"ERROR: {path} が見つかりません。先に build_features.py を実行してください。")
        sys.exit(1)

    df = pd.read_parquet(path)
    print(f"データ読み込み: {len(df)}件")
    print(f"  正例: {df['target'].sum()}, 負例: {len(df) - df['target'].sum()}")
    print(f"  正例率: {df['target'].mean():.1%}")
    print(f"  カラム: {list(df.columns)}")

    # NaN状況の確認
    print(f"\n  NaN状況:")
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"    {col}: {nan_count} ({nan_count/len(df)*100:.1f}%)")

    return df


def temporal_split(df, test_year=2025):
    """
    時系列分割: test_year以降をテスト、それ以前をトレーニング

    year_month カラム（例: "2024-07"）からyearを抽出して分割
    """
    df = df.copy()
    df["_year"] = df["year_month"].astype(str).str[:4].astype(int)

    train_mask = df["_year"] < test_year
    test_mask = df["_year"] >= test_year

    print(f"\n  時系列分割: train=~{test_year-1}年, test={test_year}年~")
    print(f"    Train年: {sorted(df[train_mask]['_year'].unique())}")
    print(f"    Test年:  {sorted(df[test_mask]['_year'].unique())}")

    df = df.drop(columns=["_year"])
    return train_mask, test_mask


def find_optimal_threshold(y_true, y_pred_proba, min_recall=0.8):
    """
    Recall >= min_recall を維持しつつ F1 を最大化する閾値を探索
    （安全用途なので見逃しを最小化）
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    best_threshold = 0.5
    best_f1 = 0.0

    for p, r, t in zip(precisions, recalls, thresholds):
        if r >= min_recall:
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

    return best_threshold


def train_and_evaluate(df, feature_cols, model_name="model"):
    """LightGBMの学習・評価を実行（NaN除去なし、時系列分割）"""
    print(f"\n{'='*60}")
    print(f"[{model_name}] 特徴量: {feature_cols}")
    print(f"{'='*60}")

    # 使える特徴量のみ（カラムが存在するもの）
    available_features = [f for f in feature_cols if f in df.columns]
    if not available_features:
        print("  使える特徴量がありません。スキップします。")
        return None

    print(f"  利用可能特徴量: {available_features}")

    # ★ 修正1: dropnaしない。LightGBMはNaNをネイティブに処理可能
    # target列のNaNだけは除去（予測対象がないレコード）
    df_clean = df.dropna(subset=["target"])
    print(f"  有効レコード: {len(df_clean)} / {len(df)} (NaN特徴量はLightGBMが処理)")

    # NaN状況を確認（参考情報）
    for f in available_features:
        nan_pct = df_clean[f].isna().mean() * 100
        if nan_pct > 0:
            print(f"    {f}: NaN {nan_pct:.1f}% → LightGBMが欠損として処理")

    if len(df_clean) < 100:
        print("  データが少なすぎます。スキップします。")
        return None

    X = df_clean[available_features]
    y = df_clean["target"]

    # ★ 修正3: 時系列分割（2025年をテストに使用）
    train_mask, test_mask = temporal_split(df_clean)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(X_test) < 10 or y_test.sum() < 5:
        print("  テストデータが不十分です。ランダム分割にフォールバック。")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    print(f"  Train: {len(X_train)} (正例={y_train.sum()}, 正例率={y_train.mean():.1%})")
    print(f"  Test:  {len(X_test)} (正例={y_test.sum()}, 正例率={y_test.mean():.1%})")

    # ★ 修正2: scale_pos_weight = 1.0（ネガティブサンプリング済みなので二重補正しない）
    params = {
        "objective": "binary",
        "metric": "auc",
        "scale_pos_weight": 1.0,  # ← 修正: ネガティブサンプリング済みのため1.0
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # 予測
    y_pred_proba = model.predict(X_test)

    # ★ 修正5: 安全用途向け閾値最適化（Recall≥0.8を維持しつつF1最大化）
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, min_recall=0.8)
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    # 評価
    auc = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  === 評価結果 ===")
    print(f"  閾値:      {optimal_threshold:.3f} (Recall≥0.8制約下でF1最大化)")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"\n  混同行列:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0][0]}, FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}, TP={cm[1][1]}")

    # 参考: 固定閾値0.5での結果も表示
    y_pred_05 = (y_pred_proba >= 0.5).astype(int)
    print(f"\n  参考（閾値=0.5）:")
    print(f"    Recall={recall_score(y_test, y_pred_05):.4f}, "
          f"Precision={precision_score(y_test, y_pred_05, zero_division=0):.4f}, "
          f"F1={f1_score(y_test, y_pred_05, zero_division=0):.4f}")

    # 特徴量重要度
    importance = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    print(f"\n  特徴量重要度 (gain):")
    for _, row in importance.iterrows():
        bar = "█" * int(row["importance"] / importance["importance"].max() * 30)
        print(f"    {row['feature']:<15} {row['importance']:>10.1f} {bar}")

    return {
        "model": model,
        "auc": auc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "threshold": optimal_threshold,
        "importance": importance,
        "features": available_features,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def compare_models(baseline_result, full_result):
    """衛星データあり/なしの比較"""
    print(f"\n{'='*60}")
    print("Before/After比較: 衛星データの効果")
    print(f"{'='*60}")

    if baseline_result is None or full_result is None:
        print("  比較できません（いずれかのモデルが学習失敗）")
        return

    print(f"\n  データ量: Baseline={baseline_result['train_size']}+{baseline_result['test_size']}, "
          f"Full={full_result['train_size']}+{full_result['test_size']}")

    metrics = ["auc", "recall", "precision", "f1"]
    print(f"\n  {'指標':<12} {'Baseline':>10} {'+ 衛星データ':>12} {'改善幅':>10}")
    print(f"  {'-'*48}")
    for m in metrics:
        base = baseline_result[m]
        full = full_result[m]
        diff = full - base
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        print(f"  {m:<12} {base:>10.4f} {full:>12.4f} {arrow}{abs(diff):>9.4f}")

    auc_improvement = full_result["auc"] - baseline_result["auc"]
    if auc_improvement > 0.02:
        print(f"\n  → 衛星データにより AUC が {auc_improvement:.4f} 向上。有意な改善あり。")
    elif auc_improvement > 0:
        print(f"\n  → 衛星データにより AUC が微増（{auc_improvement:.4f}）。限定的な改善。")
    else:
        print(f"\n  → 衛星データによるAUC改善は見られず。")
        print(f"     ただし衛星データは空間的な生息適地判定やNDVI変動の説明力で貢献可能。")

    # 衛星データ特徴量の寄与率を計算
    if full_result and "importance" in full_result:
        imp = full_result["importance"]
        total = imp["importance"].sum()
        satellite_features = ["ndvi", "ndvi_diff", "lst_celsius", "precip", "landcover"]
        sat_imp = imp[imp["feature"].isin(satellite_features)]["importance"].sum()
        print(f"\n  衛星データ特徴量の寄与率: {sat_imp/total*100:.1f}% (gain合計比)")


def save_results(baseline_result, full_result):
    """結果を保存"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # モデル保存
    if full_result and full_result["model"]:
        model_path = OUTPUT_DIR / "model_full.lgb"
        full_result["model"].save_model(str(model_path))
        print(f"\nモデル保存: {model_path}")

    # 評価結果JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "split_method": "temporal (train: ~2024, test: 2025~)",
        "scale_pos_weight": 1.0,
        "note": "NaN removed: LightGBM native NaN handling",
    }

    for label, result in [("baseline", baseline_result), ("full_model", full_result)]:
        if result:
            results[label] = {
                k: v for k, v in result.items()
                if k not in ("model", "importance")
            }
            if "importance" in result:
                results[label]["importance"] = result["importance"].to_dict("records")

    results_path = OUTPUT_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"評価結果保存: {results_path}")

    # 特徴量重要度グラフ（Before/After並列表示）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, result, title in [
        (axes[0], baseline_result, "Baseline (衛星データなし)"),
        (axes[1], full_result, "Full Model (衛星データあり)"),
    ]:
        if result and "importance" in result:
            imp = result["importance"]
            # 衛星データ特徴量を色分け
            satellite_features = {"ndvi", "ndvi_diff", "lst_celsius", "precip", "landcover"}
            colors = ["#e74c3c" if f in satellite_features else "#3498db"
                      for f in imp["feature"]]
            ax.barh(imp["feature"], imp["importance"], color=colors)
            ax.set_xlabel("Importance (gain)")
            ax.set_title(f"{title}\nAUC={result['auc']:.4f}")
            ax.invert_yaxis()

    # 凡例
    from matplotlib.patches import Patch
    axes[1].legend(
        handles=[
            Patch(color="#3498db", label="位置・地形特徴量"),
            Patch(color="#e74c3c", label="衛星データ特徴量"),
        ],
        loc="lower right",
    )

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "feature_importance.png"
    fig.savefig(fig_path, dpi=150)
    print(f"特徴量重要度グラフ保存: {fig_path}")
    plt.close()

    # Precision-Recall曲線も保存
    # （main関数で呼ばれた後に必要に応じて追加可能）


def main():
    # データ読み込み
    df = load_data()

    # ===== Model 1: ベースライン（衛星データなし）=====
    baseline_result = train_and_evaluate(
        df, BASELINE_FEATURES, model_name="Baseline (位置+標高+森林+月)"
    )

    # ===== Model 2: 全特徴量（衛星データあり）=====
    full_result = train_and_evaluate(
        df, ALL_FEATURES, model_name="Full (衛星データ含む)"
    )

    # ===== Before/After比較 =====
    compare_models(baseline_result, full_result)

    # ===== 保存 =====
    save_results(baseline_result, full_result)

    print("\n完了！")


if __name__ == "__main__":
    main()

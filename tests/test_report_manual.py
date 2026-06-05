#!/usr/bin/env python
"""
手動測試 report 生成 — 用假資料驗證 build_report 輸出

注意：這是一個獨立執行的腳本，並非 pytest 測試。所有頂層程式碼都包在
``main()`` 中並由 ``if __name__ == "__main__"`` 守護，避免 pytest 在
collection 階段執行而中斷整個測試流程。

執行方式：
    python tests/test_report_manual.py
"""

import json


def main():
    # ============================================================
    # 1. 建立假資料（模擬 route_data）
    # ============================================================

    fake_routes = [
        {
            "mode": "高鐵",
            "cost": 700,
            "duration_min": 100,
            "distance_km": 160.0,
            "transfers": 1,
            "frequency_per_day": 120,
            "convenience": 8.0,
            "source": "static",
            "fetched_at": "2026-05-31T10:00:00+08:00",
            "segments": [
                {"mode": "步行/接駁", "from": "台北", "to": "台北高鐵站",
                 "duration_min": 25, "cost": 0, "distance_km": 0.0},
                {"mode": "高鐵", "from": "台北高鐵站", "to": "台中高鐵站",
                 "duration_min": 50, "cost": 700, "distance_km": 160.0},
            ],
        },
        {
            "mode": "台鐵",
            "cost": 375,
            "duration_min": 200,
            "distance_km": 160.0,
            "transfers": 1,
            "frequency_per_day": 40,
            "convenience": 6.0,
            "source": "static",
            "fetched_at": "2026-05-31T10:00:00+08:00",
            "segments": [
                {"mode": "步行/接駁", "from": "台北", "to": "台北車站",
                 "duration_min": 25, "cost": 0, "distance_km": 0.0},
                {"mode": "台鐵", "from": "台北車站", "to": "台中車站",
                 "duration_min": 150, "cost": 375, "distance_km": 160.0},
            ],
        },
        {
            "mode": "客運",
            "cost": 260,
            "duration_min": 180,
            "distance_km": 160.0,
            "transfers": 0,
            "frequency_per_day": 60,
            "convenience": 5.0,
            "source": "static",
            "fetched_at": "2026-05-31T10:00:00+08:00",
            "segments": [
                {"mode": "客運", "from": "台北", "to": "台中",
                 "duration_min": 180, "cost": 260, "distance_km": 160.0},
            ],
        },
        {
            "mode": "租車",
            "cost": 1800,
            "duration_min": 120,
            "distance_km": 160.0,
            "transfers": 0,
            "frequency_per_day": 999,
            "convenience": 7.0,
            "source": "static",
            "fetched_at": "2026-05-31T10:00:00+08:00",
            "segments": [
                {"mode": "租車", "from": "台北", "to": "台中",
                 "duration_min": 120, "cost": 1800, "distance_km": 160.0},
            ],
        },
    ]

    # ============================================================
    # 2. 計算 features（特徵工程）
    # ============================================================

    print("=" * 70)
    print("【步驟 1】輸入假資料（4種交通方式）")
    print("=" * 70)
    for route in fake_routes:
        print(f"  {route['mode']:6} | 票價: {route['cost']:4} | 時間: {route['duration_min']:3} 分 | 便利性: {route['convenience']}")

    from features.feature_engineer import compute_features, build_dt_feature_row

    features = compute_features(fake_routes, num_people=2, luggage=1)

    print("\n" + "=" * 70)
    print("【步驟 2】特徵工程 - 計算標準化分數")
    print("=" * 70)
    for feat in features:
        print(f"\n{feat['mode']}:")
        print(f"  成本分數: {feat['cost_score']:.3f}  (原始: {feat['cost_raw']:.0f})")
        print(f"  時間分數: {feat['time_score']:.3f}  (原始: {feat['time_raw']:.0f} 分鐘)")
        print(f"  便利分數: {feat['convenience_score']:.3f}  (原始: {feat['convenience_raw']:.0f})")

    # ============================================================
    # 3. 計算評分（加權評分）
    # ============================================================

    from models.scoring_model import compute_scores

    w_cost = 5
    w_time = 5
    w_conv = 5
    results = compute_scores(features, w_cost=w_cost, w_time=w_time, w_conv=w_conv)

    print("\n" + "=" * 70)
    print(f"【步驟 3】加權評分 (權重: 成本={w_cost}, 時間={w_time}, 便利性={w_conv})")
    print("=" * 70)
    for res in results:
        print(f"排名 {res['rank']}: {res['mode']:6} | 總分: {res['total_score']:.4f} | 成本:{res['cost_score']:.2f} 時間:{res['time_score']:.2f} 便利:{res['convenience_score']:.2f}")

    # ============================================================
    # 4. 生成 XAI 報告
    # ============================================================

    from xai.report import build_report
    from models.decision_tree import FEATURE_COLS
    import pandas as pd

    # 完整的 29 特徵 DT feature_row（鍵涵蓋 FEATURE_COLS），與 ui/app.py 一致。
    route_row = pd.Series({
        "avg_transfer": 1,
        "speed": 48.0,
        "avg_cost": 375.0,
        "avg_time": 200.0,
        "avg_walking": 50.0,
        "transport_tools_norm": "LONG DISTANCE TRAIN",
        "has_car12": 0, "has_car24": 0, "has_car35": 0, "has_car57": 0,
    })
    feature_row = build_dt_feature_row(
        route_row, people=2, luggage=1, budget=2000.0,
        time_pref=5, cost_pref=5, conv_pref=3, feature_cols=FEATURE_COLS,
    )

    report = build_report(
        results=results,
        features=features,
        feature_row=feature_row,
        w_cost=w_cost,
        w_time=w_time,
        w_conv=w_conv,
        budget=2000,
        clf=None,
    )

    print("\n" + "=" * 70)
    print("【步驟 4】XAI 報告生成")
    print("=" * 70)

    print("\n【推薦原因】")
    print(f"  {report['top_reason']}")

    print("\n【特徵貢獻分解】")
    for contrib in report['contributions']:
        print(f"  {contrib['dim_zh']:6} | 權重: {contrib['weight']:.2f} | 分數: {contrib['score']:.3f} | 貢獻: {contrib['contribution']:.4f} ({contrib['share']*100:.1f}%)")

    total_contrib = sum(c['contribution'] for c in report['contributions'])
    print(f"  {'合計':6} | 總貢獻: {total_contrib:.4f}")

    print("\n【決策樹路徑】")
    if report['decision_path']:
        print(f"  {report['decision_path']}")
    else:
        print("  (無預先訓練的模型，決策路徑為 None)")

    print("\n【反事實提示】")
    if report['counterfactuals']:
        for i, tip in enumerate(report['counterfactuals'], 1):
            print(f"  {i}. {tip}")
    else:
        print("  (無反事實提示)")

    print("\n【SHAP 貢獻】")
    if report['shap']:
        print(f"  預測: {report['shap']['pred_zh']}")
        for val in report['shap']['values']:
            print(f"    {val['feature_zh']:8} | {val['value']:+.4f}")
    else:
        print("  (無 SHAP 資料，shap 套件可能未安裝)")

    # ============================================================
    # 5. 完整報告結構輸出
    # ============================================================

    print("\n" + "=" * 70)
    print("【完整報告 JSON】")
    print("=" * 70)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("✓ 報告生成完成！")
    print("=" * 70)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    main()

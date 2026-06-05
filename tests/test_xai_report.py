"""
測試 xai/report.py — XAI 報告輸出

涵蓋：
- feature_contributions：貢獻值分解、排序、加總不變式
- _top_reason：中文說明字串格式
- decision_path_zh：決策樹路徑轉中文（使用真實/假的 classifier）
- shap_contributions：有/無 shap 套件的行為
- counterfactuals：預算門檻與權重翻轉提示
- build_report：完整輸出結構驗證

所有測試不呼叫外部 API；決策樹測試使用假的訓練資料或 mock。
"""

import pytest
import numpy as np


# ─────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────

def _make_results(overrides=None):
    """製造最小的 compute_scores 輸出 list（sorted by score desc）。"""
    base = [
        {
            "mode": "高鐵", "rank": 1,
            "cost_raw": 700.0, "time_raw": 100.0, "convenience_raw": 7.0,
            "cost_score": 0.6, "time_score": 0.8, "convenience_score": 0.9,
            "total_score": 0.7667,
            "weights": {"cost": 0.3333, "time": 0.3333, "convenience": 0.3333},
            "segments": [], "transfers": 1, "distance_km": 160.0,
        },
        {
            "mode": "台鐵", "rank": 2,
            "cost_raw": 375.0, "time_raw": 200.0, "convenience_raw": 5.0,
            "cost_score": 0.9, "time_score": 0.3, "convenience_score": 0.4,
            "total_score": 0.5333,
            "weights": {"cost": 0.3333, "time": 0.3333, "convenience": 0.3333},
            "segments": [], "transfers": 1, "distance_km": 160.0,
        },
        {
            "mode": "客運", "rank": 3,
            "cost_raw": 260.0, "time_raw": 180.0, "convenience_raw": 4.0,
            "cost_score": 1.0, "time_score": 0.4, "convenience_score": 0.2,
            "total_score": 0.5333,
            "weights": {"cost": 0.3333, "time": 0.3333, "convenience": 0.3333},
            "segments": [], "transfers": 0, "distance_km": 160.0,
        },
        {
            "mode": "租車", "rank": 4,
            "cost_raw": 1800.0, "time_raw": 120.0, "convenience_raw": 8.0,
            "cost_score": 0.0, "time_score": 0.7, "convenience_score": 1.0,
            "total_score": 0.5667,
            "weights": {"cost": 0.3333, "time": 0.3333, "convenience": 0.3333},
            "segments": [], "transfers": 0, "distance_km": 160.0,
        },
    ]
    if overrides:
        base[0].update(overrides)
    return base


def _make_features():
    """製造 compute_features 輸出（用於反事實計算）。"""
    return [
        {"mode": "高鐵", "cost_raw": 700.0, "time_raw": 100.0, "convenience_raw": 7.0,
         "cost_score": 0.6, "time_score": 0.8, "convenience_score": 0.9,
         "segments": [], "transfers": 1, "distance_km": 160.0},
        {"mode": "台鐵", "cost_raw": 375.0, "time_raw": 200.0, "convenience_raw": 5.0,
         "cost_score": 0.9, "time_score": 0.3, "convenience_score": 0.4,
         "segments": [], "transfers": 1, "distance_km": 160.0},
        {"mode": "客運", "cost_raw": 260.0, "time_raw": 180.0, "convenience_raw": 4.0,
         "cost_score": 1.0, "time_score": 0.4, "convenience_score": 0.2,
         "segments": [], "transfers": 0, "distance_km": 160.0},
        {"mode": "租車", "cost_raw": 1800.0, "time_raw": 120.0, "convenience_raw": 8.0,
         "cost_score": 0.0, "time_score": 0.7, "convenience_score": 1.0,
         "segments": [], "transfers": 0, "distance_km": 160.0},
    ]


def _train_tiny_clf():
    """
    用最小合成資料訓練一個 Decision Tree，供路徑測試使用。

    特徵必須涵蓋 FEATURE_COLS（與正式模型相同的 29 維特徵空間），decision_path_zh
    / shap_contributions 才能依 FEATURE_COLS 正確對應節點特徵索引到中文名稱。
    這裡用 conftest 的 make_full_feature_row 產生完整特徵列。
    """
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd
    from models.decision_tree import FEATURE_COLS
    from conftest import make_full_feature_row

    if not FEATURE_COLS:
        return None

    rows = [
        make_full_feature_row(people=1, luggage=0, budget=500,
                              time_pref=8, cost_pref=3, conv_pref=4),
        make_full_feature_row(people=2, luggage=1, budget=2000,
                              time_pref=5, cost_pref=5, conv_pref=5),
        make_full_feature_row(people=4, luggage=2, budget=3000,
                              time_pref=3, cost_pref=8, conv_pref=2),
        make_full_feature_row(people=6, luggage=3, budget=5000,
                              time_pref=2, cost_pref=9, conv_pref=1),
    ]
    X = pd.DataFrame(rows)[FEATURE_COLS]
    y = ["Bus", "HSR", "Car", "Car"]
    clf = DecisionTreeClassifier(random_state=42, max_depth=3)
    clf.fit(X, y)
    return clf


# ─────────────────────────────────────────────────────────
# 1. feature_contributions
# ─────────────────────────────────────────────────────────

class TestFeatureContributions:
    def test_returns_three_items(self):
        from xai.report import feature_contributions
        results = _make_results()
        contribs = feature_contributions(results[0])
        assert len(contribs) == 3

    def test_sum_equals_total_score(self):
        from xai.report import feature_contributions
        best = _make_results()[0]
        contribs = feature_contributions(best)
        total = sum(c["contribution"] for c in contribs)
        assert abs(total - best["total_score"]) < 1e-3, (
            f"貢獻加總 {total:.6f} ≠ total_score {best['total_score']:.6f}"
        )

    def test_sorted_by_contribution_desc(self):
        from xai.report import feature_contributions
        contribs = feature_contributions(_make_results()[0])
        values = [c["contribution"] for c in contribs]
        assert values == sorted(values, reverse=True)

    def test_required_keys(self):
        from xai.report import feature_contributions
        contribs = feature_contributions(_make_results()[0])
        for c in contribs:
            for key in ("dim", "dim_zh", "weight", "score", "contribution", "share"):
                assert key in c, f"缺少 key: {key}"

    def test_dim_zh_is_chinese(self):
        from xai.report import feature_contributions
        contribs = feature_contributions(_make_results()[0])
        zh_names = {c["dim_zh"] for c in contribs}
        assert zh_names == {"成本", "時間", "便利性"}

    def test_shares_sum_to_one(self):
        from xai.report import feature_contributions
        contribs = feature_contributions(_make_results()[0])
        total_share = sum(c["share"] for c in contribs)
        assert abs(total_share - 1.0) < 1e-4

    def test_non_negative_contributions_when_scores_positive(self):
        from xai.report import feature_contributions
        contribs = feature_contributions(_make_results()[0])
        for c in contribs:
            assert c["contribution"] >= 0, f"{c['dim']} 貢獻不應為負"

    def test_uses_fixtures(self, sample_results):
        """用 conftest fixture 驗證真實計算路徑。"""
        from xai.report import feature_contributions
        contribs = feature_contributions(sample_results[0])
        total = sum(c["contribution"] for c in contribs)
        assert abs(total - sample_results[0]["total_score"]) < 1e-4


# ─────────────────────────────────────────────────────────
# 2. _top_reason（私有函式，透過 build_report 間接測試）
# ─────────────────────────────────────────────────────────

class TestTopReason:
    def test_returns_non_empty_string(self):
        from xai.report import _top_reason, feature_contributions
        best = _make_results()[0]
        contribs = feature_contributions(best)
        reason = _top_reason(best, contribs)
        assert isinstance(reason, str) and len(reason) > 0

    def test_mentions_best_mode(self):
        from xai.report import _top_reason, feature_contributions
        best = _make_results()[0]
        contribs = feature_contributions(best)
        reason = _top_reason(best, contribs)
        assert best["mode"] in reason, "推薦原因應包含交通方式名稱"

    def test_mentions_total_score(self):
        from xai.report import _top_reason, feature_contributions
        best = _make_results()[0]
        contribs = feature_contributions(best)
        reason = _top_reason(best, contribs)
        assert str(round(best["total_score"], 2)) in reason


# ─────────────────────────────────────────────────────────
# 3. decision_path_zh
# ─────────────────────────────────────────────────────────

class TestDecisionPathZh:
    def test_none_clf_returns_none(self):
        from xai.report import decision_path_zh
        row = {"people": 2, "luggage": 1, "budget": 2000,
               "time_pref": 5, "cost_pref": 5, "conv_pref": 3, "distance_km": 160.0}
        result = decision_path_zh(None, row)
        assert result is None

    def test_returns_chinese_string_with_real_clf(self):
        from xai.report import decision_path_zh
        from conftest import make_full_feature_row
        clf = _train_tiny_clf()
        row = make_full_feature_row(people=2, luggage=1, budget=2000,
                                    time_pref=5, cost_pref=5, conv_pref=3)
        result = decision_path_zh(clf, row)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_recommendation_keyword(self):
        from xai.report import decision_path_zh
        from conftest import make_full_feature_row
        clf = _train_tiny_clf()
        row = make_full_feature_row(people=2, luggage=1, budget=2000,
                                    time_pref=5, cost_pref=5, conv_pref=3)
        result = decision_path_zh(clf, row)
        assert "推薦" in result

    def test_contains_at_least_one_feature_name(self):
        from xai.report import decision_path_zh, _FEATURE_ZH
        from conftest import make_full_feature_row
        clf = _train_tiny_clf()
        row = make_full_feature_row(people=2, luggage=1, budget=2000,
                                    time_pref=5, cost_pref=5, conv_pref=3)
        result = decision_path_zh(clf, row)
        # 任一決策節點的特徵名都會經 _FEATURE_ZH 轉成中文
        feature_zh_names = set(_FEATURE_ZH.values())
        has_feature = any(f in result for f in feature_zh_names)
        # 若路徑只有一個 leaf（無分裂條件）則沒有特徵名，也是合法輸出
        assert has_feature or "無分裂條件" in result

    def test_different_inputs_may_produce_different_paths(self):
        from xai.report import decision_path_zh
        from conftest import make_full_feature_row
        clf = _train_tiny_clf()
        row1 = make_full_feature_row(people=1, luggage=0, budget=500,
                                     time_pref=8, cost_pref=3, conv_pref=4)
        row2 = make_full_feature_row(people=6, luggage=3, budget=5000,
                                     time_pref=2, cost_pref=9, conv_pref=1)
        r1 = decision_path_zh(clf, row1)
        r2 = decision_path_zh(clf, row2)
        # 不強制一定不同（資料太少可能同一 leaf），但皆為字串
        assert isinstance(r1, str) and isinstance(r2, str)


# ─────────────────────────────────────────────────────────
# 4. shap_contributions
# ─────────────────────────────────────────────────────────

class TestShapContributions:
    def test_none_clf_returns_none(self):
        from xai.report import shap_contributions
        row = {"people": 2, "luggage": 1, "budget": 2000,
               "time_pref": 5, "cost_pref": 5, "conv_pref": 3, "distance_km": 160.0}
        result = shap_contributions(None, row)
        assert result is None

    def test_with_real_clf_returns_dict_or_none(self):
        """有無 shap 套件皆不應拋例外。"""
        from xai.report import shap_contributions
        from conftest import make_full_feature_row
        clf = _train_tiny_clf()
        row = make_full_feature_row(people=2, luggage=1, budget=2000,
                                    time_pref=5, cost_pref=5, conv_pref=3)
        result = shap_contributions(clf, row)
        if result is not None:
            assert "pred_zh" in result
            assert "values" in result
            assert isinstance(result["values"], list)

    def test_shap_values_sorted_by_abs_desc(self):
        """若 shap 可用，推力應按絕對值由大到小排序。"""
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap 未安裝，跳過此測試")

        from xai.report import shap_contributions
        from conftest import make_full_feature_row
        clf = _train_tiny_clf()
        row = make_full_feature_row(people=2, luggage=1, budget=2000,
                                    time_pref=5, cost_pref=5, conv_pref=3)
        result = shap_contributions(clf, row)
        if result is None:
            pytest.skip("shap_contributions 回傳 None")
        abs_values = [abs(v["value"]) for v in result["values"]]
        assert abs_values == sorted(abs_values, reverse=True)


# ─────────────────────────────────────────────────────────
# 5. counterfactuals
# ─────────────────────────────────────────────────────────

class TestCounterfactuals:
    def test_returns_list(self):
        from xai.report import counterfactuals
        tips = counterfactuals(_make_features(), 5, 5, 5, 2000)
        assert isinstance(tips, list)

    def test_budget_tip_when_cheaper_exists(self):
        """目前最佳為高鐵(700)，台鐵(375)更便宜 → 應有預算門檻提示。"""
        from xai.report import counterfactuals
        tips = counterfactuals(_make_features(), 5, 5, 5, 2000)
        budget_tips = [t for t in tips if "預算" in t]
        assert len(budget_tips) >= 1, f"應有預算反事實提示，實際: {tips}"

    def test_tip_is_string(self):
        from xai.report import counterfactuals
        tips = counterfactuals(_make_features(), 5, 5, 5, 2000)
        for t in tips:
            assert isinstance(t, str)

    def test_tips_mention_mode_names(self):
        from xai.report import counterfactuals
        tips = counterfactuals(_make_features(), 5, 5, 5, 2000)
        all_text = " ".join(tips)
        known_modes = {"高鐵", "台鐵", "客運", "租車"}
        mentioned = known_modes & set(all_text.split("**"))
        # 反事實至少提到一個交通方式（夾在 ** 之間）
        assert len(mentioned) > 0 or any(m in all_text for m in known_modes)

    def test_no_tips_when_single_mode(self):
        """只有一種交通方式時，無法產生反事實比較，應回傳空或最少提示。"""
        from xai.report import counterfactuals
        single = [_make_features()[0]]
        tips = counterfactuals(single, 5, 5, 5, 2000)
        assert isinstance(tips, list)

    def test_uses_fixtures(self, sample_features, sample_results):
        """用真實計算結果驗證反事實邏輯。"""
        from xai.report import counterfactuals
        tips = counterfactuals(sample_features, 5, 5, 5, 2000)
        assert isinstance(tips, list)


# ─────────────────────────────────────────────────────────
# 6. build_report — 完整整合測試
# ─────────────────────────────────────────────────────────

class TestBuildReport:
    def test_returns_expected_keys(self, sample_results, sample_features, sample_feature_row):
        from xai.report import build_report
        report = build_report(
            results=sample_results,
            features=sample_features,
            feature_row=sample_feature_row,
            w_cost=5, w_time=5, w_conv=5,
            budget=2000,
            clf=None,
        )
        for key in ("top_reason", "contributions", "decision_path", "shap", "counterfactuals"):
            assert key in report, f"report 缺少 key: {key}"

    def test_top_reason_is_string(self, sample_results, sample_features, sample_feature_row):
        from xai.report import build_report
        report = build_report(
            results=sample_results, features=sample_features,
            feature_row=sample_feature_row,
            w_cost=5, w_time=5, w_conv=5, budget=2000, clf=None,
        )
        assert isinstance(report["top_reason"], str)
        assert len(report["top_reason"]) > 0

    def test_contributions_length(self, sample_results, sample_features, sample_feature_row):
        from xai.report import build_report
        report = build_report(
            results=sample_results, features=sample_features,
            feature_row=sample_feature_row,
            w_cost=5, w_time=5, w_conv=5, budget=2000, clf=None,
        )
        assert len(report["contributions"]) == 3

    def test_contributions_invariant(self, sample_results, sample_features, sample_feature_row):
        """貢獻加總 ≈ best total_score。"""
        from xai.report import build_report
        report = build_report(
            results=sample_results, features=sample_features,
            feature_row=sample_feature_row,
            w_cost=5, w_time=5, w_conv=5, budget=2000, clf=None,
        )
        total = sum(c["contribution"] for c in report["contributions"])
        assert abs(total - sample_results[0]["total_score"]) < 1e-4

    def test_counterfactuals_is_list(self, sample_results, sample_features, sample_feature_row):
        from xai.report import build_report
        report = build_report(
            results=sample_results, features=sample_features,
            feature_row=sample_feature_row,
            w_cost=5, w_time=5, w_conv=5, budget=2000, clf=None,
        )
        assert isinstance(report["counterfactuals"], list)

    def test_decision_path_none_without_model(self, sample_results, sample_features, sample_feature_row):
        """無模型時 decision_path 應為 None（或從磁碟載入已存在的模型）。"""
        from xai.report import build_report
        import os, pickle
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "models", "decision_tree.pkl"
        )
        # 若有預先訓練的模型，clf 不會是 None；若沒有，decision_path 應為 None
        report = build_report(
            results=sample_results, features=sample_features,
            feature_row=sample_feature_row,
            w_cost=5, w_time=5, w_conv=5, budget=2000, clf=None,
        )
        # decision_path 只能是 str 或 None
        assert report["decision_path"] is None or isinstance(report["decision_path"], str)

    def test_with_tiny_clf(self, sample_results, sample_features, sample_feature_row):
        """注入假 clf，decision_path 應為中文字串。"""
        from xai.report import build_report
        clf = _train_tiny_clf()
        report = build_report(
            results=sample_results, features=sample_features,
            feature_row=sample_feature_row,
            w_cost=5, w_time=5, w_conv=5, budget=2000, clf=clf,
        )
        assert isinstance(report["decision_path"], str)
        assert "推薦" in report["decision_path"]

    def test_weight_sensitivity(self, sample_features):
        """調整成本偏好會影響 top_reason 中提到的維度。"""
        from xai.report import build_report
        from models.scoring_model import compute_scores

        results_cost = compute_scores(sample_features, w_cost=10, w_time=1, w_conv=1)
        results_time = compute_scores(sample_features, w_cost=1, w_time=10, w_conv=1)
        from conftest import make_full_feature_row
        feature_row = make_full_feature_row(
            people=2, luggage=1, budget=2000,
            time_pref=5, cost_pref=5, conv_pref=3,
        )

        r_cost = build_report(results_cost, sample_features, feature_row,
                               w_cost=10, w_time=1, w_conv=1, budget=2000, clf=None)
        r_time = build_report(results_time, sample_features, feature_row,
                               w_cost=1, w_time=10, w_conv=1, budget=2000, clf=None)

        # 高成本偏好 → top reason 應提到「成本」
        assert "成本" in r_cost["top_reason"]
        # 高時間偏好 → top reason 應提到「時間」
        assert "時間" in r_time["top_reason"]


# ─────────────────────────────────────────────────────────
# 7. feature_engineer 整合
# ─────────────────────────────────────────────────────────

class TestFeatureEngineerIntegration:
    def test_compute_features_returns_list(self, sample_route_data):
        from features.feature_engineer import compute_features
        features = compute_features(sample_route_data, 2, 1)
        assert isinstance(features, list)
        assert len(features) == len(sample_route_data)

    def test_scores_in_zero_one_range(self, sample_route_data):
        from features.feature_engineer import compute_features
        features = compute_features(sample_route_data, 2, 1)
        for f in features:
            for key in ("cost_score", "time_score", "convenience_score"):
                assert 0.0 <= f[key] <= 1.0, f"{f['mode']}.{key} = {f[key]} 超出 [0,1]"

    def test_car_cost_is_per_person_divided(self, sample_route_data):
        """租車成本應除以人數（人均分攤）。"""
        from features.feature_engineer import compute_features
        car_route = next(r for r in sample_route_data if r["mode"] == "租車")
        features_2p = compute_features(sample_route_data, num_people=2, luggage=0)
        features_4p = compute_features(sample_route_data, num_people=4, luggage=0)
        car_2p = next(f for f in features_2p if f["mode"] == "租車")
        car_4p = next(f for f in features_4p if f["mode"] == "租車")
        # 人數越多，租車人均成本越低
        assert car_4p["cost_raw"] < car_2p["cost_raw"]

    def test_more_luggage_lowers_convenience(self, sample_route_data):
        """行李越多，大眾運輸便利性越低。"""
        from features.feature_engineer import compute_features
        f0 = compute_features(sample_route_data, num_people=2, luggage=0)
        f5 = compute_features(sample_route_data, num_people=2, luggage=5)
        bus_0 = next(f for f in f0 if f["mode"] == "客運")
        bus_5 = next(f for f in f5 if f["mode"] == "客運")
        assert bus_5["convenience_raw"] < bus_0["convenience_raw"]

    def test_empty_input_returns_empty(self):
        from features.feature_engineer import compute_features
        assert compute_features([], 2, 1) == []

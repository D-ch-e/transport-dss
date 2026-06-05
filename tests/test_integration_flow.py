"""
整合測試套件 — Transport DSS 期末專題交付版
=============================================
涵蓋：
  1. 完整系統流程（整合測試）
  2. 防呆 / Edge Case
  3. Feature Engineering
  4. Scoring Model
  5. XAI Report
  6. API 失敗 Fallback 穩定性
  7. RouteOption 結構驗證

執行方式：
  python -m pytest tests/test_integration_flow.py -v
  python -m pytest -v
"""

from __future__ import annotations

import math
import os
import sys

# 確保專案根目錄在 PYTHONPATH — 無論從哪個目錄執行 pytest 都可找到模組
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock

# ──────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def static_routes():
    """四種主要交通方式的靜態 RouteOption，符合 feature_engineer 期望的 schema。"""
    return [
        {
            "mode": "高鐵",
            "cost": 700,
            "duration_min": 100,
            "distance_km": 160.0,
            "transfers": 1,
            "frequency_per_day": 120,
            "convenience": 8.0,
            "source": "static",
            "segments": [
                {"mode": "步行/接駁", "from": "台北", "to": "台北高鐵站",
                 "duration_min": 25, "cost": 0, "distance_km": 0.0},
                {"mode": "高鐵", "from": "台北高鐵站", "to": "台中高鐵站",
                 "duration_min": 50, "cost": 700, "distance_km": 160.0},
                {"mode": "步行/接駁", "from": "台中高鐵站", "to": "台中",
                 "duration_min": 25, "cost": 0, "distance_km": 0.0},
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
            "segments": [
                {"mode": "步行/接駁", "from": "台北", "to": "台北車站",
                 "duration_min": 25, "cost": 0, "distance_km": 0.0},
                {"mode": "台鐵", "from": "台北車站", "to": "台中車站",
                 "duration_min": 150, "cost": 375, "distance_km": 160.0},
                {"mode": "步行/接駁", "from": "台中車站", "to": "台中",
                 "duration_min": 25, "cost": 0, "distance_km": 0.0},
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
            "segments": [
                {"mode": "租車", "from": "台北", "to": "台中",
                 "duration_min": 120, "cost": 1800, "distance_km": 160.0},
            ],
        },
    ]


@pytest.fixture
def computed_features(static_routes):
    from features.feature_engineer import compute_features
    return compute_features(static_routes, num_people=2, luggage=1)


@pytest.fixture
def scored_results(computed_features):
    from models.scoring_model import compute_scores
    return compute_scores(computed_features, w_cost=5, w_time=5, w_conv=5)


@pytest.fixture
def feature_row():
    """完整的 29 特徵 DT feature_row（鍵涵蓋 FEATURE_COLS），供 XAI 解釋使用。"""
    from conftest import make_full_feature_row
    return make_full_feature_row()


# ──────────────────────────────────────────────────────────────
# 1. 整合測試：完整系統流程
# ──────────────────────────────────────────────────────────────

class TestFullPipeline:
    """完整流程：routes → features → scores → XAI report。"""

    def test_pipeline_with_static_data_completes_without_exception(
        self, static_routes, feature_row
    ):
        """完整管線在靜態資料下不應拋出未處理例外。"""
        from features.feature_engineer import compute_features
        from models.scoring_model import compute_scores
        from xai.report import build_report

        features = compute_features(static_routes, num_people=2, luggage=1)
        results = compute_scores(features, w_cost=5, w_time=5, w_conv=5)
        report = build_report(
            results=results,
            features=features,
            feature_row=feature_row,
            w_cost=5, w_time=5, w_conv=5,
            budget=2000,
            clf=None,
        )
        assert report is not None, "build_report 應回傳非 None 結果"

    def test_routes_not_empty(self, static_routes):
        """routes 不應為空。"""
        assert len(static_routes) > 0, "routes 不應為空"

    def test_each_route_has_required_fields(self, static_routes):
        """每個 route 至少包含 mode / cost / duration_min。"""
        required = {"mode", "cost", "duration_min"}
        for route in static_routes:
            missing = required - route.keys()
            assert not missing, f"route 缺少欄位 {missing}: {route}"

    def test_features_produced_successfully(self, computed_features):
        """features 可以成功產生（非空）。"""
        assert len(computed_features) > 0, "compute_features 應產生至少一筆特徵"

    def test_feature_scores_in_unit_range(self, computed_features):
        """所有 feature scores 必須在 [0, 1]。"""
        for f in computed_features:
            for key in ("cost_score", "time_score", "convenience_score"):
                val = f[key]
                assert 0.0 <= val <= 1.0, (
                    f"{f['mode']} 的 {key}={val} 超出 [0, 1]"
                )

    def test_scores_produced_and_sorted(self, scored_results):
        """scores 可以產生，且依 total_score 由高到低排序。"""
        assert len(scored_results) > 0, "compute_scores 應產生至少一筆結果"
        for i in range(len(scored_results) - 1):
            assert scored_results[i]["total_score"] >= scored_results[i + 1]["total_score"], (
                "結果未依 total_score 降序排列"
            )

    def test_first_result_is_rank_one(self, scored_results):
        """排序後第一筆 rank 應為 1。"""
        assert scored_results[0]["rank"] == 1, "最佳方案的 rank 應為 1"

    def test_report_has_required_keys(self, scored_results, computed_features, feature_row):
        """report 應包含 explanation / contributions / counterfactuals。"""
        from xai.report import build_report

        report = build_report(
            results=scored_results,
            features=computed_features,
            feature_row=feature_row,
            w_cost=5, w_time=5, w_conv=5,
            budget=2000,
            clf=None,
        )
        for key in ("top_reason", "contributions", "counterfactuals"):
            assert key in report, f"report 缺少必要欄位 '{key}'"

    def test_report_top_reason_is_nonempty_string(self, scored_results, computed_features, feature_row):
        """report 的推薦原因應為非空字串。"""
        from xai.report import build_report

        report = build_report(
            results=scored_results,
            features=computed_features,
            feature_row=feature_row,
            w_cost=5, w_time=5, w_conv=5,
            budget=2000,
            clf=None,
        )
        assert isinstance(report["top_reason"], str), "top_reason 應為 str"
        assert len(report["top_reason"]) > 0, "top_reason 不應為空字串"

    def test_report_contributions_is_list(self, scored_results, computed_features, feature_row):
        """report 的 contributions 應為 list。"""
        from xai.report import build_report

        report = build_report(
            results=scored_results,
            features=computed_features,
            feature_row=feature_row,
            w_cost=5, w_time=5, w_conv=5,
            budget=2000,
            clf=None,
        )
        assert isinstance(report["contributions"], list), "contributions 應為 list"

    def test_report_counterfactuals_is_list(self, scored_results, computed_features, feature_row):
        """report 的 counterfactuals 應為 list，即使沒有反事實也不 crash。"""
        from xai.report import build_report

        report = build_report(
            results=scored_results,
            features=computed_features,
            feature_row=feature_row,
            w_cost=5, w_time=5, w_conv=5,
            budget=2000,
            clf=None,
        )
        assert isinstance(report["counterfactuals"], list), "counterfactuals 應為 list"


# ──────────────────────────────────────────────────────────────
# 2. 防呆測試（Edge Cases）
# ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """異常輸入不應讓系統 crash，應回傳合理錯誤或空結果。"""

    def test_same_origin_and_destination_returns_error(self):
        """起點 == 終點 → 驗證失敗，回傳合理錯誤訊息。"""
        from data.input_validator import validate_input

        valid, msg = validate_input("北車", "北車", num_people=2, luggage=0, budget=1000)
        assert not valid, "起點與終點相同時驗證應回傳 False"
        assert len(msg) > 0, "錯誤訊息不應為空"

    def test_extremely_low_budget_returns_error_or_valid(self):
        """極低預算（budget=1）→ 應合理處理，不 crash。"""
        from data.input_validator import validate_input

        # budget=1 是合法正整數，validator 應接受（限制在服務層）
        valid, msg = validate_input("北車", "101", num_people=1, luggage=0, budget=1)
        # 只要不 crash 即通過；valid 可為 True 或 False，但 msg 應為 str
        assert isinstance(valid, bool), "validate_input 應回傳 bool"
        assert isinstance(msg, str), "validate_input 錯誤訊息應為 str"

    def test_zero_people_returns_error(self):
        """人數 = 0 → 驗證失敗。"""
        from data.input_validator import validate_input

        valid, msg = validate_input("北車", "101", num_people=0, luggage=0, budget=1000)
        assert not valid, "人數為 0 時驗證應回傳 False"

    def test_negative_people_returns_error(self):
        """人數 = -1 → 驗證失敗。"""
        from data.input_validator import validate_input

        valid, msg = validate_input("北車", "101", num_people=-1, luggage=0, budget=1000)
        assert not valid, "人數為負數時驗證應回傳 False"

    def test_unknown_origin_returns_error(self):
        """未知起點 → 驗證失敗。"""
        from data.input_validator import validate_input

        valid, msg = validate_input("火星基地", "101", num_people=1, luggage=0, budget=1000)
        assert not valid, "未知起點時驗證應回傳 False"

    def test_unknown_destination_returns_error(self):
        """未知終點 → 驗證失敗。"""
        from data.input_validator import validate_input

        valid, msg = validate_input("北車", "未知星球", num_people=1, luggage=0, budget=1000)
        assert not valid, "未知終點時驗證應回傳 False"

    def test_empty_route_data_compute_features_returns_empty(self):
        """空 routes → compute_features 應回傳空 list，不 crash。"""
        from features.feature_engineer import compute_features

        result = compute_features([], num_people=2, luggage=1)
        assert result == [], "空 route_data 時 compute_features 應回傳 []"

    def test_empty_features_compute_scores_returns_empty(self):
        """空 features → compute_scores 應回傳空 list，不 crash。"""
        from models.scoring_model import compute_scores

        result = compute_scores([], w_cost=5, w_time=5, w_conv=5)
        assert result == [], "空 features 時 compute_scores 應回傳 []"

    def test_plan_routes_unknown_locations_returns_list(self):
        """未知起訖地 → plan_routes 應回傳 list（可為空），不 crash。"""
        from services.route_planner import plan_routes

        # 完全陌生的地名 → geocode 失敗 → static fallback 也找不到 → 回 []
        result = plan_routes("UNKNOWN_ORIGIN_XYZ", "UNKNOWN_DEST_XYZ")
        assert isinstance(result, list), "plan_routes 對未知地點應回傳 list"


# ──────────────────────────────────────────────────────────────
# 3. Feature Engineering 測試
# ──────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    """compute_features 的數值正確性與邊界條件。"""

    def test_cost_score_in_unit_range(self, computed_features):
        for f in computed_features:
            assert 0.0 <= f["cost_score"] <= 1.0, (
                f"{f['mode']} cost_score={f['cost_score']} 超出 [0, 1]"
            )

    def test_time_score_in_unit_range(self, computed_features):
        for f in computed_features:
            assert 0.0 <= f["time_score"] <= 1.0, (
                f"{f['mode']} time_score={f['time_score']} 超出 [0, 1]"
            )

    def test_convenience_score_in_unit_range(self, computed_features):
        for f in computed_features:
            assert 0.0 <= f["convenience_score"] <= 1.0, (
                f"{f['mode']} convenience_score={f['convenience_score']} 超出 [0, 1]"
            )

    def test_no_nan_in_scores(self, computed_features):
        for f in computed_features:
            for key in ("cost_score", "time_score", "convenience_score"):
                assert not math.isnan(f[key]), f"{f['mode']} {key} 為 NaN"

    def test_no_infinity_in_scores(self, computed_features):
        for f in computed_features:
            for key in ("cost_score", "time_score", "convenience_score"):
                assert math.isfinite(f[key]), f"{f['mode']} {key} 為 infinity"

    def test_more_luggage_decreases_or_maintains_convenience_for_bus(self):
        """行李增加時，客運的便利性調整值應 ≤ 行李少時。"""
        from features.feature_engineer import compute_convenience_feature

        route = {"mode": "客運", "convenience": 5.0, "transfers": 0}
        conv_low = compute_convenience_feature(route, num_people=2, luggage=0)
        conv_high = compute_convenience_feature(route, num_people=2, luggage=5)
        assert conv_high <= conv_low, (
            f"行李增加後客運便利性應 ≤ 行李少時：{conv_high} > {conv_low}"
        )

    def test_car_per_person_cost_decreases_with_more_people(self):
        """租車人均成本應在多人時 ≤ 單人時。"""
        from features.feature_engineer import compute_cost_feature

        route = {"mode": "租車", "cost": 1800}
        cost_1 = compute_cost_feature(route, num_people=1)
        cost_4 = compute_cost_feature(route, num_people=4)
        assert cost_4 <= cost_1, (
            f"租車 4 人均成本 {cost_4} 應 ≤ 單人成本 {cost_1}"
        )

    def test_minmax_normalization_does_not_crash_when_all_same(self):
        """所有 routes cost 相同（max == min）時 Min-Max 不應 crash，應輸出 1.0。"""
        from features.feature_engineer import compute_features

        same_cost_routes = [
            {"mode": "高鐵", "cost": 500, "duration_min": 90,
             "distance_km": 100.0, "transfers": 0, "convenience": 8.0, "source": "static"},
            {"mode": "台鐵", "cost": 500, "duration_min": 90,
             "distance_km": 100.0, "transfers": 0, "convenience": 6.0, "source": "static"},
        ]
        features = compute_features(same_cost_routes, num_people=1, luggage=0)
        for f in features:
            assert f["cost_score"] == 1.0, (
                f"cost 全相同時 cost_score 應為 1.0，得到 {f['cost_score']}"
            )
            assert not math.isnan(f["cost_score"]), "cost_score 不應為 NaN"

    def test_single_route_all_scores_are_one(self):
        """只有一條 route 時，Min-Max 退化：三個 score 都應為 1.0。"""
        from features.feature_engineer import compute_features

        routes = [
            {"mode": "高鐵", "cost": 700, "duration_min": 100,
             "distance_km": 160.0, "transfers": 1, "convenience": 8.0, "source": "static"},
        ]
        features = compute_features(routes, num_people=2, luggage=1)
        assert len(features) == 1
        f = features[0]
        for key in ("cost_score", "time_score", "convenience_score"):
            assert f[key] == 1.0, f"單一 route 的 {key} 應為 1.0，得到 {f[key]}"


# ──────────────────────────────────────────────────────────────
# 4. Scoring Model 測試
# ──────────────────────────────────────────────────────────────

class TestScoringModel:
    """compute_scores 的正確性與權重行為。"""

    def test_scores_not_empty(self, scored_results):
        assert len(scored_results) > 0, "compute_scores 應至少回傳一筆"

    def test_each_result_has_total_score(self, scored_results):
        for r in scored_results:
            assert "total_score" in r, f"結果缺少 total_score: {r}"

    def test_total_score_in_reasonable_range(self, scored_results):
        """total_score 由三個 [0,1] 數值加權，結果也應在 [0, 1]。"""
        for r in scored_results:
            assert 0.0 <= r["total_score"] <= 1.0, (
                f"{r['mode']} total_score={r['total_score']} 超出 [0, 1]"
            )

    def test_results_sorted_by_total_score_descending(self, scored_results):
        for i in range(len(scored_results) - 1):
            assert scored_results[i]["total_score"] >= scored_results[i + 1]["total_score"], (
                "結果未依 total_score 降序排列"
            )

    def test_weights_are_normalized(self, scored_results):
        """weights 加總應等於 1（允許四捨五入後的浮點誤差）。"""
        for r in scored_results:
            w = r["weights"]
            total_w = w["cost"] + w["time"] + w["convenience"]
            assert abs(total_w - 1.0) < 1e-3, (
                f"{r['mode']} 的 weights 加總 {total_w} ≠ 1"
            )

    def test_weight_change_affects_scores_or_ranking(self, computed_features):
        """改變權重後，total_score 或排名應有所不同。"""
        from models.scoring_model import compute_scores

        results_balanced = compute_scores(computed_features, w_cost=1, w_time=1, w_conv=1)
        results_cost_heavy = compute_scores(computed_features, w_cost=10, w_time=1, w_conv=1)

        scores_balanced = [r["total_score"] for r in results_balanced]
        scores_cost_heavy = [r["total_score"] for r in results_cost_heavy]
        # 至少有一個分數改變
        assert scores_balanced != scores_cost_heavy, (
            "權重改變後所有 total_score 相同，不合理"
        )

    def test_all_zero_weights_falls_back_gracefully(self, computed_features):
        """所有權重為 0 → 應 fallback 為等權重，不 crash，不出現 NaN。"""
        from models.scoring_model import compute_scores

        results = compute_scores(computed_features, w_cost=0, w_time=0, w_conv=0)
        assert len(results) > 0, "零權重時也應回傳結果"
        for r in results:
            assert not math.isnan(r["total_score"]), "零權重時 total_score 不應為 NaN"

    def test_no_nan_or_infinity_in_total_score(self, scored_results):
        for r in scored_results:
            assert math.isfinite(r["total_score"]), (
                f"{r['mode']} total_score={r['total_score']} 不是有限數"
            )


# ──────────────────────────────────────────────────────────────
# 5. XAI Report 測試
# ──────────────────────────────────────────────────────────────

class TestXAIReport:
    """build_report 的結構、數值正確性與降級行為。"""

    def test_report_is_dict(self, scored_results, computed_features, feature_row):
        from xai.report import build_report

        report = build_report(
            results=scored_results, features=computed_features,
            feature_row=feature_row, w_cost=5, w_time=5, w_conv=5,
            budget=2000, clf=None,
        )
        assert isinstance(report, dict), "build_report 應回傳 dict"

    def test_report_has_top_reason(self, scored_results, computed_features, feature_row):
        from xai.report import build_report

        report = build_report(
            results=scored_results, features=computed_features,
            feature_row=feature_row, w_cost=5, w_time=5, w_conv=5,
            budget=2000, clf=None,
        )
        assert "top_reason" in report and report["top_reason"], (
            "report 應包含非空的 top_reason"
        )

    def test_contributions_match_scoring_model(self, scored_results, computed_features, feature_row):
        """contributions 三項加總應等於 total_score（允許浮點誤差）。"""
        from xai.report import build_report

        report = build_report(
            results=scored_results, features=computed_features,
            feature_row=feature_row, w_cost=5, w_time=5, w_conv=5,
            budget=2000, clf=None,
        )
        contributions = report["contributions"]
        assert len(contributions) == 3, "contributions 應有三個維度"

        total_contribution = sum(c["contribution"] for c in contributions)
        best_score = scored_results[0]["total_score"]
        assert abs(total_contribution - best_score) < 1e-3, (
            f"contributions 加總 {total_contribution:.4f} ≠ total_score {best_score:.4f}"
        )

    def test_each_contribution_has_required_fields(self, scored_results, computed_features, feature_row):
        from xai.report import build_report

        report = build_report(
            results=scored_results, features=computed_features,
            feature_row=feature_row, w_cost=5, w_time=5, w_conv=5,
            budget=2000, clf=None,
        )
        for c in report["contributions"]:
            for field in ("dim", "dim_zh", "weight", "score", "contribution"):
                assert field in c, f"contribution 缺少欄位 '{field}': {c}"

    def test_counterfactuals_is_list_and_no_crash(self, scored_results, computed_features, feature_row):
        """counterfactuals 至少回傳 list，即使沒有反事實也不 crash。"""
        from xai.report import build_report

        report = build_report(
            results=scored_results, features=computed_features,
            feature_row=feature_row, w_cost=5, w_time=5, w_conv=5,
            budget=2000, clf=None,
        )
        assert isinstance(report["counterfactuals"], list), (
            "counterfactuals 應為 list"
        )

    def test_report_without_clf_does_not_crash(self, scored_results, computed_features, feature_row):
        """clf=None（模型不存在）時 build_report 不應 crash。"""
        from xai.report import build_report

        report = build_report(
            results=scored_results, features=computed_features,
            feature_row=feature_row, w_cost=5, w_time=5, w_conv=5,
            budget=2000, clf=None,
        )
        assert isinstance(report, dict), "clf=None 時 build_report 仍應回傳 dict"

    def test_shap_missing_degrades_gracefully(self, scored_results, computed_features, feature_row):
        """shap 套件不存在時 build_report 不應 crash，shap 欄位應為 None。"""
        from xai.report import build_report

        with patch.dict("sys.modules", {"shap": None}):
            report = build_report(
                results=scored_results, features=computed_features,
                feature_row=feature_row, w_cost=5, w_time=5, w_conv=5,
                budget=2000, clf=None,
            )
        assert isinstance(report, dict), "shap 不可用時 build_report 仍應回傳 dict"
        # shap 欄位存在且應為 None（降級）
        assert report.get("shap") is None, "shap 不可用時 report['shap'] 應為 None"

    def test_decision_path_returns_string_or_none(self, scored_results, computed_features, feature_row):
        """decision_path 鍵應存在，值為 str 或 None。"""
        from xai.report import build_report

        report = build_report(
            results=scored_results, features=computed_features,
            feature_row=feature_row, w_cost=5, w_time=5, w_conv=5,
            budget=2000, clf=None,
        )
        dp = report.get("decision_path")
        assert dp is None or isinstance(dp, str), (
            f"decision_path 應為 str 或 None，得到 {type(dp)}"
        )


# ──────────────────────────────────────────────────────────────
# 6. API 失敗 Fallback 穩定性測試
# ──────────────────────────────────────────────────────────────

class TestAPIFallbackStability:
    """API 失敗時系統應能 fallback 到靜態資料並完成整個管線。"""

    @pytest.fixture(autouse=True)
    def disable_realtime(self, monkeypatch):
        """所有本 class 的測試都在 realtime=False 的環境下執行。
        SETTINGS 是 frozen dataclass，需 patch 使用它的模組層參照。
        """
        from config.settings import Settings, CacheTTL
        offline_settings = Settings(
            google_maps_api_key="",
            tdx_client_id="",
            tdx_client_secret="",
            enable_realtime=False,
            ttl=CacheTTL(),
        )
        monkeypatch.setattr("config.settings.SETTINGS", offline_settings)
        monkeypatch.setattr("services.route_planner.SETTINGS", offline_settings, raising=False)
        yield

    def _run_full_pipeline(self, routes, feature_row_dict):
        from features.feature_engineer import compute_features
        from models.scoring_model import compute_scores
        from xai.report import build_report

        features = compute_features(routes, num_people=2, luggage=1)
        results = compute_scores(features, w_cost=5, w_time=5, w_conv=5)
        report = build_report(
            results=results, features=features,
            feature_row=feature_row_dict,
            w_cost=5, w_time=5, w_conv=5,
            budget=2000, clf=None,
        )
        return features, results, report

    def test_google_api_fail_does_not_crash(self, static_routes, feature_row):
        """Google Maps API 失敗時使用靜態資料，不 crash。"""
        with patch("data.clients.google_client.GoogleMapsClient.directions",
                   side_effect=Exception("Google API 模擬失敗")):
            features, results, report = self._run_full_pipeline(static_routes, feature_row)
        assert len(features) > 0, "Google API 失敗後 features 不應為空"

    def test_tdx_api_fail_does_not_crash(self, static_routes, feature_row):
        """TDX API 失敗時使用靜態資料，不 crash。"""
        with patch("data.clients.tdx_client.TDXClient.thsr_od_timetable",
                   side_effect=Exception("TDX API 模擬失敗")):
            features, results, report = self._run_full_pipeline(static_routes, feature_row)
        assert len(results) > 0, "TDX API 失敗後 results 不應為空"

    def test_youbike_api_fail_does_not_crash(self, static_routes, feature_row):
        """YouBike API 失敗時使用靜態資料，不 crash。"""
        with patch("data.clients.youbike_client.YouBikeClient.nearest_stations",
                   side_effect=Exception("YouBike API 模擬失敗")):
            features, results, report = self._run_full_pipeline(static_routes, feature_row)
        assert len(report["contributions"]) > 0, "YouBike API 失敗後 report 仍應有 contributions"

    def test_fallback_to_static_data_still_produces_routes(self):
        """plan_routes 在 API 金鑰缺失時應 fallback 到靜態資料，可產生 routes。"""
        from data.api_client import get_route_data

        routes = get_route_data("台北", "台中")
        # 靜態資料若存在應有路線；若無則只要不 crash
        assert isinstance(routes, list), "get_route_data 應回傳 list"

    def test_fallback_routes_complete_feature_engineering(self):
        """靜態 fallback 後的 routes 可以完成 feature engineering。"""
        from data.api_client import get_route_data
        from features.feature_engineer import compute_features

        routes = get_route_data("台北", "台中")
        if not routes:
            pytest.skip("靜態資料無 台北→台中 路線，跳過此測試")

        features = compute_features(routes, num_people=2, luggage=1)
        assert len(features) > 0, "靜態 fallback routes 應能產生 features"

    def test_fallback_routes_complete_scoring(self):
        """靜態 fallback 後的 routes 可以完成 scoring。"""
        from data.api_client import get_route_data
        from features.feature_engineer import compute_features
        from models.scoring_model import compute_scores

        routes = get_route_data("台北", "台中")
        if not routes:
            pytest.skip("靜態資料無 台北→台中 路線，跳過此測試")

        features = compute_features(routes, num_people=2, luggage=1)
        results = compute_scores(features, w_cost=5, w_time=5, w_conv=5)
        assert len(results) > 0, "靜態 fallback 後 scoring 應有結果"

    def test_fallback_routes_complete_xai_report(self):
        """靜態 fallback 後的 routes 可以產生 XAI report。"""
        from data.api_client import get_route_data
        from features.feature_engineer import compute_features
        from models.scoring_model import compute_scores
        from xai.report import build_report

        routes = get_route_data("台北", "台中")
        if not routes:
            pytest.skip("靜態資料無 台北→台中 路線，跳過此測試")

        features = compute_features(routes, num_people=2, luggage=1)
        results = compute_scores(features)
        from conftest import make_full_feature_row
        fr = make_full_feature_row()
        report = build_report(
            results=results, features=features, feature_row=fr,
            w_cost=5, w_time=5, w_conv=5, budget=2000, clf=None,
        )
        assert isinstance(report, dict), "靜態 fallback 後 XAI report 應為 dict"

    def test_tests_do_not_require_real_api_key(self):
        """確認測試不依賴真實 API key（環境變數空值時不 crash）。"""
        with patch.dict(os.environ, {
            "GOOGLE_MAPS_API_KEY": "",
            "TDX_CLIENT_ID": "",
            "TDX_CLIENT_SECRET": "",
        }):
            from config.settings import load_settings
            settings = load_settings()
            assert not settings.has_google, "沒有 GOOGLE_MAPS_API_KEY 時 has_google 應為 False"
            assert not settings.has_tdx, "沒有 TDX 金鑰時 has_tdx 應為 False"


# ──────────────────────────────────────────────────────────────
# 7. RouteOption 結構測試
# ──────────────────────────────────────────────────────────────

class TestRouteOptionSchema:
    """驗證 route planner 回傳格式的完整性。"""

    REQUIRED_TOP_LEVEL = {"mode", "cost", "duration_min", "distance_km",
                          "transfers", "convenience", "source", "segments"}

    REQUIRED_SEGMENT = {"mode", "from", "to", "duration_min"}

    def test_static_routes_have_all_required_fields(self, static_routes):
        """靜態 fixture 的每個 route 都應含所有必要欄位。"""
        for route in static_routes:
            missing = self.REQUIRED_TOP_LEVEL - route.keys()
            assert not missing, f"{route['mode']} 缺少欄位 {missing}"

    def test_mode_is_string(self, static_routes):
        for route in static_routes:
            assert isinstance(route["mode"], str), f"mode 應為 str: {route}"

    def test_cost_is_numeric(self, static_routes):
        for route in static_routes:
            assert isinstance(route["cost"], (int, float)), (
                f"cost 應為數值: {route}"
            )

    def test_duration_min_is_positive(self, static_routes):
        for route in static_routes:
            assert route["duration_min"] > 0, (
                f"{route['mode']} duration_min 應 > 0，得到 {route['duration_min']}"
            )

    def test_distance_km_is_non_negative(self, static_routes):
        for route in static_routes:
            assert route["distance_km"] >= 0, (
                f"{route['mode']} distance_km 應 ≥ 0"
            )

    def test_transfers_is_non_negative_int(self, static_routes):
        for route in static_routes:
            assert isinstance(route["transfers"], int), (
                f"{route['mode']} transfers 應為 int"
            )
            assert route["transfers"] >= 0, (
                f"{route['mode']} transfers 應 ≥ 0"
            )

    def test_segments_is_list(self, static_routes):
        for route in static_routes:
            assert isinstance(route["segments"], list), (
                f"{route['mode']} segments 應為 list"
            )

    def test_each_segment_has_required_fields(self, static_routes):
        for route in static_routes:
            for seg in route["segments"]:
                missing = self.REQUIRED_SEGMENT - seg.keys()
                assert not missing, (
                    f"{route['mode']} 的 segment 缺少欄位 {missing}: {seg}"
                )

    def test_segment_duration_sum_matches_top_level(self, static_routes):
        """segments 的 duration_min 加總應等於頂層 duration_min（允許微小誤差）。"""
        for route in static_routes:
            segs = route["segments"]
            if not segs:
                continue
            seg_total = sum(s.get("duration_min", 0) for s in segs)
            assert abs(seg_total - route["duration_min"]) <= 1, (
                f"{route['mode']} segments 時間加總 {seg_total} ≠ "
                f"頂層 duration_min {route['duration_min']}"
            )

    def test_segment_cost_sum_matches_top_level(self, static_routes):
        """segments 的 cost 加總應等於頂層 cost（允許微小誤差）。"""
        for route in static_routes:
            segs = route["segments"]
            if not segs:
                continue
            seg_total = sum(s.get("cost", 0) for s in segs)
            assert abs(seg_total - route["cost"]) <= 1, (
                f"{route['mode']} segments cost 加總 {seg_total} ≠ "
                f"頂層 cost {route['cost']}"
            )

    def test_transfers_is_reasonable_relative_to_segment_count(self, static_routes):
        """transfers 不應大於 segments 數量（每段之間才有可能轉乘）。"""
        for route in static_routes:
            segs = route["segments"]
            max_possible = max(0, len(segs) - 1) if segs else 0
            assert route["transfers"] <= max_possible or max_possible == 0, (
                f"{route['mode']} transfers={route['transfers']} 超過最大可能轉乘數 {max_possible}"
            )

    def test_get_route_data_returns_valid_schema(self):
        """api_client.get_route_data 回傳格式應含基本欄位。"""
        from data.api_client import get_route_data

        routes = get_route_data("台北", "台中")
        if not routes:
            pytest.skip("靜態資料無 台北→台中 路線")

        required = {"mode", "cost", "duration_min", "distance_km", "transfers", "source"}
        for r in routes:
            missing = required - r.keys()
            assert not missing, f"get_route_data 回傳 route 缺少欄位 {missing}"

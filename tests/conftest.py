"""
pytest 共用 fixtures — 載入靜態資料、建立假的 RouteOption / features / results。
"""

import sys
import os

# 確保專案根目錄在 PYTHONPATH 中，無論從哪個目錄執行 pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ─────────────────────────────────────────────────────────
# 假的 RouteOption（plan_routes 的輸出格式）
# ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_route_data():
    """四種主要交通方式的 RouteOption，符合 feature_engineer 期望的 schema。"""
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
            "fetched_at": "2026-05-31T10:00:00+08:00",
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
            "fetched_at": "2026-05-31T10:00:00+08:00",
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


@pytest.fixture
def sample_features(sample_route_data):
    """由 compute_features 產生的 features list（以 fixture 的 route_data 計算）。"""
    from features.feature_engineer import compute_features
    return compute_features(sample_route_data, num_people=2, luggage=1)


@pytest.fixture
def sample_results(sample_features):
    """由 compute_scores 產生的 results list（等權重）。"""
    from models.scoring_model import compute_scores
    return compute_scores(sample_features, w_cost=5, w_time=5, w_conv=5)


def make_full_feature_row(
    people: int = 2,
    luggage: int = 1,
    budget: float = 2000.0,
    time_pref: int = 5,
    cost_pref: int = 5,
    conv_pref: int = 3,
) -> dict:
    """
    建立完整的 29 特徵 DT feature_row（鍵涵蓋 FEATURE_COLS）。

    XAI 的 decision_path_zh / shap_contributions 需要決策樹的完整特徵
    （avg_transfer、speed、ml_*、transport_tools_norm_* one-hot 等），
    與正式環境 ui/app.py 透過 build_dt_feature_row 建立的格式一致。
    這裡合成一列 ML route_row 餵入同一個 helper，確保特徵完整。
    """
    import pandas as pd
    from features.feature_engineer import build_dt_feature_row
    from models.decision_tree import FEATURE_COLS

    route_row = pd.Series({
        "avg_transfer": 1,
        "speed": 48.0,            # 160 km / 200 min ≈ 48 km/h
        "avg_cost": 375.0,
        "avg_time": 200.0,
        "avg_walking": 50.0,
        "transport_tools_norm": "LONG DISTANCE TRAIN",  # 對應一個 one-hot 欄位
        "has_car12": 0, "has_car24": 0, "has_car35": 0, "has_car57": 0,
    })
    return build_dt_feature_row(
        route_row, people=people, luggage=luggage, budget=budget,
        time_pref=time_pref, cost_pref=cost_pref, conv_pref=conv_pref,
        feature_cols=FEATURE_COLS,
    )


@pytest.fixture
def sample_feature_row():
    """決策樹 XAI 解釋使用的完整 feature_row dict（鍵涵蓋 FEATURE_COLS）。"""
    return make_full_feature_row()

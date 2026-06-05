"""
M3 · Feature Engineer — 特徵工程模組

功能：
- 計算各交通方式的成本特徵（cost）— 租車人均分攤
- 計算各交通方式的時間特徵（time）
- 計算各交通方式的便利性特徵（convenience）— 連續線性懲罰
- 進行 Min-Max 正規化
- 將原始資料轉換成模型可比較的格式

便利性調整邏輯（來自 Colab Score_Ranking_System）：
- 行李懲罰：客運 ×0.5 > 台鐵 ×0.3 > 高鐵 ×0.25 > 租車 ×0.1
- 人數懲罰：客運 ×0.25 > 台鐵 ×0.15 > 高鐵 ×0.1 > 租車 ×(-0.05)（加分）
"""

# ── 便利性調整係數（Colab 版本） ──
# 行李懲罰權重（行李越多，大眾運輸越不方便）
LUGGAGE_PENALTY = {
    "客運": 0.5,
    "公車": 0.5,
    "大眾運輸": 0.4,
    "台鐵": 0.3,
    "捷運": 0.3,
    "高鐵": 0.25,
    "租車": 0.1,
}

# 人數懲罰權重（人數越多，大眾運輸越不方便；租車則加分）
PEOPLE_PENALTY = {
    "客運": 0.25,
    "公車": 0.25,
    "大眾運輸": 0.2,
    "台鐵": 0.15,
    "捷運": 0.15,
    "高鐵": 0.1,
    "租車": -0.05,  # 負值 = 加分
}


def _base_mode(mode: str) -> str:
    """取出複合模式的主交通方式（如 '高鐵+捷運' → '高鐵'）。"""
    return mode.split("+", 1)[0] if "+" in mode else mode


def compute_cost_feature(route: dict, num_people: int) -> float:
    """
    計算成本特徵。

    - 大眾運輸（高鐵、台鐵、客運、捷運）：票價即為每人成本
    - 租車：總租車費用 ÷ 人數 = 人均成本
    - 複合方案：取主交通方式判定
    """
    if _base_mode(route["mode"]) == "租車":
        return route["cost"] / num_people
    else:
        return route["cost"]


def compute_time_feature(route: dict) -> float:
    """
    計算時間特徵。

    直接使用預估交通時間（分鐘）。
    """
    return route["duration_min"]


def compute_convenience_feature(route: dict, num_people: int, luggage: int) -> float:
    """
    計算調整後的便利性分數（Colab 連續線性懲罰版本）。

    以資料中的基礎便利性為起點，根據行李與人數施加線性懲罰：
    - penalty = luggage × LUGGAGE_PENALTY[mode] + people × PEOPLE_PENALTY[mode]
    - 結果 = max(1, base - penalty)

    Parameters
    ----------
    route : dict
        交通方式資料，需包含 "mode" 與 "convenience" 欄位
    num_people : int
        人數
    luggage : int
        行李件數

    Returns
    -------
    float
        調整後的便利性分數（≥ 1）
    """
    base = route.get("convenience", 5)
    mode = _base_mode(route["mode"])

    # 行李懲罰
    penalty = luggage * LUGGAGE_PENALTY.get(mode, 0.2)

    # 人數懲罰（租車為負值 = 加分）
    penalty += num_people * PEOPLE_PENALTY.get(mode, 0.1)

    # 轉乘懲罰：每多一次轉乘扣 0.5 分（直達 transfers=0 不扣）
    transfers = max(0, int(route.get("transfers", 0)))
    penalty += 0.5 * transfers

    return max(1.0, base - penalty)


def compute_features(
    route_data: list[dict],
    num_people: int,
    luggage: int,
) -> list[dict]:
    """
    計算所有交通方式的三大特徵，並進行 Min-Max 正規化。

    Parameters
    ----------
    route_data : list[dict]
        各交通方式的原始資料（來自 api_client.get_route_data）
    num_people : int
        人數
    luggage : int
        行李件數

    Returns
    -------
    list[dict]
        每個 dict 包含：
        - mode: 交通方式名稱
        - cost_raw: 原始人均成本
        - time_raw: 原始時間（分鐘）
        - convenience_raw: 調整後便利性分數
        - cost_score: 正規化成本分數（0~1，越高越好，即成本越低）
        - time_score: 正規化時間分數（0~1，越高越好，即時間越短）
        - convenience_score: 正規化便利性分數（0~1，越高越好）
    """
    if not route_data:
        return []

    # 計算原始值
    features = []
    for route in route_data:
        features.append({
            "mode": route["mode"],
            "cost_raw": compute_cost_feature(route, num_people),
            "time_raw": compute_time_feature(route),
            "convenience_raw": compute_convenience_feature(
                route, num_people, luggage
            ),
            # 透傳給 UI 渲染分段路線
            "segments": route.get("segments", []),
            "transfers": int(route.get("transfers", 0)),
            "distance_km": route.get("distance_km", 0.0),
        })

    # Min-Max 正規化
    costs = [f["cost_raw"] for f in features]
    times = [f["time_raw"] for f in features]
    convs = [f["convenience_raw"] for f in features]

    cost_min, cost_max = min(costs), max(costs)
    time_min, time_max = min(times), max(times)
    conv_min, conv_max = min(convs), max(convs)

    for f in features:
        # 成本：越低越好 → 反轉
        if cost_max == cost_min:
            f["cost_score"] = 1.0
        else:
            f["cost_score"] = 1.0 - (f["cost_raw"] - cost_min) / (cost_max - cost_min)

        # 時間：越短越好 → 反轉
        if time_max == time_min:
            f["time_score"] = 1.0
        else:
            f["time_score"] = 1.0 - (f["time_raw"] - time_min) / (time_max - time_min)

        # 便利性：越高越好
        if conv_max == conv_min:
            f["convenience_score"] = 1.0
        else:
            f["convenience_score"] = (f["convenience_raw"] - conv_min) / (
                conv_max - conv_min
            )

    return features


def build_dt_feature_row(
    route_row: "pd.Series",
    people: int,
    luggage: int,
    budget: float,
    time_pref: int,
    cost_pref: int,
    conv_pref: int,
    feature_cols: list,
) -> dict:
    """
    為 DT 模型建立單條候選路線的 29 個特徵 dict。
    供 xai/report.py 的 decision_path_zh / shap_contributions 使用。

    Parameters
    ----------
    route_row : pd.Series
        來自 taiwan_route_dataset_ml.csv 的一列
    feature_cols : list
        FEATURE_COLS（從 decision_tree.py 取得），用於 one-hot 欄位列舉
    """
    import pandas as pd

    is_car = any(
        route_row.get(col, 0) == 1
        for col in ["has_car12", "has_car24", "has_car35", "has_car57"]
    )
    effective_cost = route_row["avg_cost"] / people if is_car else route_row["avg_cost"]

    features: dict = {
        "avg_transfer": route_row["avg_transfer"],
        "speed": route_row["speed"],
        "people": people,
        "luggage": luggage,
        "budget": budget,
        "time_pref": time_pref,
        "cost_pref": cost_pref,
        "conv_pref": conv_pref,
        "ml_budget_remaining": budget - effective_cost,
        "ml_budget_use_ratio": effective_cost / (budget + 1e-5),
        "ml_time_cost_score": route_row["avg_time"] * time_pref,
        "ml_walking_cost_score": route_row["avg_walking"] * conv_pref,
        "ml_money_cost_score": effective_cost * cost_pref,
    }

    for col in feature_cols:
        if col.startswith("transport_tools_norm_"):
            features[col] = 0
    target_col = f"transport_tools_norm_{route_row['transport_tools_norm']}"
    if target_col in features:
        features[target_col] = 1

    return features
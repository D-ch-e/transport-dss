"""
M4 · Decision Tree — 決策樹模型模組

使用組員預訓練的 DecisionTreeClassifier（29 特徵，binary best/not_best，準確率 92%）。

推論流程：
1. 從 data/ml/taiwan_route_dataset_ml.csv 取得 O-D 對的所有候選路線
2. 對每條候選路線建立 29 個特徵（路線統計 + 使用者情境 + one-hot 交通類型）
3. clf.predict_proba() 得到每條候選路線為「最佳」的機率
4. 套用物理限制過濾後，依機率排序回傳 top-k
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 模型與路線資料路徑
_ML_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml")

# 16 種交通工具組合的中文對照
TRANSPORT_MODE_ZH = {
    "BICYCLE": "自行車",
    "BUS": "公車/客運",
    "BUS|HIGH SPEED TRAIN": "公車+高鐵",
    "BUS|HIGH SPEED TRAIN|SUBWAY": "公車+高鐵+捷運",
    "BUS|LONG DISTANCE TRAIN": "公車+台鐵",
    "BUS|LONG DISTANCE TRAIN|SUBWAY": "公車+台鐵+捷運",
    "BUS|SUBWAY": "公車+捷運",
    "CAR|CAR_12": "租車(1-2人)",
    "CAR|CAR_24": "租車(2-4人)",
    "CAR|CAR_35": "租車(3-5人)",
    "CAR|CAR_57": "租車(5-7人)",
    "HIGH SPEED TRAIN|LONG DISTANCE TRAIN|SUBWAY": "高鐵+台鐵+捷運",
    "LONG DISTANCE TRAIN": "台鐵",
    "LONG DISTANCE TRAIN|SUBWAY": "台鐵+捷運",
    "SUBWAY": "捷運",
    "WALKING": "步行",
}

# xai/report.py 向後相容 alias
MODE_EN_TO_ZH = TRANSPORT_MODE_ZH

# 模組層級快取
_clf_cache: DecisionTreeClassifier | None = None
_feature_cols_cache: list[str] | None = None
_ml_table_cache: pd.DataFrame | None = None
_lookup_table_cache: pd.DataFrame | None = None

# FEATURE_COLS 在模組載入時讀取（供 xai/report.py import）
def _load_feature_cols() -> list[str]:
    global _feature_cols_cache
    if _feature_cols_cache is None:
        path = os.path.join(_ML_DIR, "feature_columns.pkl")
        if os.path.exists(path):
            _feature_cols_cache = joblib.load(path)
        else:
            _feature_cols_cache = []
    return _feature_cols_cache


FEATURE_COLS: list[str] = _load_feature_cols()


def load_trained_model() -> DecisionTreeClassifier | None:
    """載入預訓練決策樹模型（data/ml/decision_tree_model.pkl）。"""
    global _clf_cache
    if _clf_cache is not None:
        return _clf_cache

    path = os.path.join(_ML_DIR, "decision_tree_model.pkl")
    if not os.path.exists(path):
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _clf_cache = joblib.load(path)
    return _clf_cache


def _load_tables() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """載入 ML 路線表與 lookup 表，帶模組層級快取。"""
    global _ml_table_cache, _lookup_table_cache

    if _ml_table_cache is None:
        ml_path = os.path.join(_ML_DIR, "taiwan_route_dataset_ml.csv")
        if os.path.exists(ml_path):
            _ml_table_cache = pd.read_csv(ml_path)

    if _lookup_table_cache is None:
        lu_path = os.path.join(_ML_DIR, "taiwan_route_dataset_lookup.csv")
        if os.path.exists(lu_path):
            _lookup_table_cache = pd.read_csv(lu_path)

    return _ml_table_cache, _lookup_table_cache


def _build_feature_row(
    route_row: pd.Series,
    people: int,
    luggage: int,
    budget: float,
    time_pref: int,
    cost_pref: int,
    conv_pref: int,
) -> dict:
    """為單條候選路線建立 29 個特徵的 dict。"""
    is_car = any(
        route_row.get(col, 0) == 1
        for col in ["has_car12", "has_car24", "has_car35", "has_car57"]
    )
    effective_cost = route_row["avg_cost"] / people if is_car else route_row["avg_cost"]

    features: dict = {
        "avg_transfer": route_row["avg_transfer"],
        "speed": route_row["speed"],  # 直接使用 CSV 預計算值，與訓練分佈一致
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

    # 所有 one-hot 欄位預設 0，當前交通工具設為 1
    for col in FEATURE_COLS:
        if col.startswith("transport_tools_norm_"):
            features[col] = 0
    target_col = f"transport_tools_norm_{route_row['transport_tools_norm']}"
    if target_col in features:
        features[target_col] = 1

    return features


def _apply_physical_constraints(
    mode: str,
    prob: float,
    avg_distance: float,
    people: int,
    luggage: int,
    effective_cost: float,
    budget: float,
) -> float:
    """套用物理限制，不符合條件的候選直接設機率為 0。"""
    if effective_cost > budget:
        return 0.0
    if mode == "BICYCLE" and (people >= 3 or luggage >= 1 or avg_distance > 12):
        return 0.0
    if mode == "WALKING" and (luggage >= 1 or avg_distance > 4):
        return 0.0
    return prob


def predict_best_mode(
    people: int,
    luggage: int,
    budget: int,
    time_pref: int,
    cost_pref: int,
    conv_pref: int,
    distance_km: float,          # 保留舊簽名相容性，內部不使用
    clf: DecisionTreeClassifier | None = None,
    origin: str | None = None,
    destination: str | None = None,
    top_k: int = 3,
) -> dict | None:
    """
    使用預訓練決策樹預測最佳交通方式。

    origin / destination 須提供才能執行新版路線排名推論。
    若其中一個為 None，回傳 None（UI 有 if dt_result: 保護）。

    Returns
    -------
    dict | None
        {
          "mode_en": str,
          "mode_zh": str,
          "probabilities": {zh_name: prob, ...},   # top_k，供 UI badge 顯示
          "top_k": [
              {
                "rank": int, "mode_en": str, "mode_zh": str,
                "prob": float, "avg_cost": float, "avg_time": float,
                "avg_transfer": float, "avg_distance": float,
                "route_names": list[str],
                "_route_row": pd.Series,  # 內部用，供 XAI feature_row 建構
              }
          ]
        }
    """
    if origin is None or destination is None:
        return None

    if clf is None:
        clf = load_trained_model()
    if clf is None:
        return None

    if not FEATURE_COLS:
        return None

    ml_table, lookup_table = _load_tables()
    if ml_table is None:
        return None

    candidates = ml_table[
        (ml_table["origin"] == origin) & (ml_table["destination"] == destination)
    ]
    if candidates.empty:
        return None

    best_class_idx = int(np.where(clf.classes_ == 1)[0][0])
    results = []

    for _, row in candidates.iterrows():
        mode = row["transport_tools_norm"]
        is_car = any(
            row.get(col, 0) == 1
            for col in ["has_car12", "has_car24", "has_car35", "has_car57"]
        )
        effective_cost = row["avg_cost"] / people if is_car else row["avg_cost"]

        feature_dict = _build_feature_row(
            row, people, luggage, float(budget), time_pref, cost_pref, conv_pref
        )
        X = pd.DataFrame([feature_dict])[FEATURE_COLS]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = float(clf.predict_proba(X)[0][best_class_idx])

        prob = _apply_physical_constraints(
            mode, prob, row["avg_distance"], people, luggage, effective_cost, budget
        )

        results.append({
            "mode_en": mode,
            "mode_zh": TRANSPORT_MODE_ZH.get(mode, mode),
            "prob": prob,
            "avg_cost": round(float(row["avg_cost"]), 1),
            "avg_time": round(float(row["avg_time"]), 1),
            "avg_transfer": round(float(row["avg_transfer"]), 2),
            "avg_distance": round(float(row["avg_distance"]), 1),
            "_route_row": row,
        })

    results.sort(key=lambda x: x["prob"], reverse=True)
    top = results[:top_k]

    # 補充 lookup 路線名稱
    for item in top:
        item["rank"] = top.index(item) + 1
        if lookup_table is not None:
            match = lookup_table[
                (lookup_table["origin"] == origin)
                & (lookup_table["destination"] == destination)
                & (lookup_table["transport_tools_norm"] == item["mode_en"])
            ]
            if not match.empty:
                import ast
                raw = match.iloc[0]["route_name"]
                try:
                    names = ast.literal_eval(raw) if isinstance(raw, str) else raw
                except Exception:
                    names = [raw]
                item["route_names"] = list(dict.fromkeys(names))  # 去重保序
            else:
                item["route_names"] = []

    prob_dict = {
        item["mode_zh"]: round(item["prob"], 4)
        for item in top
        if item["prob"] > 0
    }

    return {
        "mode_en": top[0]["mode_en"],
        "mode_zh": top[0]["mode_zh"],
        "probabilities": prob_dict,
        "top_k": top,
    }

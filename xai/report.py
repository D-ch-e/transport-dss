"""
M5 · XAI Report — 可解釋層模組

將既有評分管線與決策樹模型的訊號封裝成統一的中文解釋物件，供 UI 顯示：
- 加權貢獻分解（精確解釋，與 scoring_model 一致）
- 決策樹 → 中文 IF-THEN 決策路徑
- SHAP 特徵推力（額外解釋決策樹，缺套件時優雅降級）
- 反事實提示（預算門檻、偏好權重）

主要入口：build_report(...)
"""

import numpy as np
import pandas as pd

from models.scoring_model import compute_scores
from models.decision_tree import (
    FEATURE_COLS,
    MODE_EN_TO_ZH,
    load_trained_model,
)

# 維度英中對照
_DIM_ZH = {"cost": "成本", "time": "時間", "convenience": "便利性"}

# 決策樹特徵英中對照（用於決策路徑與 SHAP）
_FEATURE_ZH = {
    # 使用者情境
    "people": "人數",
    "luggage": "行李量",
    "budget": "預算",
    "time_pref": "時間偏好",
    "cost_pref": "成本偏好",
    "conv_pref": "便利性偏好",
    # 路線統計
    "avg_transfer": "平均轉乘次數",
    "speed": "速度(km/h)",
    # 衍生特徵
    "ml_budget_remaining": "預算剩餘",
    "ml_budget_use_ratio": "預算使用比",
    "ml_time_cost_score": "時間成本分",
    "ml_walking_cost_score": "步行成本分",
    "ml_money_cost_score": "金錢成本分",
    # 交通工具 one-hot
    "transport_tools_norm_BICYCLE": "交通:自行車",
    "transport_tools_norm_BUS": "交通:公車/客運",
    "transport_tools_norm_BUS|HIGH SPEED TRAIN": "交通:公車+高鐵",
    "transport_tools_norm_BUS|HIGH SPEED TRAIN|SUBWAY": "交通:公車+高鐵+捷運",
    "transport_tools_norm_BUS|LONG DISTANCE TRAIN": "交通:公車+台鐵",
    "transport_tools_norm_BUS|LONG DISTANCE TRAIN|SUBWAY": "交通:公車+台鐵+捷運",
    "transport_tools_norm_BUS|SUBWAY": "交通:公車+捷運",
    "transport_tools_norm_CAR|CAR_12": "交通:租車(1-2人)",
    "transport_tools_norm_CAR|CAR_24": "交通:租車(2-4人)",
    "transport_tools_norm_CAR|CAR_35": "交通:租車(3-5人)",
    "transport_tools_norm_CAR|CAR_57": "交通:租車(5-7人)",
    "transport_tools_norm_HIGH SPEED TRAIN|LONG DISTANCE TRAIN|SUBWAY": "交通:高鐵+台鐵+捷運",
    "transport_tools_norm_LONG DISTANCE TRAIN": "交通:台鐵",
    "transport_tools_norm_LONG DISTANCE TRAIN|SUBWAY": "交通:台鐵+捷運",
    "transport_tools_norm_SUBWAY": "交通:捷運",
    "transport_tools_norm_WALKING": "交通:步行",
}


def feature_contributions(best: dict) -> list[dict]:
    """
    計算最佳方案各維度對加權總分的貢獻 = weight × score。

    三項貢獻相加 = best["total_score"]，與 compute_scores 完全一致。

    Returns
    -------
    list[dict]
        依貢獻由大到小排序，每項含 dim / dim_zh / contribution / share。
    """
    weights = best["weights"]
    items = []
    for dim in ("cost", "time", "convenience"):
        contribution = weights[dim] * best[f"{dim}_score"]
        items.append({
            "dim": dim,
            "dim_zh": _DIM_ZH[dim],
            "weight": round(weights[dim], 4),
            "score": round(best[f"{dim}_score"], 4),
            "contribution": round(contribution, 4),
        })

    total = sum(i["contribution"] for i in items) or 1.0
    for i in items:
        i["share"] = round(i["contribution"] / total, 4)

    items.sort(key=lambda x: x["contribution"], reverse=True)
    return items


def _top_reason(best: dict, contributions: list[dict]) -> str:
    """由貢獻分解產生中文推薦原因。"""
    top = contributions[0]
    return (
        f"在您目前的偏好下，**{top['dim_zh']}** 的權重為 {top['weight']*100:.0f}%，"
        f"且 **{best['mode']}** 的{top['dim_zh']}分數達 {top['score']:.2f}，"
        f"貢獻最大（佔綜合評分 {top['share']*100:.0f}%），"
        f"因此獲得最高的綜合評分 {best['total_score']:.2f}。"
    )


def _feature_row_to_X(feature_row: dict) -> pd.DataFrame:
    """將特徵 dict 依 FEATURE_COLS 順序轉成模型輸入（具名欄位，消除 sklearn feature-name 警告）。"""
    return pd.DataFrame([[feature_row[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)


def decision_path_zh(clf, feature_row: dict) -> str | None:
    """
    將決策樹對單一樣本的 decision path 轉成中文 IF-THEN 說明。

    Parameters
    ----------
    clf : DecisionTreeClassifier
    feature_row : dict
        鍵需涵蓋 FEATURE_COLS。

    Returns
    -------
    str | None
        例如「若 時間權重 > 6 且 預算 ≤ 1500，則推薦 高鐵」。
    """
    if clf is None:
        return None

    X = _feature_row_to_X(feature_row)
    tree = clf.tree_
    node_indicator = clf.decision_path(X)
    leaf_id = clf.apply(X)[0]
    node_index = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]

    conditions = []
    for node_id in node_index:
        if node_id == leaf_id:
            continue
        feat_name = FEATURE_COLS[tree.feature[node_id]]
        feat_zh = _FEATURE_ZH.get(feat_name, feat_name)
        threshold = tree.threshold[node_id]
        if X.iloc[0, tree.feature[node_id]] <= threshold:
            conditions.append(f"{feat_zh} ≤ {threshold:.0f}")
        else:
            conditions.append(f"{feat_zh} > {threshold:.0f}")

    pred = clf.classes_[np.argmax(tree.value[leaf_id][0])]
    pred_zh = MODE_EN_TO_ZH.get(pred, pred)

    if not conditions:
        return f"模型直接推薦 {pred_zh}（無分裂條件）。"
    return "若 " + " 且 ".join(conditions) + f"，則推薦 {pred_zh}。"


def shap_contributions(clf, feature_row: dict) -> dict | None:
    """
    對單一樣本計算決策樹的 SHAP values，回傳各特徵對預測類別的推力。

    缺少 shap 套件時回傳 None，讓 UI 優雅降級。

    Returns
    -------
    dict | None
        {
          "pred_zh": str,
          "values": [{"feature": str, "feature_zh": str, "value": float}, ...]
        }
        values 依絕對推力由大到小排序。
    """
    try:
        import shap
    except ImportError:
        return None

    if clf is None:
        return None

    X = _feature_row_to_X(feature_row)
    pred = clf.predict(X)[0]
    class_idx = list(clf.classes_).index(pred)

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X)

    # 多類別時 shap_values 為 list（每類一個陣列）或 3D 陣列
    if isinstance(sv, list):
        row = np.asarray(sv[class_idx])[0]
    else:
        sv = np.asarray(sv)
        row = sv[0, :, class_idx] if sv.ndim == 3 else sv[0]

    values = [
        {
            "feature": FEATURE_COLS[i],
            "feature_zh": _FEATURE_ZH.get(FEATURE_COLS[i], FEATURE_COLS[i]),
            "value": round(float(row[i]), 4),
        }
        for i in range(len(FEATURE_COLS))
    ]
    values.sort(key=lambda x: abs(x["value"]), reverse=True)

    return {"pred_zh": MODE_EN_TO_ZH.get(pred, pred), "values": values}


def counterfactuals(
    features: list[dict],
    w_cost: float,
    w_time: float,
    w_conv: float,
    budget: float,
) -> list[str]:
    """
    產生反事實提示，涵蓋預算門檻與偏好權重兩類可變條件。

    只取使 rank #1 翻轉的第一個臨界點，避免冗長。
    """
    tips: list[str] = []
    results = compute_scores(features, w_cost, w_time, w_conv)
    best_mode = results[0]["mode"]
    best_cost = results[0]["cost_raw"]

    # ── 預算門檻 ──
    # 找出比目前最佳便宜的替代方案；若預算下調至略低於最佳方案成本，
    # 最佳方案會超出預算而被排除，推薦改為較便宜的方案。
    cheaper = [r for r in results[1:] if r["cost_raw"] < best_cost]
    if cheaper:
        alt = min(cheaper, key=lambda r: r["cost_raw"])
        threshold = int(best_cost) - 1
        if threshold >= alt["cost_raw"]:
            tips.append(
                f"💸 若預算降至 ${threshold}/人 以下，"
                f"目前最佳的 **{best_mode}**（${best_cost:.0f}/人）將超出預算，"
                f"推薦可能改為 **{alt['mode']}**（${alt['cost_raw']:.0f}/人）。"
            )

    # ── 偏好權重 ──
    # 對三個權重各做網格掃描，找最小變動使 rank #1 翻轉。
    flip = _weight_flip(features, w_cost, w_time, w_conv, best_mode)
    if flip:
        tips.append(flip)

    return tips


def _weight_flip(features, w_cost, w_time, w_conv, best_mode) -> str | None:
    """掃描單一權重的增幅，找最小變動使最佳方案翻轉。"""
    base = {"cost": w_cost, "time": w_time, "convenience": w_conv}
    best_flip = None  # (delta, dim, new_mode)

    for dim in ("cost", "time", "convenience"):
        for delta in (0.1, 0.2, 0.3, 0.5):
            trial = dict(base)
            trial[dim] = base[dim] * (1 + delta)
            res = compute_scores(
                features, trial["cost"], trial["time"], trial["convenience"]
            )
            if res[0]["mode"] != best_mode:
                if best_flip is None or delta < best_flip[0]:
                    best_flip = (delta, dim, res[0]["mode"])
                break  # 同維度只取最小翻轉幅度

    if best_flip is None:
        return None

    delta, dim, new_mode = best_flip
    return (
        f"⚖️ 若 **{_DIM_ZH[dim]}** 權重再提高約 {delta*100:.0f}%，"
        f"推薦可能改為 **{new_mode}**。"
    )


def build_report(
    results: list[dict],
    features: list[dict],
    feature_row: dict,
    w_cost: float,
    w_time: float,
    w_conv: float,
    budget: float,
    clf=None,
    best: dict | None = None,
) -> dict:
    """
    統一封裝 XAI 解釋。

    Parameters
    ----------
    results : list[dict]
        compute_scores 的輸出（已排序）。
    features : list[dict]
        compute_features 的輸出（反事實重算用）。
    feature_row : dict
        決策樹輸入，鍵涵蓋 FEATURE_COLS。
    w_cost, w_time, w_conv, budget : float
        使用者原始偏好與預算。
    clf : optional
        決策樹模型；None 時自動載入儲存的模型。
    best : dict, optional
        指定要解釋的方案（compute_scores 的某一列）；
        None 時預設解釋排名第一名 results[0]。

    Returns
    -------
    dict
        {top_reason, contributions, decision_path, shap, counterfactuals}
    """
    if clf is None:
        clf = load_trained_model()

    best = best if best is not None else results[0]
    contributions = feature_contributions(best)

    return {
        "top_reason": _top_reason(best, contributions),
        "contributions": contributions,
        "decision_path": decision_path_zh(clf, feature_row) if feature_row else None,
        "shap": shap_contributions(clf, feature_row) if feature_row else None,
        "counterfactuals": counterfactuals(features, w_cost, w_time, w_conv, budget),
    }

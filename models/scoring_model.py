"""
M4 · Scoring Model — 加權評分模型

功能：
- 使用加權評分公式計算各交通方式的總分
- 根據使用者的成本、時間、便利性偏好權重計算
- 回傳排序後的推薦結果

評分公式：
    total_score = w_cost × cost_score + w_time × time_score + w_conv × convenience_score

說明：
- 輸入的 features 由 features.feature_engineer.compute_features() 產生，
  其中 cost_score / time_score / convenience_score 已完成 Min-Max 正規化，
  且皆為「越高越好」（成本、時間已反轉），因此此處直接加權即可，不需再反轉。
- 便利性調整、正規化等邏輯一律由 feature_engineer 負責，本模組不重複實作。
"""


def get_weights(w_cost: float, w_time: float, w_conv: float) -> dict:
    """
    將使用者偏好權重正規化為總和 1.0。

    若三者皆為 0（或總和 ≤ 0），fallback 為等權重，避免除以零產生 NaN。

    Returns
    -------
    dict
        {"cost": ..., "time": ..., "convenience": ...}，三者加總為 1.0
    """
    total = w_cost + w_time + w_conv
    if total <= 0:
        return {"cost": 1 / 3, "time": 1 / 3, "convenience": 1 / 3}
    return {
        "cost": w_cost / total,
        "time": w_time / total,
        "convenience": w_conv / total,
    }


def compute_scores(
    features: list,
    w_cost: float = 1,
    w_time: float = 1,
    w_conv: float = 1,
    budget: float = None,
) -> list:
    """
    計算各交通方式的加權總分，並依分數由高到低排序回傳。

    Parameters
    ----------
    features : list[dict]
        compute_features() 的輸出，每筆需含 cost_score / time_score /
        convenience_score（已正規化）與 cost_raw / time_raw / convenience_raw。
    w_cost, w_time, w_conv : float
        使用者偏好權重（任意非負數，會自動正規化；全為 0 時 fallback 等權重）。
    budget : float, optional
        每人預算上限。若提供，僅保留 cost_raw ≤ budget 的方案；
        若全數超出預算則回傳空 list。預設 None（不過濾）。

    Returns
    -------
    list[dict]
        依 total_score 降序排列，每筆含：
        mode, rank, cost_raw, time_raw, convenience_raw,
        cost_score, time_score, convenience_score, total_score,
        weights（正規化後權重 dict），以及透傳的 segments / transfers。
    """
    if not features:
        return []

    weights = get_weights(w_cost, w_time, w_conv)

    # 預算過濾（cost_raw 已是每人成本：租車費用在 compute_cost_feature 已除以人數）
    pool = features
    if budget is not None:
        pool = [f for f in features if f["cost_raw"] <= budget]
        if not pool:
            return []

    results = []
    for f in pool:
        # 三項分數皆已正規化且「越高越好」，直接加權求和
        total_score = (
            weights["cost"] * f["cost_score"]
            + weights["time"] * f["time_score"]
            + weights["convenience"] * f["convenience_score"]
        )
        results.append({
            "mode": f["mode"],
            "cost_raw": f["cost_raw"],
            "time_raw": f["time_raw"],
            "convenience_raw": f["convenience_raw"],
            "cost_score": f["cost_score"],
            "time_score": f["time_score"],
            "convenience_score": f["convenience_score"],
            "total_score": total_score,
            "weights": weights,
            # 透傳給 UI 渲染分段路線與比較表
            "segments": f.get("segments", []),
            "transfers": f.get("transfers", 0),
        })

    # 依總分降序排序並重新編號
    results.sort(key=lambda r: r["total_score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return results

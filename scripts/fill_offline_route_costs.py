"""
離線備援路由表產生腳本

來源：data/static/taiwan_route_dataset.csv（2136 筆景點層級路由，cost 全空）。
用途：當 ENABLE_REALTIME=false 或 geocoding 失敗時，plan_routes 會 fallback 到
      api_client.get_route_data()。城市層級的 fare_table.csv 不含 UI 的 10 個景點，
      故需要一份「景點層級」備援表。

本腳本依 mode 用與 services/route_planner.py 一致的估算公式補上 cost，
並 normalize 成 get_route_data() 的回傳 schema，輸出：
      data/static/offline_route_table.csv

欄位：origin, destination, mode, cost, duration_min, distance_km,
      transfers, frequency_per_day, convenience

執行：python scripts/fill_offline_route_costs.py
"""

import math
import os

import pandas as pd

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "static")
_SRC = os.path.join(_STATIC_DIR, "taiwan_route_dataset.csv")
_OUT = os.path.join(_STATIC_DIR, "offline_route_table.csv")

# 資料集 mode（Google 風格）→ 系統模式（中文）。
# walking / bicycling 為接駁性質，不單獨列為離線方案。
_MODE_MAP = {
    "driving": "租車",
    "transit": "大眾運輸",
}

# 租車估算（沿用 vehicle_params.json 最便宜車型與 route_planner 計價邏輯，離線略過高速里程）
_CAR_HOURLY = 250
_CAR_MILEAGE_PER_KM = 3.4


def _estimate_cost(mode_zh: str, distance_km: float, time_min: float) -> int:
    """依模式估算成本（與 route_planner 係數一致）。"""
    if mode_zh == "租車":
        rental_hours = max(1, math.ceil(time_min / 60))
        return int(round(_CAR_HOURLY * rental_hours + distance_km * _CAR_MILEAGE_PER_KM))
    # 大眾運輸：距離為基礎的粗估（捷運/公車混合），下限 20 元
    return max(20, int(round(distance_km * 3)))


def build_offline_table() -> pd.DataFrame:
    df = pd.read_csv(_SRC)
    df["mode_zh"] = df["mode"].map(_MODE_MAP)
    df = df[df["mode_zh"].notna()].copy()

    rows: list[dict] = []
    # 每個 (origin, destination, 系統模式) 取耗時最短者作代表
    grouped = df.sort_values("time_min").groupby(
        ["origin", "destination", "mode_zh"], as_index=False
    ).first()

    for _, r in grouped.iterrows():
        mode_zh = r["mode_zh"]
        distance_km = round(float(r["distance_km"]), 2)
        duration_min = int(round(float(r["time_min"])))
        transfers = int(r["transfer_count"]) if not pd.isna(r["transfer_count"]) else 0
        cost = _estimate_cost(mode_zh, distance_km, duration_min)

        if mode_zh == "租車":
            convenience, freq = 7.0, 999
        else:  # 大眾運輸
            convenience, freq = round(6.0 - min(transfers, 3), 1), 80

        rows.append({
            "origin": r["origin"],
            "destination": r["destination"],
            "mode": mode_zh,
            "cost": cost,
            "duration_min": duration_min,
            "distance_km": distance_km,
            "transfers": transfers,
            "frequency_per_day": freq,
            "convenience": convenience,
        })

    return pd.DataFrame(rows)


def main() -> None:
    out = build_offline_table()
    out.to_csv(_OUT, index=False, encoding="utf-8")
    print(f"已產生離線備援表：{_OUT}")
    print(f"  路由筆數：{len(out)}")
    print(f"  涵蓋景點對：{out.groupby(['origin', 'destination']).ngroups}")
    print(f"  模式分布：{out['mode'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

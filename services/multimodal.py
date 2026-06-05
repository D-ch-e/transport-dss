"""
services.multimodal — 多式聯運組合

D5 範圍：在既有單模式 RouteOption 之外，**額外**生成「主幹線 + YouBike 末端接駁」組合。

策略：
- 對於已生成的 高鐵 / 台鐵 / 客運 三種主幹線方案，檢查兩端景點與樞紐站間的距離；
- 若距離 > 步行可接受值（預設 800m）且該端有 YouBike 站可借/可還，
  則生成一個「主幹線 + YouBike」變體，計算新的總時間/成本/便利性。

每個 multimodal option 仍是 RouteOption 同一個 schema，方便 feature_engineer 直接吃。
"""

from __future__ import annotations

import logging
from copy import deepcopy

from data.clients.youbike_client import YouBikeClient, _haversine_m
from services.route_planner import (
    ATTRACTION_NEAR_HUBS,
    HUB_TO_ATTRACTION_TRANSFER_MIN,
    resolve_location,
)

logger = logging.getLogger(__name__)


# ── 樞紐站座標（簡表，避免又打 geocoding） ────────────────────
# 之後若擴大可從 TDX Station endpoint 抓並快取
HUB_COORDS: dict[str, dict[str, tuple[float, float]]] = {
    "THSR": {
        "台北": (25.0478, 121.5170),
        "桃園": (25.0129, 121.2151),
        "新竹": (24.8083, 121.0405),
        "台中": (24.1124, 120.6151),
        "嘉義": (23.4596, 120.3275),
        "台南": (22.9249, 120.2854),
        "左營": (22.6873, 120.3076),
    },
    "TRA": {
        "台北": (25.0478, 121.5170),
        "板橋": (25.0136, 121.4636),
        "桃園": (24.9893, 121.3144),
        "中壢": (24.9532, 121.2253),
        "新竹": (24.8019, 120.9716),
        "基隆": (25.1320, 121.7400),
        "松山": (25.0500, 121.5779),
        "南港": (25.0532, 121.6066),
        "萬華": (25.0335, 121.5004),
    },
}

# 騎乘速度（YouBike 都市混合車流，含等紅燈）
YOUBIKE_SPEED_KMH = 12.0
WALK_OK_RADIUS_M = 800.0       # 低於此距離不啟動 YouBike 替代步行
YOUBIKE_MAX_RADIUS_M = 4500.0  # 超過此距離騎乘不合理
YOUBIKE_BASE_FEE = 10          # 前 30 分免費，之後每 30 分 10 元


# ── helpers ──────────────────────────────────────────────────


def _youbike_leg_minutes(distance_m: float) -> int:
    """騎乘時間（含取/還車各 1 分鐘）。"""
    ride = (distance_m / 1000.0) / YOUBIKE_SPEED_KMH * 60
    return max(2, int(round(ride + 2)))


def _youbike_fee(distance_m: float) -> int:
    """估算費用：前 30 分鐘免費，之後每滿 30 分鐘 10 元。"""
    minutes = _youbike_leg_minutes(distance_m)
    if minutes <= 30:
        return 0
    extra_blocks = (minutes - 30 + 29) // 30  # 超過 30 分後，每 30 分一個計費區塊
    return YOUBIKE_BASE_FEE * extra_blocks


def _hub_coord(system: str, hub_name: str) -> tuple[float, float] | None:
    return HUB_COORDS.get(system, {}).get(hub_name)


def _find_youbike_leg(
    yb: YouBikeClient,
    end_point: tuple[float, float],
    *,
    want_rent: bool,
) -> tuple[dict, float] | None:
    """
    在 end_point 附近找最佳 YouBike 站。
    want_rent=True：上車端（要可借）；False：下車端（要可還）。
    回傳 (station_info, distance_m) 或 None。
    """
    lat, lng = end_point
    stations = yb.nearest_stations(
        lat, lng,
        k=1,
        max_radius_m=YOUBIKE_MAX_RADIUS_M,
        require_bikes=want_rent,
    )
    if not stations:
        return None
    s = stations[0]
    # 下車端：需有空位
    if not want_rent and s["available_return_bikes"] < 1:
        return None
    return s, float(s["distance_m"])


# ── 對外 entry：擴增 plan_routes 的結果 ───────────────────────


def augment_with_youbike(
    origin: str,
    destination: str,
    base_options: list[dict],
) -> list[dict]:
    """
    接收 plan_routes 回的單模式方案列表，append 多式聯運變體。
    回傳：原方案 + 多式聯運方案合併後的列表。
    """
    o_loc = resolve_location(origin)
    d_loc = resolve_location(destination)
    if not (o_loc and d_loc):
        return base_options

    yb = YouBikeClient()
    added: list[dict] = []

    for opt in base_options:
        mode = opt["mode"]
        if mode not in {"高鐵", "台鐵"}:
            continue  # 客運/租車不做接駁變體（客運本身已含轉乘）

        system = "THSR" if mode == "高鐵" else "TRA"
        o_hub = ATTRACTION_NEAR_HUBS.get(origin, {}).get(system)
        d_hub = ATTRACTION_NEAR_HUBS.get(destination, {}).get(system)
        if not (o_hub and d_hub):
            continue

        o_hub_pt = _hub_coord(system, o_hub)
        d_hub_pt = _hub_coord(system, d_hub)
        if not (o_hub_pt and d_hub_pt):
            continue

        # 起點 → 起站
        o_walk_m = _haversine_m(o_loc["lat"], o_loc["lng"], *o_hub_pt)
        # 終站 → 終點
        d_walk_m = _haversine_m(*d_hub_pt, d_loc["lat"], d_loc["lng"])

        # 至少一端的步行距離夠長才值得接 YouBike；且不能超過合理騎乘上限
        if max(o_walk_m, d_walk_m) < WALK_OK_RADIUS_M:
            continue

        leg_o = leg_d = None
        if WALK_OK_RADIUS_M <= o_walk_m <= YOUBIKE_MAX_RADIUS_M:
            r = _find_youbike_leg(yb, o_hub_pt, want_rent=True)
            if r:
                dist = _haversine_m(o_loc["lat"], o_loc["lng"], r[0]["lat"], r[0]["lng"])
                if dist <= YOUBIKE_MAX_RADIUS_M:
                    leg_o = (r[0], dist)
        if WALK_OK_RADIUS_M <= d_walk_m <= YOUBIKE_MAX_RADIUS_M:
            r = _find_youbike_leg(yb, d_hub_pt, want_rent=False)
            if r:
                dist = _haversine_m(r[0]["lat"], r[0]["lng"], d_loc["lat"], d_loc["lng"])
                if dist <= YOUBIKE_MAX_RADIUS_M:
                    leg_d = (r[0], dist)

        if not (leg_o or leg_d):
            continue

        # 拷貝原方案，把對應端的「步行/接駁」segment 換成 YouBike
        new_opt = deepcopy(opt)
        new_opt["mode"] = f"{mode}+YouBike"
        new_opt["source"] = opt["source"] + "+youbike"
        new_opt["transfers"] = opt["transfers"] + sum(1 for x in (leg_o, leg_d) if x)

        extra_cost = 0
        extra_time = 0
        # 重建 segments
        segs = []
        # 上車端
        if leg_o:
            st, dist = leg_o
            ride_min = _youbike_leg_minutes(dist)
            yb_fee = _youbike_fee(dist)
            extra_cost += yb_fee
            # 替換掉原本 25 分鐘的接駁
            extra_time += ride_min - HUB_TO_ATTRACTION_TRANSFER_MIN
            segs.append({
                "mode": "YouBike", "from": origin, "to": st["name"],
                "duration_min": ride_min, "distance_km": round(dist / 1000.0, 2),
                "cost": yb_fee,
                "available_bikes": st["available_rent_bikes"],
            })
        else:
            segs.append(opt["segments"][0])
        # 主幹線
        segs.append(opt["segments"][1])
        # 下車端
        if leg_d:
            st, dist = leg_d
            ride_min = _youbike_leg_minutes(dist)
            yb_fee = _youbike_fee(dist)
            extra_cost += yb_fee
            extra_time += ride_min - HUB_TO_ATTRACTION_TRANSFER_MIN
            segs.append({
                "mode": "YouBike", "from": st["name"], "to": destination,
                "duration_min": ride_min, "distance_km": round(dist / 1000.0, 2),
                "cost": yb_fee,
                "return_spaces": st["available_return_bikes"],
            })
        else:
            segs.append(opt["segments"][-1])

        new_opt["segments"] = segs
        new_opt["duration_min"] = max(1, opt["duration_min"] + extra_time)
        new_opt["cost"] = opt["cost"] + extra_cost
        # 加 YouBike 通常便利性略降（要取/還車）但比純步行接駁高
        new_opt["convenience"] = round(opt["convenience"] + 0.5, 2)

        added.append(new_opt)

    return base_options + added
